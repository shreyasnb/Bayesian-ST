"""Replay an independent EW optimization audit; timing needs explicit isolation."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from lib.beamforming_objective_audit import (INNER_STEPS,load_context,replay_trial,
    summarize,paired_summary,validation_scores,operation_counts,timed_workload,sha)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def declaration(context):
    return dict(cohort=context['cohort'],sigma2=context['config']['sigma2'],tau2=context['config']['tau2'],
        rank=1,initial_parameter='A_0=0',transport='G_t=I_n',
        validation_seed=context['config']['validation_seed'],test_seed=context['config']['test_seed'],cases=[{key:value for key,value in case.items() if key!='policy'} for case in context['cases']],
        inner_steps=list(INNER_STEPS),original_inner_steps=3,original_fixed_stepsizes={case['case']:case['policy']['great_ew']['stepsize'] for case in context['cases']},
        covariance_definition='C_t=sum beta^(t-j)*(eta_j/eta_0)*u_j*u_j^H=A_t_uncentered/eta_0; gamma=sqrt(beta)',
        objective='L_t(x)=tr(C_t)-x^H C_t x; exact minimum tr(C_t)-lambda_max(C_t)',
        relative_gap_definition='(lambda_max-x^H C_t x)/(lambda_max-lambda_min)',
        selection_criterion=('Final t=200 measured log output power, averaged over64 validation simulations'
            if context['cohort']=='stationary' else
            'For each validation simulation, arithmetic mean of measured log output power at all32 reports, then average64 simulations; no true covariance or oracle normalization'),
        stationary_primary_evaluation='t=200 only; four report times retained solely for objective diagnostics',
        test_policy='Retain original three-step baseline and independently selected count; exact common-EW mode reference',
        source_hashes=context['source_hashes'],input_hashes=context['input_hashes'],
        validation_simulations=64,test_simulations=256,wall_time_claims=False)


def checkpoint(context,case,trial,split,counts,folder,decl_hash):
    stem=split+'/'+case['case']+'/'+str(trial).zfill(3)
    csv=folder/'checkpoints'/(stem+'.csv');record=csv.with_suffix('.json')
    expected=dict(declaration_sha256=decl_hash,case=case['case'],trial=trial,split=split,inner_steps=counts)
    if record.exists():
        old=json.loads(record.read_text())
        if old['task']!=expected or sha(csv)!=old['csv_sha256']:
            raise RuntimeError('Preserve obsolete or changed audit checkpoint: '+str(record))
        return pd.read_csv(csv),old['checks']
    frame,checks=replay_trial(context,case,trial,split,counts)
    csv.parent.mkdir(parents=True,exist_ok=True)
    temporary=csv.with_suffix('.csv.tmp');frame.to_csv(temporary,index=False);temporary.replace(csv)
    write(record,dict(task=expected,csv_sha256=sha(csv),checks=checks))
    return frame,checks


def run(context,preflight=False):
    folder=ROOT/'results/beamforming_comparison_audit'/context['cohort']/'objective'
    folder.mkdir(parents=True,exist_ok=True)
    decl=declaration(context);decl_hash=signature(decl)
    if preflight:
        checks=[];frames=[]
        for case in context['cases']:
            for split in ['validation','test']:
                frame,check=replay_trial(context,case,0,split,[3]);frames.append(frame);checks.append(check)
        frame=pd.concat(frames,ignore_index=True);frame.to_csv(folder/'preflight_checkpoints.csv',index=False)
        summary=frame[frame.method.eq('great_ew')].groupby(['condition','n']).agg(
            maximum_relative_gap=('relative_objective_gap','max'),maximum_chordal=('chordal_to_mode','max'),
            maximum_absolute_gap=('objective_gap','max')).reset_index()
        report=dict(status='passed',declaration=decl,checks=checks,summary=summary.to_dict('records'))
        write(folder/'preflight.json',report);print(json.dumps(report['summary'],indent=2),flush=True);return report
    registered=folder/'preregistered.json'
    if registered.exists() and json.loads(registered.read_text())!=decl:
        raise RuntimeError('Preserve the prior declaration and use a new audit directory')
    write(registered,decl)
    manifest=folder/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['declaration_sha256']!=decl_hash:raise RuntimeError('Completed objective audit has another declaration')
        for name,digest in old['artifacts'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Changed completed objective artifact: '+name)
        print('Verified existing completed objective audit',context['cohort'],flush=True);return old
    frames=[];checks=[];selection={};old_selection_checks={}
    for case in context['cases']:
        parts=[]
        for trial in range(64):
            frame,check=checkpoint(context,case,trial,'validation',list(INNER_STEPS),folder,decl_hash)
            parts.append(frame);checks.append(check)
            if (trial+1)%16==0:print('Validation',case['case'],trial+1,'/64',flush=True)
        frame=pd.concat(parts,ignore_index=True);frames.append(frame)
        means=validation_scores(frame,case)
        selected=int(means.idxmin())
        selection[case['case']]=dict(selected_inner_steps=selected,original_inner_steps=3,
            beta=case['beta'],stepsize=case['policy']['great_ew']['stepsize'],
            measured_validation_db_by_inner_steps={str(int(k)):float(v) for k,v in means.items()},
            validation_selection_times=[200] if case['stationary'] else case['times'],
            selected_by=('Measured validation log-power at t=200 only, all64 simulations' if case['stationary']
                         else 'Measured validation log-power at all32 reports, all64 simulations'))
        baseline=frame[frame.method.eq('great_ew')&frame.inner_steps.eq(3)&frame.t.isin(case['policy']['primary_times'])].measured_power.mean()
        curves=context['validation_curves'];p=case['policy']['great_ew']
        match=curves[curves.n.eq(case['n'])&curves.condition.eq(case['condition'])
            &curves.stationary.eq(case['stationary'])&curves.family.eq('great_ew')
            &curves.inner_steps.eq(3)&np.isclose(curves.beta,case['beta'])
            &np.isclose(curves.parameter,p['stepsize'],rtol=1e-12,atol=0)]
        if len(match)!=1 or not np.isclose(baseline,float(match.iloc[0].loss),rtol=2e-11,atol=2e-12):
            raise RuntimeError('Original measured validation criterion failed replay: '+case['case'])
        old_selection_checks[case['case']]=dict(replayed_measured_power=float(baseline),saved=float(match.iloc[0].loss))
    validation=pd.concat(frames,ignore_index=True);validation.to_csv(folder/'validation_checkpoints.csv',index=False)
    summarize(validation).to_csv(folder/'validation_summary.csv',index=False)
    write(folder/'selected.json',selection)
    tests=[]
    for case in context['cases']:
        counts=sorted({3,selection[case['case']]['selected_inner_steps']})
        for trial in range(256):
            frame,check=checkpoint(context,case,trial,'test',counts,folder,decl_hash)
            tests.append(frame);checks.append(check)
            if (trial+1)%32==0:print('Test',case['case'],trial+1,'/256',flush=True)
    test=pd.concat(tests,ignore_index=True);test.to_csv(folder/'test_checkpoints.csv',index=False)
    summarize(test).to_csv(folder/'test_summary.csv',index=False)
    paired_summary(test,context['cohort']).to_csv(folder/'paired_summary.csv',index=False)
    operation_counts(context,selection).to_csv(folder/'operation_counts.csv',index=False)
    numeric_checks=dict(status='passed',trial_checks=checks,original_validation_criterion_replay=old_selection_checks,
        maximum_covariance_identity_relative_error=max(x['maximum_statistics_match_relative_error'] for x in checks),
        maximum_original_great_ew_db_error=max(x['maximum_original_great_ew_db_error'] for x in checks),
        maximum_original_mode_db_error=max(x['maximum_original_mode_db_error'] for x in checks),
        selection_never_uses_true_covariance=True,timing_performed=False,
        all_original_trials_present=True,original_files_not_modified=True)
    write(folder/'checks.json',numeric_checks)
    for mapping in (decl['source_hashes'],decl['input_hashes']):
        for name,digest in mapping.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Objective audit source/input changed during the run: '+name)
    artifacts={str(path.relative_to(ROOT)):sha(path) for path in sorted(folder.glob('*'))
        if path.is_file() and path.name!='manifest.json'}
    result=dict(status='completed',example='beamforming_comparison_audit',cohort=context['cohort'],
        config=decl,declaration_sha256=decl_hash,source_hashes=decl['source_hashes'],
        input_hashes=decl['input_hashes'],artifacts=artifacts,checks=numeric_checks,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent replay and additional optimization iterations only; all old selected outputs remain immutable')
    write(manifest,result);print('Completed objective audit',context['cohort'],flush=True);return result


def timing(context,isolation_confirmed=False):
    if not isolation_confirmed:raise RuntimeError('Timing requires the parent-confirmed isolated phase')
    folder=ROOT/'results/beamforming_comparison_audit'/context['cohort']/'objective'
    objective=json.loads((folder/'manifest.json').read_text());selected=json.loads((folder/'selected.json').read_text())
    output=folder/'timing';output.mkdir(exist_ok=True)
    protocol=dict(warmups=1,repetitions=7,seed=2026111101,blas_threads=1,
        isolation_confirmed=True,workload='All measurement updates and task decisions at the declared report indices',
        setup='Observations/scaling prepared before clock; tracker initialization and phase instrumentation included',
        warmup_trajectory='Original validation simulation0',timed_trajectory='Original test simulation0',
        phase_units='tracking_ms includes covariance+SGD or covariance+report-time EVD; decision_ms includes32 moving or4 stationary beamformer decisions',
        exact_mode='One covariance update per measurement and eigensolve only at task reporting indices',
        excluded='Stream generation, heldout/true scoring, objective and exact-reference diagnostics',
        objective_manifest_sha256=sha(folder/'manifest.json'))
    write(output/'preregistered.json',protocol);rows=[];rng=np.random.default_rng(protocol['seed'])
    for case in context['cases']:
        counts=sorted({0,3,int(selected[case['case']]['selected_inner_steps'])})
        for count in counts:timed_workload(context,case,count,split='validation')
        for repetition in range(7):
            for count in rng.permutation(counts):
                elapsed=timed_workload(context,case,int(count))
                rows.append(dict(condition=case['condition'],n=case['n'],T=case['T'],
                    method='mode_exact' if count==0 else 'great_ew',inner_steps=int(count),
                    repetition=repetition,**elapsed))
    raw=pd.DataFrame(rows);raw.to_csv(output/'raw.csv',index=False)
    summary=raw.groupby(['condition','n','T','method','inner_steps']).total_ms.agg(
        repetitions='size',median_ms='median',p90_ms=lambda x:x.quantile(.9)).reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in output.glob('*') if p.is_file() and p.name!='manifest.json'}
    write(output/'manifest.json',dict(status='completed',protocol=protocol,artifacts=artifacts,
        source_hashes=context['source_hashes'],input_hashes=context['input_hashes']))
    print('Completed isolated timing',context['cohort'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort','--scope',choices=['moving','stationary'],default='moving')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--timing',action='store_true')
    parser.add_argument('--isolation-confirmed',action='store_true')
    args=parser.parse_args();context=load_context(ROOT,args.cohort)
    if args.timing:timing(context,args.isolation_confirmed)
    else:run(context,args.preflight)

if __name__=='__main__':main()
