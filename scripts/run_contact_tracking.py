#!/usr/bin/env python3
"""Frozen, bounded mode-only contact motion study, isolated from the pilot."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.contact_tracking import (default_config,candidate_specs,validation_task,
    choose_policies,policy_id,test_task,paired_summary,INFORMATION_COLUMNS)


def portable(x):
    if isinstance(x,dict):return {k:portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def write_json(path,value):
    path.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n')


def source_hashes():
    names=['lib/contact_tracking.py','lib/contact_mechanics.py','lib/contact_study.py',
           'lib/prediction.py','lib/models.py','lib/geometry.py','scripts/run_contact_tracking.py',
           'tests/test_contact_tracking.py']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def map_jobs(function,tasks,workers,label):
    output=[];start=perf_counter()
    if workers==1:iterator=map(function,tasks)
    else:
        executor=ProcessPoolExecutor(max_workers=workers);iterator=executor.map(function,tasks,chunksize=1)
    try:
        for i,result in enumerate(iterator,1):
            output.append(result)
            if i%8==0 or i==len(tasks):print(f'{label}: {i}/{len(tasks)} ({perf_counter()-start:.1f}s)',flush=True)
    finally:
        if workers!=1:executor.shutdown()
    return output


def jobs(config,specs,conditions,trials,stage):
    return [(config,condition,trial,specs,stage) for condition in conditions for trial in range(trials)]


def flatten(results):return [row for group in results for row in group]


def expansions(selected,config):
    extra={'plane':[],'curved':[]};notes=[]
    for row in selected:
        scope=row['scope'];family=row['family'];spec=row['specification'];proposed=[]
        if family=='identity' and spec['beta']==min(config['betas']):
            proposed=[dict(spec,beta=b) for b in [.1,.05]]
        elif family=='rolling' and spec['likelihood_window']==max(config['windows']):
            proposed=[dict(spec,likelihood_window=w) for w in [32,48]]
        elif family=='velocity' and spec.get('damping',0)>0:
            if spec['beta']==min(config['velocity_betas']):proposed.extend(dict(spec,beta=b) for b in [.4,.2])
            if spec['beta']==max(config['velocity_betas']):proposed.extend(dict(spec,beta=b) for b in [.98,1.])
            if spec['lag']==min(config['lags']):proposed.append(dict(spec,lag=1))
            if spec['lag']==max(config['lags']):proposed.append(dict(spec,lag=32))
            if spec['damping']==min(config['dampings']):proposed.append(dict(spec,damping=.125))
        if proposed:
            extra[scope].extend(proposed)
            notes.append(dict(scope=scope,family=family,winning_boundary=spec,
                expansion=proposed,rule='One coordinatewise validation-only expansion; no second extension.'))
    return extra,notes


def save_validation(out,rows,curves):
    frame=pd.DataFrame(rows).copy();frame['specification']=frame.specification.map(lambda x:json.dumps(x,sort_keys=True))
    frame.to_csv(out/'validation_trials.csv',index=False)
    curveframe=pd.DataFrame(curves).copy()
    curveframe['specification']=curveframe.specification.map(lambda x:json.dumps(x,sort_keys=True))
    curveframe.to_csv(out/'validation_curves.csv',index=False)


def validate(config,out,workers):
    base=candidate_specs(config,'identity')+candidate_specs(config,'rolling')
    rows=flatten(map_jobs(validation_task,jobs(config,base,config['main_conditions'],config['validation_trials'],'base'),workers,'Identity/window validation'))
    scan=flatten(map_jobs(validation_task,jobs(config,candidate_specs(config,'velocity'),config['curved_conditions'],config['velocity_scan_trials'],'scan'),workers,'Velocity bounded scan'))
    rows.extend(scan)
    scanframe=pd.DataFrame(scan)
    ranking=scanframe.groupby('policy',sort=False).loss.mean().sort_values()
    promising=[scanframe[scanframe.policy==p].specification.iloc[0] for p in ranking.index[:config['refine_candidates']]]
    write_json(out/'velocity_refinement.json',dict(scan_trials=config['velocity_scan_trials'],
        selection='Best three pooled terminal losses on validation only',specifications=promising,
        test_observations_used=False))
    rows.extend(flatten(map_jobs(validation_task,jobs(config,promising,config['curved_conditions'],config['validation_trials'],'refine'),workers,'Velocity full validation')))
    # A stationary plane needs no learned motion. This deliberately static
    # causal velocity control is selected independently of curved conditions.
    static_velocity=[dict(prediction='velocity',beta=1.,lag=2,damping=0.)]
    rows.extend(flatten(map_jobs(validation_task,jobs(config,static_velocity,['plane'],config['validation_trials'],'refine'),workers,'Plane motion control')))
    selected,curves=choose_policies(rows,config);extra,notes=expansions(selected,config)
    write_json(out/'boundary_expansion.json',dict(expansions=notes,test_observations_used=False))
    for scope,conditions in [('plane',['plane']),('curved',config['curved_conditions'])]:
        if extra[scope]:
            rows.extend(flatten(map_jobs(validation_task,jobs(config,extra[scope],conditions,config['validation_trials'],'expansion'),workers,'One boundary expansion '+scope)))
    selected,curves=choose_policies(rows,config)
    best={scope:min([r for r in selected if r['scope']==scope],key=lambda x:x['validation_loss']) for scope in ['plane','curved']}
    sources=source_hashes()
    frozen=dict(config=config,selected=selected,best_per_scope=best,sources=sources,
        test_observations_used=False,selection_completed_before_final_test=True,
        oracle_excluded_from_selection=True,boundary_expansion=notes)
    frozen['source_hash']=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    save_validation(out,rows,curves);write_json(out/'selected.json',frozen)
    write_json(out/'preregistered.json',dict(config=config,sources=sources,selection_hash=frozen['source_hash'],
        test_observations_used=False,selection_completed_before_final_test=True,
        start_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        primary='Noisy small-sphere terminal query MSE; selected practical methods versus original identity beta=.8',
        scoring='Nine within-trajectory query errors averaged; paired intervals across independent trajectories',
        noiseless='Same positions, inputs and update scale sigma**2; only sensor error is removed'))
    return frozen


def findings(out,metrics,paired,selected,config):
    terminal=metrics[(metrics.t==config['T'])&(metrics.noise=='on')]
    means=terminal.groupby(['condition','method'])[['corrected_mse','time_average_mse','forecast_mse','corrected_chordal']].mean().reset_index()
    means.to_csv(out/'summary.csv',index=False)
    rows=['# Causal mean tracking on contact trajectories','',
        'Policies were selected on validation before the fresh held-out seeds were read. Measurement variance is fixed at 1e-6 in noisy and noiseless runs. The true-normal alignment is an unavailable diagnostic and was excluded from tuning.','',
        '| Condition | Policy | Terminal MSE | Time-average MSE | Forecast MSE |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in means.itertuples():rows.append(f'| {r.condition} | {r.method} | {r.corrected_mse:.5g} | {r.time_average_mse:.5g} | {r.forecast_mse:.5g} |')
    rows.extend(['','## What this isolates','',
        'Residual noiseless error under identity memory measures stale-state bias; its sensor likelihood scale was not changed. A causal velocity forecast can reduce or increase that bias, because motion estimated from noisy modes can itself be inaccurate. The oracle shows what exact state alignment could achieve with unavailable geometry, not what a robot can implement.','',
        'Mode equality with independently transported weighted projector least squares is checked under the same forecast and memory. Thus any Stage 1 advantage belongs to forecast/memory choices, not to retaining a posterior distribution.','',
        'The anisotropic-resistance stress scores the actual KKT mobility. It violates the isotropic projector observation law and remains outside the matched model.','',
        'All per-observation forecast/corrected errors, including startup, are in error_paths.npz. trial0_paths.npz retains raw A, priors, nominal rotations, mode projectors and physical trajectories. No confidence interval is a simultaneous-time guarantee.',''])
    (out/'findings.md').write_text('\n'.join(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results/contact_tracking/mean')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    config=default_config()
    if args.validation_only and args.test_only:parser.error('Choose validation or test only')
    if args.test_only:
        selected=json.loads((out/'selected.json').read_text())
        if selected['sources']!=source_hashes():raise RuntimeError('Stage 1 sources changed after policy freezing; rerun validation')
        if selected['config']!=config:raise RuntimeError('Frozen configuration changed')
    else:
        if (out/'metrics.csv').exists():raise RuntimeError('Preserve existing held-out results; choose a new output directory')
        selected=validate(config,out,args.workers)
    if args.validation_only:return
    from lib.contact_uncertainty_validation import require_uncertainty_gate
    gate=require_uncertainty_gate(ROOT)
    write_json(out/'test_started.json',dict(selection_hash=selected['source_hash'],gate=gate,
        source_hashes=source_hashes(),test_seed=config['test_seed']))
    tasks=[(config,condition,trial,selected) for condition in config['main_conditions']+config['stress_conditions'] for trial in range(config['test_trials'])]
    # Stream outputs: keeping each trial's full causal arrays in RAM is needless.
    records=[];diagnostics=[];motions=[];errors={};arrays={};information={};start=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i,result in enumerate(executor.map(test_task,tasks,chunksize=1),1):
            records.extend(result['metrics']);diagnostics.extend(result['diagnostics']);motions.extend(result['motions'])
            errors.update(result['errors']);arrays.update(result['arrays']);information.update(result['information'])
            if i%16==0 or i==len(tasks):print(f'Held-out mode pairs: {i}/{len(tasks)} ({perf_counter()-start:.1f}s)',flush=True)
    metrics=pd.DataFrame(records);diagnostics=pd.DataFrame(diagnostics);motions=pd.DataFrame(motions).fillna(0)
    metrics.to_csv(out/'metrics.csv',index=False);diagnostics.to_csv(out/'information.csv',index=False);motions.to_csv(out/'motion_diagnostics.csv',index=False)
    np.savez_compressed(out/'error_paths.npz',**errors)
    np.savez_compressed(out/'trial0_paths.npz',**arrays)
    for condition in config['main_conditions']+config['stress_conditions']:
        subset={key:value for key,value in information.items() if key.startswith(condition+'__')}
        np.savez_compressed(out/f'information_paths_{condition}.npz',**subset)
    write_json(out/'path_columns.json',dict(error_paths=['forecast_chordal','corrected_chordal','forecast_mse','corrected_mse','forecast_gap','corrected_gap'],information_paths=INFORMATION_COLUMNS))
    paired=paired_summary(metrics,config);paired.to_csv(out/'paired.csv',index=False)
    paired_summary(metrics,config,endpoint='time_average_mse').to_csv(out/'paired_time_average.csv',index=False)
    findings(out,metrics,paired,selected,config)
    checks=dict(ls_parameter_max=float(motions.ls_parameter_error.max()),ls_objective_max=float(motions.ls_objective_error.max()),
        ls_projector_max=float(motions.ls_projector_error.max()),all_variances_fixed=bool(motions.fixed_update_variance.eq(config['sigma']**2).all()),
        mode_reports=len(metrics),independent_trials_per_condition=config['test_trials'],conditions=5,
        source_hashes_unchanged=source_hashes()==selected['sources'],selection_hash=selected['source_hash'],
        wall_seconds=perf_counter()-start,all_errors_finite=all(np.isfinite(x).all() for x in errors.values()))
    write_json(out/'checks.json',checks)
    files=[p for p in out.iterdir() if p.is_file() and p.name!='manifest.json']
    write_json(out/'manifest.json',dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},checks=checks))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
