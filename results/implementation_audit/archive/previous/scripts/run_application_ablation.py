"""Run isolated fixed-noise application pilots and a separate noise ablation.

The default configuration is frozen before validation; selected policies and
full validation curves are saved before final test jobs start. All cached jobs
are checksummed against numeric source, versions and complete task input.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import hashlib,json,platform,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from lib.application_ablation import candidate_families,scenarios,validation_job,test_job,choose,paired_summary,noise_ablation
from lib.validation import require_gate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def run_jobs(worker,tasks,directory,signature,workers):
    directory.mkdir(parents=True,exist_ok=True);finished={};pending=[]
    for index,task in enumerate(tasks):
        taskkey=hashlib.sha256((signature+repr(task)).encode()).hexdigest()
        path=directory/f'{index:05d}.json'
        try:
            old=json.loads(path.read_text())
            assert old['key']==taskkey
            assert hashlib.sha256(json.dumps(old['result'],sort_keys=True).encode()).hexdigest()==old['sha256']
            finished[index]=old['result']
        except (OSError,ValueError,AssertionError,KeyError):pending.append((index,task,taskkey,path))
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(worker,task):(index,key,path) for index,task,key,path in pending}
        for future in as_completed(futures):
            index,key,path=futures[future];result=portable(future.result());finished[index]=result
            doc=dict(key=key,result=result,sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest())
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(doc));temporary.replace(path)
            if len(finished)%10==0 or len(finished)==len(tasks):print(directory.name,len(finished),'/',len(tasks),flush=True)
    return [finished[i] for i in range(len(tasks))]


def expansion_candidates(selected):
    """One predeclared outer-grid validation stage; never inspect test losses."""
    extra=[]
    for (app,method),p in selected.items():
        family=candidate_families(app)[method]
        seen={tuple(sorted(q.items())) for q in family}
        def add(q):
            key=tuple(sorted(q.items()))
            if key not in seen:extra.append((app,method,q));seen.add(key)
        if 'beta' in p and p['beta']==min(q['beta'] for q in family):
            if method=='Loaded covariance':
                for loading in sorted({q['loading'] for q in family}):add(dict(beta=.5,loading=loading))
            elif method=='Shrunk PCA':
                for rho in sorted({q['rho'] for q in family}):add(dict(beta=.5,rho=rho))
            else:add(dict(beta=.5))
        if method=='Loaded covariance':
            values=sorted({q['loading'] for q in family})
            if p['loading'] in (values[0],values[-1]):
                loading=values[0]/4 if p['loading']==values[0] else values[-1]*4
                for beta in sorted({q['beta'] for q in family}):add(dict(beta=beta,loading=loading))
        if method=='GROUSE':
            values=sorted(q['step'] for q in family)
            if p['step'] in (values[0],values[-1]):add(dict(step=values[0]/3 if p['step']==values[0] else values[-1]*3))
        if method=='Rolling PCA' and p['window']==min(q['window'] for q in family):add(dict(window=6))
    return extra


def boundary_status(selected,validation,config):
    """Separate structural/full-history endpoints from unresolved grid edges."""
    out={}
    for (app,method),p in selected.items():
        candidates=[r['parameters'] for r in validation if r['application']==app and r['method']==method]
        notes=[];unresolved=False
        horizon=config['T_array'] if app=='array' else config['T_field']
        for key,value in p.items():
            values=sorted({q[key] for q in candidates})
            if value not in (values[0],values[-1]):continue
            if key=='rho' or (key=='beta' and value==1.):notes.append(key+': structural endpoint')
            elif key=='window' and value>=horizon:notes.append('window: full-history saturation at this horizon')
            else:notes.append(key+': unresolved outer validation edge');unresolved=True
        out[app,method]=dict(notes=notes,unresolved=unresolved)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'results/dev_prediction/applications')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--validation-trials',type=int,default=8)
    parser.add_argument('--test-trials',type=int,default=16);parser.add_argument('--draws',type=int,default=1000)
    parser.add_argument('--burnin',type=int,default=500);parser.add_argument('--max-draws',type=int,default=4000)
    parser.add_argument('--max-burnin',type=int,default=2000);parser.add_argument('--development',action='store_true');parser.add_argument('--T',type=int,default=32);parser.add_argument('--field-T',type=int,default=128)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    gate=None if args.development else require_gate(ROOT,gate_path=ROOT/'results/dev_prediction/validation.json')
    config=dict(validation_seed=582031,test_seed=582032,noise_seed=582033,T_array=args.T,T_field=args.field_T,
        validation_trials=args.validation_trials,test_trials=args.test_trials,draws=args.draws,burnin=args.burnin,
        max_draws=args.max_draws,max_burnin=args.max_burnin,
        primary_endpoint='Terminal time, posterior-mean vs scalar-shrinkage task loss in matched drifting identity-frame data',
        array_selection_loss='Mean output power over 256 independent heldout snapshots at terminal time',
        field_selection_loss='Terminal current-observation reconstruction squared error; coefficient/spatial orthonormal basis equivalent',
        motion_access='Identity supplied to every method in all primary and mismatch cases; half-true motion supplied to every method only in separately labelled synthetic partial controls',
        policy='One matched-drift validation policy per application, frozen and reused in all test regimes; no regime-specific test tuning',
        conditional_on='Fixed true measurement variances; original application generators; no noise fits in primary comparisons',
        mean_initial_state_law='Original array deterministic orthonormal interference modes; field Gaussian-QR Haar initial rank-three state, identical across methods',
        statistics='Exploratory pointwise paired Student intervals across independent trajectory identifiers; robustness cases paired interventions, not extra independent trials',
        reference_grids={a:candidate_families(a) for a in ('array','field')},scenarios=scenarios(),
        boundary_rule='Before final testing, one extra validation stage adds beta .5 at a selected .7 boundary; loading divided/multiplied by4 at selected edges, step divided/multiplied by3, window6 at selected12. Structural beta1/rho0or1 and windows beyond horizon are not expanded. Remaining outer edges are unresolved and prohibit positive main-paper claims.',
        runtime='Measured prediction, correction, terminal moment and decision times; no regional optimization; cumulative per trajectory with terminal posterior reporting only',
        hardware=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),workers=args.workers,blas_threads=1),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        review_revision='57f520cbe860b73e13d71bae5c31391e2a009482')
    paths=['lib/application_ablation.py','scripts/run_application_ablation.py','lib/experiments.py','lib/filter.py','lib/em.py','lib/geometry.py','lib/models.py','lib/moments.py','lib/spectral_series.py','benchmarks/case1/baselines.py','benchmarks/case2/baselines.py']
    source={p:sha(ROOT/p) for p in paths}
    signature=hashlib.sha256(json.dumps(dict(config=config,sources=source),sort_keys=True).encode()).hexdigest()
    meta_path=out/'config.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['signature']==signature and all(sha(out/path)==digest for path,digest in meta['tables'].items()):
            meta.setdefault('original_run_core_hash',meta.get('core_hash'))
            meta['revalidated_core_hash']=None if gate is None else gate['core_hash'];meta['development_only']=args.development
            meta_path.write_text(json.dumps(meta,indent=2))
            print('Applications: verified cached tables');return
    (out/'predeclared_config.json').write_text(json.dumps(dict(config=config,sources=source,signature=signature),indent=2))
    tasks=[(config,a,trial,method,p) for a in ('array','field') for method,family in candidate_families(a).items() for p in family for trial in range(args.validation_trials)]
    validation=run_jobs(validation_job,tasks,out/'checkpoints/validation',signature,args.workers)
    initial_selected,_=choose(validation)
    extra=expansion_candidates(initial_selected)
    expansion_tasks=[(config,a,trial,m,p) for a,m,p in extra for trial in range(args.validation_trials)]
    if expansion_tasks:validation+=run_jobs(validation_job,expansion_tasks,out/'checkpoints/validation_expansion',signature,args.workers)
    selected,tuning=choose(validation)
    boundaries=boundary_status(selected,validation,config)
    tuning['boundary_notes']=[str(boundaries[r.application,r.method]['notes']) if r.selected else '' for r in tuning.itertuples()]
    tuning['unresolved_selected_boundary']=[bool(r.selected and boundaries[r.application,r.method]['unresolved']) for r in tuning.itertuples()]
    pd.DataFrame(validation).to_csv(out/'validation_trials.csv',index=False);tuning.to_csv(out/'tuning.csv',index=False)
    frozen=[dict(application=a,method=m,parameters=p) for (a,m),p in selected.items()]
    (out/'selected.json').write_text(json.dumps(dict(signature=signature,selected=frozen,selection_completed_before_final_test=True,boundary_notes=[dict(application=a,method=m,**b) for (a,m),b in boundaries.items()]),indent=2))
    tasks=[(config,s,trial,selected) for s in scenarios() for trial in range(args.test_trials)]
    results=run_jobs(test_job,tasks,out/'checkpoints/test',signature,args.workers)
    metrics=pd.DataFrame([row for group in results for row in group]);metrics.to_csv(out/'metrics.csv',index=False)
    paired=paired_summary(metrics)
    paired['unresolved_tuning_boundary']=[boundaries[r.application,'Posterior mean']['unresolved'] or boundaries[r.application,'Posterior mean' if r.reference=='Mode' else r.reference]['unresolved'] for r in paired.itertuples()]
    paired.loc[paired.unresolved_tuning_boundary & (paired.gain_low>0),'interpretation']='Positive pilot interval; main-paper claim withheld because tuning boundary remains unresolved'
    selected_validation_failures=tuning[(tuning.method=='Posterior mean')&tuning.selected].set_index('application').sampling_failures.to_dict()
    paired['selected_validation_sampling_failures']=paired.application.map(selected_validation_failures)
    paired.loc[(paired.posterior_sampling_failures>0)|(paired.selected_validation_sampling_failures>0),'interpretation']='Numerically unresolved: posterior sampling diagnostics failed in selected validation or test targets'
    paired.to_csv(out/'paired.csv',index=False)
    tails=metrics.groupby(['application','scenario','method']).agg(mean_loss=('loss','mean'),loss_q90=('loss',lambda x:np.quantile(x,.9)),mean_cumulative_seconds=('cumulative_seconds','mean')).reset_index()
    array=metrics[metrics.application=='array'].groupby(['scenario','method']).sinr_db.quantile(.1).rename('sinr_q10').reset_index()
    array['application']='array'
    tails=tails.merge(array,how='left',on=['application','scenario','method']);tails.to_csv(out/'tails_cost.csv',index=False)
    beta=float(selected['field','Posterior mean']['beta'])
    noise=noise_ablation(seed=config['noise_seed'],beta=beta);noise.to_csv(out/'noise.csv',index=False)
    table_names=['validation_trials.csv','tuning.csv','metrics.csv','paired.csv','tails_cost.csv','noise.csv','selected.json','predeclared_config.json']
    flags=metrics[metrics.method=='Posterior mean'].sampling_ok.astype(bool)
    checks=dict(posterior_targets=len(flags),posterior_sampling_failures=int((~flags).sum()),
        validation_posterior_failures=int((~pd.DataFrame(validation).query("method == 'Posterior mean'").sampling_ok.astype(bool)).sum()),
        maximum_matching_ew_modal_distance=float(metrics.ew_modal_distance.max()),
        selected_boundaries=[dict(application=a,method=m,parameters=selected[a,m],**b) for (a,m),b in boundaries.items() if b['notes']],
        boundary_expansion_candidates=len(extra),boundary_expansion_stage='Validation only; selected.json saved before test jobs',
        frozen_window_noise_ascent_min=float(noise[noise.frozen_ascent_checked].frozen_objective_change.min()),
        state_space_ascent_claim=False,
        status='Bounded exploratory pilot; unsupported calibrated-coverage or universal-performance claims')
    meta=dict(config=config,sources=source,signature=signature,core_hash=None if gate is None else gate['core_hash'],development_only=args.development,tables={p:sha(out/p) for p in table_names},checks=checks)
    meta_path.write_text(json.dumps(portable(meta),indent=2));print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
