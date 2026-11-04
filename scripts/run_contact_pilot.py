"""Run the independently tuned mechanical-contact replacement pilot.

Validation and test streams are frozen before execution. Historical outputs
remain unchanged. Every task checkpoint and completed table is source-checked.
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
from lib.contact_study import default_config,validation_task,test_task,select_policies,summaries


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def digest(value):return hashlib.sha256(json.dumps(portable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False));temporary.replace(path)


def jobs(function,tasks,stage,out,key,workers):
    directory=out/'checkpoints'/stage;directory.mkdir(parents=True,exist_ok=True)
    completed={};pending=[]
    for index,task in enumerate(tasks):
        path=directory/f'{index:05d}_{key[:16]}.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old.get('source_hash')==key and old.get('result_hash')==digest(old['result']):
                completed[index]=old['result'];continue
        pending.append((index,task))
    print(stage,'restored',len(completed),'/',len(tasks),flush=True)
    def retain(index,result):
        result=portable(result);completed[index]=result
        write_json(directory/f'{index:05d}_{key[:16]}.json',dict(source_hash=key,result_hash=digest(result),result=result))
        if len(completed)%8==0 or len(completed)==len(tasks):print(stage,'completed',len(completed),'/',len(tasks),flush=True)
    if workers==1:
        for index,task in pending:retain(index,function(task))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):index for index,task in pending}
            for future in as_completed(futures):retain(futures[future],future.result())
    return [completed[index] for index in range(len(tasks))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/contact_pilot')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--validation-trials',type=int,default=16)
    parser.add_argument('--test-trials',type=int,default=64)
    parser.add_argument('--T',type=int,default=600)
    parser.add_argument('--development',action='store_true',help='Explicit ungated small software smoke; no scientific claim')
    args=parser.parse_args()
    if min(args.workers,args.validation_trials,args.test_trials)<1 or args.T<3 or args.T%3:
        parser.error('Positive trial/worker counts and T divisible by three are required')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    config=default_config();config.update(validation_trials=args.validation_trials,test_trials=args.test_trials,T=args.T)
    if args.T!=600:
        if not args.development:parser.error('A changed horizon is a development protocol; use --development and a new directory')
        config['report_times']=sorted({max(3,3*int(t/3)) for t in [args.T/10,args.T*.3,args.T]})
        config['primary_time']=args.T
    gate=None
    if not args.development:
        from lib.contact_validation import require_contact_gate
        gate=require_contact_gate(ROOT)
    paths=['lib/contact_mechanics.py','lib/contact_posterior.py','lib/contact_study.py','scripts/run_contact_pilot.py',
           'lib/filter.py','lib/geometry.py','lib/models.py','lib/moments.py','lib/spectral_series.py','lib/theory.py','lib/reference.py']
    source={p:sha(ROOT/p) for p in paths}
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=config,sources=source,versions=versions))
    meta_path=out/'config.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old['source_hash']==key and all(sha(out/p)==h for p,h in old['tables'].items()):
            print('Contact pilot: verified cached tables');return
        raise RuntimeError('Completed contact evidence differs; preserve it and choose a fresh output directory')
    register=dict(config=config,configuration_hash=digest(config),sources=source,versions=versions,
        source_hash=key,reviewed_commit='bef07fe301d03ecf4bd714e0e15999514c24a2a2',
        gate=gate,development_only=args.development,workers=args.workers,
        hardware=dict(platform=platform.platform(),cpu_count=os.cpu_count(),blas_threads=1))
    frozen=out/'preregistered.json'
    if frozen.exists() and json.loads(frozen.read_text())['source_hash']!=key:
        raise RuntimeError('An incomplete frozen run has different scientific sources; use a fresh output directory')
    write_json(frozen,register);write_json(out/'status.json',dict(status='running',stage='validation',source_hash=key))
    tasks=[(config,trial,condition) for condition in config['validation_conditions'] for trial in range(config['validation_trials'])]
    raw=jobs(validation_task,tasks,'validation',out,key,args.workers)
    validation=[row for group in raw for row in group]
    selected,tuning=select_policies(validation)
    pd.DataFrame(validation).to_csv(out/'validation_trials.csv',index=False);tuning.to_csv(out/'tuning.csv',index=False)
    write_json(out/'selected.json',dict(selected=selected,source_hash=key,
        test_observations_used=False,selection_completed_before_final_test=True,
        boundary_notes='beta1, rho0/1 and ridge0 are structural endpoints; beta0.8 and largest ridge remain explicit finite-grid boundaries'))
    frames={name:[] for name in ['metrics','posterior','query_predictions','bounds','paths','trajectories']}
    for stage,conditions in [('main_test',config['main_conditions']),('exploratory_stress',config['stress_conditions'])]:
        write_json(out/'status.json',dict(status='running',stage=stage,source_hash=key))
        tasks=[(config,trial,condition,selected) for condition in conditions for trial in range(config['test_trials'])]
        results=jobs(test_task,tasks,stage,out,key,args.workers)
        for group in results:
            for name in frames:frames[name].extend(group[name])
    frames={name:pd.DataFrame(rows) for name,rows in frames.items()};frames.update(summaries(frames,config))
    selected_mean=tuning[(tuning.method=='Posterior mean')&tuning.selected.astype(bool)]
    validation_failures=selected_mean.set_index('scope').numerical_failures.to_dict()
    frames['paired']['selected_validation_numerical_failures']=[int(validation_failures['plane' if c=='plane' else 'curved']) for c in frames['paired'].condition]
    frames['paired']['numerically_resolved']=(frames['paired'].numerical_failures==0)&(frames['paired'].selected_validation_numerical_failures==0)
    if any(sha(ROOT/p)!=h for p,h in source.items()):raise RuntimeError('Scientific source changed during the frozen run')
    tables={p:sha(out/p) for p in ['validation_trials.csv','tuning.csv','selected.json','preregistered.json']}
    for name,frame in frames.items():
        path=out/f'{name}.csv';frame.to_csv(path,index=False);tables[path.name]=sha(path)
    posterior=frames['posterior'];applicable=frames['bounds'][frames['bounds'].applicable.astype(bool)]
    valid=applicable[applicable.valid.astype(bool)]
    selected_validation=tuning[tuning.selected.astype(bool)]
    checks=dict(posterior_evaluations=len(posterior),numerical_failures=int((~posterior.numerical_ok.astype(bool)).sum()),
        mean_numerical_failures=int((~posterior.mean_numerical_ok.astype(bool)).sum()),
        state_region_numerical_failures=int((~posterior.state_regions_ok.astype(bool)).sum()),
        response_interval_numerical_failures=int((~posterior.response_intervals_ok.astype(bool)).sum()),
        selected_validation_numerical_failures=int(selected_validation.numerical_failures.sum()),
        maximum_ls_parameter_error=float(posterior.ls_parameter_error.max()),
        maximum_ls_projector_error=float(posterior.ls_projector_error.max()),
        maximum_ls_objective_error=float(posterior.ls_objective_error.max()),
        theorem_applicable=len(applicable),theorem_valid=len(valid),valid_certificate_violations=int(valid.violated.sum()),
        maximum_matched_projection_identity_error=float(frames['trajectories'].query("condition != 'anisotropic'").maximum_projection_identity_error.max()),
        maximum_surface_error=float(frames['trajectories'].maximum_surface_error.max()),
        no_mcmc=True,no_noise_learning=True,positions_available_to_estimators=False)
    metadata=dict(**register,selected=selected,checks=checks,tables=tables)
    write_json(meta_path,metadata);write_json(out/'status.json',dict(status='completed',source_hash=key,checks=checks))
    print(json.dumps(portable(checks),indent=2),flush=True)
    print(frames['paired'].to_string(index=False))


if __name__=='__main__':main()
