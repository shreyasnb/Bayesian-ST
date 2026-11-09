"""Run the frozen physical comparison; no historical result is rewritten."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import hashlib,json,platform,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from lib.contact_uncertainty_study import default_config,validation_task,test_task,choose_policies,needs_expansion,summaries


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def digest(value):return hashlib.sha256(json.dumps(portable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False));temporary.replace(path)


def jobs(function,tasks,stage,key,out,workers):
    directory=out/'checkpoints'/stage;directory.mkdir(parents=True,exist_ok=True)
    complete={};pending=[]
    for index,task in enumerate(tasks):
        task_hash=digest(task);path=directory/f'{index:05d}.json'
        try:
            previous=json.loads(path.read_text())
            assert previous['source_hash']==key and previous['task_hash']==task_hash
            assert previous['result_hash']==digest(previous['result'])
            complete[index]=previous['result']
        except (OSError,ValueError,KeyError,AssertionError):pending.append((index,task,task_hash,path))
    print(stage,'restored',len(complete),'/',len(tasks),flush=True)
    def retain(index,result,task_hash,path):
        result=portable(result);save(path,dict(source_hash=key,task_hash=task_hash,result_hash=digest(result),result=result))
        complete[index]=result
        if len(complete)%4==0 or len(complete)==len(tasks):print(stage,'completed',len(complete),'/',len(tasks),flush=True)
    if workers==1:
        for index,task,task_hash,path in pending:retain(index,function(task),task_hash,path)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):(index,task_hash,path) for index,task,task_hash,path in pending}
            for future in as_completed(futures):
                index,task_hash,path=futures[future];retain(index,future.result(),task_hash,path)
    return [complete[i] for i in range(len(tasks))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/contact_uncertainty/physical')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--validation-trials',type=int,default=16);parser.add_argument('--test-trials',type=int,default=64)
    parser.add_argument('--T',type=int,default=600)
    parser.add_argument('--development',action='store_true');parser.add_argument('--small-grid',action='store_true')
    args=parser.parse_args()
    if min(args.workers,args.validation_trials,args.test_trials)<1 or args.T<6:parser.error('Positive trial counts and T>=6 required')
    if (args.T!=600 or args.small_grid) and not args.development:parser.error('Changed horizon/grids require explicit --development')
    config=default_config();config.update(validation_trials=args.validation_trials,test_trials=args.test_trials,T=args.T)
    if args.T!=600:config['report_times']=sorted({6,max(6,args.T//2),args.T});config['primary_time']=args.T
    if args.small_grid:
        config.update(betas=[1.,.8],etas=[.5,1.],process_rms=[0.,.02],beta_expansion=[],process_expansion=[])
    gate=None
    if not args.development:
        from lib.contact_uncertainty_validation import require_uncertainty_gate
        gate=require_uncertainty_gate(ROOT)
    paths=['lib/contact_uncertainty_study.py','scripts/run_contact_uncertainty.py',
        'lib/contact_process_prediction.py','lib/contact_gaussian.py','lib/contact_posterior.py',
        'lib/contact_mechanics.py','lib/contact_study.py','lib/models.py','lib/geometry.py','lib/reference.py',
        'lib/process_prediction.py','lib/reporting.py']
    sources={p:sha(ROOT/p) for p in paths}
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=config,sources=sources,versions=versions))
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    meta_path=out/'config.json'
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        if previous['source_hash']==key and all(sha(out/p)==h for p,h in previous['tables'].items()):
            if gate is not None and previous['gate']['uncertainty_hash']!=gate['uncertainty_hash']:
                raise RuntimeError('Completed evidence captured a different gate; preserve and use a fresh directory')
            print('Physical uncertainty comparison: verified cached tables');return
        raise RuntimeError('Completed evidence differs; preserve it and choose a fresh output directory')
    register=dict(config=config,sources=sources,versions=versions,source_hash=key,gate=gate,
        development_only=args.development,reviewed_commit=config['reviewed_commit'],workers=args.workers,
        hardware=dict(platform=platform.platform(),cpu_count=os.cpu_count(),blas_threads=1))
    preregistered=out/'preregistered.json'
    if preregistered.exists() and json.loads(preregistered.read_text())['source_hash']!=key:
        raise RuntimeError('Scientific sources differ from an incomplete frozen run; choose a new directory')
    save(preregistered,register);save(out/'status.json',dict(status='running',stage='initial_validation',source_hash=key))
    tasks=[(config,c,j,'initial') for c in config['main_conditions'] for j in range(config['validation_trials'])]
    results=jobs(validation_task,tasks,'initial_validation',key,out,args.workers)
    raw=[row for group in results for row in group]
    selected,tuning=choose_policies(raw,config);expansion=needs_expansion(selected,config)
    if args.small_grid:expansion=dict(beta=False,process=False)
    if any(expansion.values()):
        expanded=dict(config,_expand_beta=expansion['beta'],_expand_process=expansion['process'])
        tasks=[(expanded,c,j,'expansion') for c in config['main_conditions'] for j in range(config['validation_trials'])]
        extra=jobs(validation_task,tasks,'expanded_validation',key,out,args.workers)
        raw.extend(row for group in extra for row in group);selected,tuning=choose_policies(raw,config)
    table=pd.DataFrame(raw);table.to_csv(out/'validation_trials.csv',index=False);tuning.to_csv(out/'tuning.csv',index=False)
    save(out/'selected.json',dict(source_hash=key,selected=selected,expansion=expansion,
        selection_completed_before_test=True,test_observations_used=False,
        interval_score_winners=tuning[tuning.selected_by_interval_score.astype(bool)].to_dict('records'),
        boundary_scope='One predeclared validation expansion only; remaining edges stay visible, not proof of global optimality'))
    save(out/'status.json',dict(status='running',stage='final_test',source_hash=key))
    frames={name:[] for name in ['metrics','components','directional','shape_comparison','startup']}
    for stage,conditions in [('main_test',config['main_conditions']),('observation_mismatch',config['stress_conditions'])]:
        tasks=[(config,c,j,selected) for c in conditions for j in range(config['test_trials'])]
        output=jobs(test_task,tasks,stage,key,out,args.workers)
        for group in output:
            for name in frames:frames[name].extend(group[name])
    frames={name:pd.DataFrame(rows) for name,rows in frames.items()};frames.update(summaries(frames,config))
    if any(sha(ROOT/name)!=h for name,h in sources.items()):raise RuntimeError('Scientific source changed during the frozen run')
    tables={name:sha(out/name) for name in ['validation_trials.csv','tuning.csv','selected.json','preregistered.json']}
    for name,frame in frames.items():
        path=out/f'{name}.csv';frame.to_csv(path,index=False);tables[path.name]=sha(path)
    m=frames['metrics'];uncertain=m[~m.method.isin(['Point mode','Point mean'])]
    checks=dict(report_rows=len(m),mean_failures=int((~uncertain.mean_ok.astype(bool)).sum()),
        state_failures=int((~uncertain.state_ok.astype(bool)).sum()),interval_failures=int((~uncertain.interval_ok.astype(bool)).sum()),
        selected_validation_mean_failures=int(tuning[tuning.selected.astype(bool)].mean_failures.sum()),
        selected_validation_interval_failures=int(tuning[tuning.selected.astype(bool)].interval_failures.sum()),
        failed_trajectories=int((table.completed_steps<config['T']).sum()),
        maximum_inverse_residual=float(m.maximum_inverse_residual.max()),
        maximum_moment_uncertainty=float(m.maximum_moment_uncertainty.max()),
        no_clipped_intervals=True,no_true_geometry_in_filters=True,
        optional_decision_test='Not run here; parent assesses predeclared primary evidence before any fresh decision experiment')
    save(meta_path,dict(**register,selected=selected,expansion=expansion,tables=tables,checks=checks))
    save(out/'status.json',dict(status='completed',source_hash=key,checks=checks))
    print(json.dumps(portable(checks),indent=2),flush=True)
    print(frames['paired'].query('primary').to_string(index=False))


if __name__=='__main__':main()
