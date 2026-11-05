"""Frozen validation/test stages for the revised contact uncertainty study."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,hashlib,json,platform,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from lib.contact_tracking_uncertainty import (default_config,policies_from_stage1,validation_task,
    choose_policies,boundary_expansion,selected_validation_task,test_task,summarize)


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.ndarray):return portable(value.tolist())
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
            previous=json.loads(path.read_text());assert previous['source_hash']==key and previous['task_hash']==task_hash
            assert previous['result_hash']==digest(previous['result']);complete[index]=previous['result']
        except (OSError,ValueError,KeyError,AssertionError):pending.append((index,task,task_hash,path))
    print(stage,'restored',len(complete),'/',len(tasks),flush=True)
    def retain(index,result,task_hash,path):
        arrays=result.pop('arrays',None) if isinstance(result,dict) else None
        if arrays:
            # Trial zero was preassigned before test. No selection by visual appeal.
            condition=tasks[index][1];folder=out/'trajectories';folder.mkdir(exist_ok=True)
            target=folder/f'{condition}_trial0.npz';np.savez_compressed(target,**arrays)
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
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/contact_tracking/uncertainty')
    parser.add_argument('--mean-selected',type=Path,default=ROOT/'results/contact_tracking/mean/selected.json')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--validation-only',action='store_true');parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--development',action='store_true');parser.add_argument('--small-grid',action='store_true')
    parser.add_argument('--validation-trials',type=int,default=16);parser.add_argument('--test-trials',type=int,default=64)
    parser.add_argument('--T',type=int,default=600)
    args=parser.parse_args()
    if args.validation_only and args.test_only:parser.error('Choose one stage')
    if min(args.workers,args.validation_trials,args.test_trials)<1 or args.T<6:parser.error('Positive counts and T>=6 required')
    if (args.T!=600 or args.small_grid or args.validation_trials!=16 or args.test_trials!=64) and not args.development:
        parser.error('Changed experiment requires --development and separate output')
    config=default_config();config.update(validation_trials=args.validation_trials,test_trials=args.test_trials,T=args.T)
    if args.T!=600:
        config.update(report_times=sorted({6,max(6,args.T//2),args.T}),story_times=sorted({6,max(6,args.T//2),args.T}),primary_time=args.T)
    if args.small_grid:config.update(temperatures=[1.,4.],kappas=[500.,None])
    mean_document=json.loads(args.mean_selected.read_text());policies=policies_from_stage1(mean_document)
    # Selection files are numerical sources for this stage, not inferred from tests.
    paths=['lib/contact_tracking_uncertainty.py','scripts/run_contact_tracking_uncertainty.py',
        'lib/contact_tracking.py','lib/contact_process_prediction.py','lib/contact_gaussian.py',
        'lib/contact_posterior.py','lib/contact_mechanics.py','lib/contact_study.py','lib/contact_uncertainty_study.py',
        'lib/models.py','lib/geometry.py','lib/prediction.py']
    sources={p:sha(ROOT/p) for p in paths};versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=config,sources=sources,versions=versions,mean_selection_hash=sha(args.mean_selected),policies=policies))
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    gate=None
    if not args.development and not args.validation_only:
        from lib.contact_uncertainty_validation import require_uncertainty_gate
        gate=require_uncertainty_gate(ROOT)
    meta=out/'config.json'
    if meta.exists():
        previous=json.loads(meta.read_text())
        if previous['source_hash']==key and all(sha(out/p)==h for p,h in previous['tables'].items()):
            print('Verified completed uncertainty comparison');return
        raise RuntimeError('Completed evidence differs; use a fresh directory')
    register=dict(config=config,sources=sources,versions=versions,source_hash=key,
        mean_selection_hash=sha(args.mean_selected),policies=policies,development_only=args.development,
        workers=args.workers,hardware=dict(platform=platform.platform(),cpu_count=os.cpu_count(),blas_threads=1))
    pre=out/'preregistered.json'
    if pre.exists() and json.loads(pre.read_text())['source_hash']!=key:
        raise RuntimeError('Scientific sources differ from frozen run; preserve it and use a new directory')
    save(pre,register)
    selected_path=out/'selected.json'
    if not args.test_only:
        save(out/'status.json',dict(status='running',stage='initial_validation',source_hash=key))
        tasks=[(config,c,j,policies,'initial') for c in config['main_conditions'] for j in range(config['validation_trials'])]
        output=jobs(validation_task,tasks,'initial_validation',key,out,args.workers)
        rows=[row for group in output for row in group];selected,tuning=choose_policies(rows,config)
        expansion=boundary_expansion(selected,config)
        if args.small_grid:expansion=dict(temperatures=[],kappas=[])
        if any(expansion.values()):
            expanded=dict(config,_temperatures=expansion['temperatures'],_kappas=expansion['kappas'])
            tasks=[(expanded,c,j,policies,'expanded') for c in config['main_conditions'] for j in range(config['validation_trials'])]
            output=jobs(validation_task,tasks,'expanded_validation',key,out,args.workers)
            rows.extend(row for group in output for row in group);selected,tuning=choose_policies(rows,config)
        pd.DataFrame(rows).to_csv(out/'validation_trials.csv',index=False);tuning.to_csv(out/'tuning.csv',index=False)
        if not all(p['selection_resolved'] for p in selected):raise ArithmeticError('No resolved validation winner for every family')
        # Retain proper interval quality for selected process/Gaussian policies without using it to retune.
        tasks=[(config,c,j,policies,selected) for c in config['main_conditions'] for j in range(config['validation_trials'])]
        output=jobs(selected_validation_task,tasks,'selected_validation',key,out,args.workers)
        pd.DataFrame([row for group in output for row in group]).to_csv(out/'selected_validation.csv',index=False)
        save(selected_path,dict(source_hash=key,selected=selected,expansion=expansion,policies=policies,
            selection_completed_before_test=True,test_observations_used=False,
            point_regularization='rho minimized analytically across paired validation records for selected mean/forecast/process family',
            remaining_boundaries='One extension only, not a proof of global optimality'))
        save(out/'status.json',dict(status='validation_completed',source_hash=key,heldout_opened=False))
        if args.validation_only:return
    else:
        document=json.loads(selected_path.read_text())
        if document['source_hash']!=key:raise RuntimeError('Selected policies do not match source/config hashes')
        selected=document['selected'];expansion=document['expansion']
    if not args.development:
        from lib.contact_uncertainty_validation import require_uncertainty_gate
        gate=require_uncertainty_gate(ROOT)
    # Gate/configuration and selected policy bytes are frozen before first test task.
    save(out/'test_preregistered.json',dict(**register,gate=gate,selected_hash=sha(selected_path),heldout_opened_after_selection=True))
    save(out/'status.json',dict(status='running',stage='heldout',source_hash=key))
    frames={name:[] for name in ['metrics','components','directional','shape_comparison','story']}
    for stage,conditions in [('main_test',config['main_conditions']),('mismatch_test',config['stress_conditions'])]:
        tasks=[(config,c,j,policies,selected) for c in conditions for j in range(config['test_trials'])]
        output=jobs(test_task,tasks,stage,key,out,args.workers)
        for group in output:
            for name in frames:frames[name].extend(group[name])
    frames={name:pd.DataFrame(rows) for name,rows in frames.items()};frames.update(summarize(frames,config))
    if any(sha(ROOT/p)!=h for p,h in sources.items()):raise RuntimeError('Scientific sources changed during run')
    tables={name:sha(out/name) for name in ['preregistered.json','selected.json','test_preregistered.json','validation_trials.csv','tuning.csv','selected_validation.csv']}
    for name,frame in frames.items():
        path=out/f'{name}.csv';frame.to_csv(path,index=False);tables[path.name]=sha(path)
    for path in sorted((out/'trajectories').glob('*.npz')):tables[str(path.relative_to(out))]=sha(path)
    m=frames['metrics'];checks=dict(rows=len(m),mean_failures=int((~m.mean_ok.astype(bool)).sum()),
        interval_failures=int((~m.interval_ok.astype(bool)).sum()),state_failures=int((~m.state_ok.astype(bool)).sum()),
        failures=int(m.get('report_failure',pd.Series(dtype=object)).notna().sum()),
        no_truth_in_estimation=True,unchanged_true_sensor_variance=True,
        maximum_inverse_residual=float(m.maximum_inverse_residual.max()),
        maximum_moment_uncertainty=float(m.maximum_moment_uncertainty.max()))
    save(meta,dict(**register,gate=gate,tables=tables,selected=selected,expansion=expansion,checks=checks))
    save(out/'status.json',dict(status='completed',source_hash=key,checks=checks))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
