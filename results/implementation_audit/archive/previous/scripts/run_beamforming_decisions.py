"""Reproducible passive-complex array decisions with frozen validation."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import hashlib
import json
import platform
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.beamforming_decisions import (default_config,validation_task,choose_policies,
    boundary_expansion,test_task,paired_summary)


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [portable(v) for v in value]
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def digest(value):return hashlib.sha256(json.dumps(portable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n');temporary.replace(path)


def run_jobs(function,tasks,stage,key,out,workers):
    cache=out/'checkpoints'/stage;cache.mkdir(parents=True,exist_ok=True);complete={};pending=[]
    for index,task in enumerate(tasks):
        path=cache/f'{index:05d}.json';signature=digest(task)
        try:
            old=json.loads(path.read_text())
            assert old['source_hash']==key and old['task_hash']==signature
            assert old['result_hash']==digest(old['result'])
            for name,expected in old.get('artifacts',{}).items():assert sha(out/name)==expected
            complete[index]=old['result']
        except (OSError,ValueError,KeyError,AssertionError):pending.append((index,task,signature,path))
    print(stage,'restored',len(complete),'/',len(tasks),flush=True)
    def retain(index,result,signature,path):
        story=result.pop('story',None) if isinstance(result,dict) else None;artifacts={}
        if story:
            target=out/'story.npz';np.savez_compressed(target,**story);artifacts[target.name]=sha(target)
        result=portable(result);save(path,dict(source_hash=key,task_hash=signature,result_hash=digest(result),result=result,artifacts=artifacts))
        complete[index]=result
        if len(complete)%8==0 or len(complete)==len(tasks):print(stage,len(complete),'/',len(tasks),flush=True)
    if workers==1:
        for index,task,signature,path in pending:retain(index,function(task),signature,path)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):(index,signature,path) for index,task,signature,path in pending}
            for future in as_completed(futures):
                index,signature,path=futures[future];retain(index,future.result(),signature,path)
    return [complete[i] for i in range(len(tasks))]


def summary_tables(frames,config):
    m=frames['metrics'];summary=[]
    for (condition,s,t,method),g in m.groupby(['condition','s','t','method'],sort=False):
        row=dict(condition=condition,s=s,t=t,method=method,trials=len(g),mean_failures=int((~g.mean_ok.astype(bool)).sum()))
        for field in ['true_power','heldout_power','posterior_power','posterior_excess','projection_loss','posterior_risk',
            'update_seconds','mode_seconds','moment_seconds','decision_seconds']:
            row[field]=float(g[field].mean())
        summary.append(row)
    coverage=[]
    c=frames['coverage'];power=frames['power_components']
    for (condition,s,t),g in c.groupby(['condition','s','t'],sort=False):
        component=power[(power.condition==condition)&(power.s==s)&(power.t==t)]
        within=component.groupby('trial')[['covered','width','interval_score']].mean()
        coverage.append(dict(condition=condition,s=s,t=t,trials=len(g),
            ball_coverage=float(g.ball_covered.mean()),hpd_coverage=float(g.hpd_covered.mean()),
            ball_mass_reference=float(g.ball_mass_reference.mean()),hpd_mass_reference=float(g.hpd_mass_reference.mean()),
            unresolved=int((~g.numerical_ok.astype(bool)).sum()),power_coverage=float(within.covered.mean()),
            power_width=float(within.width.mean()),power_interval_score=float(within.interval_score.mean())))
    return dict(summary=pd.DataFrame(summary),coverage_summary=pd.DataFrame(coverage),
        paired=paired_summary(m),risk=m[m.method=='mean'][['condition','s','q','n','trial','t','posterior_risk','projection_loss','mean_ok']].copy())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--validation-only',action='store_true')
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('At least one worker required')
    if args.validation_only and args.test_only:parser.error('Choose one stage')
    config=default_config(args.profile)
    out=args.output or ROOT/'results/beamforming_unified'/args.profile/'decisions'
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    source_names=['lib/beamforming_protocol.py','lib/beamforming_decisions.py','lib/beamforming_numerics.py',
        'scripts/run_beamforming_decisions.py','lib/models.py','lib/geometry.py','lib/theory.py',
        'lib/spectral_series.py','benchmarks/case1/baselines.py','benchmarks/case2/baselines.py',
        'tests/test_beamforming_decisions.py']
    sources={name:sha(ROOT/name) for name in source_names}
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=config,sources=sources,versions=versions))
    meta=out/'config.json'
    if meta.exists():
        existing=json.loads(meta.read_text())
        if existing['source_hash']==key and all(sha(out/name)==expected for name,expected in existing['tables'].items()):
            print('Verified completed decisions cache');return
        raise RuntimeError('Preserve completed evidence; choose a new output directory')
    declaration=dict(config=config,sources=sources,versions=versions,source_hash=key,
        hardware=dict(platform=platform.platform(),cpu_count=os.cpu_count(),blas_threads=1),
        seeds=dict(validation=config['validation_seed'],test=config['test_seed']),
        profile=args.profile,selection_frozen_before_test=True)
    register=out/'preregistered.json'
    if register.exists() and json.loads(register.read_text())['source_hash']!=key:
        raise RuntimeError('Sources changed during a declared run; preserve it and choose a new output directory')
    save(register,declaration)
    selected_path=out/'selected.json'
    if not args.test_only:
        save(out/'status.json',dict(status='running',stage='validation',source_hash=key))
        tasks=[(config,c,s,j) for c in config['conditions'] for s in config['noises'] for j in range(config['validation_trials'])]
        output=run_jobs(validation_task,tasks,'validation_initial',key,out,args.workers)
        rows=[row for group in output for row in group];selected,tuning=choose_policies(rows,config)
        pd.DataFrame(rows).to_csv(out/'validation_initial.csv',index=False)
        expansion=boundary_expansion(selected,config)
        expanded=dict(config)
        if any(expansion.values()):
            for field,extra in expansion.items():expanded[field]=list(dict.fromkeys(config[field]+extra))
            tasks=[(expanded,c,s,j) for c in config['conditions'] for s in config['noises'] for j in range(config['validation_trials'])]
            output=run_jobs(validation_task,tasks,'validation_expanded',key,out,args.workers)
            rows=[row for group in output for row in group];selected,tuning=choose_policies(rows,config)
        if not all(p['resolved'] for p in selected):raise ArithmeticError('At least one validation family is unresolved')
        pd.DataFrame(rows).to_csv(out/'validation_trials.csv',index=False);tuning.to_csv(out/'tuning.csv',index=False)
        save(selected_path,dict(source_hash=key,selected=selected,initial_config=config,effective_config=expanded,
            expansion=expansion,one_extension_only=True,selection_completed_before_test=True,test_observations_used=False))
        save(out/'status.json',dict(status='validation_completed',source_hash=key))
        if args.validation_only:return
    else:
        document=json.loads(selected_path.read_text())
        if document['source_hash']!=key:raise RuntimeError('Frozen selections do not match sources/configuration')
        selected=document['selected'];expanded=document['effective_config'];expansion=document['expansion']
    gate=None
    if args.profile=='paper':
        from lib.beamforming_validation import require_beamforming_gate
        gate=require_beamforming_gate(ROOT)
    save(out/'test_preregistered.json',dict(**declaration,gate=gate,selected_hash=sha(selected_path),
        heldout_opened_after_selection=True))
    save(out/'status.json',dict(status='running',stage='heldout',source_hash=key))
    start=perf_counter();tasks=[(config,c,s,j,selected) for c in config['conditions'] for s in config['noises'] for j in range(config['test_trials'])]
    output=run_jobs(test_task,tasks,'test',key,out,args.workers)
    names=['metrics','shape','spectra','coverage','power_components','scans','weight_beampattern','projection_cloud']
    frames={name:pd.DataFrame([row for group in output for row in group[name]]) for name in names}
    frames.update(summary_tables(frames,config))
    if any(sha(ROOT/name)!=expected for name,expected in sources.items()):raise RuntimeError('Scientific source changed during execution')
    tables={name:sha(out/name) for name in ['preregistered.json','selected.json','test_preregistered.json','validation_initial.csv','validation_trials.csv','tuning.csv']}
    for name,frame in frames.items():
        path=out/f'{name}.csv';frame.to_csv(path,index=False);tables[path.name]=sha(path)
    if (out/'story.npz').exists():tables['story.npz']=sha(out/'story.npz')
    checks=dict(trajectories_per_condition=config['test_trials'],conditions=len(config['conditions']),noise_regimes=len(config['noises']),
        metric_rows=len(frames['metrics']),mean_failures=int((~frames['metrics'].mean_ok.astype(bool)).sum()),
        tail_reports=len(frames['coverage']),tail_unresolved=int((~frames['coverage'].numerical_ok.astype(bool)).sum()),
        one_snapshot_max_anisotropy=float(frames['shape'][frames['shape'].t==1].anisotropy.max()),
        maximum_unit_gain_error=float(frames['metrics'].unit_gain_error.max()),
        maximum_expected_power_identity_error=float(frames['metrics'].optimality_identity_error.max()),
        maximum_mode_pca_error=float(frames['metrics'].pca_projector_error.max()),
        heldout_wall_seconds=perf_counter()-start,
        numerical_status='All flags retained; iid tail checks and mean truncation have different error scopes')
    save(meta,dict(**declaration,gate=gate,selected=selected,expansion=expansion,tables=tables,checks=checks))
    save(out/'manifest.json',dict(source_hash=key,sources=sources,tables=tables,checks=checks))
    save(out/'status.json',dict(status='completed',source_hash=key,checks=checks))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
