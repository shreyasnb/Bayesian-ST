"""Portable verify, pilot and frozen-paper stages for final beamforming data."""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import argparse
import hashlib
import json
import os
import platform
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t
from lib.final_beamforming import (
    SOURCE_COMMIT, METHODS, METHOD_INFORMATION, configuration, validation_case,
    evaluate_episode, episode_average, paired_summary, block_episode)

SOURCES=[
    'lib/final_beamforming.py','scripts/run_final_beamforming.py',
    'tests/test_final_beamforming.py','benchmarks/common/great.py',
    'lib/final_theory.py','lib/array_geometry_study.py',
    'lib/beamforming_protocol.py','lib/beamforming_numerics.py',
    'lib/beamforming_decisions.py','lib/spectral_series.py',
    'lib/geometry.py','lib/models.py',
    'benchmarks/case1/baselines.py','benchmarks/case2/baselines.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(portable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def task(arguments):
    kind,cfg,payload=arguments
    begin=perf_counter()
    if kind=='validation':
        result=validation_case(cfg,payload['n'],payload['condition'],payload['stationary'])
    elif kind=='test':
        policy=payload['policy'];trial=payload['trial']
        result=evaluate_episode(cfg,policy,trial,payload['stationary'],payload['story'])
    else:
        result=block_episode(cfg,payload['condition'],payload['trial'])
    return result,perf_counter()-begin


def execute_tasks(tasks,folder,source_hash,workers,phase):
    """Content-addressed worker records; obsolete records are not overwritten."""
    folder.mkdir(parents=True,exist_ok=True)
    completed={};pending=[];timings=[]
    for index,arguments in enumerate(tasks):
        path=folder/(f'{index:05d}.json')
        signature=fingerprint(arguments)
        if path.exists():
            old=json.loads(path.read_text())
            if old['source_hash']!=source_hash or old['task_hash']!=signature:
                raise RuntimeError('Preserve obsolete worker data and use a fresh output directory: '+str(path))
            if old['result_hash']!=fingerprint(old['result']):
                raise RuntimeError('Changed worker result: '+str(path))
            arrays={}
            if old.get('arrays'):
                target=folder/old['arrays']
                if sha(target)!=old['arrays_hash']:raise RuntimeError('Changed worker array')
                with np.load(target,allow_pickle=False) as data:arrays={name:data[name] for name in data.files}
            result=old['result']
            if arguments[0]=='test':result=(result,arrays)
            completed[index]=result
            timings.append(dict(phase=phase,index=index,seconds=old['seconds']))
        else:pending.append((index,arguments,path,signature))
    print(phase,'restored',len(completed),'/',len(tasks),flush=True)
    def retain(index,arguments,path,signature,output):
        result,seconds=output
        arrays={}
        if arguments[0]=='test':
            result,arrays=result
        result=portable(result)
        value=dict(source_hash=source_hash,task_hash=signature,result=result,
                   result_hash=fingerprint(result),seconds=seconds)
        if arrays:
            target=folder/(f'{index:05d}.npz')
            np.savez_compressed(target,**arrays)
            value.update(arrays=target.name,arrays_hash=sha(target))
        write(path,value)
        completed[index]=(result,arrays) if arguments[0]=='test' else result
        timings.append(dict(phase=phase,index=index,seconds=seconds))
        if len(completed)%16==0 or len(completed)==len(tasks):
            print(phase,'completed',len(completed),'/',len(tasks),flush=True)
    if workers==1:
        for index,arguments,path,signature in pending:
            retain(index,arguments,path,signature,task(arguments))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(task,arguments):(index,arguments,path,signature)
                     for index,arguments,path,signature in pending}
            for future in as_completed(futures):
                index,arguments,path,signature=futures[future]
                retain(index,arguments,path,signature,future.result())
    return [completed[index] for index in range(len(tasks))],timings


def summary(frame,groups,endpoint='heldout_ratio_db'):
    rows=[]
    for keys,group in frame.groupby(groups,sort=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        values=group[endpoint].to_numpy()
        finite=np.isfinite(values)
        x=values[finite]
        half=float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
        mean=float(x.mean()) if len(x) else np.nan
        rows.append(dict(zip(groups,keys),episodes=group.trial.nunique(),
            resolved_episodes=int(finite.sum()),mean=mean,lower=mean-half,upper=mean+half,
            median=float(np.median(x)) if len(x) else np.nan,
            numerical_ok=bool(finite.all() and group.numerical_ok.astype(bool).all()),
            endpoint=endpoint,interval='Pointwise Student 95% across independent episodes; exploratory cells'))
    return pd.DataFrame(rows)


def legacy_evidence():
    """Reuse the existing n=8,16 data under its original geometry/count labels."""
    from lib.simulation_validation import verify_component
    old=verify_component(ROOT,'paper','beamforming')
    folder=ROOT/'results/simulation_redesign/paper/beamforming'
    return dict(source_commit='6e78cc9911e7e78452633f13dd947afa669c3470',
        original_cohorts=old['config']['cohorts'],
        original_laws=old['config']['laws'],
        original_horizons=old['config']['horizons'],
        original_sensor_variances=old['config']['noises'],
        files={str((folder/name).relative_to(ROOT)):sha(folder/name)
               for name in ['manifest.json','metrics.csv','paired.csv','summary.csv']},
        scope='Historical n=8,16 fixed-angle/Haar evidence retained separately. Its geometry and repetition counts are not relabeled as the new overlap-controlled dimensional cohorts.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['verify','pilot','paper'],default='verify')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    cfg=configuration(args.stage)
    data=args.output or ROOT/'results/beamforming/data'/args.stage/'performance'
    tables=(data/'tables' if args.output else ROOT/'results/beamforming/tables'/args.stage/'performance')
    data.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    if args.stage=='paper':
        from lib.final_validation import require_gate
        gate=require_gate(ROOT)
    else:gate=None
    sources={name:sha(ROOT/name) for name in SOURCES}
    source_hash=fingerprint(sources)
    declaration=dict(config=cfg,sources=sources,source_hash=source_hash)
    manifest=data/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['config']!=cfg or old['sources']!=sources:
            raise RuntimeError('Preserve obsolete results and use a fresh output directory')
        for name,digest in old['artifacts'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Changed saved artifact: '+name)
        print('Verified completed',args.stage,'beamforming performance');return
    registered=data/'preregistered.json'
    if registered.exists():
        prior=json.loads(registered.read_text())
        if prior['config']!=cfg or prior['sources']!=sources:
            raise RuntimeError('Preserve the earlier declaration; use a fresh output directory')
    else:write(registered,dict(declaration,registered_utc=datetime.now(timezone.utc).isoformat()))
    begin=perf_counter()
    validation_tasks=[('validation',cfg,dict(n=n,condition='static',stationary=True))
                      for n in cfg['dimensions']]
    validation_tasks += [('validation',cfg,dict(n=cfg['moving_n'],condition=condition,stationary=False))
                         for condition in cfg['moving_conditions']]
    validation,timings=execute_tasks(validation_tasks,data/'checkpoints/validation',
        source_hash,args.workers,'validation')
    policies=[item[0] for item in validation]
    curves=[row for item in validation for row in item[1]]
    write(data/'selected.json',policies)
    pd.DataFrame(curves).to_csv(tables/'validation_curves.csv',index=False)
    tests=[]
    for policy in policies:
        for trial in range(cfg['test_episodes']):
            story=trial==cfg['story']['episode'] and (
                policy['n']==cfg['story']['stationary_n'] if policy['stationary'] else True)
            tests.append(('test',cfg,dict(policy=policy,trial=trial,
                                         stationary=policy['stationary'],story=story)))
    results,clocks=execute_tasks(tests,data/'checkpoints/test',source_hash,args.workers,'test')
    timings+=clocks
    rows=[row for result,arrays in results for row in result]
    stories={}
    for result,arrays in results:
        for name,value in arrays.items():
            if name in stories and not np.array_equal(stories[name],value):
                raise RuntimeError('Conflicting predeclared story arrays: '+name)
            stories[name]=value
    np.savez_compressed(data/'stories.npz',**stories)
    frame=pd.DataFrame(rows)
    static=frame[frame.stationary.astype(bool)].copy()
    moving=frame[~frame.stationary.astype(bool)].copy()
    episodes=pd.DataFrame(episode_average(moving.to_dict('records'),cfg['primary_burnin']))
    static.to_csv(data/'stationary.csv',index=False)
    moving.to_csv(data/'moving_checkpoints.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    episodes.to_csv(tables/'moving_episodes.csv',index=False)
    summary(static,['n','condition','t','method']).to_csv(tables/'stationary_summary.csv',index=False)
    summary(episodes,['n','condition','method']).to_csv(tables/'moving_summary.csv',index=False)
    summary(moving,['n','condition','t','method']).to_csv(tables/'moving_time_summary.csv',index=False)
    paired_summary(static,['n','condition','t']).to_csv(tables/'paired_stationary.csv',index=False)
    paired_summary(episodes,['n','condition']).to_csv(tables/'paired_moving.csv',index=False)
    block_tasks=[('block',cfg,dict(condition=condition,trial=trial))
                 for condition in cfg['block_conditions'] for trial in range(cfg['block_test_episodes'])]
    blocks,clocks=execute_tasks(block_tasks,data/'checkpoints/blocks',source_hash,args.workers,'blocks')
    timings+=clocks
    block=pd.DataFrame([row for result in blocks for row in result])
    block['heldout_ratio_db']=10*np.log10(block.heldout_ratio)
    block['true_ratio_db']=10*np.log10(block.true_ratio)
    block.to_csv(data/'block_updates.csv',index=False)
    summary(block,['kind','condition','n','block_length']).to_csv(tables/'block_summary.csv',index=False)
    pd.DataFrame(timings).to_csv(tables/'execution_times.csv',index=False)
    legacy=legacy_evidence();write(data/'legacy_evidence.json',legacy)
    primary=frame[frame.method.eq('mean')]
    checks=dict(stationary_episodes=cfg['test_episodes']*len(cfg['dimensions']),
        moving_episodes=cfg['test_episodes']*len(cfg['moving_conditions']),
        stationary_metric_rows=len(static),moving_checkpoint_rows=len(moving),
        moving_episode_rows=len(episodes),block_episodes=len(block_tasks),
        native_mean_calls=len(primary),native_mean_unresolved=int((~primary.numerical_ok.astype(bool)).sum()),
        unresolved_method_rows=int((~frame.numerical_ok.astype(bool)).sum()),
        maximum_unit_gain_error=float(frame.unit_gain_error.max()),
        maximum_exact_block_parameter_error=float(block.maximum_parameter_error.max()),
        maximum_native_order=float(primary.mean_order.max()),
        maximum_native_mean_error_bound=float(primary.mean_error_bound.max()),
        native_cap_extensions=int(primary.mean_cap_extension.astype(bool).sum()),
        n128_task_completed=bool(128 in static.n.unique()),
        section_iso_density_fitted=False,
        section_scope='h=U3*x, x real and on S2. Column1 is the modal complex eigenvector, column2 the next-largest parameter eigenvector, column3 the smallest. This is a density section, not a full posterior or marginal.',
        total_seconds=perf_counter()-begin,
        timing_scope='Concurrent execution metadata/pilot cost only; isolated cadence runner supplies final runtime comparisons')
    write(data/'checks.json',checks)
    write(data/'method_information.json',METHOD_INFORMATION)
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):
        raise RuntimeError('Scientific producers changed during the stage; preserve results before rerunning')
    artifacts={str(path.relative_to(ROOT)):sha(path) for folder in [data,tables]
        for path in sorted(folder.rglob('*')) if path.is_file()
        and 'checkpoints' not in path.parts and path.name not in ['manifest.json','status.json']}
    artifacts.update(legacy['files'])
    result=dict(status='completed',example='beamforming',stage=args.stage,component='performance',
        config=cfg,sources=sources,artifacts=artifacts,checks=checks,
        created_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        gate_science_hash=gate.get('science_hash') if gate else None,
        probability_scope='Pointwise episode confidence intervals; no simultaneous or universal algorithm-ranking claim')
    write(manifest,result)
    write(data/'status.json',dict(status='completed',checks=checks))
    print(json.dumps(portable(checks),indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
