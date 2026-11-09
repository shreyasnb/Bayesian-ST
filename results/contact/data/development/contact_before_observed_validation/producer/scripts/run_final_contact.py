"""Portable fixed-scale contact verification, pilot and frozen paper stages."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from datetime import datetime,timezone
from time import perf_counter
import hashlib,json,sys,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.final_contact import (configuration,validation_episode,select_policies,test_episode,
    block_episode,anisotropy_fixture)

SOURCES=['lib/final_contact.py','scripts/run_final_contact.py','tests/test_final_contact.py',
    'lib/final_theory.py','lib/models.py','lib/geometry.py','lib/contact_mechanics.py',
    'lib/contact_posterior.py','lib/contact_process_prediction.py','lib/spectral_series.py',
    'benchmarks/common/great.py','benchmarks/common/__init__.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n');temporary.replace(path)
def digest(value):return hashlib.sha256(json.dumps(portable(value),sort_keys=True).encode()).hexdigest()


def invoke(task):
    kind,cfg,condition,trial,selection=task
    if kind=='validation':return validation_episode(cfg,condition,trial)
    if kind=='test':return test_episode(cfg,condition,trial,selection)
    return block_episode(cfg,condition,trial,selection)


def execute(tasks,folder,source_hash,workers):
    folder.mkdir(parents=True,exist_ok=True);completed={};pending=[]
    for index,task in enumerate(tasks):
        path=folder/f'{index:05d}.json';task_hash=digest(task)
        try:
            previous=json.loads(path.read_text())
            assert previous['source_hash']==source_hash and previous['task_hash']==task_hash
            assert digest(previous['result'])==previous['result_hash']
            completed[index]=previous['result']
        except (FileNotFoundError,ValueError,KeyError,AssertionError):
            pending.append((index,task,path,task_hash))
    print(folder.name,'restored',len(completed),'/',len(tasks),flush=True)
    def retain(index,result,path,task_hash):
        result=portable(result)
        write(path,dict(source_hash=source_hash,task_hash=task_hash,
            result_hash=digest(result),result=result));completed[index]=result
        if len(completed)%16==0 or len(completed)==len(tasks):
            print(folder.name,len(completed),'/',len(tasks),flush=True)
    if workers==1:
        for index,task,path,task_hash in pending:retain(index,invoke(task),path,task_hash)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(invoke,task):(index,path,task_hash) for index,task,path,task_hash in pending}
            for future in as_completed(futures):
                index,path,task_hash=futures[future];retain(index,future.result(),path,task_hash)
    return [completed[index] for index in range(len(tasks))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['verify','pilot','paper'],default='verify')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg=configuration(args.stage)
    data=args.output or ROOT/'results/contact/data'/args.stage/'performance'
    tables=ROOT/'results/contact/tables'/args.stage/'performance'
    if args.output is not None:tables=data/'tables'
    data.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    if args.stage=='paper':
        from lib.final_validation import require_gate
        gate=require_gate(ROOT)
    else:gate=None
    sources={name:sha(ROOT/name) for name in SOURCES}
    existing=data/'manifest.json'
    if existing.exists():
        prior=json.loads(existing.read_text())
        if prior['config']!=cfg or prior['sources']!=sources:
            raise RuntimeError('Preserve obsolete contact outputs and use a fresh directory')
        for name,expected in prior['artifacts'].items():
            if sha(ROOT/name)!=expected:raise RuntimeError('Changed contact artifact: '+name)
        print('Verified cached final contact',args.stage);return
    write(data/'preregistered.json',dict(config=cfg,sources=sources,registered_utc=datetime.now(timezone.utc).isoformat()))
    start=perf_counter();key=digest(dict(config=cfg,sources=sources))
    tasks=[('validation',cfg,condition,trial,None) for condition in cfg['conditions']
        for trial in range(cfg['validation_episodes'])]
    records=execute(tasks,data/'checkpoints/validation',key,args.workers)
    validation=[row for group in records for row in group]
    selection=select_policies(validation,cfg);write(data/'selected.json',selection)
    tasks=[('test',cfg,condition,trial,selection) for condition in cfg['conditions']
        for trial in range(cfg['test_episodes'])]
    results=execute(tasks,data/'checkpoints/test',key,args.workers)
    metrics=[];stories={};diagnostics=[]
    for result in results:
        metrics.extend(result['metrics']);diagnostics.append(result['diagnostics']);stories.update(result['story'])
    np.savez_compressed(data/'story.npz',**{name:np.asarray(value) for name,value in stories.items()})
    arrays,anisotropy=anisotropy_fixture(cfg)
    np.savez_compressed(data/'anisotropy.npz',**arrays);write(data/'anisotropy_checks.json',anisotropy)
    tasks=[('blocks',cfg,condition,trial,selection) for condition in ['plane','sphere']
        for trial in range(cfg['block_episodes'])]
    blocks=[row for group in execute(tasks,data/'checkpoints/blocks',key,args.workers) for row in group]
    frame=pd.DataFrame(metrics)
    from lib.final_contact import paired_table
    frames=dict(metrics=frame,paired=pd.DataFrame(paired_table(frame)),blocks=pd.DataFrame(blocks),
        mechanics=pd.DataFrame(diagnostics))
    # Structured policy values remain JSON; the CSV is a readable tuning table.
    flat=[dict(r,policy=json.dumps(r['policy'],sort_keys=True),diagnostics=json.dumps(r.get('diagnostics',{}),sort_keys=True))
          for r in validation]
    frames['validation']=pd.DataFrame(flat)
    for name,table in frames.items():table.to_csv(tables/(name+'.csv'),index=False)
    elapsed=perf_counter()-start
    checks=dict(episodes=cfg['test_episodes']*len(cfg['conditions']),metric_rows=len(frame),
        validation_episodes=cfg['validation_episodes']*len(cfg['conditions']),
        numerical_failures=int((~frame.numerical_ok.astype(bool)).sum()),
        maximum_mean_resolution_error=float(frame.moment_resolution_error.max()),
        maximum_command_excess_identity_error=float(frame.command_excess_identity_error.max()),
        maximum_kkt_residual=float(frames['mechanics'].maximum_kkt_residual.max()),
        maximum_sphere_drift_identity_error=float(frames['mechanics'].maximum_sphere_drift_identity_error.max()),
        maximum_block_parameter_error=float(frames['blocks'].exact_parameter_error.max()),
        anisotropy=anisotropy,seconds=elapsed,
        pilot_projected_paper_seconds=elapsed*(256/max(1,cfg['test_episodes'])) if args.stage=='pilot' else None,
        elapsed_scope='Concurrent execution metadata only; isolated actual-cadence timings are separate.')
    write(data/'checks.json',checks)
    if any(sha(ROOT/name)!=expected for name,expected in sources.items()):
        raise RuntimeError('Contact scientific sources changed during generation')
    artifacts={str(path.relative_to(ROOT)):sha(path) for folder in [data,tables]
        for path in sorted(folder.glob('*')) if path.is_file() and path.name!='manifest.json'}
    manifest=dict(status='completed',example='contact',stage=args.stage,component='performance',
        config=cfg,sources=sources,artifacts=artifacts,checks=checks,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        validation_hash=gate.get('science_hash') if gate else None,
        assumptions='Maintained ideal bilateral contact; known scalar resistance; direct isotropic velocity noise.',
        primary_endpoint='Episode-average held-out clean-response MSE; common queries averaged within episode.',
        intervals='Paired pointwise95% Student across independent episodes; no multiplicity adjustment.',
        information_by_method=dict(mean=['u','v','measured G'],mode=['u','v','measured G'],
            isotropic=['same A as mean'],regularized=['u','v','measured G','validation rho'],
            great_window=['u','measured G','validation window/step/inner count'],
            great_ew=['u','measured G','validation memory/step/inner count']),
        ties='mode_unique describes the common Bingham mode; fixed coordinate eigensolver tie convention is flagged.')
    write(existing,manifest)
    print(json.dumps(portable(checks),indent=2),flush=True)


if __name__=='__main__':main()
