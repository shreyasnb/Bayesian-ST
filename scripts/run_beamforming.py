from pathlib import Path
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import hashlib
import json
import sys
from time import perf_counter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.final_beamforming import evaluate_episode,generate_stream,heldout_snapshots
from lib.final_prediction import adaptation_trial


def primary_stream(cfg,episode):
    entropy=np.random.SeedSequence([cfg['test_seed']+cfg['seed_offset'],32,episode,31]).generate_state(1)[0]
    stream=generate_stream(cfg['primary_beam_n'],cfg['primary_beam_T'],'noise_change',int(entropy),
        sigma2=.2,tau2=1.,overlap=.5,dt=cfg['primary_beam_dt'])
    stream['P']=np.einsum('ti,tj->tij',stream['h'],stream['h'].conj())
    stream['dt']=cfg['primary_beam_dt'];stream['G']=None
    stream['heldout']={t:heldout_snapshots(stream,t+1,cfg['heldout_draws'])
        for t in range(cfg['primary_report_every']-1,len(stream['U']),cfg['primary_report_every'])}
    return stream


def worker(job):
    experiment,cfg,policy,trial=job
    if experiment=='noise':
        rows,events,arrays=adaptation_trial('beamforming',primary_stream(cfg,trial),policy,cfg,
            trial,'test','noise_change')
        return rows
    policy=dict(policy)
    if experiment=='stationary':policy['times']=[200]
    return evaluate_episode(cfg,policy,trial,stationary=experiment=='stationary')[0]


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['all','stationary','moving','noise'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'results/rerun/beamforming/data')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/beamforming.json')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--smoke',action='store_true',help='One simulation per frozen condition; retains paper seeds and horizons.')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    config_bytes=args.config.read_bytes()
    config_sha256=hashlib.sha256(config_bytes).hexdigest()
    release=json.loads(config_bytes)
    experiments=['stationary','moving','noise'] if args.experiment=='all' else [args.experiment]
    for experiment in experiments:
        if (args.output/f'{experiment}.csv.gz').exists():
            raise FileExistsError(f'Refusing to overwrite {args.output/experiment}; use a fresh output directory')
        cfg=release[experiment]['config'];selected=release[experiment]['selected']
        count=1 if args.smoke else cfg['test_episodes']
        policies=[selected] if experiment=='noise' else selected
        jobs=[(experiment,cfg,policy,trial) for policy in policies for trial in range(count)]
        start=perf_counter();rows=[]
        if args.workers==1:
            output=map(worker,jobs)
            for i,part in enumerate(output,1):
                rows.extend(part);print(f'{experiment}: {i}/{len(jobs)}',flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
                for i,part in enumerate(pool.map(worker,jobs,chunksize=1),1):
                    rows.extend(part);print(f'{experiment}: {i}/{len(jobs)}',flush=True)
        frame=pd.DataFrame(rows)
        if experiment=='noise':
            flags=['mean_ok','score_numerical_ok','fit_numerical_ok']
            drop=[c for c in frame if c.endswith('_seconds')]
            frame=frame.drop(columns=drop)
        else:flags=['numerical_ok','mean_ok']
        for flag in flags:
            if not frame[flag].astype(bool).all():raise ArithmeticError('Unresolved '+flag)
        path=args.output/f'{experiment}.csv.gz'
        frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
        write_json(args.output/f'config_{experiment}.json',cfg)
        write_json(args.output/f'policies_{experiment}.json',selected)
        write_json(args.output/f'run_{experiment}.json',dict(status='completed',smoke=args.smoke,
            simulations_per_condition=count,conditions=len(policies),rows=len(frame),
            seconds=perf_counter()-start,source_commit=release['source_commit'],
            config_sha256=config_sha256,
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


if __name__=='__main__':main()
