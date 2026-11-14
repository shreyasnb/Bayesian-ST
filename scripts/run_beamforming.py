from pathlib import Path
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import sys
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['all','stationary','moving','noise'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'results/rerun/beamforming/data')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/beamforming.json')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    release=json.loads(args.config.read_text())
    experiments=['stationary','moving','noise'] if args.experiment=='all' else [args.experiment]
    for experiment in experiments:
        path=args.output/f'{experiment}.csv.gz'
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}; use a fresh output directory')
    for experiment in experiments:
        cfg=release[experiment]['config'];selected=release[experiment]['selected']
        count=cfg['test_episodes']
        policies=[selected] if experiment=='noise' else selected
        jobs=[(experiment,cfg,policy,trial) for policy in policies for trial in range(count)]
        rows=[]
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


if __name__=='__main__':main()
