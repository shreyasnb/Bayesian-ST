"""Run or resume paired conditional tracking and prediction uncertainty in n=8."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib import drift_study
from lib.drift_study import run_trajectory,summarize,configurations
from lib.validation import require_gate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    from lib import moments
    files={name:sha(ROOT/name) for name in ['lib/drift_study.py','scripts/run_drift_study.py','lib/filter.py',
        'lib/geometry.py','lib/models.py','lib/em.py','lib/theory.py']}
    # Sampling dependencies only: adding an unrelated rank-one backend must not
    # invalidate already completed general-rank posterior trajectories.
    functions={name:hashlib.sha256(inspect.getsource(getattr(moments,name)).encode()).hexdigest()
               for name in ['_hermitian','effective_sample_size','_batch_mcse','MomentResult',
                            'sample_bingham','chain_diagnostics']}
    return dict(files=files,sampling_functions=functions)


def canonical(value):
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,np.generic):return canonical(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def payload_hash(value):return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modal-trials',type=int,default=64)
    parser.add_argument('--posterior-trials',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--draws',type=int,default=1500)
    parser.add_argument('--burnin',type=int,default=1000)
    parser.add_argument('--max-draws',type=int,default=12000)
    parser.add_argument('--max-burnin',type=int,default=4000)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    if args.modal_trials<args.posterior_trials or args.posterior_trials<1 or args.workers<1:
        parser.error('Require modal_trials >= posterior_trials >= 1 and workers >= 1')
    config=dict(seed=74013,n=8,k=3,T=128,s=.03,input_norm=2.,slow_rate=.003,fast_rate=.025,
        forgetting=.97,delta=.05,report_times=[8,16,32,64,96,128],
        modal_trials=args.modal_trials,posterior_trials=args.posterior_trials,
        draws=args.draws,burnin=args.burnin,max_draws=args.max_draws,max_burnin=args.max_burnin)
    source=sources();versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=payload_hash(dict(config=config,sources=source,versions=versions))
    gate=require_gate(ROOT);out=ROOT/'results';meta=out/'drift_study_config.json'
    if meta.exists() and not args.refresh:
        saved=json.loads(meta.read_text())
        if saved.get('source_hash')==key and all(sha(out/f'drift_study_{name}.csv')==h for name,h in saved['tables'].items()):
            print('Drift study: loading verified cached tables');return
    checkpoint=out/'checkpoints/drift';checkpoint.mkdir(parents=True,exist_ok=True)
    finished={};pending=[]
    for trial in range(config['modal_trials']):
        path=checkpoint/f'trial_{trial:03d}_{key[:16]}.json'
        if path.exists() and not args.refresh:
            saved=json.loads(path.read_text())
            if saved.get('source_hash')==key and saved.get('result_hash')==payload_hash(saved['result']):
                finished[trial]=saved['result'];continue
        pending.append(trial)
    print(f'Drift trajectories: {len(finished)}/{config["modal_trials"]} restored',flush=True)
    def retain(trial,result):
        path=checkpoint/f'trial_{trial:03d}_{key[:16]}.json';tmp=path.with_suffix('.tmp')
        document=dict(source_hash=key,result_hash=payload_hash(result),result=canonical(result))
        tmp.write_text(json.dumps(document,allow_nan=False));tmp.replace(path);finished[trial]=result
        print(f'Drift trajectories: {len(finished)}/{config["modal_trials"]} complete',flush=True)
    if args.workers==1:
        for trial in pending:retain(trial,run_trajectory((config,trial)))
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(run_trajectory,(config,trial)):trial for trial in pending}
            for future in as_completed(futures):retain(futures[future],future.result())
    if source!=sources():raise RuntimeError('A numerical dependency changed during execution; rerun')
    frames={name:pd.DataFrame([row for trial in range(config['modal_trials']) for row in finished[trial][name]])
            for name in ['modal','bounds','posterior','cloud']}
    frames['coverage']=summarize(frames['posterior'],frames['bounds'])
    tables={}
    for name,frame in frames.items():
        path=out/f'drift_study_{name}.csv';frame.to_csv(path,index=False);tables[name]=sha(path)
    maximum=max(r['max_equivariance'] for r in finished.values())
    record=dict(source_hash=key,config=config,protocols=configurations(config['slow_rate'],config['fast_rate'],config['forgetting']),
        source=source,versions=versions,tables=tables,core_hash=gate['core_hash'],workers=args.workers,
        checks=dict(maximum_parameter_equivariance_error=maximum,
                    maximum_sample_risk_identity_error=float(frames['posterior'].sample_risk_identity_error.max()),
                    posterior_evaluations=len(frames['posterior']),
                    sampling_failures=int((~frames['posterior'].sampling_ok.astype(bool)).sum())),
        interpretation=dict(
            observations='Conditional u_t=P_t v_t+Gaussian noise; all components and preassigned inputs observed',
            input_protocol='Coordinate-cycle patterns rotated by the prescribed true frame; independent of hidden subspace',
            state_process='Haar initial rank-three subspace and deterministic orthogonal motion',
            coupling='Static and exact-known controls are orthogonally coupled; static draws are transported exactly',
            prediction='Fresh fixed query e0, response component e1^T P e0; not the training u component',
            bands='Pointwise clean-response posterior quantiles at reporting times; not simultaneous path coverage',
            theorem='Fixed true variance; deterministic inputs/motion conditional on each initial state; delta=.05 at each stated fixed time',
            coverage='Wilson intervals across independent trajectories; wide at 16 posterior repetitions',
            approximation='G=I with physical drift and beta<1 are approximate prediction; no calibrated coverage guarantee'))
    meta.write_text(json.dumps(record,indent=2))
    print(frames['bounds'].groupby(['protocol','t'])[['valid','chordal_error']].mean().to_string())
    print(frames['coverage'][frames['coverage'].t==config['T']].to_string(index=False))


if __name__=='__main__':main()
