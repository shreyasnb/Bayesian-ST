"""Run or load the controlled, shared-data low-sample decision study."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.decision_study import run_decision_study,mean_projector
from lib.validation import require_gate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=400)
    parser.add_argument('--validation-trials',type=int,default=160)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    gate=require_gate(ROOT);out=ROOT/'results';out.mkdir(exist_ok=True)
    config=dict(seed=86021,trials=args.trials,validation_trials=args.validation_trials)
    sources={name:sha(ROOT/name) for name in ['lib/decision_study.py','scripts/run_decision_study.py','benchmarks/case2/baselines.py','benchmarks/case1/baselines.py']}
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=hashlib.sha256(json.dumps(dict(config=config,sources=sources,versions=versions,core_hash=gate['core_hash']),sort_keys=True).encode()).hexdigest()
    meta=out/'decision_study_config.json'
    if meta.exists() and not args.refresh:
        saved=json.loads(meta.read_text())
        if saved.get('source_hash')==key and all(sha(out/f'decision_study_{name}.csv')==digest for name,digest in saved['tables'].items()):
            print('Decision study: loading verified cached tables')
            return
    # Independent low-dimensional numerical reference before simulation.
    from lib.moments import quadrature_gr13
    from lib.decision_study import _beam_weights
    from benchmarks.case1 import mvdr
    R=np.array([[2.,.2,0],[.2,1.,.1],[0.,.1,.5]])
    Q=np.eye(3)
    assert np.max(abs(_beam_weights(R,Q)-np.array([mvdr(R,q) for q in Q])))<1e-12
    for A in [np.zeros((3,3)),np.diag([0.,-8.,-80.]),np.array([[1.,.4,0],[.4,0.,.2],[0.,.2,-3.]])]:
        assert np.max(abs(mean_projector(A)-quadrature_gr13(A,96).mean))<1e-8
    result=run_decision_study(**config)
    tables={}
    for name,frame in result.items():
        if isinstance(frame,pd.DataFrame):
            path=out/f'decision_study_{name}.csv';frame.to_csv(path,index=False);tables[name]=sha(path)
    assert result['accuracy'].reference_difference.max()<=result['config']['tolerance']
    assert sources=={name:sha(ROOT/name) for name in sources},'Source changed during simulation'
    record=dict(source_hash=key,sources=sources,versions=versions,core_hash=gate['core_hash'],
                requested_config=config,config=result['config'],tables=tables,
                checks=dict(maximum_quadrature_difference=float(result['accuracy'].reference_difference.max()),
                            evaluated_posteriors=len(result['accuracy']),coverage_trials=len(result['coverage']),
                            posterior_risk_nonnegative=bool(result['metrics'].posterior_risk.ge(-1e-10).all()),
                            maximum_unit_gain_error=float(result['metrics'].unit_gain_error.max()),
                            minimum_posterior_beamforming_excess=float(result['metrics'].beamforming_posterior_excess.min())))
    meta.write_text(json.dumps(record,indent=2))
    print(result['coverage_summary'].to_string(index=False))
    print(result['paired'][result['paired'].reference.isin(['GROUSE','Shrunk PCA','Loaded covariance'])].to_string(index=False))


if __name__=='__main__':main()
