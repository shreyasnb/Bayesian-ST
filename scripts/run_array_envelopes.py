"""Project a sampled 95% state HPD region through the array response (Eq.33).

A central functional quantile is not the image of a state credible region; at
extremal responses these can behave very differently. This script saves both.
"""
from pathlib import Path
import os
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
import sys,json,hashlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from lib.experiments import _array_data
from lib.filter import BinghamFilter
from lib.geometry import projector
from lib.theory import regional_bound
from lib.validation import require_gate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def job(task):
    scenario,case,rate,metrics=task
    U,X,a,R,G=_array_data(1729+100*case,72,rate,.25,'Matched')
    f=BinghamFilter(16,3,.3,2.,beta=float(metrics[0]['beta']),complex_data=True)
    by_t={int(r['t']):r for r in metrics};rows=[]
    angles=np.linspace(-70,70,281)
    steering=np.exp(1j*np.pi*np.outer(np.arange(16),np.sin(np.deg2rad(angles))))/4
    for t,u in enumerate(U,1):
        f.update(u,G=G)
        if t not in by_t:continue
        r=by_t[t];h=float(r['hpd_threshold']);threshold=float(np.trace(f.A@projector(f.basis)).real-h)
        contains=bool(np.trace(f.A@(projector(f.basis)-projector(X[t-1]))).real<=h)
        for i,angle in enumerate(angles):
            v=steering[:,i];B=np.outer(v,v.conj())
            upper=regional_bound(B,f.A,threshold,3);negative=regional_bound(-B,f.A,threshold,3)
            low=.3+2*max(0.,-negative.upper);high=.3+2*min(1.,upper.upper)
            truth=float(np.vdot(v,R[t-1]@v).real)
            if contains and not low-1e-6<=truth<=high+1e-6:
                raise ArithmeticError('State-region response certificate missed an included truth')
            rows.append(dict(scenario=scenario,t=t,angle=angle,lower=low,upper=high,truth=truth,
                optimization_gap=max(upper.gap,negative.gap),state_in_hpd=contains,
                threshold=h,threshold_mcse=float(r['hpd_threshold_mcse'])))
        print('HPD response envelope:',scenario,t,flush=True)
    return rows


def main():
    gate=require_gate(ROOT);out=ROOT/'results'
    sources={p:sha(ROOT/p) for p in ['scripts/run_array_envelopes.py','lib/experiments.py','lib/filter.py','lib/models.py','lib/geometry.py','lib/theory.py']}
    inputs={'case1_metrics.csv':sha(out/'case1_metrics.csv')}
    key=hashlib.sha256(json.dumps(dict(sources=sources,inputs=inputs,core_hash=gate['core_hash']),sort_keys=True).encode()).hexdigest()
    meta=out/'array_envelope_config.json';path=out/'array_envelope_bounds.csv'
    if meta.exists():
        old=json.loads(meta.read_text())
        if old['source_hash']==key and old['table_sha256']==sha(path):print('HPD envelopes: verified cache');return
    m=pd.read_csv(out/'case1_metrics.csv');m=m[(m['repeat']==0)&(m.method=='Posterior mean')]
    tasks=[(name,case,rate,m[m.scenario==name].to_dict('records')) for name,case,rate in [('Static',0,0.),('Drift',3,.012)]]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        results=list(pool.map(job,tasks))
    frame=pd.DataFrame([row for group in results for row in group]);frame.to_csv(path,index=False)
    assert frame.optimization_gap.max()<1e-5
    meta.write_text(json.dumps(dict(source_hash=key,sources=sources,inputs=inputs,core_hash=gate['core_hash'],table_sha256=sha(path),
        maximum_optimization_gap=float(frame.optimization_gap.max()),
        threshold_source='Original validated posterior HPD quantile for the identical target; its Monte Carlo error is retained.',
        interpretation='Outer response bounds over one estimated state HPD region; if the state lies in the region all evaluated angular responses are enclosed. No frequentist guarantee or between-grid interpolation certificate.'),indent=2))


if __name__=='__main__':main()
