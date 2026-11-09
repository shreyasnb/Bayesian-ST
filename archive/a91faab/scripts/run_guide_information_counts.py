"""Fisher-score error across SNR and genuine within-dataset snapshot counts."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib,json,platform,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from lib.beamforming_theory import passive_fisher_information
from lib.guide_information import score_statistics
from lib.guide_information_counts import gaussian_stream,score_prefixes

SOURCES=['lib/guide_information_counts.py','scripts/run_guide_information_counts.py',
    'tests/test_guide_information_counts.py','lib/beamforming_theory.py','lib/guide_information.py']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(profile):
    return dict(profile=profile,source_commit='6e78cc9911e7e78452633f13dd947afa669c3470',
        seed=2026110903,n=3,k=1,complex_fields=[False,True],sensor_variances=[.2,1.,4.],
        signal_variance=1.3,observations=[1,4,16,64],episodes=512 if profile=='smoke' else 8192,
        thresholds=dict(maximum_product_z=7.,maximum_score_mean_z=7.),
        variance_convention='s and tau2 are variances; each real/imag coordinate has variance s/2 for proper-complex sensor noise',
        coordinates='Canonical X=e1 and Q=[e2,e3]; vec_F Z (real), [Re vec_F Z, Im vec_F Z] (complex)',
        independence='Independent datasets and independent snapshots within a dataset; same dataset prefixes across L and same underlying Gaussian draws across sensor variances are common random numbers. Cells are correlated, not extra independent replications.',
        interpretation='Expected total information is L times the one-snapshot Fisher matrix. Monte Carlo error is estimated across independent datasets. The L axis counts observations in each dataset, never simulation episodes or posterior draws.',
        selection='All fields, variances, counts, seeds and precision diagnostics are frozen before the paper run; statistical failures remain visible.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','paper'],default='smoke')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg=configuration(args.profile)
    out=args.output or ROOT/'results/implementation_audit'/args.profile/'information_counts'
    out.mkdir(parents=True,exist_ok=True);sources={p:sha(ROOT/p) for p in SOURCES}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old['sources']!=sources or old['config']!=cfg:raise RuntimeError('Preserve previous results; choose a fresh output directory')
        for p,digest in old['tables'].items():
            if sha(out/p)!=digest:raise RuntimeError('Changed count-study artifact '+p)
        print('Verified cached count study');return
    gate=None
    if args.profile=='paper':
        from lib.implementation_validation import require_gate
        gate=require_gate(ROOT)
    declaration=dict(config=cfg,sources=sources,source_commit=cfg['source_commit'])
    (out/'preregistered.json').write_text(json.dumps(declaration,indent=2)+'\n')
    start=perf_counter();summary=[];matrices=[]
    n,k,q=cfg['n'],cfg['k'],cfg['signal_variance'];X=np.eye(n)[:,:k];Q=np.eye(n)[:,k:]
    for cx in cfg['complex_fields']:
        for s in cfg['sensor_variances']:
            # Reinitialize this field's independent stream for each s to use
            # declared common Gaussian draws across the SNR comparison.
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],int(cx),0 if args.profile=='smoke' else 1]))
            U=gaussian_stream(cfg['episodes'],max(cfg['observations']),n,k,s,q,rng,cx)
            prefixes=score_prefixes(U,X,s,q,cfg['observations'],cx,Q)
            for L,scores in prefixes.items():
                expected=passive_fisher_information(n,k,s,q,observations=L,complex_data=cx)
                empirical,mean,mcse=score_statistics(scores)
                scale=max(float(expected[0,0]),1.)*1e-13
                z=float(np.max(abs(empirical-expected)/np.maximum(mcse,scale)))
                mean_z=float(np.max(abs(mean)/np.maximum(np.sqrt(np.diag(expected)/len(scores)),scale)))
                base=dict(field='complex' if cx else 'real',complex_data=cx,n=n,k=k,s=s,tau2=q,
                    snr=q/s,observations=L,episodes=len(scores))
                summary.append(dict(**base,dimension=len(expected),expected_information=float(expected[0,0]),
                    relative_frobenius_error=float(np.linalg.norm(empirical-expected,'fro')/np.linalg.norm(expected,'fro')),
                    relative_operator_error=float(np.linalg.norm(empirical-expected,2)/np.linalg.norm(expected,2)),
                    maximum_product_z=z,maximum_score_mean_z=mean_z,
                    status='passed' if z<cfg['thresholds']['maximum_product_z'] and mean_z<cfg['thresholds']['maximum_score_mean_z'] else 'inconclusive'))
                for i in range(len(expected)):
                    for j in range(len(expected)):
                        matrices.append(dict(**base,i=i,j=j,expected=expected[i,j],score_covariance=empirical[i,j],product_mcse=mcse[i,j]))
            print('Count information', 'complex' if cx else 'real','s',s,'done',flush=True)
    frame=pd.DataFrame(summary);frame.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(matrices).to_csv(out/'matrices.csv',index=False)
    if sources!={p:sha(ROOT/p) for p in SOURCES}:raise RuntimeError('Count-study producers changed during execution')
    checks=dict(cells=len(frame),unresolved_cells=int(frame.status.ne('passed').sum()),
        maximum_score_product_z=float(frame.maximum_product_z.max()),maximum_score_mean_z=float(frame.maximum_score_mean_z.max()),
        maximum_relative_frobenius_error=float(frame.relative_frobenius_error.max()),
        minimum_expected_information=float(frame.expected_information.min()),maximum_expected_information=float(frame.expected_information.max()))
    metadata=dict(**declaration,status='completed',checks=checks,numerical_ok=bool(np.isfinite(frame.select_dtypes('number')).all().all()),
        gate_sha256=None if gate is None else sha(ROOT/'results/implementation_audit/validation/validation.json'),
        seconds=perf_counter()-start,created_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(),numpy=np.__version__),
        tables={p.name:sha(p) for p in sorted(out.glob('*.csv'))},
        scope='Mathematical Gaussian score experiment only; sampling error does not establish an estimator performance advantage or a uniform confidence guarantee.')
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
