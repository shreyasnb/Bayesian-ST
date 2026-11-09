"""Check contact-pilot deterministic moments, tails and high concentrations.

This numerical development protocol changes no plant parameters and uses no
final test trajectories. It checks the exact normal-line complement, analytic
pole/ridge integrals, existing spectral moments and continuous region CDFs.
"""
from pathlib import Path
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import hashlib
import json
import platform
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from scipy.special import dawsn,erf
from lib.contact_posterior import normal_quadrature,contact_mean,contact_summary
from lib.contact_mechanics import generate_contact
from lib.filter import BinghamFilter


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def queries(v0=.02):
    pairs=np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]])/np.sqrt(2)
    return v0*np.vstack([np.eye(3),pairs])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/contact_pilot/numerics')
    parser.add_argument('--quick',action='store_true',help='One representative beta per condition')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    checks=[]
    for kappa in [80000.,1e6]:
        root=np.sqrt(kappa)
        for kind,C in [('pole',np.diag([0.,-kappa,-kappa])),('ridge',np.diag([0.,0.,-kappa]))]:
            evaluated=normal_quadrature(C)
            if kind=='pole':
                logz=np.log(dawsn(root))-.5*np.log(kappa)
                top=1/(2*root*dawsn(root))-1/(2*kappa)
                M=np.diag([top,(1-top)/2,(1-top)/2])
            else:
                logz=.5*np.log(np.pi)+np.log(erf(root))-np.log(2)-.5*np.log(kappa)
                bottom=1/(2*kappa)-np.exp(-kappa)/(np.sqrt(np.pi)*root*erf(root))
                M=np.diag([(1-bottom)/2,(1-bottom)/2,bottom])
            error=float(np.max(abs(M-evaluated['mean'])))
            log_error=float(abs(logz-evaluated['log_normalizer']))
            assert error<3e-10 and log_error<3e-10
            checks.append(dict(reference=kind,kappa=kappa,mean_error=error,log_normalizer_error=log_error,
                               radial_tail_bound=evaluated['omitted_radial_relative_bound']))
    rows=[];started=perf_counter();W=queries();development_seed=6245199
    betas=[.8] if args.quick else [1.,.98,.95,.9,.8]
    for condition in ['plane','sphere_large','sphere_small','weak','anisotropic']:
        data=generate_contact(development_seed,0,condition)
        for beta in ([1.] if condition=='plane' else betas):
            tracker=BinghamFilter(3,2,1e-6,beta=beta)
            for t,(u,v) in enumerate(zip(data['U'],data['V']),1):
                tracker.update(u,v=v)
                if t not in [60,180,600]:continue
                result=contact_summary(tracker.A,W,true_normal=data['normals'][t-1])
                row=dict(condition=condition,beta=beta,t=t,**result['numerics'],
                         ball_radius=result['ball_radius'],hpd_threshold=result['hpd_threshold'],
                         normalized_mean_interval_width=float(np.mean(result['response_upper']-result['response_lower'])/.02))
                rows.append(row)
                print(condition,beta,t,'span',round(row['span'],1),'numeric',row['numerical_ok'],flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'resolution_checks.csv',index=False)
    assert frame.numerical_ok.all(),'Numerical development targets unresolved; inspect resolution_checks.csv'
    sources={p:sha(ROOT/p) for p in ['lib/contact_posterior.py','scripts/check_contact_posterior.py',
        'lib/contact_mechanics.py','lib/filter.py','lib/models.py','lib/moments.py','lib/reference.py','lib/spectral_series.py']}
    result=dict(status='passed',independent_analytic_checks=checks,development_seed=development_seed,
        development_scope='Numerical resolutions only; no test truth used to tune the model or uncertainty',
        physical_targets=len(frame),maximum_span=float(frame.span.max()),
        maximum_mean_resolution_error=float(frame.mean_resolution_error.max()),
        maximum_quantile_scaled_error=float(frame.quantile_scaled_error.max()),
        maximum_quantile_relative_error=float(frame.quantile_relative_error.max()),
        maximum_region_integration_error=float(frame.region_integration_error.max()),
        maximum_region_radius_scaled_error=float(frame.region_radius_scaled_error.max()),
        maximum_region_hpd_scaled_error=float(frame.region_hpd_scaled_error.max()),
        maximum_moment_region_logz_error=float(frame.moment_region_logz_error.max()),
        minimum_angular_order=int(frame.quantile_angular_order.min()),maximum_angular_order=int(frame.quantile_angular_order.max()),
        seconds=perf_counter()-started,sources=sources,
        tables={'resolution_checks.csv':sha(out/'resolution_checks.csv')},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        limitations='Resolution changes and QUADPACK error estimates are not rigorous integration/roundoff enclosures',
        backend_scope='Existing spectral means through span 600; concentration-adapted real Gr(1,3) complement quadrature beyond. No numerical core changed. No MCMC.',
        quantile_method='Continuous radial functional CDF using analytic threshold intervals and Dawson integrals; warped angular quadrature with width-relative resolution gate',
        noise_intervals='Clean intervals only; no Gaussian moment surrogate for observation intervals',
        high_concentration_reason='The existing positive-series tail gate requires order+2 greater than the shifted span; a default cap4096 cannot certify span80000 and extending its quadratic recurrence is unnecessary in this dimension')
    (out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','independent_analytic_checks']},indent=2))


if __name__=='__main__':main()
