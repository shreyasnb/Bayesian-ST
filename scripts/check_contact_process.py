"""Independent contact-process numerical checks and bounded development timings."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import hashlib,json,platform,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.contact_process_prediction import (kernel_parameters,kappa_from_chordal_rms,
    fast_moments,invert_contact_mean,predict_contact)
from lib.contact_posterior import contact_mean
from lib.contact_mechanics import generate_contact
from lib.geometry import center
from lib.models import increment


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=ROOT/'results/contact_uncertainty/numerics/process';out.mkdir(parents=True,exist_ok=True)
    references=[];rng=np.random.default_rng(743029)
    spectra=[[0,0,0],[0,.2,1],[0,2,6],[0,20,100],[0,100,80000],
             [0,80000,80000],[0,0,80000],[0,0,1e6]]
    for values in spectra:
        Q=np.linalg.qr(rng.normal(size=(3,3)))[0];A=Q@np.diag(values)@Q.T
        f=fast_moments(A,checked=True);m,d=contact_mean(A,strict=True)
        fit=invert_contact_mean(f.mean)
        references.append(dict(spectrum=values,mean_error=float(np.max(abs(f.mean-m))),
            log_normalizer_error=abs(f.log_normalizer-d['log_normalizer']),
            backend_resolution_error=f.resolution_error,inverse_residual=fit.residual,
            inverse_backend_error=fit.moment_uncertainty,inverse_iterations=fit.iterations,
            inverse_seconds=fit.seconds,numerical_ok=f.numerical_ok and fit.converged))
    rows=[]
    data=generate_contact(743030,0,'sphere_small',T=180)
    for rms in [0.,.0025,.005,.01,.02,.04,.08]:
        kappa=kappa_from_chordal_rms(rms);kernel=kernel_parameters(kappa)
        A=np.zeros((3,3));M=2*np.eye(3)/3;start=perf_counter();residual=[];error=[];iterations=[]
        for u,v in zip(data['U'],data['V']):
            p=predict_contact(A,kappa,previous_mean=M)
            residual.append(p.fit.residual);error.append(p.fit.moment_uncertainty);iterations.append(p.fit.iterations)
            A=center(p.prior+increment(u,1e-6,v=v));M=fast_moments(A).mean
        rows.append(dict(requested_rms=rms,kappa=kappa,actual_rms=kernel.chordal_rms,
            rho=kernel.rho,tangent_variance=kernel.angle_variance,
            kernel_integration_error=kernel.integration_error,observations=len(data['U']),
            cumulative_prediction_correction_mean_seconds=perf_counter()-start,
            maximum_inverse_residual=max(residual),maximum_inverse_backend_error=max(error),
            mean_inverse_iterations=float(np.mean(iterations))))
    pd.DataFrame(rows).to_csv(out/'timings.csv',index=False)
    assert all(row['numerical_ok'] for row in references)
    assert max(row['mean_error'] for row in references)<3e-8
    assert max(row['log_normalizer_error'] for row in references)<3e-8
    record=dict(status='passed',development_seeds=[743029,743030],references=references,
        timings=rows,sources={p:sha(ROOT/p) for p in ['lib/contact_process_prediction.py',
            'scripts/check_contact_process.py','lib/contact_posterior.py','lib/process_prediction.py',
            'lib/contact_mechanics.py','lib/models.py','lib/geometry.py']},
        tables={'timings.csv':sha(out/'timings.csv')},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        scope='Exact kernel moment; approximate Bingham density closure. Resolution errors exclude rigorous roundoff enclosures. Per-step unchecked angular budgets are development-validated, not individually certified.',
        cost='Prediction, correction and posterior mean combined at every step; inversion is nested within prediction. Timings are cloud implementation measurements, not real-time guarantees.')
    (out/'checks.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
