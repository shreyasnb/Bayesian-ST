"""Independent finite-budget checks for the rank-one spectral series backend.

No application performance ranking is a test. Tail bounds are truncation bounds,
not interval-arithmetic enclosures. Run from any directory with Python 3.12.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
from time import perf_counter
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from scipy.special import i0e,i1e
from numpy.testing import assert_allclose
from lib.spectral_series import rank_one_bingham
from lib.moments import quadrature_gr13,sample_bingham,chain_diagnostics
from lib.reference import rank_one_regions_gr13


def unitary(rng,n,complex_data):
    A=rng.normal(size=(n,n))
    if complex_data:A=A+1j*rng.normal(size=(n,n))
    return np.linalg.qr(A)[0]


def main():
    checks={};rng=np.random.default_rng(5117)
    for complex_data in (False,True):
        result=rank_one_bingham(7.2*np.eye(8),complex_data=complex_data)
        assert_allclose(result.mean,np.eye(8)/8,atol=1e-15)
        assert result.log_normalizer==7.2 and result.order==0
    errors=[]
    for d in (.1,3.,80.,600.):
        result=rank_one_bingham(np.diag([d,0.]),tol=1e-12)
        exact_log=d/2+np.log(i0e(d/2))+d/2
        exact_high=.5+.5*i1e(d/2)/i0e(d/2)
        error=max(abs(result.log_normalizer-exact_log),np.max(abs(result.mean.diagonal()-[exact_high,1-exact_high])))
        assert error<2e-10
        errors.append(float(error))
        result=rank_one_bingham(np.diag([d,0.]),complex_data=True,tol=1e-12)
        exact_log=d+np.log(-np.expm1(-d))-np.log(d)
        exact_high=1/(-np.expm1(-d))-1/d
        error=max(abs(result.log_normalizer-exact_log),np.max(abs(result.mean.diagonal()-[exact_high,1-exact_high])))
        assert error<2e-10
        errors.append(float(error))
    checks['analytic_normalizers_max_error']=max(errors)
    A=np.diag([0.,-8.,-80.]);result=rank_one_bingham(A,tol=1e-12)
    q=quadrature_gr13(A,96);continuous=rank_one_regions_gr13(A)
    assert_allclose(result.mean,q.mean,atol=2e-11)
    assert_allclose(result.log_normalizer,continuous.log_normalizer,atol=2e-11)
    checks['appendix_I_mean_error']=float(np.max(abs(result.mean-q.mean)))
    checks['appendix_I_log_normalizer_error']=float(abs(result.log_normalizer-continuous.log_normalizer))
    fd=[]
    for complex_data in (False,True):
        A=rng.normal(size=(5,5));A=(A+A.T)/2
        if complex_data:
            B=rng.normal(size=(5,5));A=A+1j*(B-B.T)/2
        H=rng.normal(size=(5,5));H=(H+H.T)/2
        if complex_data:
            B=rng.normal(size=(5,5));H=H+1j*(B-B.T)/2
        r=rank_one_bingham(A,complex_data=complex_data,tol=1e-12)
        h=1e-5
        plus=rank_one_bingham(A+h*H,complex_data=complex_data,fixed_order=r.order)
        minus=rank_one_bingham(A-h*H,complex_data=complex_data,fixed_order=r.order)
        numerical=(plus.log_normalizer-minus.log_normalizer)/(2*h)
        analytic=np.trace(r.mean@H).real
        fd.append(float(abs(numerical-analytic)))
        assert abs(numerical-analytic)<3e-9
        U=unitary(rng,5,complex_data)
        transformed=rank_one_bingham(U@A@U.conj().T,complex_data=complex_data,tol=1e-12)
        shifted=rank_one_bingham(A+17.2*np.eye(5),complex_data=complex_data,tol=1e-12)
        assert_allclose(transformed.mean,U@r.mean@U.conj().T,atol=3e-11)
        assert_allclose(transformed.log_normalizer,r.log_normalizer,atol=3e-11)
        assert_allclose(shifted.mean,r.mean,atol=3e-11)
        assert_allclose(shifted.log_normalizer,r.log_normalizer+17.2,atol=3e-11)
        assert abs(np.trace(r.mean)-1)<1e-11
    checks['fixed_order_gradient_max_error']=max(fd)
    for complex_data in (False,True):
        n=8;H=np.diag(np.linspace(-1,1,n));e=1e-3
        plus=rank_one_bingham(e*H,complex_data=complex_data,tol=1e-13).mean
        minus=rank_one_bingham(-e*H,complex_data=complex_data,tol=1e-13).mean
        coefficient=1/(n*(n+1)) if complex_data else 2/(n*(n+2))
        assert_allclose((plus-minus)/(2*e),coefficient*H,atol=3e-10)
        spectrum=np.array([-3.,-3.,0.,0.,0.,2.,2.,2.])
        r=rank_one_bingham(np.diag(spectrum),complex_data=complex_data)
        diagonal=r.mean.diagonal()
        for value in np.unique(spectrum):
            assert np.ptp(diagonal[spectrum==value].real)<1e-14
    failures=0
    for callback in [lambda:rank_one_bingham(np.eye(4),k=2),
                     lambda:rank_one_bingham(np.diag([0.,600.]),max_order=8),
                     lambda:rank_one_bingham(np.ones((2,3))),
                     lambda:rank_one_bingham(np.array([[0.,1.],[0.,0.]]))]:
        try:callback()
        except (ValueError,ArithmeticError):failures+=1
    assert failures==4
    checks['invalid_or_unresolved_inputs_rejected']=failures
    independent=[]
    for complex_data in (False,True):
        A=np.diag(np.linspace(-8.,0.,8))
        result=rank_one_bingham(A,complex_data=complex_data)
        chains=[sample_bingham(A,1,seed,draws=12000,burnin=3000,complex_data=complex_data)
                for seed in (231,232)]
        diagnostics=chain_diagnostics(chains)
        differences=[float(np.max(abs(result.mean-chain.mean))) for chain in chains]
        assert diagnostics['max_rhat']<1.1
        assert diagnostics['min_ess']>30
        assert max(differences)<max(.02,5*diagnostics['max_mcse'])
        independent.append(dict(complex_data=complex_data,diagnostics=diagnostics,
                                maximum_series_chain_difference=max(differences)))
    checks['independent_chains']=independent
    timings=[]
    for n,span in [(3,8.),(8,20.),(8,600.),(16,20.),(16,600.),(64,20.)]:
        A=np.diag(np.linspace(-span,0,n))
        for complex_data in (False,True):
            r=rank_one_bingham(A,complex_data=complex_data,tol=1e-10)
            start=perf_counter()
            for _ in range(5):rank_one_bingham(A,complex_data=complex_data,tol=1e-10)
            elapsed=(perf_counter()-start)/5
            assert r.accuracy_met and r.moment_error_bound<=1e-10 and r.trace_error<1e-10
            timings.append(dict(n=n,span=span,complex_data=complex_data,seconds=elapsed,
                                order=r.order,tail_bound=r.relative_tail_bound,
                                moment_error_bound=r.moment_error_bound,trace_error=r.trace_error))
    checks['timings']=timings
    checks['scope']='Rank-one real and complex; truncation bounds exclude floating-point rounding; no Hessian/Fisher certificate'
    checks['status']='passed'
    checks['versions']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    checks['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                            ('lib/spectral_series.py','scripts/check_spectral_series.py')}
    path=ROOT/'results/spectral_series_checks.json'
    path.write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
