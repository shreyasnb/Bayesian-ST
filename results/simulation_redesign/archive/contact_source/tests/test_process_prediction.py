"""Explicit process law, inverse moments, and approximation limits."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from lib.geometry import center
from lib.moments import mean_bingham,log_normalizer
from lib.process_prediction import (transition_coefficients,invert_bingham_mean,
                                    stochastic_predict,moment_and_normalizer)


def test_transition_limits_and_moment_invariants():
    for n,k,complex_data in [(2,1,False),(4,1,True),(4,3,False)]:
        zero=transition_coefficients(n,k,0.,complex_data)
        assert_allclose([zero.a,zero.b,zero.rho],[k/n,k/n,0.],atol=1e-12)
        finite=transition_coefficients(n,k,20.,complex_data)
        assert 0<finite.rho<1
        assert_allclose(k*finite.a+(n-k)*finite.b,k,atol=3e-11)
        limit=transition_coefficients(n,k,np.inf,complex_data)
        assert_allclose([limit.a,limit.b,limit.rho],[1,0,1])
        A=np.diag(np.linspace(-3,2,n))
        prediction=stochastic_predict(A,k,20.,complex_data=complex_data)
        eig=np.linalg.eigvalsh(prediction.target_mean)
        assert eig.min()>0 and eig.max()<1
        assert_allclose(eig.sum(),k,atol=1e-10)
        assert prediction.fit.converged and prediction.fit.residual<1e-8
        assert_allclose(np.trace(prediction.prior),0,atol=1e-12)
    # Finite concentrations approach the deterministic limit; this does not
    # merely exercise the special infinity branch.
    approaching=[transition_coefficients(2,1,q).rho for q in (20.,200.,2000.,1e5)]
    assert np.all(np.diff(approaching)>0)
    assert approaching[-1]>.99998


def test_inverse_moments_real_complex_and_complements():
    for n,k,complex_data in [(2,1,False),(3,1,False),(4,1,True),(4,3,True)]:
        A=center(np.diag(np.linspace(-4,3,n)))
        M,psi,uncertainty=moment_and_normalizer(A,k,complex_data)
        assert_allclose(M,mean_bingham(A,k,complex_data=complex_data),atol=3e-10)
        assert_allclose(psi,log_normalizer(A,k,complex_data=complex_data),atol=3e-10)
        fit=invert_bingham_mean(M,k,complex_data=complex_data)
        assert fit.converged and fit.residual+fit.moment_uncertainty<=1e-8
        assert fit.objective<=fit.initial_objective+1e-8
        assert_allclose(fit.A,A,atol=3e-6)
        warm=invert_bingham_mean(M,k,initial=A,complex_data=complex_data)
        assert warm.iterations<=fit.iterations


def test_unitary_equivariance_and_deterministic_prediction():
    rng=np.random.default_rng(701251)
    for complex_data in (False,True):
        Q=rng.normal(size=(3,3))
        if complex_data:Q=Q+1j*rng.normal(size=(3,3))
        Q=np.linalg.qr(Q)[0];A=np.diag([-2.,0.,2.])
        rotated=Q@A@Q.conj().T
        p=stochastic_predict(A,1,6.,complex_data=complex_data)
        q=stochastic_predict(rotated,1,6.,complex_data=complex_data)
        assert_allclose(q.target_mean,Q@p.target_mean@Q.conj().T,atol=1e-11)
        assert_allclose(q.prior,Q@p.prior@Q.conj().T,atol=1e-7)
        exact=stochastic_predict(A,1,np.inf,G=Q,complex_data=complex_data)
        assert_allclose(exact.prior,rotated,atol=1e-12)
        assert not exact.approximate_density
        uniform=stochastic_predict(A,1,0.,G=Q,complex_data=complex_data)
        assert_allclose(uniform.prior,0,atol=1e-12)


def test_transition_means_against_independent_direct_sampling():
    rng=np.random.default_rng(701252);N=60000;kappa=6.
    # Real Gr(1,2): doubled projective angle is von Mises(kappa/2).
    theta=rng.vonmises(0,kappa/2,N)/2
    X=np.column_stack([np.cos(theta),np.sin(theta)])
    empirical=X.T@X/N
    real=transition_coefficients(2,1,kappa)
    assert_allclose(empirical,np.diag([real.a,real.b]),atol=.005)
    # Complex Gr(1,2): |x_1|^2 has a truncated exponential density.
    t=np.log1p(rng.random(N)*np.expm1(kappa))/kappa
    phase=rng.uniform(0,2*np.pi,N)
    X=np.column_stack([np.sqrt(t),np.sqrt(1-t)*np.exp(1j*phase)])
    empirical=X.T@X.conj()/N
    complex_coeff=transition_coefficients(2,1,kappa,True)
    assert_allclose(empirical,np.diag([complex_coeff.a,complex_coeff.b]),atol=.005)


def test_reject_unsupported_and_unresolved_process_fits():
    with pytest.raises(NotImplementedError):transition_coefficients(8,3,10.)
    with pytest.raises(NotImplementedError):stochastic_predict(np.zeros((8,8)),3,10.)
    with pytest.raises(ValueError):transition_coefficients(3,1,-1.)
    with pytest.raises(ValueError):invert_bingham_mean(np.diag([1.,0.]))
    with pytest.raises(ValueError):invert_bingham_mean(np.eye(3)/2)
    with pytest.raises(ValueError):stochastic_predict(np.eye(3),1,2.,G=np.ones((3,3)))
    A=np.diag([-6.,0.,6.]);M=moment_and_normalizer(A)[0]
    with pytest.raises(ArithmeticError):invert_bingham_mean(M,maxiter=1)
