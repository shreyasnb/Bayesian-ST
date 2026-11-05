"""Independent kernel, inverse, derivative and likelihood checks for Gr(2,3)."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import logsumexp

from lib.geometry import center
from lib.models import increment,constant
from lib.contact_posterior import contact_mean,normal_quadrature
from lib.process_prediction import stochastic_predict
from lib.contact_process_prediction import (kernel_parameters,kappa_from_chordal_rms,
    kernel_tangent_variance,fast_moments,invert_contact_mean,predict_contact,predictive_log_score,_spectral)


def test_watson_kernel_and_angular_variance_against_independent_quad():
    for kappa in [0.,.01,4.,20.,1000.,160000.]:
        p=kernel_parameters(kappa)
        if kappa<100:
            z=quad(lambda q:np.exp(kappa*(q*q-1)),0,1,epsabs=1e-12)[0]
            a=quad(lambda q:q*q*np.exp(kappa*(q*q-1)),0,1,epsabs=1e-12)[0]/z
            variance=quad(lambda q:np.arccos(q)**2*np.exp(kappa*(q*q-1))/2,0,1,epsabs=1e-12)[0]/z
            assert_allclose([p.a,p.angle_variance],[a,variance],atol=2e-11)
        assert_allclose(p.a+2*p.b,1,atol=1e-14)
        assert_allclose(p.rho,p.a-p.b,atol=1e-14)
        assert_allclose(p.chordal_rms**2,1-p.a,atol=1e-14)
        assert kernel_tangent_variance(kappa)==p.angle_variance
    assert_allclose(kernel_parameters(0).angle_variance,(np.pi-2)/2)
    assert kernel_parameters(np.inf).angle_variance==0


def test_exact_dispersion_conversion_and_limits():
    for rms in [.0025,.005,.01,.02,.04,.08,.5]:
        kappa=kappa_from_chordal_rms(rms)
        assert_allclose(kernel_parameters(kappa).chordal_rms,rms,atol=2e-12)
    assert np.isposinf(kappa_from_chordal_rms(0.))
    assert kappa_from_chordal_rms(np.sqrt(2/3))==0
    with pytest.raises(ValueError):kappa_from_chordal_rms(.9)
    with pytest.raises(ValueError):kernel_parameters(-1.)


def test_fast_radial_backend_against_contact_integration_and_shift():
    rng=np.random.default_rng(743021)
    for values in [[0.,0.,0.],[0.,2.,6.],[0.,20.,100.],[0.,100.,80000.],
                   [0.,80000.,80000.],[0.,0.,80000.],[0.,0.,1e6]]:
        Q=np.linalg.qr(rng.normal(size=(3,3)))[0];A=Q@np.diag(values)@Q.T
        fast=fast_moments(A,checked=True);reference,d=contact_mean(A,strict=True)
        assert fast.numerical_ok and fast.resolution_checked
        assert_allclose(fast.mean,reference,atol=3e-8)
        assert_allclose(fast.log_normalizer,d['log_normalizer'],atol=3e-8)
        shifted=fast_moments(A+17*np.eye(3),checked=True)
        assert_allclose(shifted.mean,fast.mean,atol=3e-9)
        assert_allclose(shifted.log_normalizer,fast.log_normalizer+34,atol=3e-8)
        assert_allclose(np.trace(fast.mean),2,atol=2e-12)
    unchecked=fast_moments(np.zeros((3,3)))
    assert not unchecked.resolution_checked and np.isnan(unchecked.resolution_error)


def test_log_normalizer_gradient_and_covariance_hessian():
    A=np.array([[2.,.2,-.1],[.2,-1.,.3],[-.1,.3,.5]])
    H=np.array([[.3,-.2,.1],[-.2,-.4,.5],[.1,.5,.7]])
    h=1e-5;f=fast_moments(A,True)
    derivative=(fast_moments(A+h*H).log_normalizer-fast_moments(A-h*H).log_normalizer)/(2*h)
    assert_allclose(derivative,np.trace(f.mean@H),atol=1e-9)
    gaps=np.array([2.,6.]);mu,_,cov=_spectral(gaps)
    for j in range(2):
        e=np.eye(2)[j]*h
        derivative=(_spectral(gaps+e)[0][1:]-_spectral(gaps-e)[0][1:])/(2*h)
        assert_allclose(derivative,-cov[:,j],atol=1e-10)
    assert np.linalg.eigvalsh(cov).min()>0


def test_inverse_roundtrips_broad_concentrated_and_repeated_spectra():
    for values in [[0.,0.,0.],[0.,2.,6.],[0.,20.,100.],[0.,100.,80000.],
                   [0.,80000.,80000.],[0.,0.,80000.]]:
        A=center(np.diag(values));target=fast_moments(A,True).mean
        fit=invert_contact_mean(target)
        assert fit.converged and fit.residual+fit.moment_uncertainty<=1e-8
        assert fit.objective<=fit.initial_objective+5e-8
        assert_allclose(fast_moments(fit.A,True).mean,target,atol=1e-8)
        assert_allclose(np.trace(fit.A),0,atol=1e-10)
    with pytest.raises(ValueError):invert_contact_mean(np.eye(3))
    with pytest.raises(ValueError):invert_contact_mean(np.diag([0.,1.,1.]))
    with pytest.raises(ArithmeticError):invert_contact_mean(fast_moments(np.diag([0.,2.,6.])).mean,maxiter=1)


def test_prediction_matches_existing_process_and_transport():
    rng=np.random.default_rng(743022);Q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    A=np.diag([-2.,0.,2.])
    for kappa in [0.,4.,20.]:
        p=predict_contact(A,kappa,G=Q)
        old=stochastic_predict(A,2,kappa,G=Q)
        assert p.fit.converged and p.approximate_density==(kappa>0)
        assert_allclose(p.target_mean,old.target_mean,atol=2e-10)
        assert_allclose(fast_moments(p.prior).mean,old.target_mean,atol=1e-8)
        assert_allclose(2*p.coefficients.a+p.coefficients.b,2,atol=1e-12)
        rotated=predict_contact(Q@A@Q.T,kappa)
        assert_allclose(rotated.prior,p.prior,atol=3e-8)
    exact=predict_contact(A,np.inf,G=Q)
    assert not exact.approximate_density and exact.fit.iterations==0
    assert_allclose(exact.prior,Q@A@Q.T,atol=1e-14)


def test_pre_correction_predictive_score_has_constants_and_shift_restoration():
    A=np.array([[1.,.3,0.],[.3,-2.,.1],[0.,.1,.5]])
    u=np.array([.2,-.1,.8]);v=np.array([1.,.2,-.1]);s=.12
    B=increment(u,s,v=v);q=normal_quadrature(-A,80,384)
    n=q['normal_nodes'];P=np.eye(3)-np.einsum('ni,nj->nij',n,n)
    energies=np.einsum('ij,nji->n',B,P)
    independent=constant(u,2,s,v=v)+logsumexp(np.log(q['weights'])+energies)
    score=predictive_log_score(A,u,v,s)
    assert_allclose(score,independent,atol=2e-10)
    assert_allclose(predictive_log_score(A+11*np.eye(3),u,v,s),score,atol=2e-11)
