"""Independent deterministic references for high-SNR contact uncertainty."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import dawsn,erf,ndtri

from lib.contact_posterior import (normal_quadrature,contact_mean,
                                   response_quantiles,contact_summary)
from lib.moments import mean_bingham,log_normalizer


def queries():
    pairs=np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]])/np.sqrt(2)
    return .02*np.vstack([np.eye(3),pairs])


def test_high_concentration_pole_and_ridge_against_exact_integrals():
    for kappa in [80000.,1e6]:
        # Uniform sphere squared coordinates have independent exact integrals.
        root=np.sqrt(kappa)
        pole=normal_quadrature(np.diag([0.,-kappa,-kappa]))
        exact_log=np.log(dawsn(root))-.5*np.log(kappa)
        top=1/(2*root*dawsn(root))-1/(2*kappa)
        assert_allclose(pole['log_normalizer'],exact_log,atol=2e-10)
        assert_allclose(pole['mean'],np.diag([top,(1-top)/2,(1-top)/2]),atol=3e-11)
        assert pole['omitted_radial_relative_bound']<1e-12
        ridge=normal_quadrature(np.diag([0.,0.,-kappa]))
        exact_log=.5*np.log(np.pi)+np.log(erf(root))-np.log(2)-.5*np.log(kappa)
        bottom=1/(2*kappa)-np.exp(-kappa)/(np.sqrt(np.pi)*root*erf(root))
        assert_allclose(ridge['log_normalizer'],exact_log,atol=3e-10)
        assert_allclose(ridge['mean'],np.diag([(1-bottom)/2,(1-bottom)/2,bottom]),atol=3e-10)


def test_existing_spectral_backend_and_complement_duality_agree():
    A=np.diag([0.,8.,80.])
    M,diagnostics=contact_mean(A,strict=True)
    assert 'existing spectral series' in diagnostics['mean_backend']
    assert_allclose(M,mean_bingham(A,2),atol=2e-10)
    assert_allclose(diagnostics['log_normalizer'],log_normalizer(A,2),atol=2e-10)
    assert_allclose(np.trace(M),2.,atol=2e-10)
    assert np.linalg.eigvalsh(M).min()>0 and np.linalg.eigvalsh(M).max()<1


def test_shift_equivariance_and_highly_anisotropic_means():
    rng=np.random.default_rng(829415)
    Q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    A=np.diag([0.,100.,80000.])
    M,d=contact_mean(A,strict=True)
    transformed,e=contact_mean(Q@A@Q.T+13*np.eye(3),strict=True)
    assert_allclose(transformed,Q@M@Q.T,atol=2e-10)
    assert_allclose(e['log_normalizer'],d['log_normalizer']+26,atol=2e-9)
    assert d['mean_resolution_error']<1e-8


def test_uniform_functional_quantiles_and_hpd_atom_are_explicit():
    # For uniform n on S2, n_1 is uniform[-1,1], so 1-n_1^2 has
    # quantile 1-(1-p)^2. This checks a functional CDF independently.
    W=np.array([[.02,0.,0.]])
    lo,hi,d=response_quantiles(np.zeros((3,3)),W)
    assert d['response_quantiles_ok']
    assert_allclose([lo[0,0],hi[0,0]],.02*(1-(1-np.array([.025,.975]))**2),atol=2e-6)
    assert_allclose(lo[0,1:],-hi[0,1:],atol=1e-11)
    result=contact_summary(np.zeros((3,3)),W,strict=True)
    assert_allclose(result['mean'],2*np.eye(3)/3,atol=1e-12)
    assert result['hpd_threshold']==0
    assert result['numerics']['hpd_probability']==1.
    assert not result['numerics']['mode_unique']


def test_continuous_high_concentration_quantiles_and_state_regions():
    kappa=80000.;A=np.diag([0.,kappa,kappa]);W=.02*np.eye(3)
    result=contact_summary(A,W,true_normal=np.array([1.,0.,0.]),strict=True)
    lo,hi=result['response_lower'],result['response_upper']
    # The scaled polar loss converges to Exp(1), and transverse components
    # to independent N(0,1/(2*kappa)). Corrections are O(1/kappa).
    assert_allclose(np.array([lo[0,0],hi[0,0]])/.02,
                    -np.log(1-np.array([.025,.975]))/kappa,rtol=3e-5,atol=2e-10)
    assert_allclose(hi[1,0]/.02,ndtri(.975)/np.sqrt(2*kappa),rtol=5e-5)
    assert_allclose(lo[1,0],-hi[1,0],atol=2e-11)
    assert_allclose(result['ball_radius'],.0061193862004,atol=2e-12)
    assert_allclose(result['hpd_threshold'],2.99575099758,atol=2e-10)
    assert result['ball_covered'] and result['hpd_covered']
    assert result['numerics']['region_integration_error']<1e-7
    assert result['numerics']['mean_numerical_ok']
    assert result['numerics']['regions_numerical_ok']
    assert result['numerics']['region_radius_scaled_error']<1e-6
    assert result['numerics']['region_hpd_scaled_error']<1e-5


def test_resolution_failure_is_retained_and_no_noisy_interval_shortcut():
    lo,hi,d=response_quantiles(np.zeros((3,3)),queries(),angular_orders=(32,64),max_order=64)
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert not d['response_quantiles_ok']
    with pytest.raises(ValueError,match='clean intervals only'):
        contact_summary(np.eye(3),queries(),noise_variance=1e-6)
    with pytest.raises(ValueError):normal_quadrature(np.ones((3,2)))
