"""Independent complex-line integration, sampling and process checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.polynomial.legendre import leggauss
from lib.beamforming_numerics import (complex_moments,sample_complex_bingham,
    sample_complex_transition,transition_coefficients_complex,transition_log_normalizer,
    invert_complex_mean,predict_complex,cp1_power_cdf,cp1_regions,posterior_report,
    _spectral_covariance,sample_complex_transitions,sample_complex_laplace)
from lib.spectral_series import rank_one_bingham
from lib.geometry import center


def unitary(n,rng):
    Q,R=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    return Q


def simplex3(lam,order=96):
    """Independent Cartesian simplex integration, including its Jacobian."""
    nodes,weights=leggauss(order);x=(nodes+1)/2;w=weights/2
    first,second=np.meshgrid(x,x,indexing='ij')
    X=np.stack([first,(1-first)*second,(1-first)*(1-second)],axis=-1).reshape(-1,3)
    base=(2*np.outer(w,w)*(1-first)).ravel()
    mass=base*np.exp(X@np.asarray(lam)-max(lam));Z=mass.sum();mass/=Z
    return X.T@mass,np.einsum('b,bi,bj->ij',mass,X,X),float(max(lam)+np.log(Z))


def test_native_complex_cp1_formula_matches_series_and_shift():
    rng=np.random.default_rng(89127);Q=unitary(2,rng)
    for gap in [0.,1e-7,.4,8.,200.]:
        A=Q@np.diag([-gap/2,gap/2])@Q.conj().T
        exact=complex_moments(A);series=rank_one_bingham(A,complex_data=True,tol=1e-12)
        assert_allclose(exact.mean,series.mean,atol=2e-12)
        assert_allclose(exact.log_normalizer,series.log_normalizer,atol=3e-12)
        shifted=complex_moments(A+13*np.eye(2))
        assert_allclose(shifted.mean,exact.mean,atol=2e-13)
        assert_allclose(shifted.log_normalizer,exact.log_normalizer+13,atol=2e-13)


def test_complex_simplex_covariance_matches_independent_quadrature():
    for lam in [[3.,3.,0.],[0.,.001,40.],[0.,0.,0.]]:
        expected,second,logz=simplex3(lam)
        result=complex_moments(np.diag(lam),tol=1e-12)
        assert_allclose(result.mean.diagonal().real,expected,atol=4e-12)
        assert_allclose(result.log_normalizer,logz,atol=3e-12)
        covariance=_spectral_covariance(lam,result.mean.diagonal().real)
        assert_allclose(covariance,second-np.outer(expected,expected),atol=4e-11)
        assert_allclose(covariance@np.ones(3),0,atol=2e-12)


def test_iid_complex_sampler_matches_cp1_continuous_cdf_and_phase_measure():
    rng=np.random.default_rng(625711);Q=unitary(2,rng);A=Q@np.diag([-.7,.7])@Q.conj().T
    samples=sample_complex_bingham(A,30000,rng)
    query=np.array([1.,1j])/np.sqrt(2);power=abs(samples.vectors@query.conj())**2
    for p in [.1,.3,.7,.9]:
        assert abs(np.mean(power<=p)-cp1_power_cdf(A,query,p))<.012
    assert_allclose(np.sum(abs(samples.vectors)**2,axis=1),1,atol=1e-12)
    assert samples.diagnostics['proposal']=='truncated exponential box'
    assert samples.diagnostics['acceptance']==1.
    # Native complex phases have circular variation, unlike real-sphere draws.
    coordinates=samples.vectors@Q.conj()
    assert np.var(np.angle(coordinates[:,0]*coordinates[:,1].conj()))>3.


def test_cp1_cdf_resolves_hostile_boundary_layer_and_agrees_with_bloch_integral():
    from scipy.integrate import quad
    from scipy.special import i0e
    gap=1e6;y=1-1/gap
    expected=np.exp(gap*(y-1))*(-np.expm1(-gap*y))/(-np.expm1(-gap))
    assert_allclose(cp1_power_cdf(np.diag([0.,gap]),[0.,1.],y),expected,atol=1e-13)
    assert_allclose(cp1_power_cdf(np.diag([0.,gap]),[1.,0.],1/gap),-np.expm1(-1),atol=1e-13)
    # Independently integrate the vMF Bloch-coordinate marginal in its ordinary
    # (well-resolved) regime; this is not the radial-quantile implementation.
    for gap in [.01,4.,100.,1000.]:
        overlap=.23;query=np.sqrt([1-overlap,overlap]);c=gap/2
        cosine=2*overlap-1;sine=np.sqrt(1-cosine*cosine)
        for x in [.2,.23,.3]:
            def density(z):
                b=c*sine*np.sqrt(max(0.,1-z*z))
                return np.exp(c*cosine*z+b-c)*i0e(b)
            reference=quad(density,-1,2*x-1,epsabs=1e-12,epsrel=1e-11,limit=300)[0]/(-np.expm1(-2*c)/c)
            assert_allclose(cp1_power_cdf(np.diag([0.,gap]),query,x),reference,atol=3e-9)
    rng=np.random.default_rng(625121);query=np.sqrt([.77,.23]);A=np.diag([0.,1e6])
    draws=sample_complex_bingham(A,100000,rng).vectors
    power=abs(draws@query.conj())**2
    for x in [.2295,.23,.2305]:
        assert abs(cp1_power_cdf(A,query,x)-np.mean(power<=x))<.012


def test_iid_envelopes_have_correct_normalization_and_high_dimension_mean():
    rng=np.random.default_rng(725531)
    for spectrum in [np.zeros(8),np.linspace(0,5,8),np.r_[20.,20.,np.zeros(6)],np.linspace(0,400,8)]:
        A=np.diag(spectrum);sample=sample_complex_bingham(A,20000,rng);forward=complex_moments(A)
        assert np.linalg.norm(sample.mean-forward.mean,2)<.018
        log_accept=forward.log_normalizer-max(spectrum)
        if sample.diagnostics['proposal']=='truncated exponential box':
            log_accept+=sample.diagnostics['log_box_to_uniform_acceptance_ratio']
        expected=np.exp(log_accept);N=sample.diagnostics['proposals']
        assert abs(sample.diagnostics['acceptance']-expected)<6*np.sqrt(max(expected*(1-expected),1/N)/N)+.001
    failed=sample_complex_bingham(np.diag(np.linspace(0,5,8)),100,rng,max_proposals=100,strict=False)
    assert not failed.diagnostics['numerical_ok']
    with pytest.raises(ArithmeticError):
        sample_complex_bingham(np.diag(np.linspace(0,5,8)),100,np.random.default_rng(981),max_proposals=100)


def test_complex_transition_radial_measure_coefficients_and_log_normalizer():
    rng=np.random.default_rng(625734);n=8;h=unitary(n,rng)[:,0];P=np.outer(h,h.conj())
    for kappa in [0.,1e-7,4.,20.,100.]:
        coefficients=transition_coefficients_complex(n,kappa)
        result=rank_one_bingham(kappa*P,complex_data=True,tol=1e-12)
        assert_allclose(result.log_normalizer,transition_log_normalizer(n,kappa),atol=3e-12)
        assert_allclose(result.mean,coefficients.a*P+coefficients.b*(np.eye(n)-P),atol=4e-12)
        draws=sample_complex_transition(h,kappa,rng,size=30000)
        empirical=draws.T@draws.conj()/len(draws)
        assert np.linalg.norm(empirical-result.mean,2)<.015
        assert_allclose(np.sum(abs(draws)**2,axis=1),1,atol=2e-12)
    assert_allclose(sample_complex_transition(h,np.inf,rng,size=5),np.broadcast_to(h,(5,n)))
    assert_allclose(transition_coefficients_complex(n,0).a,1/n,atol=1e-15)
    assert transition_log_normalizer(n,0)==0.


def test_inverse_mean_uses_complex_covariance_and_converges_after_rotation():
    rng=np.random.default_rng(782112)
    for lam in [[3.,3.,0.],[0.,.001,40.]]:
        target=simplex3(lam)[0];Q=unitary(3,rng);M=(Q*target)@Q.conj().T
        fit=invert_complex_mean(M)
        assert fit.converged and fit.residual+fit.moment_uncertainty<=1e-8
        assert_allclose(complex_moments(fit.A,tol=1e-12).mean,M,atol=1e-8)
        assert fit.objective<=fit.initial_objective+5e-8
    for n in [2,8]:
        Q=unitary(n,rng);A=center(Q@np.diag(np.linspace(0,30,n))@Q.conj().T)
        result=predict_complex(A,20.)
        assert result.fit.converged
        assert_allclose(complex_moments(result.prior).mean,result.target_mean,atol=1e-8)
        deterministic=predict_complex(A,np.inf,G=Q)
        assert_allclose(deterministic.prior,Q@A@Q.conj().T,atol=2e-12)
        assert not deterministic.approximate_density


def test_batch_transition_mean_and_same_parameter_laplace_shape():
    rng=np.random.default_rng(457175);A=np.diag(np.linspace(0,30,8));forward=complex_moments(A)
    old=sample_complex_bingham(A,30000,rng).vectors
    propagated=sample_complex_transitions(old,20.,rng)
    kernel=transition_coefficients_complex(8,20.)
    target=kernel.rho*forward.mean+(1-kernel.rho)*np.eye(8)/8
    assert np.linalg.norm(propagated.T@propagated.conj()/len(old)-target,2)<.015
    local=sample_complex_laplace(np.diag([200.,0.,0.]),20000,rng)
    assert local.diagnostics['local_approximation_ok']
    assert_allclose(np.sum(abs(local.vectors)**2,axis=1),1.,atol=2e-12)
    mean=complex_moments(np.diag([200.,0.,0.])).mean
    assert np.linalg.norm(local.mean-mean,2)<.003
    broad=sample_complex_laplace(np.diag([.2,0.,0.]),4000,rng)
    assert not broad.diagnostics['local_approximation_ok']
    with pytest.raises(ArithmeticError):sample_complex_laplace(np.zeros((3,3)),10,rng)


def test_posterior_report_has_independent_tail_checks_and_valid_uniform_ties():
    rng=np.random.default_rng(715533);W=np.eye(2,dtype=complex)
    report=posterior_report(np.diag([0.,4.]),W,truth=np.array([1.,0.]),draws=8192,rng=rng)
    assert report['quadratic_quantiles'].shape==(2,5)
    assert report['numerics']['numerical_ok']
    for i,w in enumerate(W):
        for j,probability in enumerate(report['quantile_probabilities']):
            assert abs(cp1_power_cdf(np.diag([0.,4.]),w,report['quadratic_quantiles'][i,j])-probability)<.02
    exact=cp1_regions(np.diag([0.,4.]))
    assert_allclose(report['ball_radius'],exact['ball_radius'],atol=0)
    uniform=posterior_report(np.zeros((8,8)),np.eye(8)[:2],draws=8192,seed=815731)
    assert_allclose(uniform['ball_radius']**14,.95,atol=2e-15)
    assert 'Haar-centred ball' in uniform['numerics']['hpd_tie_kind']
    assert_allclose(uniform['hpd_constraint_threshold'],1-uniform['ball_radius']**2)
    assert np.linalg.matrix_rank(uniform['hpd_constraint_matrix'],tol=1e-10)==1
    # Projection-risk identity uses the mean; posterior state mode alone would
    # incorrectly produce zero risk.
    assert np.all(uniform['risks']>0)
