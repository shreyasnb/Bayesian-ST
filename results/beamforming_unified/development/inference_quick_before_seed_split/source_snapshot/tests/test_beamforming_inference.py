"""Independent checks of passive ordered-power and process conditionals."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import gammainc, gammaln
from scipy.stats import invgamma

from lib.beamforming_inference import (PowerPrior,ordered_prior_normalizer,
    ordered_prior_log_density,truncated_inverse_gamma,variance_conditionals,
    passive_log_likelihood,path_state_parameter,transition_overlap,
    kappa_log_density,sample_kappa,joint_covariance_mean,stationary_gibbs,
    full_path_gibbs,stationary_reference_n2,recorded_prior_em,joint_predictive_score,
    JointResult,joint_forecast_covariance)
from lib.beamforming_protocol import proper_complex,haar_line
from lib.beamforming_numerics import transition_log_normalizer


def fixture(n=3,T=5,seed=79):
    rng=np.random.default_rng(seed);h=haar_line(rng,n)
    U=proper_complex(rng,T)[:,None]*h+proper_complex(rng,(T,n),.2)
    return U,h,rng


def test_ordered_joint_prior_has_one_global_normalizer():
    p=PowerPrior();expected=quad(lambda b:invgamma.pdf(b,p.b_shape,scale=p.b_scale)
        *invgamma.sf(b,p.a_shape,scale=p.a_scale),0,np.inf,epsabs=1e-11)[0]
    assert_allclose(ordered_prior_normalizer(p),expected,atol=2e-11)
    a,b=2.,.4
    direct=invgamma.logpdf(a,p.a_shape,scale=p.a_scale)+invgamma.logpdf(b,p.b_shape,scale=p.b_scale)-np.log(expected)
    assert_allclose(ordered_prior_log_density(a,b,p),direct,atol=2e-12)
    assert ordered_prior_log_density(b,a,p)==-np.inf


def test_truncated_inverse_gamma_preserves_the_conditional_cdf():
    rng=np.random.default_rng(993)
    for lower,upper in [(.4,np.inf),(0.,.3),(.4,.7)]:
        x=np.array([truncated_inverse_gamma(5.,2.,rng,lower,upper) for _ in range(4000)])
        cdf=invgamma.cdf(x,5.,scale=2.)
        lo=invgamma.cdf(lower,5.,scale=2.);hi=invgamma.cdf(upper,5.,scale=2.)
        uniform=(cdf-lo)/(hi-lo)
        assert np.all((x>lower)&(x<upper))
        assert abs(uniform.mean()-.5)<.02
        assert abs(np.mean(uniform<.25)-.25)<.025


def test_complex_variance_counts_and_energy_match_full_likelihood_ratios():
    U,h,_=fixture();p=PowerPrior();v=variance_conditionals(U,h,p);T,n=U.shape
    assert v['a_shape']==p.a_shape+T
    assert v['b_shape']==p.b_shape+(n-1)*T
    assert_allclose(v['parallel_energy']+v['perpendicular_energy'],np.sum(abs(U)**2),atol=1e-12)
    for a,b in [(1.2,.2),(2.,.7)]:
        joint=passive_log_likelihood(U,h,a,b)+ordered_prior_log_density(a,b,p)
        shape_log=-(v['a_shape']+1)*np.log(a)-v['a_scale']/a
        shape_log+=-(v['b_shape']+1)*np.log(b)-v['b_scale']/b
        constant=joint-shape_log
        if a==1.2:reference=constant
        else:assert_allclose(constant,reference,atol=1e-11)


def test_moving_state_conditional_has_both_neighbors_and_correct_transports():
    U,_,rng=fixture(n=3,T=4)
    states=np.array([haar_line(rng,3) for _ in U])
    G=np.array([np.linalg.qr(proper_complex(rng,(3,3)))[0] for _ in U])
    a,b,kappa=1.4,.3,7.
    def logjoint(path):return passive_log_likelihood(U,path,a,b)+kappa*transition_overlap(path,G)
    for index in range(len(U)):
        A=path_state_parameter(U,states,index,a,b,kappa,G)
        other=states.copy();other[index]=haar_line(rng,3)
        difference=logjoint(states)-logjoint(other)
        conditional=np.vdot(states[index],A@states[index]).real-np.vdot(other[index],A@other[index]).real
        assert_allclose(difference,conditional,atol=2e-11)


def test_kappa_conditional_uses_exactly_T_minus_one_normalizers():
    p=PowerPrior();n,m,S=8,7,5.7;k1,k2=20.,31.
    expected=(p.kappa_shape-1)*np.log(k1/k2)-p.kappa_rate*(k1-k2)
    expected+=(k1-k2)*S-m*(transition_log_normalizer(n,k1)-transition_log_normalizer(n,k2))
    assert_allclose(kappa_log_density(k1,S,m,n,p)-kappa_log_density(k2,S,m,n,p),expected,atol=2e-12)
    # With no transitions, sampling log kappa must retain the Jacobian and
    # target the original proper gamma prior, not a shape-minus-one density.
    rng=np.random.default_rng(178);x=50.;values=[]
    for j in range(4500):
        x=sample_kappa(x,0.,0,n,p,rng)
        if j>=500:values.append(x)
    assert abs(np.mean(values)-p.kappa_shape/p.kappa_rate)<2.


def test_joint_covariance_averages_power_state_correlation():
    a=np.array([2.,4.]);b=np.array([.2,.6]);h=np.eye(2,dtype=complex)
    R=joint_covariance_mean(a,b,h)
    assert_allclose(R,np.diag([1.3,2.1]),atol=1e-12)
    separable=b.mean()*np.eye(2)+(a-b).mean()*(h.T@h.conj()/2)
    assert np.linalg.norm(R-separable)>.5


def test_future_covariance_preserves_joint_power_concentration_state_dependence():
    from lib.beamforming_numerics import transition_coefficients_complex
    a=np.array([[2.,4.]]);b=np.array([[.2,.6]]);k=np.array([[1.,100.]])
    h=np.eye(2,dtype=complex)[None,:,None,:]
    result=JointResult(a,b,k,h,np.zeros((2,2)),np.eye(2)/2,{})
    expected=np.zeros((2,2),complex)
    for aa,bb,kk,v in zip(a.ravel(),b.ravel(),k.ravel(),h.reshape(2,2)):
        rho=transition_coefficients_complex(2,kk).rho
        expected+=bb*np.eye(2)+(aa-bb)*(rho*np.outer(v,v.conj())+(1-rho)*np.eye(2)/2)
    assert_allclose(joint_forecast_covariance(result),expected/2,atol=1e-13)


def test_stationary_joint_gibbs_agrees_with_independent_complex_CP1_grid():
    U,_,_=fixture(n=2,T=8,seed=22)
    exact=stationary_reference_n2(U)
    assert exact['numerical_ok'] and exact['refinement_error']<1e-7
    result=stationary_gibbs(U,draws=1024,burnin=256,chains=3,seed=567)
    assert result.diagnostics['completed']
    assert abs(result.a.mean()-exact['a'])<.07
    assert abs(result.b.mean()-exact['b'])<.025
    assert np.linalg.norm(result.covariance_mean-exact['covariance'])<.07
    assert_allclose(np.trace(result.projector_mean),1,atol=2e-12)


def test_frozen_prior_EM_is_normalized_and_has_only_its_fixed_ascent_claim():
    U,_,_=fixture(n=3,T=6)
    result=recorded_prior_em(U,iterations=3)
    assert result['ascent_verified']
    assert np.all(np.diff(result['objectives'])>=-1e-8)
    assert 'frozen' in result['objective']
    assert result['s']>0 and result['q']>=0


def test_prospective_fit_does_not_see_future_data_and_scores_full_constants():
    U,_,_=fixture(n=2,T=9)
    result=stationary_gibbs(U[:8],draws=128,burnin=64,chains=2,seed=711)
    score=joint_predictive_score(result,U[8])
    h=result.states.reshape(-1,2);a=result.a.ravel();b=result.b.ravel()
    logs=np.array([passive_log_likelihood(U[8:9],state,pa,pb) for state,pa,pb in zip(h,a,b)])
    from scipy.special import logsumexp
    assert_allclose(score['log_score'],logsumexp(logs)-np.log(len(logs)),atol=1e-12)
    altered=U.copy();altered[8]*=100
    unchanged=stationary_gibbs(altered[:8],draws=128,burnin=64,chains=2,seed=711)
    assert_allclose(unchanged.a,result.a,atol=0)
    assert_allclose(unchanged.states,result.states,atol=0)


def test_short_full_path_gibbs_keeps_order_and_reports_unresolved_mixing():
    U,_,_=fixture(n=2,T=5)
    result=full_path_gibbs(U,draws=32,burnin=16,chains=2,seed=837)
    assert result.diagnostics['completed']
    assert not result.diagnostics['mixing_ok']
    assert result.states.shape==(2,32,5,2)
    assert np.all(result.a>result.b) and np.all(result.b>0) and np.all(result.kappa>0)
