"""Independent complex-line references and passive tracking conventions."""
import numpy as np
from numpy.testing import assert_allclose
from lib.beamforming_reference import (bloch,bloch_projectors,grid_basis,
    kernel_multipliers,ComplexLineReference,compare_reference)
from lib.beamforming_tracking import (configuration,signature_projectors,scatter,
    matched_isotropization,physical_data,fixed_time_record,matched_process_data)
from lib.beamforming_numerics import complex_moments,cp1_power_cdf
from lib.beamforming_protocol import haar_line
from lib.models import increment,constant
from lib.geometry import center


def test_bloch_projector_and_overlap_are_native_complex_lines():
    rng=np.random.default_rng(821)
    for _ in range(8):
        h=haar_line(rng,2);g=haar_line(rng,2);r=bloch(h);v=bloch(g)
        assert_allclose(np.linalg.norm(r),1,atol=1e-14)
        assert_allclose(bloch_projectors(r),np.outer(h,h.conj()),atol=1e-14)
        assert_allclose(abs(np.vdot(h,g))**2,(1+r@v)/2,atol=1e-14)
        assert_allclose(bloch(np.exp(.73j)*h),r,atol=1e-14)
        assert np.linalg.norm(bloch(np.array([-np.conj(h[1]),np.conj(h[0])]))+r)<1e-13


def test_full_harmonics_normalized_and_transition_is_vmf_not_watson():
    nodes,w,Y,_,degrees=grid_basis(4,10,20)
    assert_allclose(Y.T@(w[:,None]*Y),np.eye(25),atol=4e-15)
    assert set(degrees)==set(range(5))  # Odd degrees are essential on CP1.
    for kappa in [0.,4.,20.]:
        multipliers=kernel_multipliers(kappa,4)
        rho=0. if kappa==0 else 1/np.tanh(kappa/2)-2/kappa
        assert_allclose(multipliers[1],rho,atol=2e-14)
    assert_allclose(kernel_multipliers(np.inf,4),1)


def test_continuous_uniform_scalar_cdf_and_analytic_quantiles():
    ref=ComplexLineReference(0.,degree=4,z_order=10,phi_order=20)
    rng=np.random.default_rng(623);queries=np.asarray([haar_line(rng,2) for _ in range(5)])
    probabilities=np.array([.025,.25,.5,.75,.975])
    assert_allclose(ref.quadratic_quantiles(queries),np.broadcast_to(probabilities,(5,5)),atol=2e-14)
    assert_allclose(ref.quadratic_cdf(queries,np.linspace(.1,.9,5)),np.linspace(.1,.9,5),atol=2e-14)
    report=ref.summary(queries)
    assert_allclose(report['mean'],np.eye(2)/2,atol=1e-14)
    assert_allclose(report['ball_radius']**2,.95,atol=2e-14)
    assert report['uniform_density']
    assert_allclose(report['numerics']['hpd_mass'],.95,atol=1e-14)
    assert 'Haar' in report['numerics']['hpd_tie_kind']


def test_one_observation_reference_matches_independent_complex_cdf_and_likelihood():
    ref=ComplexLineReference(4.,degree=20,z_order=40,phi_order=80)
    u=np.array([.3+.4j,-.5+.2j]);s=.2;q=1.
    B=increment(u,s,q,complex_data=True);score=ref.update(u,s,q)
    exact=complex_moments(B)
    assert_allclose(ref.mean(),exact.mean,atol=1e-13)
    assert_allclose(score,constant(u,1,s,q,complex_data=True)+exact.log_normalizer,atol=1e-13)
    queries=np.array([[1,0],[1,1j],[.7+.3j,-.2+.5j]],complex)
    queries/=np.linalg.norm(queries,axis=1)[:,None]
    for p in [.05,.4,.9]:
        values=np.full(3,p)
        independent=np.asarray([cp1_power_cdf(B,w,p) for w in queries])
        assert_allclose(ref.quadratic_cdf(queries,values),independent,atol=2e-12)


def test_reference_prediction_matches_exact_first_moment_and_uniform_reset():
    ref=ComplexLineReference(4.,degree=16,z_order=32,phi_order=64)
    ref.update(np.array([.5+.3j,-.2j]),.2,1.);before=ref.mean()
    rho=1/np.tanh(2)-.5;ref.predict()
    assert_allclose(ref.mean(),rho*before+(1-rho)*np.eye(2)/2,atol=2e-14)
    reset=ComplexLineReference(0.,degree=16,z_order=32,phi_order=64)
    reset.update(np.array([.5+.3j,-.2j]),.2,1.);reset.predict()
    assert_allclose(reset.mean(),np.eye(2)/2,atol=2e-14)
    u=np.array([-.2+.3j,.4j]);reset.update(u,.2,1.)
    assert_allclose(reset.mean(),complex_moments(increment(u,.2,1.,complex_data=True)).mean,atol=1e-13)


def test_passive_mode_equals_independently_weighted_scatter_and_scan_bound():
    rng=np.random.default_rng(9022);n=4;U=rng.normal(size=(17,n))+1j*rng.normal(size=(17,n))
    h=np.asarray([haar_line(rng,n) for _ in range(17)]);s=.2;q=1.;beta=.8
    A=np.zeros((n,n),complex);R=np.zeros_like(A);mass=0.
    for u in U:
        A=center(beta*A+increment(u,s,q,complex_data=True))
        R=beta*R+np.outer(u,u.conj());mass=beta*mass+1
    direct,total=scatter(U,beta)
    assert_allclose(direct,R/mass,atol=2e-15);assert_allclose(total,mass,atol=2e-15)
    assert_allclose(A,center(q/(s*(s+q))*mass*direct),atol=3e-14)
    record,P,mode=fixed_time_record(U,h,s,q,beta,.05)
    assert record['scan_error']<=record['scan_lipschitz_limit']+1e-14
    assert not record['valid'] and np.isnan(record['bound'])
    assert_allclose(P,np.outer(mode,mode.conj()))


def test_matched_isotropization_preserves_mode_distance_and_single_snapshot_equality():
    rng=np.random.default_rng(183);u=rng.normal(size=8)+1j*rng.normal(size=8)
    A=center(increment(u,.2,1.,complex_data=True));M=complex_moments(A).mean
    h=u/np.linalg.norm(u);P=np.outer(h,h.conj());iso,rho=matched_isotropization(M,P)
    assert_allclose(M,iso,atol=2e-12)
    assert_allclose(np.trace(P@M),np.trace(P@iso),atol=1e-14)
    assert 0<rho<1


def test_generators_keep_physical_and_stochastic_laws_distinct():
    cfg=configuration('quick');cfg.update(physical_T=16,process_T=16)
    physical=physical_data(cfg,cfg['validation_seed'],0,.2)
    assert_allclose(np.sum(abs(physical['h'])**2,axis=1),1,atol=3e-15)
    assert physical['U'].shape==(16,8)
    assert 'z' not in physical
    matched=matched_process_data(cfg,cfg['process_development_seed'],0,8,4.)
    assert_allclose(np.sum(abs(matched['h'])**2,axis=1),1,atol=3e-15)
    assert 'theta' not in matched  # An arbitrary complex line is not a ULA arrival angle.
