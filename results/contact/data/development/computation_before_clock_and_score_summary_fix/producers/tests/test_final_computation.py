"""Independent cost-accounting and production-action correctness checks."""
from dataclasses import replace
import numpy as np
from lib.final_computation import (configuration,synthetic_case,run_pipeline,
    action,array_bytes,extract,numerical_accuracy,contact_command)
from lib.models import increment
from lib.geometry import center


def small_config():
    cfg=configuration('verify');cfg.update(observations=8,learning_block=4,
        dimensions=[3],spans=[0.,80.],em_iterations=[1])
    return cfg


def test_cadence_preserves_exact_assimilation_and_endpoint_action():
    cfg=small_config()
    for example in ('beamforming','contact'):
        case=synthetic_case(example,3,cfg);A=np.zeros((3,3),case.U.dtype)
        for j,u in enumerate(case.U):
            A=center(case.beta*A+increment(u,case.sigma2[j],
                None if case.V is not None else case.tau2[j],
                None if case.V is None else case.V[j],example=='beamforming'))
        for method in ('mean','mode','isotropic'):
            every=run_pipeline(case,method,1,cfg);endpoint=run_pipeline(case,method,8,cfg)
            assert every['actions']==8 and endpoint['actions']==1
            np.testing.assert_allclose(endpoint['parameter'],A,atol=1e-12)
            np.testing.assert_allclose(every['final_action'],endpoint['final_action'],atol=1e-12)
            assert endpoint['wall_seconds']>=sum(endpoint[k] for k in (
                'correction_seconds','action_seconds','fitting_seconds','report_seconds'))
            assert endpoint['mean_ok']


def test_point_beam_action_matches_independent_positive_definite_solve():
    rng=np.random.default_rng(218)
    h=rng.normal(size=8)+1j*rng.normal(size=8);h/=np.linalg.norm(h)
    a=rng.normal(size=8)+1j*rng.normal(size=8);a/=np.linalg.norm(a)
    P=np.outer(h,h.conj())
    for rho in (0.,.4,1.):
        s,q=.17,2.3;M=rho*P+(1-rho)*np.eye(8)/8
        reference=np.linalg.solve(s*np.eye(8)+q*M,a);reference/=np.vdot(a,reference)
        w=action(M,'beamforming',a,s,q,h,rho)
        np.testing.assert_allclose(w,reference,atol=2e-14)
        np.testing.assert_allclose(np.vdot(w,a),1.,atol=1e-14)


def test_matched_isotropic_actions_use_exact_axial_inverse():
    cfg=small_config();rng=np.random.default_rng(27)
    for example in ('beamforming','contact'):
        case=synthetic_case(example,3,cfg);Q=np.linalg.qr(rng.normal(size=(3,3)))[0]
        A=(Q*np.array([-3.,-.4,3.4]))@Q.T
        if example=='beamforming':A=A.astype(complex)
        M,_,payload=extract(A,example,'isotropic',cfg)
        if example=='beamforming':
            actual=action(M,example,case.desired,.2,1.,payload)
            expected=np.linalg.solve(.2*np.eye(3)+M,case.desired);expected/=np.vdot(case.desired,expected)
        else:
            actual=contact_command(M,case.desired,payload)
            expected=np.linalg.solve(np.eye(3)+10*(np.eye(3)-M),case.desired)
        np.testing.assert_allclose(actual,expected,atol=1e-13)


def test_adaptive_cycle_charges_fits_once_and_keeps_complete_partition():
    cfg=small_config();case=synthetic_case('beamforming',3,cfg)
    result=run_pipeline(case,'adaptive_mean',4,cfg,em_iterations=1)
    assert result['fits']==2 and result['actions']==2
    assert result['fit_ok'] and result['mean_ok']
    assert result['fitting_seconds']>0
    assert result['correction_seconds']>0
    total=sum(result[k] for k in ('correction_seconds','action_seconds','report_seconds','fitting_seconds','overhead_seconds'))
    assert abs(total-result['wall_seconds'])<1e-12
    assert result['overhead_seconds']>=0
    assert result['predictive_scores']==2 and result['predictive_score_ok']


def test_physical_coordinate_transport_and_excluded_observation_action():
    cfg=small_config();case=synthetic_case('contact',3,cfg)
    from scipy.linalg import expm
    G=np.array([expm((j+1)*.03*np.array([[0.,-1.,.5],[1.,0.,-.3],[-.5,.3,0.]])) for j in range(8)])
    moved=replace(case,G=G);expected=np.zeros((3,3))
    for j,u in enumerate(case.U):expected=center(case.beta*G[j]@expected@G[j].T+increment(u,case.sigma2[j],v=case.V[j]))
    result=run_pipeline(moved,'mode',8,cfg)
    np.testing.assert_allclose(result['parameter'],expected,atol=1e-12)
    beam=synthetic_case('beamforming',3,cfg)
    changed=beam.U.copy();changed[-1]*=1.2
    original=run_pipeline(beam,'adaptive_mean',8,cfg,em_iterations=1)
    altered=run_pipeline(replace(beam,U=changed),'adaptive_mean',8,cfg,em_iterations=1)
    np.testing.assert_allclose(original['final_action'],altered['final_action'],atol=1e-13)
    assert np.linalg.norm(original['parameter']-altered['parameter'])>1e-4


def test_persistent_bytes_do_not_count_same_state_twice():
    A=np.zeros((7,7),complex);B=np.zeros(7)
    assert array_bytes(dict(A=A,same=A,buffer=[B,A]))==A.nbytes+B.nbytes
    assert array_bytes([A[:2],A[2:]])==A.nbytes


def test_independent_complex_accuracy_separates_second_moment_from_mean():
    cfg=small_config();case=synthetic_case('beamforming',3,cfg)
    rows,_=numerical_accuracy('beamforming',[case],cfg)
    for row in rows:
        assert row['mean_ok'] and row['normalizer_ok'] and row['reference_ok']
        assert row['second_moment_evaluated'] and row['second_moment_ok']
        assert row['tail_evaluated'] is False and row['tail_ok'] is None
        assert row['mean_reference_error']<1e-8
