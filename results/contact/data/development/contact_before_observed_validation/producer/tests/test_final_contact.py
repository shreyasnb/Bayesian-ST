"""Independent mechanics, task and delayed-output contracts for final contact."""
import numpy as np

from lib.final_contact import (configuration,generate_contact_trial,generate_static_trial,
    ConditionalTracker,bingham_records,isotropic_mean,mode_projector,query_directions,
    response_loss,soft_commands,command_cost,anisotropy_fixture,block_episode,great_records)
from lib.contact_posterior import normal_quadrature


def test_kkt_observation_and_sphere_drift_are_independent_identities():
    cfg=configuration('verify')
    for condition in ('plane','sphere','fast_sphere'):
        data=generate_contact_trial(dict(cfg,sigma2=0.),condition,3)
        assert np.max(abs(data['U']-np.einsum('tij,tj->ti',data['P'],data['V'])))<2e-14
        assert data['diagnostics']['maximum_kkt_residual']<1e-13
        assert data['diagnostics']['maximum_surface_error']<1e-13
        assert data['diagnostics']['maximum_sphere_drift_identity_error']<1e-13
        fine=generate_contact_trial(dict(cfg,sigma2=0.),condition,3,substeps=2)
        assert np.max(abs(data['positions']-fine['positions']))<1e-8
        if condition!='plane':
            assert data['diagnostics']['finite_difference_drift_relative_error']<.003
    # Independently measured, noncommuting rotations transport the state and
    # velocities together; no surface-normal alignment is supplied.
    data=generate_contact_trial(cfg,'sphere',2)
    R=data['R']
    assert np.max(abs(data['G'][1:]-R[1:].transpose(0,2,1)@R[:-1]))<1e-13
    assert np.max(abs(data['clean']-np.einsum('tji,tj->ti',R,data['world_clean'])))<1e-13


def test_stationary_input_energy_and_noise_are_paired_without_truth_input():
    cfg=configuration('verify')
    balanced=generate_static_trial(cfg,'static_balanced',5)
    concentrated=generate_static_trial(cfg,'static_concentrated',5)
    assert np.array_equal(balanced['normal'],concentrated['normal'])
    assert np.array_equal(balanced['standard_noise'],concentrated['standard_noise'])
    assert np.allclose(np.linalg.norm(balanced['V'],axis=1),cfg['v0'])
    assert np.allclose(np.linalg.norm(concentrated['V'],axis=1),cfg['v0'])
    assert abs(balanced['diagnostics']['input_energy']-concentrated['diagnostics']['input_energy'])<1e-14
    # Estimation uses just u,v,G; changing scoring-only fields changes no A.
    first,As=bingham_records(cfg,balanced,1.,[4,16,32])
    changed=dict(balanced,P=np.zeros_like(balanced['P']),normal=np.zeros_like(balanced['normal']))
    second,Bs=bingham_records(cfg,changed,1.,[4,16,32])
    assert np.array_equal(As,Bs)
    assert all(np.array_equal(first[t]['M'],second[t]['M']) for t in first)


def test_isotropic_control_and_soft_expected_cost_use_the_projector_mixture():
    rng=np.random.default_rng(77)
    normals=rng.normal(size=(5,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
    projectors=np.eye(3)-np.einsum('bi,bj->bij',normals,normals)
    weights=np.array([.1,.2,.3,.15,.25]);M=np.einsum('b,bij->ij',weights,projectors)
    eig,E=np.linalg.eigh(M);P=E[:,1:]@E[:,1:].T
    iso,alpha,xi=isotropic_mean(M,P)
    assert abs(np.trace(iso)-2)<1e-13
    assert np.linalg.eigvalsh(iso).min()>=0 and np.linalg.eigvalsh(iso).max()<=1
    assert abs(np.trace(P@(iso-M)))<1e-13
    Q=query_directions(.02)
    for lam in [0.,1.,10.]:
        W=soft_commands(M,Q,lam)
        direct=sum(w*command_cost(W,truth,Q,lam) for w,truth in zip(weights,projectors))
        formula=np.mean(np.sum((W-Q)**2,axis=1)+lam*np.einsum('qi,ij,qj->q',W,np.eye(3)-M,W))
        assert abs(direct-formula)<1e-17
        if lam==0:assert np.allclose(W,Q)
    assert np.allclose(soft_commands(2*np.eye(3)/3,Q,10.),Q/(1+10/3))
    # Squaring I-M produces a different, incorrect expectation.
    W=soft_commands(M,Q,10.)
    wrong=np.mean(np.sum((W-Q)**2,axis=1)+10*np.sum((W@(np.eye(3)-M))**2,axis=1))
    correct=sum(w*command_cost(W,truth,Q,10.) for w,truth in zip(weights,projectors))
    assert abs(correct-wrong)>1e-6


def test_actual_data_axial_density_matches_overlap_and_directional_risk():
    arrays,checks=anisotropy_fixture(configuration('verify'))
    assert checks['data_generated'] and checks['numerical_ok']
    assert checks['selected_episode']==0 and checks['selected_time']==16
    assert checks['isotropic_mean_match_error']<1e-10
    assert checks['isotropic_overlap_error']<1e-10
    assert abs(np.trace(arrays['normal_A']))<1e-12
    assert abs(np.trace(arrays['iso_normal_A']))<1e-12
    # Independent quadrature of full squared vector response error has the
    # trace-risk formula; a directional covariance is not inferred from it.
    quad=normal_quadrature(arrays['normal_A'],80,256)
    P=np.eye(3)-np.einsum('bi,bj->bij',quad['normal_nodes'],quad['normal_nodes'])
    M=arrays['plane_mean'];queries=arrays['unit_queries'][::50]
    error=np.einsum('qj,bji->bqi',queries,P-M)
    direct=np.einsum('b,bqi,bqi->q',quad['weights'],error,error)
    formula=np.einsum('qi,ij,qj->q',queries,M-M@M,queries)
    assert np.max(abs(direct-formula))<1e-10


def test_completed_blocks_match_singletons_without_using_future_outputs():
    cfg=configuration('verify')
    policies=[dict(condition='sphere',beta=.95)]
    rows=block_episode(cfg,'sphere',0,policies)
    assert [r['block_length'] for r in rows]==[1,4,16,32]
    for row in rows:
        L=row['block_length']
        assert row['exact_parameter_error']<1e-10
        assert row['first_data_output_observation']==L
        assert row['action_outputs']==cfg['T']//L
        assert row['maximum_output_age_observations']==L-1
        assert row['beta_block']==.95**L
        assert row['common_endpoint_count']==2
        assert row['common_endpoint_observations']==[32,64]
        assert abs(row['common_endpoint_response_mse']-row['common_endpoint_singleton_response_mse'])<1e-16
    assert abs(rows[0]['singleton_response_mse']-rows[0]['delayed_response_mse'])<1e-16


def test_great_reference_receives_responses_and_frames_not_recorded_inputs():
    cfg=configuration('verify');data=generate_contact_trial(cfg,'plane',0)
    policy=dict(window=16,beta=None,stepsize=10.,inner_steps=1)
    first,_=great_records(data,policy,[4,16,64])
    changed=dict(data,V=1e6*np.ones_like(data['V']),P=np.zeros_like(data['P']))
    second,_=great_records(changed,policy,[4,16,64])
    assert all(np.array_equal(first[t],second[t]) for t in first)


def test_conditional_mode_minimizes_independently_evaluated_projector_residual():
    cfg=configuration('verify')
    data=generate_static_trial(cfg,'static_balanced',7)
    tracker=ConditionalTracker(data['sigma2'])
    for u,v in zip(data['U'],data['V']):tracker.update(u,v)
    P,normal,gap=mode_projector(tracker.A)
    assert gap>0
    def residual(h):
        candidate=np.eye(3)-np.outer(h,h)
        return float(np.sum((data['U']-data['V']@candidate.T)**2))
    optimum=residual(normal)
    rng=np.random.default_rng(92)
    alternatives=rng.normal(size=(256,3));alternatives/=np.linalg.norm(alternatives,axis=1)[:,None]
    assert min(residual(h) for h in alternatives)>=optimum-1e-13
    tangent=np.cross(normal,np.eye(3)[np.argmin(abs(normal))]);tangent/=np.linalg.norm(tangent)
    step=1e-5
    plus=np.cos(step)*normal+np.sin(step)*tangent
    minus=np.cos(step)*normal-np.sin(step)*tangent
    assert abs((residual(plus)-residual(minus))/(2*step))<1e-9
