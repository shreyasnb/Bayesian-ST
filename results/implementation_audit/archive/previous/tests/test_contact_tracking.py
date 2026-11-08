"""Causality, fixed-noise ablations and independent contact-mode checks."""
from copy import deepcopy
import numpy as np
import pytest
from numpy.testing import assert_allclose

from lib.contact_tracking import (default_config, make_tracker, normal_alignment,
    replay, plant_data, mobility_maps, candidate_specs, choose_policies)
from lib.contact_study import query_truth, query_vectors, response_loss
from lib.geometry import projector


def config(T=30):
    c=default_config();c.update(T=T,report_times=[5,15,T]);return c


@pytest.mark.parametrize('condition',['plane','sphere_small','anisotropic'])
def test_tight_frame_scoring_agrees_with_independent_kkt(condition):
    c=config(); data=plant_data(c,4210,2,condition);H=mobility_maps(data,c['d'])
    queries=query_vectors(c['v0']);normal=np.array([1.,2.,-1.])/np.sqrt(6)
    P=np.eye(3)-np.outer(normal,normal)
    for t in [1,15,30]:
        actual,_=query_truth(data,t,queries,c['d'])
        assert_allclose(queries@H[t-1].T,actual,atol=2e-17)
        assert_allclose(c['v0']**2/3*np.sum((P-H[t-1])**2),
                        response_loss(P,queries,actual),atol=2e-19)


@pytest.mark.parametrize('spec',[
    dict(prediction='identity',beta=.8),
    dict(prediction='rolling',beta=1.,likelihood_window=3),
    dict(prediction='velocity',beta=.8,lag=2,damping=.5),
    dict(prediction='oracle',beta=1.)])
def test_replay_matches_transported_projector_least_squares(spec):
    c=config();data=plant_data(c,4211,0,'sphere_small')
    path=replay(c,data,spec)
    assert path['ls_parameter_error']<1e-8
    assert path['ls_projector_error']<1e-8
    assert path['ls_objective_error']<1e-12
    assert_allclose(path['G']@np.swapaxes(path['G'],1,2),np.broadcast_to(np.eye(3),(30,3,3)),atol=1e-12)
    assert path['forecast_gap'][0]==0


def test_forecast_uses_only_past_data_and_truth_is_scoring_only():
    c=config();data=plant_data(c,4212,0,'sphere_small')
    spec=dict(prediction='velocity',beta=.8,lag=2,damping=.5)
    original=replay(c,data,spec); changed=deepcopy(data)
    changed['U'][10:]+=np.array([.01,.02,-.03])
    # Arbitrary geometry changes cannot enter the practical raw recursion.
    changed['normals']=data['normals'][::-1].copy()
    other=replay(c,changed,spec)
    assert_allclose(original['prior'][:11],other['prior'][:11],atol=0)
    assert_allclose(original['G'][:11],other['G'][:11],atol=0)
    assert np.linalg.norm(original['A'][10]-other['A'][10])>1


def test_noiseless_ablation_keeps_scaling_and_mechanics_fixed():
    c=config();data=plant_data(c,4213,0,'sphere_small');copy=deepcopy(data)
    spec=dict(prediction='identity',beta=.8)
    path=replay(c,data,spec,noiseless=True)
    copy['U']=data['velocities'].copy()
    reference=replay(c,copy,spec,noiseless=False)
    assert path['fixed_update_variance']==1e-6
    assert_allclose(path['A'],reference['A'],atol=0)
    assert_allclose(data['positions'],copy['positions'],atol=0)
    noiseless=replay(c,data,dict(prediction='oracle',beta=1.),noiseless=True)
    assert noiseless['corrected_chordal'][-1]<2e-7


def test_minimal_plane_alignment_respects_normal_sign():
    rng=np.random.default_rng(4214)
    for _ in range(20):
        a=rng.normal(size=3);a/=np.linalg.norm(a)
        b=rng.normal(size=3);b/=np.linalg.norm(b)
        G=normal_alignment(a,b)
        assert_allclose(G.T@G,np.eye(3),atol=1e-14)
        assert_allclose(G@np.outer(a,a)@G.T,np.outer(b,b),atol=1e-14)
        assert_allclose(G,normal_alignment(-a,b),atol=1e-14)


def test_initialization_cap_and_cut_locus_are_explicit():
    c=config(); f=make_tracker(c,dict(prediction='velocity',beta=.8,lag=2,damping=1.,speed_cap=.001))
    assert f.predict().shape==(3,3)
    assert f.motion_diagnostics['reason']=='initialization'
    f._prepared=False
    # Two planes at the Grassmann cut locus cannot define a unique velocity.
    f.history.extend([np.eye(3)[:,:2],np.eye(3)[:,:2],np.eye(3)[:,1:]])
    f.predict();assert f.motion_diagnostics['reason']=='cut_locus_identity'
    f._prepared=False;f.history.clear()
    a=.3;R=np.array([[np.cos(a),0,-np.sin(a)],[0,1,0],[np.sin(a),0,np.cos(a)]])
    f.history.extend([np.eye(3)[:,:2],np.eye(3)[:,:2],R@np.eye(3)[:,:2]])
    f.predict();assert f.motion_diagnostics['capped']
    assert f.motion_diagnostics['speed']==.001


def test_validation_requires_full_refinement_and_never_selects_oracle():
    c=config();c['validation_trials']=2;rows=[]
    for trial in range(2):
        for condition in c['curved_conditions']:
            for beta,loss in [(.6,2.),(.8,1.)]:
                rows.append(dict(scope='curved',condition=condition,trial=trial,stage='refine',
                    family='velocity',policy=str(beta),specification=dict(prediction='velocity',beta=beta,lag=2,damping=.5),
                    loss=loss,time_average_mse=loss))
    rows.append(dict(rows[0],policy='incomplete',loss=0.))
    selected,curves=choose_policies(rows,c)
    assert selected[0]['policy']=='0.8'
    assert len(candidate_specs(c,'velocity'))==36
