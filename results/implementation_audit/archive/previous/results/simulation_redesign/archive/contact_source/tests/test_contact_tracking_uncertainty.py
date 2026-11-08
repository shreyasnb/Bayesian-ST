"""Information, scoring and control checks for the revised contact comparison."""
import numpy as np
from lib.contact_tracking_uncertainty import (default_config,shared_inputs,estimate_path,
    path_scores,shrinkage_coefficients,choose_policies,boundary_expansion,policies_from_stage1)
from lib.contact_mechanics import generate_contact
from lib.contact_study import query_vectors,query_truth
from lib.contact_process_prediction import fast_moments,predictive_log_score
from lib.geometry import center
from lib.models import increment


def setup(T=18):
    config=default_config();config.update(T=T,report_times=[6,T],story_times=[6,T])
    data=generate_contact(774311,0,'sphere_small',T=T)
    policies=dict(raw=dict(prediction='identity',beta=.8),motion=dict(prediction='velocity',beta=.9,lag=2,damping=.5))
    shared=shared_inputs(config,data['U'],data['V'],policies)
    return config,data,shared,policies


def test_temperature_preserves_raw_recursion_and_mode_but_not_spread():
    config,data,shared,_=setup()
    plain=estimate_path(config,data['U'],data['V'],dict(kind='raw',temperature=1.),shared)
    broad=estimate_path(config,data['U'],data['V'],dict(kind='raw',temperature=4.),shared)
    np.testing.assert_allclose(plain['A'],shared['raw']['A'],atol=1e-10)
    np.testing.assert_allclose(broad['A'],plain['A']/4,atol=1e-10)
    np.testing.assert_allclose(broad['mode'],plain['mode'],atol=1e-10)
    assert np.linalg.norm(broad['mean'][-1]-plain['mean'][-1])>1e-4
    # Temperature changes prior law only, never the actual physical noise likelihood.
    j=11
    expected=predictive_log_score(shared['raw']['prior'][j]/4,data['U'][j],data['V'][j],config['sigma']**2)
    np.testing.assert_allclose(-broad['nll'][j],expected,atol=1e-10)


def test_current_observation_cannot_change_its_own_shared_forecast():
    config,data,shared,policies=setup()
    U=data['U'].copy();U[10]+=np.array([.1,-.2,.3])
    altered=shared_inputs(config,U,data['V'],policies)
    np.testing.assert_allclose(shared['motion']['G'][:11],altered['motion']['G'][:11],atol=1e-12)
    assert np.linalg.norm(shared['motion']['G'][11:]-altered['motion']['G'][11:])>.01
    for kind in ['process','gaussian']:
        spec=dict(kind=kind,nominal='motion',kappa=2000.)
        before=estimate_path(config,data['U'],data['V'],spec,shared)
        after=estimate_path(config,U,data['V'],spec,altered)
        np.testing.assert_allclose(before['prior_mean'][:11],after['prior_mean'][:11],atol=2e-10)
        assert before['clocks']['shared_forecast_seconds']>0


def test_deterministic_process_is_exact_transport_and_proper_score():
    config,data,shared,_=setup()
    path=estimate_path(config,data['U'],data['V'],dict(kind='process',nominal='motion',kappa=None),shared)
    A=np.zeros((3,3))
    for j,(u,v,G) in enumerate(zip(data['U'],data['V'],shared['motion']['G'])):
        prior=center(G@A@G.T)
        np.testing.assert_allclose(path['prior'][j],prior,atol=1e-9)
        np.testing.assert_allclose(-path['nll'][j],predictive_log_score(prior,u,v,config['sigma']**2),atol=2e-10)
        A=center(prior+increment(u,config['sigma']**2,v=v))
    assert path['diagnostics']['maximum_inverse_residual']==0


def test_tight_frame_loss_matches_nine_query_loss_in_mismatch_and_shrink_optimum():
    config,data,shared,_=setup()
    path=estimate_path(config,data['U'],data['V'],dict(kind='raw'),shared)
    for condition in ['sphere_small','anisotropic']:
        plant=generate_contact(774311,0,condition,T=config['T'])
        scores=path_scores(config,plant,path,.7);Q=query_vectors(config['v0'])
        truth,_=query_truth(plant,config['T'],Q,config['d'])
        expected=np.mean(np.sum((Q@path['mean'][-1].T-truth)**2,axis=1))
        np.testing.assert_allclose(scores['mean_mse'][-1],expected,atol=1e-18)
        coeff=shrinkage_coefficients(config,plant,path)
        rho=float(np.clip(-coeff['shrink_b']/coeff['shrink_a'],0,1))
        grid=np.linspace(0,1,101)
        losses=[np.mean(np.sum((r*Q@path['mode'][-1].T-truth)**2,axis=1)) for r in grid]
        optimum=np.mean(np.sum((rho*Q@path['mode'][-1].T-truth)**2,axis=1))
        assert optimum<=min(losses)+1e-18


def test_validation_selection_never_drops_failed_trial_or_uses_test_truth():
    config=default_config();config.update(validation_trials=2,curved_conditions=['sphere_small'])
    rows=[]
    for T in [1.,4.]:
        for j in range(2):
            rows.append(dict(scope='curved',method='temperature',temperature=T,trial=j,
                interval_score=1. if T==1. else .8,predictive_nll=-2.,terminal_mse=.01,time_mse=.02,
                mode_mse=.01,completed_steps=600,mean_ok=True,interval_ok=not(T==4. and j==1),
                shrink_a=2.,shrink_b=-1.,candidate_seconds=.1))
    selected,tuning=choose_policies(rows,config)
    assert selected[0]['temperature']==1.
    assert selected[0]['rho']==.5
    assert not tuning.loc[tuning.temperature==4,'resolved'].iloc[0]
    assert boundary_expansion([dict(temperature=8.),dict(kappa=20.)],config)==dict(temperatures=[16.,32.],kappas=[5.,2.])
    document=dict(best_per_scope={},selected=[])
    for s in ['plane','curved']:
        row=dict(scope=s,family='velocity',specification=dict(prediction='velocity',beta=.9,lag=8,damping=.5))
        document['best_per_scope'][s]=row;document['selected'].append(row)
    assert policies_from_stage1(document)['curved']['motion']['prediction']=='velocity'
    document['best_per_scope']['curved']['specification']=dict(prediction='oracle')
    try:policies_from_stage1(document)
    except ValueError:pass
    else:raise AssertionError('Oracle passed into practical posterior comparison')
