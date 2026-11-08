"""Loss aggregation, policy isolation and the true predictive likelihood."""
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from lib.contact_uncertainty_study import (interval_score,default_config,candidate_specs,
    choose_policies,_bingham_path,_direction_rows)
from lib.models import increment,constant
from lib.geometry import center
from lib.contact_posterior import contact_mean


def test_interval_score_uses_width_and_both_tail_penalties():
    lower=np.array([0.,0.,0.]);upper=np.array([1.,1.,1.]);truth=np.array([.5,-.5,1.25])
    assert_allclose(interval_score(lower,upper,truth),[1.,21.,11.])


def test_generalized_law_scores_true_likelihood_before_scaled_correction():
    cfg=default_config();cfg.update(T=6,report_times=[6])
    rng=np.random.default_rng(875);V=.02*np.eye(3)[np.arange(6)%3]
    P=np.eye(3)-np.ones((3,3))/3;U=V@P+.001*rng.normal(size=(6,3))
    data=dict(U=U,V=V,tangent_projectors=np.broadcast_to(P,(6,3,3)))
    spec=dict(method='Scaled Bingham',beta=.8,eta=.25)
    result=_bingham_path(cfg,data,spec,[6]);assert result['diagnostics']['completed_steps']==6
    A=np.zeros((3,3));scores=[]
    for u,v in zip(U,V):
        prior=.8*A;B=increment(u,1e-6,v=v)
        psi0=contact_mean(prior,strict=True)[1]['log_normalizer']
        psi1=contact_mean(prior+B,strict=True)[1]['log_normalizer']
        scores.append(-(psi1-psi0+constant(u,2,1e-6,v=v)))
        A=center(prior+.25*B)
    assert_allclose(result['nll'],scores,atol=2e-7)
    assert_allclose(result['snapshots'][6]['A'],A,atol=1e-11)


def test_policy_selection_averages_conditions_within_validation_trajectory():
    cfg=default_config();cfg.update(validation_trials=2,curved_conditions=['a','b'])
    rows=[]
    for trial in range(2):
        for condition in ['a','b']:
            for beta,nll,score in [(1.,2.,1.),(.8,1.,3.)]:
                rows.append(dict(scope='curved',method='Scaled Bingham',condition=condition,trial=trial,
                    beta=beta,eta=1.,predictive_nll=nll,interval_score=score,clean_mse=1.,
                    completed_steps=600,mean_ok=True,state_ok=True,interval_ok=True,candidate_seconds=1.))
    selected,tuning=choose_policies(rows,cfg)
    assert selected[0]['beta']==.8
    assert tuning[tuning.selected_by_interval_score].beta.iloc[0]==1.
    assert tuning.independent_trials.eq(2).all() and tuning.evaluations.eq(4).all()
    bad=list(rows);bad=[dict(row,completed_steps=599) if row['beta']==.8 and row['trial']==0 else row for row in bad]
    assert choose_policies(bad,cfg)[0][0]['beta']==1.


def test_expansion_does_not_duplicate_initial_candidates_and_plane_is_fair():
    cfg=default_config();cfg.update(_expand_beta=True,_expand_process=False)
    assert len(candidate_specs(cfg,'plane'))==len(candidate_specs(cfg,'curved'))
    assert {x['beta'] for x in candidate_specs(cfg,'plane',extra=True)}=={.1,.05}
    extra=candidate_specs(cfg,'curved',extra=True)
    assert {x['beta'] for x in extra}=={.1,.05}
    assert all(x['method']=='Scaled Bingham' for x in extra)


def test_directional_bias_is_invariant_to_antipodal_estimator_sign():
    normal=np.array([1.,0.,0.]);E=np.eye(3)[:,1:];true=np.array([np.cos(.1),np.sin(.1),0.])
    previous=np.array([np.cos(.08),np.sin(.08),0.]);cov=.01*(np.eye(3)-np.outer(normal,normal))
    a=_direction_rows(normal,E,cov,cov,np.ones(2),true,previous)
    b=_direction_rows(-normal,-E,cov,cov,np.ones(2),true,previous)
    assert_allclose([r['tangent_error'] for r in a],[r['tangent_error'] for r in b],atol=1e-14)
    assert a[0]['tangent_error']>0
