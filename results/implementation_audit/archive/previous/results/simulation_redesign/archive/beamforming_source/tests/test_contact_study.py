"""Physical query targets, observation access, independent LS and tuning units."""
import numpy as np
from numpy.testing import assert_allclose
from lib.contact_study import (query_vectors,plant_data,default_config,query_truth,
                              weighted_projector_ls,_snapshots,ridge_map,select_policies)
from lib.geometry import projector
from lib.filter import BinghamFilter


def test_nine_fixed_queries_and_actual_anisotropic_mobility():
    q=query_vectors();assert q.shape==(9,3)
    assert_allclose(np.linalg.norm(q,axis=1),.02,atol=1e-16)
    config=default_config();config['T']=6
    matched=plant_data(config,6245102,0,'sphere_small')
    physical,geometric=query_truth(matched,6,q,config['d'])
    assert_allclose(physical,geometric,atol=1e-14)
    stress=plant_data(config,6245102,0,'anisotropic')
    physical,geometric=query_truth(stress,6,q,config['d'])
    assert np.linalg.norm(physical-geometric)>.001
    assert_allclose(physical@stress['normals'][5],0,atol=1e-15)


def test_weighted_projector_ls_matches_conditional_mode_without_increment_formula():
    rng=np.random.default_rng(824011);U=rng.normal(size=(30,3));V=rng.normal(size=(30,3))
    f=BinghamFilter(3,2,.2,tau2=None,beta=.9)
    for u,v in zip(U,V):f.update(u,v=v)
    P,S,values=weighted_projector_ls(U,V,.9)
    assert_allclose(P,projector(f.basis),atol=2e-13)
    # Check the least-squares objective directly for an independent projector.
    Q=np.linalg.qr(rng.normal(size=(3,2)))[0];candidate=Q@Q.T
    weights=.9**np.arange(29,-1,-1)
    direct=lambda R:np.sum(weights*np.sum((U-V@R.T)**2,axis=1))
    assert_allclose(direct(candidate)-direct(P),np.trace(S@(P-candidate)),atol=2e-13)


def test_estimators_use_only_force_and_velocity_streams():
    config=default_config();config['T']=12
    data=plant_data(config,6245102,0,'sphere_large')
    a=_snapshots(data,[.8,1.],config['sigma']**2,[12])
    # Geometry and positions can be removed entirely from the estimator input.
    b=_snapshots(dict(U=data['U'],V=data['V']),[.8,1.],config['sigma']**2,[12])
    for beta in a:
        assert_allclose(a[beta][12]['A'],b[beta][12]['A'],atol=0)
        assert_allclose(ridge_map(a[beta][12],.01),ridge_map(b[beta][12],.01),atol=0)
        assert a[beta][12]['ls_projector_error']<1e-8


def test_policy_selection_keeps_plane_separate_and_averages_condition_within_trial():
    rows=[]
    for scope,conditions in [('plane',['plane']),('curved',['sphere_large','sphere_small','weak'])]:
        for trial in range(3):
            for condition in conditions:
                for beta in ([1.] if scope=='plane' else [1.,.8]):
                    rows.append(dict(scope=scope,condition=condition,trial=trial,method='Posterior mean',
                        beta=beta,loss=(0. if beta==.8 else 1.)+trial*.01,numerical_ok=True))
    selected,tuning=select_policies(rows)
    assert next(p for p in selected if p['scope']=='plane')['beta']==1.
    assert next(p for p in selected if p['scope']=='curved')['beta']==.8
    assert (tuning.independent_trials==3).all()
    assert tuning[tuning.scope=='curved'].evaluations.eq(9).all()


def test_scalar_shrinkage_uses_one_continuous_pooled_validation_optimum():
    rows=[]
    for scope,conditions in [('plane',['plane']),('curved',['sphere_large','sphere_small','weak'])]:
        for trial in range(4):
            for condition in conditions:
                optimum=.99-.002*trial
                for rho in [0.,.25,.5,.75,1.]:
                    rows.append(dict(scope=scope,condition=condition,trial=trial,method='Scalar shrinkage',
                        beta=1.,rho=rho,loss=2*(rho-optimum)**2+.1,numerical_ok=True))
    selected,tuning=select_policies(rows)
    for row in selected:assert_allclose(row['rho'],.987,atol=1e-12)
    assert (tuning[tuning.selected].optimization=='Analytic common rho from pooled independent validation trajectories').all()
