"""Independent draft-5 audit of decisions, duality and available information.

These tests check mathematical contracts and causality, not superiority over
another estimator or a desired Monte Carlo result.
"""
import copy
import numpy as np
import pytest
from lib.contact_geometry_study import (configuration,projector_from_normal,
    static_paths,dynamic_paths,mean_projector,isotropized_projector_mean,
    soft_action,soft_cost,conditional_soft_excess,ridge_map,_record)


def _observed_frame_case(T=8):
    """One data record in two measured frames, without an estimator oracle."""
    rng=np.random.default_rng(75483021);cfg=configuration('quick')
    cfg['dynamic_times']=[T]
    normal=np.array([1.,2.,3.])/np.sqrt(14);P=projector_from_normal(normal)
    V=cfg['v0']*rng.normal(size=(T,3))
    U=V@P+cfg['sigma_dynamic']*rng.normal(size=(T,3))
    R=[np.eye(3)]
    for _ in range(1,T):
        Q,_=np.linalg.qr(rng.normal(size=(3,3)))
        Q[:,0]*=np.linalg.det(Q);R.append(Q)
    R=np.asarray(R);G=np.asarray([np.eye(3)]+[R[j].T@R[j-1] for j in range(1,T)])
    world=dict(U=U,V=V,G=np.broadcast_to(np.eye(3),(T,3,3)),P=np.broadcast_to(P,(T,3,3)))
    wrist=dict(U=np.einsum('tji,tj->ti',R,U),V=np.einsum('tji,tj->ti',R,V),G=G,
        P=np.einsum('tji,jk,tkl->til',R,P,R),
        normal=np.einsum('tji,j->ti',R,normal),positions=np.zeros((T,3)))
    return cfg,world,wrist,R


def test_nonprojector_soft_penalty_equals_actual_projector_mixture():
    weights=np.array([.2,.3,.5]);Ps=np.array([np.eye(3)-np.outer(e,e) for e in np.eye(3)])
    M=np.einsum('b,bij->ij',weights,Ps)
    W=np.array([[.2,.1,-.05],[0.,.7,.13],[.03,-.08,.5]])
    lam=7.;v0=.8
    direct=v0*v0/3*(np.linalg.norm(W-np.eye(3),'fro')**2+
        lam*sum(p*np.linalg.norm((np.eye(3)-P)@W,'fro')**2 for p,P in zip(weights,Ps)))
    assert abs(soft_cost(W,M,lam,v0)-direct)<1e-12
    # The squared complement of the mean would omit posterior uncertainty.
    wrong=v0*v0/3*(np.linalg.norm(W-np.eye(3),'fro')**2+lam*np.linalg.norm((np.eye(3)-M)@W,'fro')**2)
    assert abs(direct-wrong)>.01
    optimum=soft_action(M,lam);H=np.eye(3)+lam*(np.eye(3)-M)
    assert np.linalg.norm(H@optimum-np.eye(3))<1e-12
    comparison=np.diag([.3,.85,.9]);alternative=soft_action(comparison,lam)
    excess,error=conditional_soft_excess(M,comparison,lam,v0)
    manual=v0*v0/3*np.trace((alternative-optimum).T@H@(alternative-optimum))
    assert abs(excess-manual)<1e-12 and error<1e-12 and excess>=0


def test_normal_residual_update_is_negative_plane_parameter():
    rng=np.random.default_rng(75483022);n=5;T=7;s=.17
    normal=rng.normal(size=n);normal/=np.linalg.norm(normal);P=projector_from_normal(normal)
    V=rng.normal(size=(T,n));U=V@P+np.sqrt(s)*rng.normal(size=(T,n))
    path,_=static_paths(U,V,s);normal_parameter=np.zeros((n,n))
    for record,u,v in zip(path,U,V):
        residual=v-u
        # Independent normal-line likelihood from its observed residual.
        normal_parameter+=(np.outer(residual,v)+np.outer(v,residual)-np.outer(v,v))/(2*s)
        C=normal_parameter-np.trace(normal_parameter)*np.eye(n)/n
        assert np.linalg.norm(C+record['A'],'fro')<2e-13
    e,E=np.linalg.eigh(normal_parameter)
    estimated_normal=E[:,-1]
    plane_normal=np.linalg.eigh(path[-1]['A'])[1][:,0]
    assert abs(abs(estimated_normal@plane_normal)-1)<1e-12


def test_future_observations_cannot_change_prefix_parameters():
    cfg,world,wrist,R=_observed_frame_case(8)
    _,first=dynamic_paths(cfg,wrist,beta=.9,report_times=[4])
    changed=copy.deepcopy(wrist)
    changed['U'][4:]+=np.array([.08,-.04,.06])
    changed['V'][4:]*=-1.3
    _,second=dynamic_paths(cfg,changed,beta=.9,report_times=[4])
    assert np.array_equal(np.asarray([r['A'] for r in first[:4]]),np.asarray([r['A'] for r in second[:4]]))
    assert np.linalg.norm(first[-1]['A']-second[-1]['A'])>1


def test_hidden_normal_and_position_only_change_scoring():
    cfg,world,wrist,R=_observed_frame_case(6)
    rows,first=dynamic_paths(cfg,wrist,beta=.95,report_times=[6])
    changed=copy.deepcopy(wrist)
    alternative=projector_from_normal(np.array([1.,0.,0.]))
    changed['P']=np.broadcast_to(alternative,changed['P'].shape)
    changed['normal']=np.full_like(changed['normal'],999.)
    changed['positions']=np.full_like(changed['positions'],-123.)
    altered,second=dynamic_paths(cfg,changed,beta=.95,report_times=[6])
    assert np.array_equal(np.asarray([r['A'] for r in first]),np.asarray([r['A'] for r in second]))
    left=next(r['response_mse'] for r in rows if r['method']=='mean')
    right=next(r['response_mse'] for r in altered if r['method']=='mean')
    assert abs(left-right)>1e-8


def test_measured_frame_transport_agrees_with_world_frame_recursion():
    cfg,world,wrist,R=_observed_frame_case(8)
    _,w=dynamic_paths(cfg,world,beta=.9,report_times=[8])
    _,b=dynamic_paths(cfg,wrist,beta=.9,report_times=[8])
    for j in range(len(w)):
        assert np.linalg.norm(b[j]['A']-R[j].T@w[j]['A']@R[j],'fro')<2e-10
        assert np.linalg.norm(np.outer(b[j]['mode'],b[j]['mode'])-
            R[j].T@np.outer(w[j]['mode'],w[j]['mode'])@R[j],'fro')<2e-10


def test_uniform_and_higherdimensional_complement_means_keep_their_rank_trace():
    for n in [3,8,16]:
        uniform,d=mean_projector(np.zeros((n,n)))
        assert d['numerical_ok'] and np.linalg.norm(uniform-(1-1/n)*np.eye(n))<1e-12
        A=np.diag(np.r_[-12.,np.zeros(n-1)])
        M,d=mean_projector(A);shifted,sd=mean_projector(A+7*np.eye(n))
        assert d['numerical_ok'] and sd['numerical_ok']
        assert abs(np.trace(M)-(n-1))<1e-10
        assert np.linalg.eigvalsh(M)[0]>0 and np.linalg.eigvalsh(M)[-1]<1
        assert np.linalg.norm(M-shifted)<1e-10
        assert abs(sd['log_normalizer']-d['log_normalizer']-7*(n-1))<1e-9
        iso,rho=isotropized_projector_mean(M,projector_from_normal(np.eye(n)[0]))
        assert 0<=rho<=1 and np.linalg.norm(iso-M)<1e-10


def test_unconstrained_ridge_map_is_not_a_soft_admissibility_projector():
    D=ridge_map(dict(C=np.eye(3),Y=2*np.eye(3)),0.)
    assert np.linalg.eigvalsh(D)[-1]>1
    with pytest.raises(ValueError):soft_action(D,1.)
    with pytest.raises(ValueError):soft_action(np.diag([-.1,.7,.8]),1.)
    with pytest.raises(ValueError):soft_action(np.eye(3),-1.)
    cfg=configuration('quick');P=projector_from_normal(np.array([0.,0.,1.]))
    record=_record(D,P,np.diag([0.,0.,-1.]),'ridge',cfg)
    assert np.isfinite(record['response_mse'])
    assert np.isnan(record['normal_chordal_error'])
    for lam in cfg['lambdas']:
        assert np.isnan(record['cost_'+str(int(lam))])
