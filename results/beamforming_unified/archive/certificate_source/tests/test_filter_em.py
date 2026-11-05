import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal
from lib.geometry import orth, projector, center, gaussian, risk
from lib.models import increment, constant, log_likelihood
from lib.filter import BinghamFilter
from lib.em import Record, fit_noise, marginal_mstep
from lib.moments import quadrature


def test_likelihoods_real_and_complex():
    rng=np.random.default_rng(5)
    for complex_data in [False,True]:
        n,k,s,tau=4,2,.4,1.2
        X=orth(gaussian(rng,(n,k),complex_data)); P=projector(X)
        u=gaussian(rng,n,complex_data); v=gaussian(rng,n,complex_data)
        for known in [False,True]:
            vv=v if known else None
            R=s*np.eye(n) if known else s*np.eye(n)+tau*P
            residual=u-P@v if known else u
            f=1 if complex_data else .5
            expected=-f*(n*np.log(np.pi if complex_data else 2*np.pi)+np.linalg.slogdet(R)[1]+np.vdot(residual,np.linalg.solve(R,residual)).real)
            assert_allclose(log_likelihood(u,P,s,tau,vv,complex_data),expected,atol=1e-11)


def test_exact_prediction_and_once_only_correction():
    rng=np.random.default_rng(7); n,k=5,2
    A=center(rng.normal(size=(n,n))); G=orth(rng.normal(size=(n,n)))
    f=BinghamFilter(n,k,.2,beta=.8,A0=A,window=3)
    u,v=rng.normal(size=(2,n))
    f.update(u,v,G)
    assert_allclose(f.records[-1].prior,center(.8*G@A@G.T),atol=1e-13)
    assert_allclose(f.A,center(.8*G@A@G.T+increment(u,.2,v=v)),atol=1e-13)
    for _ in range(4): f.update(u,v)
    assert len(f.records)==3
    old=[r.prior.copy() for r in f.records]
    f.update(u,v,fit=True,mean=lambda A,k: quadrature(A,k).mean) if n==3 else None
    assert all(np.array_equal(a,r.prior) for a,r in zip(old,f.records))


def records(conditional,seed=13,L=18):
    rng=np.random.default_rng(seed); out=[]
    for j in range(L):
        angle=.7*j/L; x=np.array([np.cos(angle),np.sin(angle),0.]); P=np.outer(x,x)
        prior=2*P-np.eye(3)*2/3
        v=rng.normal(size=3)
        u=P@v+np.sqrt(.3)*rng.normal(size=3) if conditional else x*rng.normal()+np.sqrt(.3)*rng.normal(size=3)
        out.append(Record(prior,u,v if conditional else None))
    return out


def test_em_ascent_frozen_distinct_priors():
    for conditional in [False,True]:
        rs=records(conditional); copies=[r.prior.copy() for r in rs]
        fit=fit_noise(rs,1,.9,None if conditional else .7,
                      lambda A,k:quadrature(A,k,order=40).mean,iterations=5,
                      log_normalizer=lambda A:quadrature(A,order=40).log_normalizer)
        assert fit.accepted==5 and fit.rejected==0
        assert np.min(np.diff(fit.objectives))>=-1e-8
        assert fit.s>0 and (fit.tau2 is None or fit.tau2>=0)
        assert all(np.array_equal(a,r.prior) for a,r in zip(copies,rs))
        assert not np.array_equal(rs[0].prior,rs[-1].prior)


def test_uncertainty_term_and_variance_boundary():
    M=np.diag([.6,.3,.1]);v=np.array([1.,2.,3.]);u=np.zeros(3)
    expected=np.vdot(u-M@v,u-M@v).real+risk(M,v)
    assert_allclose(expected,np.vdot(v,M@v))
    assert expected>np.linalg.norm(u-M@v)**2
    assert marginal_mstep(0,0,3,1,.2)==(.2,0.)
    s,tau=marginal_mstep(.1,2,3,1,.2)
    assert_allclose([s,tau],[4.1/3,0])
    assert_allclose(marginal_mstep(1,.1,3,1,.2),[.2,.8])


def test_filter_fit_does_not_replay_history():
    f=BinghamFilter(3,1,.8,tau2=1.,window=4)
    rng=np.random.default_rng(4)
    for u in rng.normal(size=(3,3)):f.update(u)
    u=rng.normal(size=3);old=f.A.copy()
    f.update(u,fit=True,mean=lambda A,k:quadrature(A,k,order=32).mean,iterations=2,
             log_normalizer=lambda A:quadrature(A,order=32).log_normalizer)
    assert_allclose(f.A,center(old+increment(u,f.s,f.tau2)),atol=1e-12)
    assert_allclose(f.records[-1].prior,old,atol=1e-12)
