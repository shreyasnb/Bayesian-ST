import numpy as np
import pytest
from scipy.linalg import expm

from lib.guide_tracking import (marginal_certificate,conditional_certificate,
    certificate_observation,marginal_preflight)
from lib.models import increment


def projector(x):
    return np.outer(x,x.conj())


def test_explicit_marginal_constants_initialization_and_infinite_memory():
    n=3;T=37;s=.3;q=1.2;beta=.97;delta=.04
    Ps=np.broadcast_to(np.diag([1.,0.,0.]),(T,n,n))
    A0=np.diag([-10.,-9.,8.])
    for complex_data in [False,True]:
        c=marginal_certificate(Ps.astype(complex) if complex_data else Ps,s,q,beta,delta,A0=A0,complex_data=complex_data)
        w=beta**np.arange(T);eta=(1 if complex_data else .5)*q/(s*(s+q))
        h=(2 if complex_data else 1)*n*np.log(9)+np.log(2/delta)
        assert c['initialization']==pytest.approx(beta**T*9.)
        assert c['gamma']==pytest.approx(eta*q*w.sum())
        assert c['noise']==pytest.approx(4*eta*(s+q)*(np.sqrt((w@w)*h)+h))
        assert c['drift']==0
        p=marginal_preflight(n,s,q,200000,beta,delta,complex_data)
        assert p['ratio']==pytest.approx(p['limit'],rel=1e-12)
        assert c['epsilon']/c['gamma']==pytest.approx(beta**T*9/c['gamma']+marginal_preflight(n,s,q,T,beta,delta,complex_data)['ratio'])


def test_noncommuting_transport_and_nonuniform_prior_marginal_decomposition():
    rng=np.random.default_rng(90);T,n=7,4
    matrices=rng.normal(size=(T,n,n));G=np.array([expm(.2*(a-a.T)) for a in matrices])
    X=np.linalg.qr(rng.normal(size=(n,2)))[0];Ps=np.broadcast_to(X@X.T,(T,n,n)).copy()
    U=rng.normal(size=(T,n));A0=np.diag([-4.,1.,3.,5.]);s=.4;q=1.1;beta=.8
    c=marginal_certificate(Ps,s,q,beta,A0=A0,G=G)
    report=certificate_observation(c,U)
    assert report['decomposition_error']<2e-13
    A=A0.copy()
    for u,g in zip(U,G):A=beta*g@A@g.T+increment(u,s,q)
    # Its scalar gauge is immaterial to the subspace and removed consistently.
    A-=beta**T*c['initialization_gauge']*np.eye(n)
    expected=c['initial_matrix']+c['eta']*sum(w*np.outer(t@u,t@u) for w,t,u in zip(c['weights'],c['transports'],U))
    np.testing.assert_allclose(A,expected,rtol=1e-13,atol=1e-13)
    P=np.linalg.eigh(A)[1][:,-2:];P=P@P.T
    assert report['modal_error']==pytest.approx(np.linalg.norm(P-Ps[-1],'fro')/np.sqrt(2))


def test_conditional_real_printed_noise_drift_and_decomposition():
    rng=np.random.default_rng(19);T=9;theta=np.linspace(0,.8,T)
    Ps=np.array([projector(np.array([np.cos(t),np.sin(t),0.])) for t in theta])
    V=rng.normal(size=(T,3));U=np.einsum('tij,tj->ti',Ps,V)+rng.normal(size=(T,3))*.2
    c=conditional_certificate(Ps,V,.04,beta=.9)
    w=.9**np.arange(T-1,-1,-1)
    expected=sum(wj*np.linalg.norm(P-Ps[-1],2)*np.dot(v,v)/.04 for wj,P,v in zip(w,Ps,V))
    assert c['drift']==pytest.approx(expected)
    D=sum(wj*wj*np.outer(v,v) for wj,v in zip(w,V))
    expected_noise=(2*np.sqrt(3)+np.sqrt(2*np.log(20)))*np.sqrt(np.linalg.eigvalsh(D)[-1]/.04)
    assert c['noise']==pytest.approx(expected_noise)
    assert certificate_observation(c,U,V)['decomposition_error']<2e-12


def test_native_complex_conditional_is_not_real_bound_or_marginal_constant():
    rng=np.random.default_rng(40);T=8;n=3
    H=rng.normal(size=(T,n))+1j*rng.normal(size=(T,n));H/=np.linalg.norm(H,axis=1)[:,None]
    Ps=np.einsum('ti,tj->tij',H,H.conj());V=rng.normal(size=(T,n))+1j*rng.normal(size=(T,n))
    U=np.einsum('tij,tj->ti',Ps,V)+.1*(rng.normal(size=(T,n))+1j*rng.normal(size=(T,n)))
    with pytest.raises(ValueError):conditional_certificate(Ps,V,.02)
    c=conditional_certificate(Ps,V,.02,complex_data=True)
    assert not c['manuscript_verified']
    assert 'Derived conservative' in c['formula_name']
    assert certificate_observation(c,U,V)['decomposition_error']<2e-11
    assert c['eta']==50.


def test_invalid_conditions_remain_nan_and_zero_information_is_visible():
    T=24;Ps=np.broadcast_to(np.diag([0.,0.,1.]),(T,3,3));V=np.broadcast_to([1.,0.,0.],(T,3))
    c=conditional_certificate(Ps,V,.2)
    assert c['gamma']==0 and not c['valid'] and not c['informative'] and np.isnan(c['bound'])
    p=marginal_preflight(8,.2,1.,100000,beta=.98,complex_data=True)
    assert p['permanently_vacuous'] and p['limit']>.5


def test_long_static_bound_and_geometric_ceiling_are_distinct():
    T=16384;Ps=np.broadcast_to(np.diag([1.,0.,0.]),(T,3,3))
    c=marginal_certificate(Ps,.2,1.,complex_data=False)
    assert c['valid'] and c['informative'] and c['bound']<c['ceiling']
    assert c['bound']==pytest.approx(2*np.sqrt(c['k'])*c['epsilon']/c['gamma'])
    c=marginal_certificate(np.eye(3)-Ps[:2048],.2,1.,complex_data=False)
    assert c['valid'] and not c['informative'] and c['bound']>c['ceiling']


def test_matched_anisotropy_control_is_equivalent_to_modal_loading():
    from lib.beamforming_decisions import isotropized_mean
    from lib.beamforming_numerics import complex_moments
    from lib.array_geometry_study import distortionless_weight
    rng=np.random.default_rng(32);Q=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    A=(Q*np.array([-12.,-5.,-1.,0.]))@Q.conj().T
    P=projector(Q[:,-1]);M=complex_moments(A).mean;Mi,_=isotropized_mean(M,P)
    alpha=np.trace(P@M).real;xi=(1-alpha)/3;s=.2;q=1.4
    expected=(s+q*xi)*np.eye(4)+q*(alpha-xi)*P
    np.testing.assert_allclose(s*np.eye(4)+q*Mi,expected,atol=1e-12)
    loading=(s+q*xi)/(alpha-xi)-s
    a=rng.normal(size=4)+1j*rng.normal(size=4);a/=np.linalg.norm(a)
    np.testing.assert_allclose(distortionless_weight(expected,a),distortionless_weight((s+loading)*np.eye(4)+q*P,a),atol=2e-12)
    uniform,_=isotropized_mean(np.eye(4)/4,P)
    np.testing.assert_allclose(uniform,np.eye(4)/4,atol=1e-14)
