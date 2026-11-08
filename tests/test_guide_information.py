"""Independent Gaussian, tangent-coordinate and continuous-measure checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import null_space
from scipy.special import logsumexp

from lib.beamforming_theory import passive_fisher_information, passive_tangent_basis
from lib.guide_information import (algebra_checks, chart_basis, conditional_fisher,
    expected_nll, finite_hessian, gaussian_kl, independent_moments, native_moments)
from lib.models import constant, increment
from lib.theory import fisher_information, tracking_bound


def test_real_rank_two_and_complex_information_from_covariance_derivatives():
    rng=np.random.default_rng(4912)
    for cx in (False,True):
        X=np.linalg.qr(rng.normal(size=(5,2))+(1j*rng.normal(size=(5,2)) if cx else 0))[0]
        Q=null_space(X.conj().T);s,q=.6,1.4
        R=s*np.eye(5)+q*X@X.conj().T
        inverse=np.linalg.inv(R)
        D=q*passive_tangent_basis(X,cx,Q)
        direct=np.array([[(1. if cx else .5)*np.trace(inverse@a@inverse@b).real
            for b in D] for a in D])
        F=passive_fisher_information(5,2,s,q,complex_data=cx)
        assert_allclose(direct,F,atol=2e-13)
        assert F.shape==((12,12) if cx else (6,6))
        z=np.zeros(len(F));z[0]=.07
        Y=chart_basis(X,z,cx,Q)
        assert_allclose(Y.conj().T@Y,np.eye(2),atol=1e-13)
        expected=finite_hessian(lambda a:expected_nll(X,a,s,q,cx,None,Q),len(F),.001)
        KL=finite_hessian(lambda a:gaussian_kl(X,a,s,q,cx,None,Q),len(F),.001)
        assert_allclose(expected,F,atol=2e-5)
        assert_allclose(KL,F,atol=2e-5)


def test_conditional_recorded_inputs_and_independent_channel_additivity():
    rng=np.random.default_rng(3910);X=np.linalg.qr(rng.normal(size=(5,2)))[0]
    Q=null_space(X.T);V=rng.normal(size=(8,5));s=.7
    F=conditional_fisher(X,V,s,False,Q)
    assert_allclose(F,fisher_information(X,V.T,s),atol=1e-12)
    H=finite_hessian(lambda z:expected_nll(X,z,s,V=V,complement=Q),len(F),.001)
    assert_allclose(H,F,atol=7e-5)
    first,second=V[:3],V[3:]
    assert_allclose(conditional_fisher(X,first,s,False,Q)+conditional_fisher(X,second,s,False,Q),F,atol=1e-12)
    marginal=passive_fisher_information(5,2,s,1.4,observations=3,complex_data=False)
    mixed=finite_hessian(lambda z:3*gaussian_kl(X,z,s,1.4,False,None,Q)
        +gaussian_kl(X,z,s,1.4,False,V,Q),len(F),.001)
    assert_allclose(mixed,marginal+F,atol=9e-5)


def test_real_and_complex_quadrature_are_distinct_native_references():
    A=np.array([[.3,.2,0.],[.2,-.7,.1],[0.,.1,1.1]])
    realM,realz=independent_moments(A,1,False,48)
    complexM,complexz=independent_moments(A.astype(complex),1,True,48)
    assert np.linalg.norm(realM-complexM)>.01
    for cx,k in [(False,1),(False,2),(True,1)]:
        M,z=independent_moments(A,k,cx,48)
        M2,z2=independent_moments(A,k,cx,72)
        native,nativez=native_moments(A,k,cx)
        assert_allclose(M,M2,atol=2e-12)
        assert_allclose(z,z2,atol=2e-12)
        assert_allclose(M,native,atol=2e-11)
        assert_allclose(z,nativez,atol=2e-11)
        shifted,zshift=independent_moments(A+2.7*np.eye(3),k,cx,48)
        assert_allclose(shifted,M,atol=2e-12)
        assert_allclose(zshift-z,2.7*k,atol=2e-12)


def test_fixed_state_block_evidence_against_direct_rank_two_gaussian_integration():
    rng=np.random.default_rng(6894);s,q=.9,1.2;n,k=3,2
    U=rng.normal(size=(3,n));A0=np.diag([.2,-.1,.4])
    x,w=np.polynomial.legendre.leggauss(40);phi=np.arange(160)*2*np.pi/160
    vectors=np.stack(np.broadcast_arrays(np.sqrt(1-x*x)[:,None]*np.cos(phi),
        np.sqrt(1-x*x)[:,None]*np.sin(phi),x[:,None]),axis=-1).reshape(-1,3)
    P=np.eye(3)-np.einsum('bi,bj->bij',vectors,vectors)
    prior_weights=np.broadcast_to(w[:,None]/320,(40,160)).ravel()
    prior_log=np.einsum('ij,bji->b',A0,P)+np.log(prior_weights)
    direct_log=prior_log.copy()
    R=s*np.eye(3)+q*P
    for u in U:
        quadratic=np.einsum('i,bi->b',u,np.linalg.solve(R,np.broadcast_to(u,(len(P),3))[...,None])[...,0])
        direct_log+=-.5*(3*np.log(2*np.pi)+np.linalg.slogdet(R)[1]+quadratic)
    direct=logsumexp(direct_log)-logsumexp(prior_log)
    B=sum((increment(u,s,q) for u in U),np.zeros((3,3)))
    _,psi0=native_moments(A0,k)
    _,psi1=native_moments(A0+B,k)
    normalizer=sum(constant(u,k,s,q) for u in U)+psi1-psi0
    assert_allclose(direct,normalizer,atol=2e-11)


def test_all_low_dimensional_algebra_and_trace_constants():
    for cx,k in [(False,1),(False,2),(True,1)]:
        rows=algebra_checks(np.random.default_rng(3101+int(cx)+k),cx,k)
        assert all(row['status']=='passed' for row in rows),rows


def test_realized_bingham_curvature_uses_gaps_not_expected_information():
    A=np.diag([4.,1.,-2.]);X=np.eye(3)[:,:1];Q=np.eye(3)[:,1:]
    for cx in (False,True):
        d=4 if cx else 2
        def loss(z):
            Y=chart_basis(X,z,cx,Q)
            return -float(np.trace(A@Y@Y.conj().T).real)
        H=finite_hessian(loss,d,.001)
        expected=np.diag([6.,12.,6.,12.] if cx else [6.,12.])
        assert_allclose(H,expected,atol=2e-5)
        assert np.ptp(np.linalg.eigvalsh(H))>5.9


def test_matched_isotropy_covariance_is_equivalent_to_modal_loading():
    rng=np.random.default_rng(9120);X=np.linalg.qr(rng.normal(size=(6,2)))[0]
    P=X@X.T;n,k=6,2;s,q=.3,1.7;alpha=.7;xi=(k-k*alpha)/(n-k)
    M=alpha*P+xi*(np.eye(n)-P)
    R=s*np.eye(n)+q*M
    direct=(s+q*xi)*np.eye(n)+q*(alpha-xi)*P
    ell=(s+q*xi)/(alpha-xi)-s
    loaded=(s+ell)*np.eye(n)+q*P
    a=rng.normal(size=n);a/=np.linalg.norm(a)
    w=np.linalg.solve(R,a);w/=a@w
    w2=np.linalg.solve(loaded,a);w2/=a@w2
    assert_allclose(R,direct,atol=1e-14)
    assert_allclose(w,w2,atol=1e-13)
    flat=k/n
    Rflat=s*np.eye(n)+q*(flat*P+flat*(np.eye(n)-P))
    assert_allclose(Rflat,(s+q*flat)*np.eye(n),atol=1e-14)
    isotropic=np.linalg.solve(Rflat,a);isotropic/=a@isotropic
    assert_allclose(isotropic,a,atol=1e-13)


def test_complex_arrays_are_not_sent_to_the_old_real_bound():
    X=np.broadcast_to(np.eye(3,dtype=complex)[:,:1],(5,3,1))
    with pytest.raises(ValueError,match='reals'):
        tracking_bound(X,np.ones((5,3),complex),.5)
