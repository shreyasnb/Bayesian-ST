import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import multivariate_normal

from lib.final_theory import (
    exact_aggregate,sequential_parameter,align_block,sharp_marginal_certificate,
    sharp_conditional_certificate,natural_curvature,tangent_coordinate,
    global_kl_identity,independent_nodes,native_response_covariance)
from lib.geometry import center
from lib.models import increment
from lib.guide_information import chart_basis,finite_hessian
from lib.block_em import moments
from lib.guide_tracking import certificate_observation


def _normal(rng,shape,complex_data):
    z=rng.normal(size=shape)
    return (z+1j*rng.normal(size=shape))/np.sqrt(2) if complex_data else z


@pytest.mark.parametrize('complex_data',[False,True])
def test_discounted_block_matches_individual_gaussian_factors_and_predictions(complex_data):
    rng=np.random.default_rng(803);n=3;L=16
    U=_normal(rng,(L,n),complex_data);V=_normal(rng,(L,n),complex_data)
    A0=np.diag([.7,-.4,-.3]).astype(complex if complex_data else float)
    s=np.linspace(.3,.8,L);q=np.linspace(.2,1.1,L)
    for conditional in (False,True):
        vs=V if conditional else None
        A=exact_aggregate(A0,U,s,q,V=vs,beta=.87,complex_data=complex_data)
        sequential=sequential_parameter(A0,U,s,q,V=vs,beta=.87,complex_data=complex_data)
        assert np.linalg.norm(A-sequential)<1e-11
        M,psi,_=moments(A,1,complex_data);Mseq,zseq,_=moments(sequential,1,complex_data)
        assert np.linalg.norm(M-Mseq)<1e-10 and abs(psi-zseq)<1e-10
        # The likelihood ratio is computed directly from Gaussian densities,
        # not by summing the implementation's increment routine.
        P=np.diag([1.,0.,0.]);Q=np.diag([0.,1.,0.])
        direct=0.
        for j,u in enumerate(U):
            covariance=s[j]*np.eye(n) if conditional else s[j]*np.eye(n)+q[j]*P
            other=s[j]*np.eye(n) if conditional else s[j]*np.eye(n)+q[j]*Q
            mean=P@V[j] if conditional else np.zeros(n)
            other_mean=Q@V[j] if conditional else np.zeros(n)
            if complex_data:
                def density(R,mu):
                    return multivariate_normal.logpdf(np.r_[u.real,u.imag],
                        mean=np.r_[mu.real,mu.imag],
                        cov=.5*np.block([[R.real,-R.imag],[R.imag,R.real]]))
            else:
                def density(R,mu):
                    return multivariate_normal.logpdf(u,mean=mu,cov=R)
            direct+=.87**(L-1-j)*(density(covariance,mean)-density(other,other_mean))
        actual=np.trace((A-.87**L*A0)@(P-Q)).real
        assert abs(actual-direct)<1e-10
        assert np.linalg.norm(M@U[0]-Mseq@U[0])<1e-10


@pytest.mark.parametrize('complex_data',[False,True])
def test_noncommuting_recorded_transport_aligns_responses_inputs_and_boundary(complex_data):
    rng=np.random.default_rng(804);L=4;n=3
    rotations=[]
    for _ in range(L):
        Z=_normal(rng,(n,n),complex_data)
        rotations.append(expm(.2*(Z-Z.conj().T)))
    G=np.array(rotations);U=_normal(rng,(L,n),complex_data)
    V=_normal(rng,(L,n),complex_data);A0=np.diag([2.,-.5,-1.5])
    A=exact_aggregate(A0,U,.4,V=V,beta=.8,G=G,complex_data=complex_data)
    sequential=sequential_parameter(A0,U,.4,V=V,beta=.8,G=G,complex_data=complex_data)
    assert np.linalg.norm(A-sequential)<1e-10
    aligned=align_block(A0,U,V,G,complex_data=complex_data)
    block=exact_aggregate(aligned['A0'],aligned['U'],.4,V=aligned['V'],
                          beta=.8,complex_data=complex_data)
    assert np.linalg.norm(block-A)<1e-10
    assert np.linalg.norm(G[0]@G[1]-G[1]@G[0])>1e-4
    # Neither normalization by L nor discounting only once is the same recursion.
    wrong=center(.8*aligned['A0']+sum(increment(u,.4,v=v,complex_data=complex_data)
                  for u,v in zip(aligned['U'],aligned['V'])))
    assert np.linalg.norm(wrong-A)>.1


def test_covariance_averaging_changes_nonuniform_prior_mode_and_mean():
    A0=np.diag([1.5,0.,-1.5]);U=np.tile([0.,2.,0.],(4,1))
    full=exact_aggregate(A0,U,1.,1.)
    averaged=center(A0+sum(increment(u,1.,1.) for u in U)/len(U))
    h=np.linalg.eigh(full)[1][:,-1];g=np.linalg.eigh(averaged)[1][:,-1]
    assert abs(h@g)<1e-12
    M,_,_=moments(full,1);Q,_,_=moments(averaged,1)
    assert np.linalg.norm(M-Q)>.2
    B=increment(np.zeros(3),.2,v=np.array([1.,0.,0.]))
    assert np.linalg.eigvalsh(B)[0]<0  # no PSD projection of conditional A


@pytest.mark.parametrize('complex_data',[False,True])
def test_global_kl_matches_gaussian_closed_identity_in_same_input_law(complex_data):
    rng=np.random.default_rng(805)
    X=np.linalg.qr(_normal(rng,(5,2),complex_data))[0]
    Y=np.linalg.qr(_normal(rng,(5,2),complex_data))[0]
    for V in [None,_normal(rng,(7,5),complex_data)]:
        direct,formula=global_kl_identity(X,Y,.7,1.2,V,complex_data)
        assert abs(direct-formula)<1e-10


def test_sharp_conditional_aggregate_drift_and_reference_information_gap():
    theta=np.linspace(0.,.7,8)
    H=np.c_[np.sin(theta),np.zeros(8),np.cos(theta)]
    P=np.einsum('ti,tj->tij',H,H);V=np.eye(3)[np.arange(8)%3]
    A0=np.diag([-4.,0.,1.])
    c=sharp_conditional_certificate(P,V,.03,A0=A0)
    reference=sum((P[-1]@np.outer(v,v)+np.outer(v,v)@P[-1]-np.outer(v,v))/.06 for v in V)
    population=sum((p@np.outer(v,v)+np.outer(v,v)@p-np.outer(v,v))/.06 for p,v in zip(P,V))
    assert np.linalg.norm(c['reference']-reference)<1e-10
    assert abs(c['drift']-np.linalg.norm(population-reference,2))<1e-10
    assert c['drift']<=c['separate_norm_drift']+1e-10
    assert c['gap_information_error']<1e-10
    assert abs(c['initialization']-2.5)<1e-10
    U=np.einsum('tij,tj->ti',P,V)+.1
    result=certificate_observation(c,U,V)
    assert result['decomposition_error']<1e-10
    # A long, well-excited state makes the exact sharp expression testable.
    P=np.broadcast_to(np.diag([0.,0.,1.]),(1024,3,3))
    V=np.eye(3)[np.arange(1024)%3]
    c=sharp_conditional_certificate(P,V,.03)
    assert c['valid']
    assert c['bound']==pytest.approx(c['ceiling']*c['epsilon']/(c['gamma']-c['epsilon']))
    assert c['bound']<c['loose_bound']


@pytest.mark.parametrize('complex_data',[False,True])
def test_marginal_sharp_bound_zero_signal_and_nonuniform_gauge(complex_data):
    P=np.broadcast_to(np.diag([1.,0.,0.]),(4096,3,3))
    c=sharp_marginal_certificate(P,.2,1.,complex_data=complex_data)
    assert c['gap_information_error']<1e-10
    assert c['valid'] and c['informative']
    assert c['bound']<c['loose_bound']
    zero=sharp_marginal_certificate(P,.2,0.,complex_data=complex_data,A0=np.diag([-4.,0.,1.]))
    assert zero['initialization']==pytest.approx(2.5)
    assert not zero['valid'] and np.isnan(zero['bound']) and zero['gamma']==0


def test_invariant_and_graph_density_hessians_have_distinct_volume_term():
    A=np.diag([2.,-1.,-3.]);X=np.eye(3)[:,:1];Q=np.eye(3)[:,1:]
    def invariant(z):
        Y=chart_basis(X,z,False,Q)
        return -np.trace(A@Y@Y.T)
    def graph(z):
        return invariant(z)+1.5*np.log1p(z@z)
    H=natural_curvature(A)['H']
    assert np.max(abs(finite_hessian(invariant,2,1e-4)-H))<1e-6
    assert np.max(abs(finite_hessian(graph,2,1e-4)-H-3*np.eye(2)))<1e-6
    coordinate,local=tangent_coordinate(np.array([np.cos(.2),np.sin(.2),0.]),X[:,0])
    assert local and np.linalg.norm(coordinate)==pytest.approx(.2)


@pytest.mark.parametrize('complex_data,k',[(False,1),(False,2),(True,1)])
def test_full_response_covariance_requires_second_projector_moment(complex_data,k):
    A=np.diag([1.1,-.3,-.8]);v=np.array([.6,.2,-.4])
    if complex_data:v=v.astype(complex)+1j*np.array([.1,-.3,.2])
    ref=independent_nodes(A,k,complex_data,48)
    response=np.einsum('bij,j->bi',ref['projectors'],v)
    mu=ref['mean']@v
    covariance=np.einsum('b,bi,bj->ij',ref['weights'],response-mu,(response-mu).conj())
    native,M,_=native_response_covariance(A,k,v,complex_data)
    assert np.linalg.norm(native-covariance)<2e-7
    assert abs(np.trace(covariance).real-np.vdot(v,(M-M@M)@v).real)<1e-10
    assert np.linalg.norm(covariance-(M-M@M))>.01
