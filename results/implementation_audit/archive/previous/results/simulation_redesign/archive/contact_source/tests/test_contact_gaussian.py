"""Independent geometry, Jacobian, mapped-law and causal Gaussian checks."""
from copy import deepcopy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.polynomial.hermite_e import hermegauss
from lib.contact_gaussian import (tangent_basis,measurement,measurement_jacobian,
    exponential_normal,parallel_rotation,normal_coordinates,gaussian_summary,
    laplace_state,laplace_summary,RecursiveTangentGaussian,_MappedGaussian)
from lib.contact_posterior import contact_summary


def queries():
    pair=np.array([[1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]])/np.sqrt(2)
    return .02*np.vstack([np.eye(3),pair])


def test_measurement_jacobian_and_geodesic_parallel_transport():
    n=np.array([1.,2.,-3.]);n/=np.linalg.norm(n);E=tangent_basis(n);v=np.array([.01,.03,-.02])
    h=1e-6
    numeric=np.column_stack([(measurement(exponential_normal(n,h*e),v)-
                             measurement(exponential_normal(n,-h*e),v))/(2*h) for e in E.T])
    assert_allclose(measurement_jacobian(n,v,E),numeric,atol=1e-11)
    x=E@np.array([.2,-.1]);R=parallel_rotation(n,x);m=exponential_normal(n,x)
    assert_allclose(R.T@R,np.eye(3),atol=1e-14)
    assert_allclose(R@n,m,atol=1e-14)
    assert_allclose((R@E).T@m,0,atol=1e-14)
    coordinates,angle=normal_coordinates(n,-m,E)
    assert_allclose(coordinates,[.2,-.1],atol=1e-14)
    assert_allclose(angle,np.linalg.norm(x),atol=1e-14)


def test_mapped_gaussian_mean_matches_independent_hermite_integration():
    n=np.array([1.,2.,3.]);n/=np.linalg.norm(n);E=tangent_basis(n)
    P=np.array([[.13,.04],[.04,.03]])
    law=_MappedGaussian(n,P,E,128);M=law.mean()
    points,weights=hermegauss(32);weights=weights/np.sqrt(2*np.pi)
    z=np.array(np.meshgrid(points,points)).reshape(2,-1).T
    w=(weights[:,None]*weights).ravel();x=z@np.linalg.cholesky(P).T@E.T
    r=np.linalg.norm(x,axis=1)
    normals=np.cos(r)[:,None]*n+np.sinc(r/np.pi)[:,None]*x
    reference=np.eye(3)-np.einsum('b,bi,bj->ij',w,normals,normals)
    assert_allclose(M,reference,atol=2e-12)
    assert_allclose(np.trace(M),2,atol=1e-14)
    assert np.linalg.eigvalsh(M).min()>0 and np.linalg.eigvalsh(M).max()<1


def test_continuous_response_intervals_and_cut_mass_against_iid_gaussians():
    rng=np.random.default_rng(734025);n=np.array([1.,0.,0.]);E=tangent_basis(n)
    P=np.diag([.15,.03]);W=queries()
    result=gaussian_summary(n,P,W,basis=E)
    z=rng.multivariate_normal(np.zeros(2),P,size=160000);r=np.linalg.norm(z,axis=1)
    normals=np.cos(r)[:,None]*n+np.sinc(r/np.pi)[:,None]*(z@E.T)
    responses=W[None,:,:]-np.einsum('bi,qi->bq',normals,W)[:,:,None]*normals[:,None,:]
    empirical=np.quantile(responses,[.025,.975],axis=0)
    assert_allclose(result['response_lower'],empirical[0],atol=.00012)
    assert_allclose(result['response_upper'],empirical[1],atol=.00012)
    exact_mass=_MappedGaussian(n,P,E,256).cut_locus_mass()
    assert abs(np.mean(r>=np.pi/2)-exact_mass)<.001
    assert result['numerics']['integration_numerical_ok']
    assert 'not invariant-density' in result['numerics']['hpd_kind']


def test_laplace_matches_concentrated_bingham_but_singular_law_fails():
    A=np.diag([0.,80000.,80000.]);W=queries()
    approximation=laplace_summary(A,W);resolved=contact_summary(A,W)
    assert approximation['numerics']['numerical_ok']
    assert_allclose(approximation['mean'],resolved['mean'],atol=3e-10)
    assert_allclose(approximation['response_lower'],resolved['response_lower'],atol=2e-9)
    assert_allclose(approximation['response_upper'],resolved['response_upper'],atol=2e-9)
    with pytest.raises(ArithmeticError,match='Singular'):
        laplace_state(np.zeros((3,3)))
    failed=laplace_summary(np.zeros((3,3)),W)
    assert not failed['numerics']['numerical_ok']
    assert failed['numerics']['singular_precision']
    assert np.isnan(failed['response_lower']).all()


def test_broad_and_deterministic_limits_remain_explicit():
    W=queries();n=np.array([0.,0.,1.])
    broad=gaussian_summary(n,.6*np.eye(2),W)
    assert broad['numerics']['cut_locus_mass']>.01
    assert not broad['numerics']['local_approximation_ok']
    assert not broad['numerics']['numerical_ok']
    exact=gaussian_summary(n,np.zeros((2,2)),W)
    truth=W-np.outer(W@n,n)
    assert_allclose(exact['mean'],np.diag([1.,1.,0.]),atol=1e-14)
    assert_allclose(exact['response_lower'],truth,atol=2e-12)
    assert_allclose(exact['response_upper'],truth,atol=2e-12)
    assert exact['ball_radius']==0
    degenerate=gaussian_summary(n,np.diag([.01,0.]),W,true_normal=n)
    assert degenerate['numerics']['degenerate_covariance']
    assert degenerate['numerics']['local_ellipse_dimension']==1
    assert degenerate['hpd_covered']


def test_basis_rotation_and_antipodal_equivariance():
    n=np.array([1.,2.,3.]);n/=np.linalg.norm(n);E=tangent_basis(n)
    P=np.array([[.02,.003],[.003,.006]]);W=queries()
    angle=.37;Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    a=gaussian_summary(n,P,W,basis=E,angular_orders=(64,128))
    b=gaussian_summary(-n,Q.T@P@Q,W,basis=-E@Q,angular_orders=(64,128))
    assert_allclose(a['mean'],b['mean'],atol=2e-12)
    assert_allclose(a['response_lower'],b['response_lower'],atol=3e-6)
    assert_allclose(a['response_upper'],b['response_upper'],atol=3e-6)


def test_recursive_filter_causality_joseph_covariance_and_warmup_model():
    rng=np.random.default_rng(734026);f=RecursiveTangentGaussian(1e-6,np.inf,warmup=6)
    n=np.array([1.,2.,3.]);n/=np.linalg.norm(n)
    for t in range(10):
        v=.02*np.eye(3)[t%3];u=measurement(n,v)+.001*rng.normal(size=3)
        score=f.predictive_log_score(u,v)
        assert np.isfinite(score);f.update(u,v)
    assert f.initialized and f.t==10
    assert np.linalg.eigvalsh(f.covariance).min()>0
    assert_allclose(f.basis.T@f.normal,0,atol=1e-14)
    a,b=deepcopy(f),deepcopy(f);a.predict();b.predict()
    assert_allclose(a.normal,b.normal,atol=0);assert_allclose(a.covariance,b.covariance,atol=0)
    old=a.covariance.copy();a.predict();assert_allclose(a.covariance,old,atol=0)
    a.update(np.zeros(3),np.ones(3)*.02);b.update(np.ones(3)*.005,np.ones(3)*.02)
    assert not np.allclose(a.normal,b.normal)
    assert a.timings['warmup_seconds']>0


def test_recursive_state_is_invariant_to_tangent_basis_and_antipodal_sign():
    A=np.diag([0.,300.,1000.]);a=RecursiveTangentGaussian(.01,np.inf);a.initialize(A)
    b=deepcopy(a);angle=.71
    Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    b.normal=-a.normal;b.basis=-a.basis@Q;b.covariance=Q.T@a.covariance@Q
    for t in range(5):
        v=np.array([.5,.1+t*.03,-.3]);u=np.array([.1,.2,.3])
        assert_allclose(a.predictive_log_score(u,v),b.predictive_log_score(u,v),atol=1e-11)
        a.update(u,v);b.update(u,v)
        assert_allclose(np.outer(a.normal,a.normal),np.outer(b.normal,b.normal),atol=2e-12)
        assert_allclose(a.basis@a.covariance@a.basis.T,b.basis@b.covariance@b.basis.T,atol=2e-12)
