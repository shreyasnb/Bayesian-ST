"""Independent theoretical and physical checks for the revised contact study."""
import numpy as np
import pytest
from lib.contact_geometry_study import (configuration,command_program,dynamic_data,
    projector_from_normal,normal_mode,mean_projector,isotropized_projector_mean,
    paper_certificate,soft_action,soft_cost,conditional_soft_excess,TangentGaussian)
from lib.contact_gaussian import measurement,measurement_jacobian,tangent_basis,exponential_normal
from lib.theory import fisher_information


def test_equal_energy_commands_have_different_local_fisher_information():
    n=np.array([0.,0.,1.]);X=np.eye(3)[:,:2];s=.01**2
    a=command_program(3,30,'balanced',.02);b=command_program(3,30,'unbalanced',.02)
    np.testing.assert_allclose(np.sum(a*a,axis=1),np.sum(b*b,axis=1),rtol=1e-13)
    Fa=fisher_information(X,a.T,s);Fb=fisher_information(X,b.T,s)
    assert np.linalg.matrix_rank(b)==3
    assert np.linalg.eigvalsh(Fa)[0] > 10*np.linalg.eigvalsh(Fb)[0]


def test_mechanical_wrist_transform_preserves_projection_observation_law():
    cfg=configuration('quick');cfg['dynamic_T']=12
    data=dynamic_data(cfg,'sphere',0)
    np.testing.assert_allclose(data['R'].transpose(0,2,1)@data['R'],np.broadcast_to(np.eye(3),data['R'].shape),atol=1e-12)
    np.testing.assert_allclose(data['G'][1:],data['R'][1:].transpose(0,2,1)@data['R'][:-1],atol=1e-12)
    assert data['physical_diagnostics']['maximum_projection_identity_error']<1e-12
    clean_body=np.einsum('tij,tj->ti',data['P'],data['V'])
    clean_world=np.array([projector_from_normal(n)@v for n,v in zip(data['world_normal'],data['world_V'])])
    np.testing.assert_allclose(clean_body,np.einsum('tji,tj->ti',data['R'],clean_world),atol=1e-12)
    altered=dict(cfg,sigma_dynamic=.008);other=dynamic_data(altered,'sphere',0)
    np.testing.assert_array_equal(data['R'],other['R'])
    np.testing.assert_array_equal(data['normal'],other['normal'])
    np.testing.assert_allclose(other['U']-clean_body,2*(data['U']-clean_body),atol=1e-12)


def test_matched_isotropization_preserves_normal_mode_and_expected_distance():
    A=np.diag([0.,8.,80.]);P,_,_=normal_mode(A);M,d=mean_projector(A)
    iso,rho=isotropized_projector_mean(M,P)
    assert d['numerical_ok']
    normalP=np.eye(3)-P
    np.testing.assert_allclose(np.trace(normalP@(np.eye(3)-M)),np.trace(normalP@(np.eye(3)-iso)),atol=1e-12)
    assert np.linalg.norm(M-iso)>1e-3
    assert 0<rho<1


def test_printed_paper_certificate_constants_and_invalid_condition():
    T=3072;v0=.02;s=.01**2;delta=.05
    X=np.broadcast_to(np.array([0.,0.,1.])[None,:,None],(T,3,1))
    V=command_program(3,T,'balanced',v0);b=paper_certificate(X,V,s,delta=delta)
    gamma=T*v0*v0/(3*s)
    noise=(2*np.sqrt(3)+np.sqrt(2*np.log(1/delta)))*np.sqrt(gamma)
    np.testing.assert_allclose(b['gamma'],gamma,rtol=1e-12)
    np.testing.assert_allclose(b['noise'],noise,rtol=1e-12)
    np.testing.assert_allclose(b['bound'],2*noise/gamma,rtol=1e-12)
    assert b['valid'] and b['drift']==0
    short=paper_certificate(X[:3],V[:3],s,delta=delta)
    assert not short['valid'] and np.isnan(short['bound'])


def test_conditional_contact_jacobian_is_independent_geodesic_derivative():
    n=np.array([1.,2.,3.]);n/=np.linalg.norm(n);E=tangent_basis(n);v=np.array([.02,-.01,.03])
    J=measurement_jacobian(n,v,E);h=1e-6
    numerical=np.column_stack([(measurement(exponential_normal(n,h*E[:,j]),v)-
        measurement(exponential_normal(n,-h*E[:,j]),v))/(2*h) for j in range(2)])
    np.testing.assert_allclose(J,numerical,atol=1e-10)


def test_nonprojector_soft_expected_cost_and_map_eligibility():
    D=np.diag([.2,.8,1.]);P=np.diag([0.,1.,1.]);lam=7.;v0=.02
    W=soft_action(D,lam);comparison=soft_action(P,lam)
    excess,error=conditional_soft_excess(D,P,lam,v0)
    H=np.eye(3)+lam*(np.eye(3)-D)
    target=v0*v0/3*np.trace((comparison-W).T@H@(comparison-W))
    np.testing.assert_allclose(excess,target,atol=1e-15)
    assert error<1e-15 and excess>0
    np.testing.assert_allclose(H@W,np.eye(3),atol=1e-12)
    # Squaring I-D would produce the wrong expected cost for a mean projector.
    wrong=v0*v0/3*(np.linalg.norm(W-np.eye(3),'fro')**2+lam*np.linalg.norm((np.eye(3)-D)@W,'fro')**2)
    assert abs(soft_cost(W,D,lam,v0)-wrong)>1e-5
    with pytest.raises(ValueError):soft_action(np.diag([1.2,.5,.3]),lam)
    with pytest.raises(ValueError):TangentGaussian(.01,-1.)
