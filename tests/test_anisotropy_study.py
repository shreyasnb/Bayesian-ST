"""Independent measure, complement, response and soft-decision checks."""
import numpy as np
import pytest
from lib.anisotropy_study import (PRIMARY_PARAMETERS,continuous_moments,
    continuous_regions,continuous_region_mass,sphere_rule,response_summary,
    axial_parameter,soft_action,soft_cost,query_bounds)
from lib.spectral_series import rank_one_bingham
from lib.reference import rank_one_regions_gr13


def test_native_mean_logz_against_continuous_sphere_and_shift():
    A=PRIMARY_PARAMETERS[0];native=rank_one_bingham(A,tol=1e-11)
    direct=continuous_moments(A,epsabs=1e-10)
    assert np.linalg.norm(native.mean-direct['mean'],2)<2e-10
    assert abs(native.log_normalizer-direct['log_normalizer'])<2e-10
    shifted=continuous_moments(A+17*np.eye(3),epsabs=1e-10)
    assert np.linalg.norm(shifted['mean']-direct['mean'])<1e-12
    assert abs(shifted['log_normalizer']-direct['log_normalizer']-17)<1e-11
    uniform=continuous_moments(np.zeros((3,3)))
    assert abs(uniform['log_normalizer'])<1e-12
    assert np.linalg.norm(uniform['mean']-np.eye(3)/3)<1e-12


def test_coordinate_permutation_preserves_entire_distance_law():
    A,B=PRIMARY_PARAMETERS;swap=np.eye(3)[[0,2,1]]
    assert np.array_equal(B,swap@A@swap.T)
    Ma=continuous_moments(A)['mean'];Mb=continuous_moments(B)['mean']
    assert np.linalg.norm(Mb-swap@Ma@swap.T)<1e-12
    # Independent direct quadrature in the two original coordinate orders.
    from lib.anisotropy_study import _continuous_mass
    for radius in [.1,.3,.5,.8]:
        a=_continuous_mass(8.,80.,'ball',radius,1e-10)
        b=_continuous_mass(80.,8.,'ball',radius,1e-10)
        assert abs(a-b)<2e-10
    assert abs(np.trace(np.diag([1.,0,0])@(Ma-Mb)))<1e-12


def test_continuous_regions_against_independent_dawson_reference():
    A=PRIMARY_PARAMETERS[0];reference=rank_one_regions_gr13(A,epsabs=1e-11)
    direct=continuous_regions(A,epsabs=1e-10)
    assert abs(direct['radius']-reference.radius)<2e-9
    assert abs(direct['hpd_loss']-reference.hpd_loss)<2e-9
    assert abs(continuous_region_mass(A,'ball',direct['radius'])-.95)<1e-9
    assert abs(continuous_region_mass(A,'hpd',direct['hpd_loss'])-.95)<1e-9
    bounds=query_bounds(A,direct['radius'],direct['hpd_loss'])
    assert abs(bounds[1]['ball_bound']-direct['radius'])<2e-8
    assert abs(bounds[2]['ball_bound']-direct['radius'])<2e-8
    assert abs(bounds[1]['hpd_bound']-np.sqrt(direct['hpd_loss']/8))<2e-8
    assert abs(bounds[2]['hpd_bound']-np.sqrt(direct['hpd_loss']/80))<2e-8


def test_complement_response_risk_and_full_covariance_trace():
    A=PRIMARY_PARAMETERS[0];M=rank_one_bingham(A,tol=1e-11).mean
    rule=sphere_rule(A,64,128);v=np.array([.3,.4,np.sqrt(.75)])
    response=response_summary(M,[v]);normal=rule['nodes']*(rule['nodes']@v)[:,None]
    plane=v-normal;mean_normal=M@v;mean_plane=(np.eye(3)-M)@v
    cn=normal-mean_normal;cp=plane-mean_plane
    covariance=(cn.T*rule['weights'])@cn
    assert abs(rule['weights'].sum()-1)<1e-12
    assert np.linalg.norm(rule['mean']-M)<2e-10
    assert abs(rule['weights']@np.sum(cp*cp,axis=1)-response['risk'][0])<2e-10
    assert abs(np.trace(covariance)-response['risk'][0])<2e-10
    assert np.linalg.norm(covariance-(M-M@M))>1e-3
    # Every stored quadrature node is a unit normal, not a 3D subspace basis.
    assert np.max(abs(np.sum(rule['nodes']**2,axis=1)-1))<1e-12


def test_axial_control_matches_rms_but_not_general_shape():
    A=PRIMARY_PARAMETERS[0];M=rank_one_bingham(A,tol=1e-11).mean
    kappa,axial=axial_parameter(M,np.eye(3)[0])
    assert kappa>0
    assert abs(axial[0,0]-M[0,0])<2e-10
    assert abs(axial[1,1]-axial[2,2])<1e-12
    # An existing axial parameter is recovered, including its identity gauge.
    original=np.diag([0.,-8.,-8.]);known=rank_one_bingham(original,tol=1e-11).mean
    fitted,mean=axial_parameter(known,np.eye(3)[0])
    assert abs(fitted-8)<2e-8
    assert np.linalg.norm(mean-known)<2e-10


def test_soft_action_identity_against_direct_projector_quadrature():
    A=PRIMARY_PARAMETERS[0];M=rank_one_bingham(A,tol=1e-11).mean
    v=np.ones(3)/np.sqrt(3);penalty=20.;optimal=soft_action(M,v,penalty)
    alternative=np.array([.04,.2,.5]);H=np.eye(3)+penalty*M
    excess=soft_cost(M,v,alternative,penalty)-soft_cost(M,v,optimal,penalty)
    assert abs(excess-(alternative-optimal)@H@(alternative-optimal))<1e-12
    rule=sphere_rule(A,64,128)
    direct=np.sum((optimal-v)**2)+penalty*(rule['weights']@((rule['nodes']@optimal)**2))
    assert abs(direct-soft_cost(M,v,optimal,penalty))<2e-10
    assert np.linalg.norm(H@optimal-v)<1e-12


def test_rotation_and_invalid_measure_inputs():
    A=PRIMARY_PARAMETERS[0]
    Q,_=np.linalg.qr(np.array([[1.,2.,3.],[4.,-1.,1.],[2.,0.,1.]]))
    direct=continuous_moments(A);rotated=continuous_moments(Q@A@Q.T)
    assert np.linalg.norm(rotated['mean']-Q@direct['mean']@Q.T)<1e-11
    assert abs(rotated['log_normalizer']-direct['log_normalizer'])<1e-11
    with pytest.raises(ValueError):continuous_moments(np.eye(4))
    with pytest.raises(ValueError):continuous_regions(np.zeros((3,3)))
    with pytest.raises(ValueError):soft_action(np.eye(3),np.ones(3),penalty=-1)
