"""Mechanics, physical response identity and observation-schedule integration."""
import numpy as np
from numpy.testing import assert_allclose
from lib.contact_mechanics import (solve_contact_velocity,generate_contact,
                                   integration_check,PLANE_NORMAL)


def test_kkt_force_balance_matches_independent_isotropic_projection():
    rng=np.random.default_rng(10431)
    for _ in range(16):
        n=rng.normal(size=3);n/=np.linalg.norm(n);F=rng.normal(size=3)
        velocity,reaction=solve_contact_velocity(F,n,10.)
        assert_allclose(velocity,(np.eye(3)-np.outer(n,n))@F/10,atol=2e-15)
        assert_allclose(n@velocity,0,atol=2e-15)
        assert_allclose(10*velocity-n*reaction,F,atol=2e-14)
        assert_allclose(F@velocity,10*(velocity@velocity),atol=2e-14)


def test_plane_observations_use_instantaneous_solver_velocity():
    d=generate_contact(411,2,'plane',T=12)
    P=np.eye(3)-np.outer(PLANE_NORMAL,PLANE_NORMAL)
    assert_allclose(d['velocities'],d['V']@P,atol=2e-17)
    assert_allclose(d['positions']@PLANE_NORMAL,0,atol=2e-17)
    assert_allclose(d['V'][:3],.02*np.eye(3),atol=0)
    assert np.linalg.norm(d['U']-d['velocities'])>0
    # Position records are noiseless scoring data, not a purported encoder stream.
    assert 'not encoder' in d['diagnostics']['sensing']


def test_sphere_normals_come_from_position_and_halved_step_converges():
    errors=[]
    for substeps in [1,2,4]:
        d=generate_contact(412,0,'sphere_small',T=120,dt=.2,substeps=substeps,sigma=0)
        assert_allclose(np.linalg.norm(d['path'],axis=1),.05,atol=2e-17)
        assert_allclose(d['normals'],d['positions']/.05,atol=4e-16)
        assert_allclose(np.einsum('tij,tj->ti',d['tangent_projectors'],d['V']),d['velocities'],atol=2e-17)
        errors.append(d['terminal_position'])
    assert np.linalg.norm(errors[1]-errors[2])<np.linalg.norm(errors[0]-errors[1])/8
    check=integration_check(412,0,'sphere_small',T=120)
    assert check['same_observation_schedule']
    assert check['maximum_position_difference']<1e-9
    assert check['maximum_velocity_difference']<1e-9


def test_anisotropic_stress_satisfies_mechanics_but_not_scalar_model():
    d=generate_contact(413,0,'anisotropic',T=24)
    assert d['diagnostics']['maximum_force_balance_residual']<1e-14
    assert d['diagnostics']['maximum_constraint_residual']<1e-15
    assert d['diagnostics']['mean_projection_identity_error']>.001
    for n,v,F,lam in zip(d['normals'],d['velocities'],d['forces'],d['reactions']):
        assert_allclose(F@v,v@d['resistance']@v,atol=1e-16)


def test_halving_integration_preserves_noise_force_and_initial_orientation():
    a=generate_contact(414,3,'weak',T=24,substeps=1)
    b=generate_contact(414,3,'weak',T=24,substeps=2)
    assert_allclose(a['U']-a['velocities'],b['U']-b['velocities'],atol=4e-18)
    assert_allclose(a['V'],b['V'],atol=0)
    assert_allclose(a['normals'][0],b['normals'][0],atol=0)
    c=generate_contact(414,4,'weak',T=24)
    assert not np.allclose(a['normals'][0],c['normals'][0])
