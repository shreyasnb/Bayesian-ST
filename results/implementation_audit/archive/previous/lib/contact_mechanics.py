"""Idealized bilateral guide mechanics for the robotic contact pilot.

The plant solves force balance and the velocity constraint independently of
the projector observation model. Geometry is used by the plant and scoring;
the estimator receives only the noisy instantaneous velocity and applied force.
There is no friction cone, separation, impact, inertia, or position sensor model.
"""
from __future__ import annotations

import numpy as np


PLANE_NORMAL=np.array([1.,2.,3.])/np.sqrt(14.)
CONDITIONS={
    'plane':dict(name='Plane',radius=None,weak=False,anisotropic=False),
    'sphere_large':dict(name='Sphere, R=0.20 m',radius=.20,weak=False,anisotropic=False),
    'sphere_small':dict(name='Sphere, R=0.05 m',radius=.05,weak=False,anisotropic=False),
    'weak':dict(name='Weak excitation',radius=.05,weak=True,anisotropic=False),
    'anisotropic':dict(name='Anisotropic resistance',radius=.05,weak=False,anisotropic=True),
}


def condition_spec(condition):
    """Return a copied, explicit plant specification, allowing named aliases."""
    if isinstance(condition,str):
        aliases={'large':'sphere_large','small':'sphere_small','sphere_020':'sphere_large',
                 'sphere_005':'sphere_small','sphere_005_weak':'weak',
                 'resistance_mismatch':'anisotropic'}
        key=aliases.get(condition,condition)
        if key not in CONDITIONS:raise ValueError(f'Unknown contact condition: {condition}')
        return dict(CONDITIONS[key],key=key)
    if not isinstance(condition,dict):raise ValueError('Condition must be a name or dictionary')
    spec=dict(radius=None,weak=False,anisotropic=False,name='Custom guide',key='custom')
    spec.update(condition)
    if spec['radius'] is not None and (not np.isfinite(spec['radius']) or spec['radius']<=0):
        raise ValueError('Sphere radius must be positive or None for a plane')
    return spec


def solve_contact_velocity(force,normal,resistance):
    """Solve the 4x4 KKT system for velocity and signed normal reaction.

    The reaction multiplier may have either sign because contact is bilateral.
    ``resistance`` is a positive scalar or a symmetric positive-definite matrix.
    """
    force=np.asarray(force,dtype=float);normal=np.asarray(normal,dtype=float)
    if force.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(force).all() or not np.isfinite(normal).all():
        raise ValueError('Force and unit normal must be finite three-vectors')
    if not np.isclose(normal@normal,1.,atol=1e-10,rtol=0):
        raise ValueError('The surface normal must have unit length')
    D=np.asarray(resistance,dtype=float)
    if D.ndim==0:D=float(D)*np.eye(3)
    if (D.shape!=(3,3) or not np.isfinite(D).all() or
            not np.allclose(D,D.T,atol=1e-12,rtol=0) or np.linalg.eigvalsh(D)[0]<=0):
        raise ValueError('Resistance must be symmetric positive definite')
    K=np.zeros((4,4));K[:3,:3]=D;K[:3,3]=-normal;K[3,:3]=normal
    answer=np.linalg.solve(K,np.r_[force,0.])
    return answer[:3],float(answer[3])


def force_program(tau,v0=.02,weak=False):
    """Preassigned bounded world-frame free-velocity program in m/s."""
    value=v0*np.array([np.cos(.3*tau),np.sin(.3*tau),.5*np.sin(.21*tau+.4)])
    if weak:value[1:]*=.2
    return value


def _normal(position,spec):
    return PLANE_NORMAL.copy() if spec['radius'] is None else position/np.linalg.norm(position)


def _project(position,spec):
    if spec['radius'] is None:
        return position-PLANE_NORMAL*(PLANE_NORMAL@position)
    norm=np.linalg.norm(position)
    if norm<=0:raise ArithmeticError('Sphere integration reached the centre')
    return float(spec['radius'])*position/norm


def _tangent_basis(normal):
    # Geometry for scoring only, independent of all inference implementations.
    axis=np.eye(3)[:,int(np.argmin(abs(normal)))]
    first=np.cross(normal,axis);first/=np.linalg.norm(first)
    return np.column_stack([first,np.cross(normal,first)])


def _step(position,tau,h,D,d,v0,spec,plane_input=None):
    def velocity(p,time):
        free=plane_input if plane_input is not None else force_program(time,v0,spec['weak'])
        return solve_contact_velocity(d*free,_normal(p,spec),D)[0]
    a=velocity(position,tau)
    b=velocity(position+h*a/2,tau+h/2)
    c=velocity(position+h*b/2,tau+h/2)
    e=velocity(position+h*c,tau+h)
    updated=_project(position+h*(a+2*b+2*c+e)/6,spec)
    travel=h*(np.linalg.norm(a)+2*np.linalg.norm(b)+2*np.linalg.norm(c)+np.linalg.norm(e))/6
    return updated,float(travel)


def generate_contact(seed,trial,condition,T=600,dt=.05,d=10.,v0=.02,
                     sigma=.001,substeps=1):
    """Generate constrained motion and instantaneous observations.

    Observations are at times ``arange(T)*dt``; integration covers ``[0,T*dt]``.
    Doubling ``substeps`` halves only the plant integration step, preserving
    observation times, initial orientation, force program and measurement noise.
    The same seed/trial across conditions gives paired interventions on initial
    orientation and noise; independent trial IDs remain the statistical units.

    The physical force is d*v even under anisotropic resistance. Its measured
    free velocity remains F/d, deliberately creating observation-model mismatch.
    """
    spec=condition_spec(condition)
    if (not isinstance(T,(int,np.integer)) or T<1 or not isinstance(substeps,(int,np.integer)) or substeps<1
            or not all(np.isfinite(x) and x>0 for x in (dt,d,v0))
            or not np.isfinite(sigma) or sigma<0):
        raise ValueError('Require positive T, dt, d, v0, substeps and nonnegative sigma')
    rng_initial=np.random.default_rng(np.random.SeedSequence([int(seed),int(trial),9021]))
    initial=rng_initial.normal(size=3);initial/=np.linalg.norm(initial)
    position=np.zeros(3) if spec['radius'] is None else float(spec['radius'])*initial
    D=d*np.diag([1.,1.5,2.]) if spec['anisotropic'] else d*np.eye(3)
    times=np.arange(T)*dt;positions=[];normals=[];bases=[];free=[];velocities=[];reactions=[]
    path=[position.copy()];travel=0.;balance_errors=[];constraint_errors=[];model_errors=[]
    for j,tau in enumerate(times):
        n=_normal(position,spec)
        v=v0*np.eye(3)[:,j%3] if spec['radius'] is None else force_program(tau,v0,spec['weak'])
        speed,reaction=solve_contact_velocity(d*v,n,D)
        positions.append(position.copy());normals.append(n);bases.append(_tangent_basis(n))
        free.append(v);velocities.append(speed);reactions.append(reaction)
        constraint_errors.append(abs(n@speed))
        balance_errors.append(np.linalg.norm(D@speed-n*reaction-d*v))
        model_errors.append(np.linalg.norm(speed-(np.eye(3)-np.outer(n,n))@v))
        for sub in range(substeps):
            position,length=_step(position,tau+sub*dt/substeps,dt/substeps,D,d,v0,spec,
                                  v if spec['radius'] is None else None)
            travel+=length
        path.append(position.copy())
    positions=np.asarray(positions);normals=np.asarray(normals);X=np.asarray(bases)
    free=np.asarray(free);velocities=np.asarray(velocities);path=np.asarray(path)
    rng_noise=np.random.default_rng(np.random.SeedSequence([int(seed),int(trial),9022]))
    noise=sigma*rng_noise.normal(size=(T,3));observations=velocities+noise
    P=np.eye(3)-np.einsum('ti,tj->tij',normals,normals)
    all_normals=np.vstack([normals,_normal(position,spec)])
    rotation=np.arccos(np.clip(np.sum(all_normals[:-1]*all_normals[1:],axis=1),-1.,1.))
    surface_error=abs(path@PLANE_NORMAL) if spec['radius'] is None else abs(np.linalg.norm(path,axis=1)-spec['radius'])
    excitation=np.linalg.eigvalsh(free.T@free)
    diagnostics=dict(surface_travel=float(travel),normal_rotation=float(rotation.sum()),
        maximum_normal_step=float(rotation.max(initial=0.)),
        net_tangent_rotation=float(np.arccos(np.clip(abs(normals[0]@normals[-1]),0.,1.))),
        maximum_surface_error=float(surface_error.max(initial=0.)),
        maximum_constraint_residual=float(max(constraint_errors)),
        maximum_force_balance_residual=float(max(balance_errors)),
        maximum_projection_identity_error=float(max(model_errors)),
        mean_projection_identity_error=float(np.mean(model_errors)),
        input_excitation_eigenvalues=excitation.tolist(),
        minimum_input_excitation=float(excitation[0]),maximum_speed=float(np.linalg.norm(velocities,axis=1).max()),
        reaction_min=float(min(reactions)),reaction_max=float(max(reactions)),
        physical_interval=float(T*dt),last_observation_time=float(times[-1]),
        integration_step=float(dt/substeps),observation_dt=float(dt),
        condition=spec['name'],model_matched=not spec['anisotropic'],
        sensing='Only instantaneous noisy velocity and known applied force; positions are plant/scoring records, not encoder measurements')
    return dict(positions=positions,normals=normals,tangent_projectors=P,X=X,
        free_velocities=free,V=free,forces=d*free,velocities=velocities,reactions=np.asarray(reactions),
        observations=observations,U=observations,times=times,path=path,
        terminal_position=position.copy(),resistance=D,diagnostics=diagnostics,condition=spec)


def integration_check(seed,trial,condition,**kwargs):
    """Compare dt and dt/2 plant steps at exactly the same observation times."""
    coarse=generate_contact(seed,trial,condition,substeps=1,**kwargs)
    fine=generate_contact(seed,trial,condition,substeps=2,**kwargs)
    assert np.array_equal(coarse['times'],fine['times'])
    return dict(maximum_position_difference=float(np.max(np.linalg.norm(coarse['positions']-fine['positions'],axis=1))),
                terminal_position_difference=float(np.linalg.norm(coarse['terminal_position']-fine['terminal_position'])),
                maximum_velocity_difference=float(np.max(np.linalg.norm(coarse['velocities']-fine['velocities'],axis=1))),
                maximum_projector_difference=float(np.max(np.linalg.norm(coarse['tangent_projectors']-fine['tangent_projectors'],axis=(1,2)))),
                same_observation_schedule=True,coarse=coarse['diagnostics'],fine=fine['diagnostics'])
