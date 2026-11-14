from __future__ import annotations
import numpy as np


PLANE_NORMAL=np.array([1.,2.,3.])/np.sqrt(14.)


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
