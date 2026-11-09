"""Shared design for the isolated native-complex passive array study.

Component-specific numerical budgets are declared before their paper runs.
Only observations enter estimation; steering truth and latent amplitudes are
simulation/scoring records. Angles are radians inside every numerical API.
"""
from pathlib import Path
import numpy as np

BASE_COMMIT='3b47f8d7765d125fff6a1bbe8c8e9d43a8627517'
N=8
Q=1.
NOISES=(1.,.2)
SNAPSHOTS=(1,2,4,8,16,32,64)
DESIRED_ANGLE=np.deg2rad(-20.)
INTERFERER_ANGLE=np.deg2rad(20.)
MOTION_AMPLITUDE=np.deg2rad(12.)
MOTION_PERIOD=128


def profile(name):
    if name not in ('quick','paper'):
        raise ValueError('Profile must be quick or paper')
    return dict(name=name,n=N,k=1,q=Q,noises=list(NOISES),snapshots=list(SNAPSHOTS),
        validation_trials=8 if name=='quick' else 128,
        test_trials=16 if name=='quick' else 256,
        desired_angle=DESIRED_ANGLE,interferer_angle=INTERFERER_ANGLE,
        motion_amplitude=MOTION_AMPLITUDE,motion_period=MOTION_PERIOD,
        base_commit=BASE_COMMIT,
        model='u=h*z+e; z~CN(0,q), e~CN(0,sI); latent z and h unavailable to estimator',
        prior='Uniform on all native complex lines; not restricted to the ULA angle curve')


def output_directory(root,name,component):
    profile(name)
    return Path(root)/'results/beamforming_unified'/name/component


def physical_angles(T,moving=False):
    if T<1:raise ValueError('Positive trajectory length required')
    return (INTERFERER_ANGLE+MOTION_AMPLITUDE*np.sin(2*np.pi*np.arange(T)/MOTION_PERIOD)
            if moving else np.full(T,INTERFERER_ANGLE))


def proper_complex(rng,shape,variance=1.):
    if not np.isfinite(variance) or variance<0:raise ValueError('Nonnegative variance required')
    return np.sqrt(variance/2)*(rng.normal(size=shape)+1j*rng.normal(size=shape))


def haar_line(rng,n=N):
    h=proper_complex(rng,n)
    return h/np.linalg.norm(h)


def ula(n,theta):
    """Unit half-wavelength ULA signature, theta in radians."""
    return np.exp(1j*np.pi*np.arange(n)*np.sin(theta))/np.sqrt(n)
