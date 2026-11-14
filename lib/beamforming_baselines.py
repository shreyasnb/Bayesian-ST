import numpy as np
from benchmarks.case1.baselines import mvdr
def distortionless_weight(R,a):
    """MVDR with a final complex constraint normalization for roundoff.

    The Hermitian quadratic denominator is real in exact arithmetic. A nearly
    singular loaded SCM can leave a small imaginary solve error; restoring
    w^H a=1 makes every compared action satisfy the same constraint.
    """
    w=mvdr(R,a);gain=np.vdot(w,a)
    if abs(gain)==0:raise ArithmeticError('Zero distortionless gain')
    return w/np.conj(gain)

def covariance_shrinkage(S,alpha):
    """Continuous relative-loading family, including exact infinite loading."""
    S=np.asarray(S,complex);n=len(S)
    if not 0<=alpha<=1:raise ValueError('alpha must be in[0,1]')
    mu=float(np.trace(S).real/n)
    return (1-alpha)*S+alpha*mu*np.eye(n)
