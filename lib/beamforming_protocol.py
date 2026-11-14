import numpy as np
N=32

def proper_complex(rng,shape,variance=1.):
    if not np.isfinite(variance) or variance<0:raise ValueError('Nonnegative variance required')
    return np.sqrt(variance/2)*(rng.normal(size=shape)+1j*rng.normal(size=shape))


def haar_line(rng,n=N):
    h=proper_complex(rng,n)
    return h/np.linalg.norm(h)


def ula(n,theta):
    """Unit half-wavelength ULA signature, theta in radians."""
    return np.exp(1j*np.pi*np.arange(n)*np.sin(theta))/np.sqrt(n)
