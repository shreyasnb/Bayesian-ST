"""Normalized observation likelihoods, including complex Gaussian constants."""
import numpy as np
from .geometry import hermitian


def increment(u, s, tau2=None, v=None, complex_data=False):
    if not np.isfinite(s) or s <= 0:
        raise ValueError('s must be positive')
    f = 1. if complex_data else .5
    if v is not None:
        return hermitian(f / s * (np.outer(u, v.conj()) + np.outer(v, u.conj()) - np.outer(v, v.conj())))
    if tau2 is None or not np.isfinite(tau2) or tau2 < 0:
        raise ValueError('tau2 must be nonnegative')
    return f * tau2 / (s * (s + tau2)) * np.outer(u, u.conj())


def constant(u, k, s, tau2=None, v=None, complex_data=False):
    f = 1. if complex_data else .5
    n = len(u)
    norm = n * np.log(np.pi if complex_data else 2 * np.pi)
    logdet = n * np.log(s) if v is not None else (n-k)*np.log(s) + k*np.log(s+tau2)
    return -f * (norm + logdet + np.vdot(u,u).real / s)


def log_likelihood(u, P, s, tau2=None, v=None, complex_data=False):
    B = increment(u,s,tau2,v,complex_data)
    return constant(u,round(np.trace(P).real),s,tau2,v,complex_data) + np.trace(B @ P).real
