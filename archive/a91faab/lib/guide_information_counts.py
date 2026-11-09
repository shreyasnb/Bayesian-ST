"""Measurement-score aggregation for an actual observation-count experiment."""
import numpy as np
from .beamforming_theory import passive_scores


def score_prefixes(U, X, s, tau2, counts, complex_data=False, complement=None):
    """Return summed scores for each dataset's first L independent snapshots.

    U has shape (independent datasets, observations per dataset, N). The count
    axis is never a count of posterior draws, tangent coordinates or repeated
    Monte Carlo datasets. Prefixes use common random numbers across L.
    """
    U = np.asarray(U)
    counts = tuple(counts)
    if U.ndim != 3 or U.shape[-1] != len(X) or not np.isfinite(U).all():
        raise ValueError('U must contain finite row-wise independent observation streams')
    if not counts or any(not isinstance(L, (int, np.integer)) or not 1 <= L <= U.shape[1] for L in counts):
        raise ValueError('Observation counts must be positive integers within the stream')
    if len(set(counts)) != len(counts):
        raise ValueError('Each requested count must be unique')
    values = passive_scores(U.reshape(-1,U.shape[-1]),X,s,tau2,complex_data,complement)
    values = values.reshape(U.shape[0],U.shape[1],-1).cumsum(axis=1)
    return {L:values[:,L-1].copy() for L in counts}


def gaussian_stream(episodes, observations, n, k, s, tau2, rng, complex_data=False):
    """Draw passive snapshots at the fixed canonical subspace.

    Hidden Gaussian coefficients are used only to simulate measurements; they
    are never substituted as the recorded inputs of a conditional likelihood.
    """
    def normal(shape):
        value=rng.normal(size=shape)
        return (value+1j*rng.normal(size=shape))/np.sqrt(2) if complex_data else value
    X=np.eye(n)[:,:k]
    latent=np.sqrt(tau2)*normal((episodes,observations,k))
    noise=np.sqrt(s)*normal((episodes,observations,n))
    return latent@X.T+noise
