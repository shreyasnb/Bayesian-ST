"""Sample-covariance beamforming (Van Trees, 2002)."""
from __future__ import annotations

import numpy as np


def mvdr(covariance, steering):
    covariance = np.asarray(covariance)
    steering = np.asarray(steering)
    if covariance.shape != (steering.size, steering.size) or steering.ndim != 1:
        raise ValueError("Expected an n by n covariance and n-vector steering")
    if not np.allclose(covariance, covariance.conj().T, atol=1e-10):
        raise ValueError("Covariance must be Hermitian")
    if np.linalg.eigvalsh(covariance)[0] <= 0:
        raise ValueError("Covariance must be positive definite")
    solution = np.linalg.solve(covariance, steering)
    denominator = np.vdot(steering, solution).real
    if denominator <= 0:
        raise ValueError("Steering vector must be nonzero")
    return solution / denominator


def output_power(weight, covariance):
    return float(np.vdot(weight, covariance @ weight).real)


def sample_output_power(weight, snapshots):
    snapshots = np.asarray(snapshots)
    if snapshots.ndim != 2 or snapshots.shape[0] == 0:
        raise ValueError("Expected nonempty row-wise snapshots")
    return float(np.mean(np.abs(snapshots @ np.asarray(weight).conj()) ** 2))
