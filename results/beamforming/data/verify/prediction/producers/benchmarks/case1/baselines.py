"""Sample-covariance beamforming references (Van Trees, 2002).

The oracle is obtained by calling ``mvdr`` with the generating covariance. It
must not participate in hyperparameter selection.
"""
from __future__ import annotations

import numpy as np


def _hermitian(matrix):
    return (matrix + matrix.conj().T) / 2


def mvdr(covariance, steering):
    """Return the distortionless minimum-variance weight for an HPD covariance."""
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
    """Population interference-plus-noise output power."""
    return float(np.vdot(weight, covariance @ weight).real)


def sample_output_power(weight, snapshots):
    """Mean output power on independent row-wise snapshots (m by n)."""
    snapshots = np.asarray(snapshots)
    if snapshots.ndim != 2 or snapshots.shape[0] == 0:
        raise ValueError("Expected nonempty row-wise snapshots")
    return float(np.mean(np.abs(snapshots @ np.asarray(weight).conj()) ** 2))


class LoadedCovariance:
    """Bias-corrected EW sample covariance with relative diagonal loading.

    After t updates, covariance is sum(beta**(t-j) u_j u_j^H) divided by
    sum(beta**(t-j)), plus loading * trace(raw)/n * I. ``floor`` supplies a
    strictly positive loading scale for zero or degenerate data. There is no
    centering: the observation model has known zero mean.
    """
    def __init__(self, n, beta=0.98, loading=0.1, floor=1e-10):
        if n < 2 or not 0 < beta <= 1 or loading <= 0 or floor <= 0:
            raise ValueError("Invalid covariance estimator settings")
        self.n, self.beta = int(n), float(beta)
        self.loading, self.floor = float(loading), float(floor)
        self.scatter = np.zeros((n, n), dtype=complex)
        self.mass = 0.0

    def update(self, observation):
        observation = np.asarray(observation)
        if observation.shape != (self.n,) or not np.isfinite(observation).all():
            raise ValueError("Expected a finite n-vector")
        self.scatter = self.beta * self.scatter + np.outer(observation, observation.conj())
        self.mass = self.beta * self.mass + 1
        return self.covariance

    @property
    def raw_covariance(self):
        return _hermitian(self.scatter / self.mass) if self.mass else self.scatter.copy()

    @property
    def covariance(self):
        raw = self.raw_covariance
        scale = max(float(np.trace(raw).real / self.n), self.floor)
        return raw + self.loading * scale * np.eye(self.n)

    def transform(self, motion):
        """Transport retained data through a supplied unitary frame motion."""
        motion = np.asarray(motion)
        if motion.shape != (self.n, self.n) or not np.allclose(motion.conj().T @ motion, np.eye(self.n)):
            raise ValueError("Motion must be unitary")
        self.scatter = _hermitian(motion @ self.scatter @ motion.conj().T)
