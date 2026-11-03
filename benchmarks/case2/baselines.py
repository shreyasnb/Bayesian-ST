"""Full-observation GROUSE and covariance PCA references.

GROUSE follows the rank-one Grassmann geodesic update of Balzano et al. (2010)
for the fully observed least-squares loss. It is not GREAT or GeRoST.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def _basis(covariance, k):
    return np.linalg.eigh((covariance + covariance.conj().T) / 2)[1][:, -k:]


def _settings(n, k):
    if not 0 < k < n:
        raise ValueError("Require 0 < k < n")


def _motion(motion, n):
    motion = np.asarray(motion)
    if motion.shape != (n, n) or not np.allclose(motion.conj().T @ motion, np.eye(n)):
        raise ValueError("Motion must be unitary")
    return motion


def reconstruct(observation, basis, s, tau2, Phi=None):
    """Wiener reconstruction using a point subspace and fixed known variances."""
    if s <= 0 or tau2 < 0:
        raise ValueError("Require s > 0 and tau2 >= 0")
    observation, basis = np.asarray(observation), np.asarray(basis)
    estimate = tau2 / (s + tau2) * (basis @ (basis.conj().T @ observation))
    return estimate if Phi is None else np.asarray(Phi) @ estimate


class EWCovariancePCA:
    """Leading eigenspace of an exponentially weighted zero-mean covariance."""
    def __init__(self, n, k, beta=0.98):
        _settings(n, k)
        if not 0 < beta <= 1:
            raise ValueError("Require 0 < beta <= 1")
        self.n, self.k, self.beta = int(n), int(k), float(beta)
        self.scatter = np.zeros((n, n))
        self.mass = 0.0
        self.basis = np.eye(n, k)

    def update(self, observation):
        observation = np.asarray(observation)
        if observation.shape != (self.n,) or not np.isfinite(observation).all():
            raise ValueError("Expected a finite n-vector")
        self.scatter = self.beta * self.scatter + np.outer(observation, observation.conj())
        self.mass = self.beta * self.mass + 1
        self.basis = _basis(self.scatter, self.k)
        return self.basis.copy()

    @property
    def covariance(self):
        return self.scatter / self.mass if self.mass else self.scatter.copy()

    def transform(self, motion):
        motion = _motion(motion, self.n)
        self.scatter = motion @ self.scatter @ motion.conj().T
        self.basis = motion @ self.basis


class RollingPCA:
    """Leading eigenspace of the last ``window`` complete observations."""
    def __init__(self, n, k, window=32):
        _settings(n, k)
        if window < k:
            raise ValueError("Window must be at least k")
        self.n, self.k, self.window = int(n), int(k), int(window)
        self.observations = deque(maxlen=self.window)
        self.basis = np.eye(n, k)

    def update(self, observation):
        observation = np.asarray(observation)
        if observation.shape != (self.n,) or not np.isfinite(observation).all():
            raise ValueError("Expected a finite n-vector")
        self.observations.append(observation.copy())
        samples = np.stack(self.observations, axis=1)
        self.basis = _basis(samples @ samples.conj().T, self.k)
        return self.basis.copy()

    def transform(self, motion):
        motion = _motion(motion, self.n)
        self.observations = deque((motion @ u for u in self.observations), maxlen=self.window)
        self.basis = motion @ self.basis


class GROUSE:
    """Rank-one Grassmann geodesic descent with complete observations.

    If w = X^H u, p = Xw, r = u-p, sigma = ||r|| ||w||, the update is
    X + [(cos(step*sigma)-1)p/||p|| + sin(step*sigma)r/||r||] w^H/||w||.
    This is the exact geodesic step for the negative gradient of one-half
    squared residual loss. Step size is constant and selected on validation
    trajectories. Zero projections/residuals have zero gradient.
    """
    def __init__(self, n, k, step=0.02, rng=None):
        _settings(n, k)
        if step <= 0:
            raise ValueError("Step must be positive")
        self.n, self.k, self.step = int(n), int(k), float(step)
        rng = np.random.default_rng(rng)
        self.basis = np.linalg.qr(rng.normal(size=(n, k)))[0]

    def update(self, observation):
        observation = np.asarray(observation)
        if observation.shape != (self.n,) or not np.isfinite(observation).all():
            raise ValueError("Expected a finite n-vector")
        coefficients = self.basis.conj().T @ observation
        projection = self.basis @ coefficients
        residual = observation - projection
        norm_w, norm_r = np.linalg.norm(coefficients), np.linalg.norm(residual)
        if norm_w > 1e-14 and norm_r > 1e-14:
            angle = self.step * norm_w * norm_r
            direction = ((np.cos(angle) - 1) * projection / norm_w
                         + np.sin(angle) * residual / norm_r)
            self.basis = self.basis + np.outer(direction, coefficients.conj() / norm_w)
            # Removes only floating-point drift in the exact geodesic formula.
            self.basis = np.linalg.qr(self.basis)[0]
        return self.basis.copy()

    def transform(self, motion):
        self.basis = _motion(motion, self.n) @ self.basis
