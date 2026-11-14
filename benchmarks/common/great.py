"""GREAT subspace tracking."""
from collections import deque
import numbers

import numpy as np


def _basis(X):
    X = np.asarray(X)
    if (X.ndim != 2 or not 0 < X.shape[1] < X.shape[0]
            or not np.isfinite(X).all()):
        raise ValueError('Use a finite nontrivial column basis')
    if not np.allclose(X.conj().T @ X, np.eye(X.shape[1]), atol=2e-10, rtol=0):
        raise ValueError('Basis columns must be orthonormal')
    return X


def grassmann_exp(X, tangent):
    X = _basis(X)
    tangent = np.asarray(tangent)
    if tangent.shape != X.shape or not np.isfinite(tangent).all():
        raise ValueError('Tangent must be finite and have the basis shape')
    scale = max(1., float(np.linalg.norm(tangent)))
    if np.linalg.norm(X.conj().T @ tangent) > 2e-9 * scale:
        raise ValueError('Use a horizontal Grassmann tangent')
    tangent = tangent - X @ (X.conj().T @ tangent)
    U, singular, Vh = np.linalg.svd(tangent, full_matrices=False)
    V = Vh.conj().T
    Y = ((X @ V) * np.cos(singular)) @ Vh + (U * np.sin(singular)) @ Vh
    if np.linalg.norm(Y.conj().T @ Y - np.eye(X.shape[1])) > 5e-12:
        Y = np.linalg.qr(Y, mode='reduced')[0]
    return Y


class GREAT:
    def __init__(self, n, k, stepsize=None, inner_steps=5, window=None,
                 beta=None, complex_data=False, initial_basis=None, record_objective=False):
        if (not isinstance(n, numbers.Integral) or not isinstance(k, numbers.Integral)
                or not 0 < k < n):
            raise ValueError('Require integer 0 < k < n')
        if stepsize is not None and (not np.isfinite(stepsize) or stepsize <= 0):
            raise ValueError('Use a finite positive gradient step size or None')
        if not isinstance(inner_steps, numbers.Integral) or inner_steps < 1:
            raise ValueError('Use a positive finite number of gradient steps')
        if window is None and beta is None:
            window = 64
        if (window is None) == (beta is None):
            raise ValueError('Specify exactly one of sliding window or EW beta')
        if window is not None and (not isinstance(window, numbers.Integral) or window < 1):
            raise ValueError('Use a positive integer window')
        if beta is not None and (not np.isfinite(beta) or not 0 < beta <= 1):
            raise ValueError('Require 0 < beta <= 1')
        self.n, self.k = int(n), int(k)
        self.stepsize = None if stepsize is None else float(stepsize)
        self.inner_steps = int(inner_steps)
        self.last_stepsize = None
        self.window = None if window is None else int(window)
        self.beta = None if beta is None else float(beta)
        self.gamma = None if beta is None else float(np.sqrt(beta))
        self.complex_data = bool(complex_data)
        self.record_objective = bool(record_objective)
        dtype = complex if self.complex_data else float
        X = np.eye(n, dtype=dtype)[:, :k] if initial_basis is None else np.asarray(initial_basis)
        if not self.complex_data and np.iscomplexobj(X):
            if np.max(abs(X.imag)) > 1e-13:
                raise ValueError('Complex initialization requires complex_data=True')
            X = X.real
        X = np.array(X, dtype=dtype, copy=True)
        if X.shape != (n, k):
            raise ValueError('Initial basis has the wrong shape')
        self.basis = _basis(X).copy()
        self.covariance = np.zeros((n, n), dtype=dtype)
        self.effective_mass = 0.
        self.observations = 0
        self.total_gradient_steps = 0
        self.descent_violations = 0
        self.step_losses = []
        self._history = deque()

    def update(self, u, G=None):
        u = np.asarray(u)
        if u.shape != (self.n,) or not np.isfinite(u).all():
            raise ValueError('Use one finite measured vector of length n')
        if not self.complex_data and np.iscomplexobj(u):
            if np.max(abs(u.imag)) > 1e-13:
                raise ValueError('Complex observations require complex_data=True')
            u = u.real
        dtype = complex if self.complex_data else float
        u = np.asarray(u, dtype=dtype)
        if G is not None:
            G = np.asarray(G)
            if (G.shape != (self.n, self.n) or not np.isfinite(G).all()
                    or not np.allclose(G.conj().T @ G, np.eye(self.n), atol=2e-10, rtol=0)):
                raise ValueError('G must be a supplied orthogonal/unitary coordinate change')
            if not self.complex_data and np.iscomplexobj(G):
                if np.max(abs(G.imag)) > 1e-13:
                    raise ValueError('Complex coordinate changes require complex_data=True')
                G = G.real

            self.basis = G @ self.basis
            self.covariance = G @ self.covariance @ G.conj().T
            if self.window is not None:
                self._history = deque(G @ old for old in self._history)
        if self.beta is not None:
            self.covariance *= self.beta
            self.effective_mass = self.beta * self.effective_mass + 1
        else:
            self._history.append(u.copy())
            if len(self._history) > self.window:
                old = self._history.popleft()
                self.covariance -= np.outer(old, old.conj())
            self.effective_mass = float(len(self._history))
        self.covariance += np.outer(u, u.conj())
        self.covariance = (self.covariance + self.covariance.conj().T) / 2
        self.observations += 1
        self.step_losses = []
        self.last_stepsize = None
        count = self.observations if self.window is None else len(self._history)
        if count < self.k:
            return self.basis.copy()
        stepsize = self.stepsize
        if stepsize is None:
            if self.window is not None:
                W = np.asarray(self._history).T
                self.covariance = W @ W.conj().T
            largest = float(np.linalg.eigvalsh(self.covariance)[-1])
            stepsize = 1. / max(4. * largest, 1e-8)
        self.last_stepsize = stepsize
        CX = self.covariance @ self.basis
        trace = np.trace(self.covariance).real
        self.step_losses = ([float(trace - np.vdot(self.basis, CX).real)]
                            if self.record_objective else [])
        for step in range(self.inner_steps):
            gradient = -2 * (CX - self.basis @ (self.basis.conj().T @ CX))
            self.basis = grassmann_exp(self.basis, -stepsize * gradient)
            if step < self.inner_steps - 1 or self.record_objective:
                CX = self.covariance @ self.basis
            if self.record_objective:
                loss = float(trace - np.vdot(self.basis, CX).real)
                if loss > self.step_losses[-1] + 1e-10 * max(1., abs(self.step_losses[-1])):
                    self.descent_violations += 1
                self.step_losses.append(loss)
            self.total_gradient_steps += 1
        return self.basis.copy()

    @property
    def projector(self):
        return self.basis @ self.basis.conj().T

    def diagnostics(self):
        return dict(observations=self.observations, total_gradient_steps=self.total_gradient_steps,
                    descent_violations=self.descent_violations if self.record_objective else None,
                    descent_monitored=self.record_objective, effective_mass=self.effective_mass,
                    window=self.window, beta=self.beta, gamma=self.gamma,
                    stepsize=self.stepsize, last_stepsize=self.last_stepsize,
                    complex_extension=self.complex_data, covariance_normalized=False,
                    method='GREAT, complex extension' if self.complex_data else 'GREAT',
                    last_step_losses=list(self.step_losses))
