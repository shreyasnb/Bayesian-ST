"""GREAT finite-step Grassmann tracking and an explicit complex extension.

The objective is tr(C) - tr(X* C X), with an unnormalized window or discounted
covariance C. At each new observation the previous estimate warm-starts a fixed
number of Riemannian-gradient/exponential-map steps. There is no eigensolve.

The real sliding-window construction and EW column weighting are specified in
the supplied final simulation guide, which cites arXiv:2412.09052v2 Algorithm 1,
Sections IV-B/C and Remark 9. Primary-source accessibility and code provenance
are recorded in docs/great_reference.md; direct paper/author-code matching is
not silently claimed when those services cannot be reached.
"""
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


def _covariance(C, n):
    C = np.asarray(C)
    if C.shape != (n, n) or not np.isfinite(C).all():
        raise ValueError('Covariance must be a finite square matrix')
    if not np.allclose(C, C.conj().T, atol=2e-10, rtol=1e-12):
        raise ValueError('Covariance must be Hermitian')
    return C


def residual_objective(X, covariance):
    """Squared projection residual for the supplied unnormalized covariance."""
    X = _basis(X)
    C = _covariance(covariance, len(X))
    return float((np.trace(C) - np.trace(X.conj().T @ C @ X)).real)


def riemannian_gradient(X, covariance):
    """Grassmann gradient -2(I-XX*) C X in the horizontal representation."""
    X = _basis(X)
    C = _covariance(covariance, len(X))
    CX = C @ X
    return -2 * (CX - X @ (X.conj().T @ CX))


def grassmann_exp(X, tangent):
    """Exact Grassmann exponential for a horizontal tangent matrix.

    SVD is of an n-by-k tangent, not of the n-by-n data covariance. The formula
    includes zero singular directions and is valid for rank-deficient tangents.
    Numerical horizontal projection removes roundoff in X* tangent.
    """
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
    # This repair is normally inactive; it does not replace the exponential
    # update by a generic retraction or optimize the covariance eigenvectors.
    if np.linalg.norm(Y.conj().T @ Y - np.eye(X.shape[1])) > 5e-12:
        Y = np.linalg.qr(Y, mode='reduced')[0]
    return Y


class GREAT:
    """Warm-started finite-gradient subspace tracker.

    Select a positive sliding window or beta in (0, 1]. For EW covariance
    weights beta**age, Remark 9's data-column weight is gamma=sqrt(beta).
    stepsize multiplies the raw, unnormalized gradient; no hidden division by
    the window size or effective mass is performed. Step size, memory and
    number of inner steps must be selected on independent validation episodes.

    update receives measured u only, plus optional supplied coordinate G.
    In contact this is a response-only reference and does not receive v.
    """

    def __init__(self, n, k, stepsize=1e-3, inner_steps=1, window=None,
                 beta=None, complex_data=False, initial_basis=None, record_objective=False):
        if (not isinstance(n, numbers.Integral) or not isinstance(k, numbers.Integral)
                or not 0 < k < n):
            raise ValueError('Require integer 0 < k < n')
        if not np.isfinite(stepsize) or stepsize <= 0:
            raise ValueError('Use a finite positive gradient step size')
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
        self.stepsize, self.inner_steps = float(stepsize), int(inner_steps)
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
        """Assimilate one measurement and take the declared finite steps."""
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
            # Estimator and historical optimization dataset share the frame.
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
        CX = self.covariance @ self.basis
        trace = np.trace(self.covariance).real
        self.step_losses = ([float(trace - np.vdot(self.basis, CX).real)]
                            if self.record_objective else [])
        for step in range(self.inner_steps):
            gradient = -2 * (CX - self.basis @ (self.basis.conj().T @ CX))
            self.basis = grassmann_exp(self.basis, -self.stepsize * gradient)
            if step < self.inner_steps - 1 or self.record_objective:
                CX = self.covariance @ self.basis
            if self.record_objective:
                loss = float(trace - np.vdot(self.basis, CX).real)
                if loss > self.step_losses[-1] + 1e-10 * max(1., abs(self.step_losses[-1])):
                    self.descent_violations += 1
                self.step_losses.append(loss)
            self.total_gradient_steps += 1
        self.observations += 1
        return self.basis.copy()

    @property
    def projector(self):
        return self.basis @ self.basis.conj().T

    def diagnostics(self):
        return dict(observations=self.observations, total_gradient_steps=self.total_gradient_steps,
                    descent_violations=self.descent_violations if self.record_objective else None,
                    descent_monitored=self.record_objective, effective_mass=self.effective_mass,
                    window=self.window, beta=self.beta, gamma=self.gamma,
                    complex_extension=self.complex_data, covariance_normalized=False,
                    method='GREAT, complex extension' if self.complex_data else 'GREAT',
                    last_step_losses=list(self.step_losses))
