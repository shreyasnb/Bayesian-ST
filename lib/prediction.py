"""Fixed-noise likelihood memory and causal Grassmann velocity prediction.

Identity/rolling memory and learned rotations are local approximations to
unknown dynamics. A learned rotation uses past observations and is not covered
automatically by the deterministic-transformation tracking theorem.
"""
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from .em import Record
from .geometry import center, mode
from .models import increment


def _basis_pair(Y, Z):
    Y, Z = np.asarray(Y), np.asarray(Z)
    if (Y.ndim != 2 or Y.shape != Z.shape or not 0 < Y.shape[1] < Y.shape[0]
            or not np.isfinite(Y).all() or not np.isfinite(Z).all()):
        raise ValueError('Expected finite n by k bases with 0 < k < n')
    eye = np.eye(Y.shape[1])
    if not (np.allclose(Y.conj().T @ Y, eye, atol=1e-10, rtol=0)
            and np.allclose(Z.conj().T @ Z, eye, atol=1e-10, rtol=0)):
        raise ValueError('Bases must have orthonormal columns')
    return Y, Z


def geodesic_generator(Y, Z, cut_locus_margin=1e-6):
    """Return the minimal skew-Hermitian Grassmann generator and angles.

    Away from the cut locus, ``exp(L) P_Y exp(L)^* = P_Z``. The ambient
    extension acts only on the planes spanned by the principal vectors and
    normals. Changing either basis within its subspace leaves L unchanged.
    Near-zero angles use the continuous ratio theta/sin(theta), avoiding an
    unstable construction of a unit normal from a tiny residual.
    """
    Y, Z = _basis_pair(Y, Z)
    if not np.isfinite(cut_locus_margin) or not 0 < cut_locus_margin < np.pi / 2:
        raise ValueError('cut_locus_margin must be between zero and pi/2')
    U, cosine, Vh = np.linalg.svd(Y.conj().T @ Z)
    cosine = np.clip(cosine, 0., 1.)
    Yp, Zp = Y @ U, Z @ Vh.conj().T
    residual = Zp - Yp * cosine
    residual -= Yp @ (Yp.conj().T @ residual)
    sine = np.linalg.norm(residual, axis=0)
    angles = np.arctan2(sine, cosine)
    if np.any(angles >= np.pi / 2 - cut_locus_margin):
        raise ValueError('Principal angle is at or too close to the cut locus')
    tangent = residual / np.sinc(angles / np.pi)
    tangent[:, sine <= 8 * np.finfo(float).eps] = 0
    L = tangent @ Yp.conj().T - Yp @ tangent.conj().T
    return (L - L.conj().T) / 2, angles


@dataclass(frozen=True)
class MotionForecast:
    rotation: np.ndarray
    generator: np.ndarray
    principal_angles: np.ndarray
    active: bool
    reason: str
    capped: bool
    speed: float


def geodesic_forecast(Y, Z, lag=1, damping=1., speed_cap=None,
                      cut_locus_margin=1e-6):
    """Continue a fitted short geodesic by damping/lag of its displacement.

    ``speed_cap`` bounds the largest forecast principal rotation per step,
    after damping. It is a declared numerical policy, not a fitted process law.
    The returned generator is the uncapped full Y-to-Z generator; ``rotation``
    uses the damped/capped step. Zero damping returns exact identity even for
    subspaces at the cut locus, since no velocity estimate is needed.
    """
    Y, Z = _basis_pair(Y, Z)
    if not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError('lag must be a positive integer')
    if not np.isfinite(damping) or not 0 <= damping <= 1:
        raise ValueError('damping must lie in [0,1]')
    if speed_cap is not None and (not np.isfinite(speed_cap) or speed_cap <= 0):
        raise ValueError('speed_cap must be positive or None')
    n, k = Y.shape
    if damping == 0:
        return MotionForecast(np.eye(n, dtype=np.result_type(Y, Z)),
                              np.zeros((n, n), dtype=np.result_type(Y, Z)),
                              np.zeros(k), False, 'zero_damping', False, 0.)
    L, angles = geodesic_generator(Y, Z, cut_locus_margin)
    scale = float(damping / lag)
    speed = float(scale * angles.max(initial=0.))
    capped = bool(speed_cap is not None and speed > speed_cap)
    if capped:
        scale *= float(speed_cap / speed)
        speed = float(speed_cap)
    return MotionForecast(expm(scale * L), L, angles,
                          bool(speed > 0), 'forecast' if speed > 0 else 'stationary',
                          capped, speed)


def mean_observation_age(beta):
    """Long-run mean age of normalized exponential weights (in samples)."""
    if not np.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError('beta must lie in (0,1]')
    return np.inf if beta == 1 else float(beta / (1 - beta))


class PredictiveBinghamFilter:
    """One likelihood correction after a causal fixed-noise prediction.

    ``identity`` reproduces the original exponential recursion and accepts an
    optional supplied orthogonal/unitary G. ``velocity`` estimates G from two
    past modal bases separated by ``lag`` samples. ``rolling`` retains exactly
    the last ``likelihood_window`` increments, uses beta=1, and permits neither
    supplied motion nor noise refits. Each prior is copied before correction.

    Call ``predict`` to inspect the next prior, then ``update`` to assimilate
    exactly once. Repeated predictions are idempotent. Prediction time excludes
    correction/eigendecomposition; update time excludes prediction. These are
    cumulative totals, with ``last_*_seconds`` for the most recent operation.
    """
    def __init__(self, n, k, s, tau2=None, beta=1., A0=None, window=32,
                 complex_data=False, prediction='identity', lag=8, damping=.5,
                 likelihood_window=16, speed_cap=None, cut_locus_margin=1e-6):
        if (not isinstance(n, (int, np.integer)) or not isinstance(k, (int, np.integer))
                or not 0 < k < n or not np.isfinite(s) or s <= 0
                or not np.isfinite(beta) or not 0 < beta <= 1
                or not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError('Invalid dimension, rank, variance, forgetting or storage window')
        if tau2 is not None and (not np.isfinite(tau2) or tau2 < 0):
            raise ValueError('tau2 must be nonnegative or None')
        if prediction not in ('identity', 'velocity', 'rolling'):
            raise ValueError('prediction must be identity, velocity or rolling')
        if not isinstance(lag, (int, np.integer)) or lag < 1:
            raise ValueError('lag must be a positive integer')
        if not np.isfinite(damping) or not 0 <= damping <= 1:
            raise ValueError('damping must lie in [0,1]')
        if not isinstance(likelihood_window, (int, np.integer)) or likelihood_window < 1:
            raise ValueError('likelihood_window must be a positive integer')
        if speed_cap is not None and (not np.isfinite(speed_cap) or speed_cap <= 0):
            raise ValueError('speed_cap must be positive or None')
        if not np.isfinite(cut_locus_margin) or not 0 < cut_locus_margin < np.pi / 2:
            raise ValueError('Invalid cut-locus margin')
        self.n, self.k, self.s, self.tau2 = n, k, float(s), tau2
        self._fixed_noise = (self.s, self.tau2)
        self.beta, self.complex_data, self.prediction = float(beta), bool(complex_data), prediction
        self.lag, self.damping = int(lag), float(damping)
        self.likelihood_window, self.speed_cap = int(likelihood_window), speed_cap
        self.cut_locus_margin = float(cut_locus_margin)
        dtype = complex if complex_data else float
        initial = np.zeros((n, n), dtype=dtype) if A0 is None else np.asarray(A0)
        if initial.shape != (n, n) or not np.isfinite(initial).all():
            raise ValueError('A0 must be a finite n by n parameter')
        self.A = center(initial)
        if prediction == 'rolling' and (beta != 1 or np.linalg.norm(self.A) > 1e-12):
            raise ValueError('Rolling likelihood requires beta=1 and a uniform initial prior')
        self.records = deque(maxlen=int(window))
        self.increments = deque(maxlen=self.likelihood_window)
        self.history = deque(maxlen=self.lag + 1)
        self.t = 0
        self.prior = self.A.copy()
        self.last_motion = np.eye(n, dtype=self.A.dtype)
        self.motion_diagnostics = {}
        self.prediction_seconds = self.update_seconds = 0.
        self.last_prediction_seconds = self.last_update_seconds = 0.
        self._prepared = False

    @property
    def basis(self):
        return mode(self.A, self.k)

    def predict(self, G=None):
        """Prepare A_t^- using only corrections through t-1; no u_t argument."""
        if self._prepared:
            if G is not None and not np.allclose(G, self.last_motion, atol=1e-12, rtol=0):
                raise ValueError('A different rotation was supplied after preparing the prior')
            return self.prior.copy()
        start = perf_counter()
        eye = np.eye(self.n, dtype=self.A.dtype)
        diagnostic = dict(active=False, reason='identity', capped=False, max_angle=0., speed=0.)
        if G is not None:
            G = np.asarray(G)
            if self.prediction != 'identity':
                raise ValueError('Supplied motion is only supported by the identity/supplied rule')
            if (G.shape != (self.n, self.n) or not np.isfinite(G).all()
                    or not np.allclose(G.conj().T @ G, eye, atol=1e-10, rtol=0)):
                raise ValueError('Supplied motion must be orthogonal/unitary')
            rotation = G
            diagnostic['reason'] = 'supplied'
        elif self.prediction == 'velocity' and self.damping > 0:
            rotation = eye
            diagnostic['reason'] = 'initialization'
            if len(self.history) >= self.lag + 1:
                try:
                    forecast = geodesic_forecast(self.history[0], self.history[-1],
                        self.lag, self.damping, self.speed_cap, self.cut_locus_margin)
                    rotation = forecast.rotation
                    diagnostic = dict(active=forecast.active, reason=forecast.reason,
                        capped=forecast.capped, max_angle=float(forecast.principal_angles.max(initial=0.)),
                        speed=forecast.speed)
                except ValueError as error:
                    if 'cut locus' not in str(error):
                        raise
                    diagnostic['reason'] = 'cut_locus_identity'
        else:
            rotation = eye
            if self.prediction == 'velocity':
                diagnostic['reason'] = 'zero_damping'
        if self.prediction == 'rolling':
            # Removing the oldest increment BEFORE assimilation makes this the
            # exact prior for the new length-W local-likelihood product.
            outgoing = self.increments[0] if len(self.increments) == self.likelihood_window else 0
            self.prior = center(self.A - outgoing)
            diagnostic['reason'] = 'rolling'
        else:
            self.prior = center(self.beta * rotation @ self.A @ rotation.conj().T)
        self.last_motion = rotation.copy()
        self.motion_diagnostics = diagnostic
        self.last_prediction_seconds = perf_counter() - start
        self.prediction_seconds += self.last_prediction_seconds
        self._prepared = True
        return self.prior.copy()

    def update(self, u, v=None, G=None):
        if (self.s, self.tau2) != self._fixed_noise:
            raise ValueError('This prediction control requires fixed noise parameters; use a separate noise-fit ablation')
        u = np.asarray(u)
        if u.shape != (self.n,) or not np.isfinite(u).all():
            raise ValueError('Expected a finite observation of dimension n')
        if v is not None:
            v = np.asarray(v)
            if v.shape != (self.n,) or not np.isfinite(v).all():
                raise ValueError('Expected a finite projected input of dimension n')
        prior = self.predict(G)
        start = perf_counter()
        B = increment(u, self.s, self.tau2, v, self.complex_data)
        self.records.append(Record(prior.copy(), u.copy(), None if v is None else v.copy()))
        self.A = center(prior + B)
        if self.prediction == 'rolling':
            self.increments.append(B.copy())
        basis = self.basis
        self.history.append(basis.copy())
        self.t += 1
        self._prepared = False
        self.last_update_seconds = perf_counter() - start
        self.update_seconds += self.last_update_seconds
        return basis

    def recompute_window(self):
        """Independent queue recomputation for rolling-bookkeeping checks."""
        if self.prediction != 'rolling':
            raise ValueError('Only the rolling rule retains likelihood increments')
        return center(sum(self.increments, np.zeros_like(self.A)))
