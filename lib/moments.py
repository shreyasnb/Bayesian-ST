"""Numerical moments for real and complex Grassmann Bingham laws.

The sampler targets exp(tr(A X X*)) with respect to invariant Grassmann
measure. It does not replace numerical diagnostics with an accuracy guarantee.
Small real rank-one examples additionally admit deterministic quadrature.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp


def _hermitian(A):
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
        raise ValueError("A must be a square matrix of dimension at least two")
    if not np.all(np.isfinite(A)) or not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError("A must be finite and symmetric/Hermitian")
    return (A + A.conj().T) / 2


def effective_sample_size(values):
    """FFT autocorrelations and Geyer's initial-positive paired sequence.

    The estimate is a chain diagnostic, not proof that burn-in was sufficient.
    Constant statistics are omitted by callers and return their full count here.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 2:
        return float(len(y))
    y = y - y.mean()
    if np.dot(y, y) <= np.finfo(float).eps * len(y):
        return float(len(y))
    size = 1 << (2 * len(y) - 1).bit_length()
    f = np.fft.rfft(y, size)
    ac = np.fft.irfft(f * f.conj(), size)[:len(y)]
    ac /= ac[0]
    pairs = ac[1: 2 * ((len(y) - 1) // 2) + 1].reshape(-1, 2).sum(axis=1)
    positive = np.flatnonzero(pairs <= 0)
    if positive.size:
        pairs = pairs[:positive[0]]
    if pairs.size:
        pairs = np.minimum.accumulate(pairs)
    tau = max(1.0, 1 + 2 * pairs.sum())
    return float(np.clip(len(y) / tau, 1, len(y)))


def _batch_mcse(values):
    values = np.asarray(values)
    batches = max(2, int(np.sqrt(len(values))))
    width = len(values) // batches
    if width < 2:
        return np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    bm = values[:batches * width].reshape(batches, width, -1).mean(axis=1)
    return bm.std(axis=0, ddof=1) / np.sqrt(batches)


@dataclass
class MomentResult:
    bases: np.ndarray
    mean: np.ndarray
    raw_mean: np.ndarray
    acceptance: float
    ess: float
    mcse: float
    step_size: float
    burnin: int
    thin: int
    spectral_mcse: np.ndarray

    @property
    def draws(self):
        return len(self.bases)

    @property
    def diagnostics(self):
        return {"draws": self.draws, "burnin": self.burnin, "thin": self.thin,
                "acceptance": self.acceptance, "ess": self.ess,
                "mcse": self.mcse, "step_size": self.step_size}


def sample_bingham(A, k, rng=None, draws=1000, burnin=500, thin=1,
                   complex_data=False, initial=None, step_size=None,
                   uniform_probability=0.1):
    """Sample real/complex rank-k Bingham projectors without a normalizer.

    Each sweep updates all columns. A conditional uniform-sphere proposal is
    mixed with an isotropic geodesic proposal in the orthogonal complement of
    the entire current basis. Both kernels are symmetric on subspaces; their
    Metropolis ratios contain only the target energy difference. Periodic
    random right rotations preserve the projector and improve basis mixing.
    Step size adaptation stops at the end of burn-in. Identity shifts have no
    numerical effect because the spectrum is centered before simulation.

    ``mean`` averages row norms in the eigenbasis, including exact symmetry on
    repeated-eigenvalue blocks (Appendix G.1). ``raw_mean`` averages the actual
    sampled projectors. ``mcse`` is the largest spectral-entry batch-mean
    standard error. It does not estimate all possible credible-tail errors.
    """
    A = _hermitian(A)
    n = len(A)
    if not isinstance(k, (int, np.integer)) or not 0 < k < n:
        raise ValueError("k must satisfy 0 < k < n")
    if draws < 2 or burnin < 0 or thin < 1:
        raise ValueError("draws >= 2, burnin >= 0 and thin >= 1 are required")
    if not 0 <= uniform_probability <= 1:
        raise ValueError("uniform_probability must lie in [0, 1]")
    complex_data = bool(complex_data or np.iscomplexobj(A))
    rng = np.random.default_rng(rng)
    eigenvalues, U = np.linalg.eigh(A)
    eigenvalues -= eigenvalues.mean()
    dtype = complex if complex_data else float
    if initial is None:
        X = np.eye(n, dtype=dtype)[:, -k:].copy()
    else:
        initial = np.asarray(initial, dtype=dtype)
        if initial.shape != (n, k) or np.linalg.matrix_rank(initial) != k:
            raise ValueError("initial must be a full-rank (n,k) basis")
        X = np.linalg.qr(U.conj().T @ initial)[0]
    if step_size is None:
        step_size = min(0.6, np.sqrt((n - k) / max(np.ptp(eigenvalues), 1.0)))
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    step_size = float(step_size)
    adapt = initial is None or burnin > 0
    retained = np.empty((draws, n, k), dtype=dtype)
    row_norms = np.empty((draws, n))
    accepted = proposed = local_acc = local_num = 0

    def gaussian(shape):
        g = rng.standard_normal(shape)
        if complex_data:
            g = (g + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        return g

    total = burnin + draws * thin
    count = 0
    for sweep in range(total):
        for i in range(k):
            old = X[:, i].copy()
            g = gaussian(n)
            local = rng.random() >= uniform_probability
            if local:
                # Uniform tangent direction to the conditional projective space.
                g -= X @ (X.conj().T @ g)
                norm = np.linalg.norm(g)
                if norm < 1e-14:
                    continue
                angle = rng.normal(scale=step_size)
                candidate = np.cos(angle) * old + np.sin(angle) * g / norm
            else:
                others = np.delete(X, i, axis=1)
                g -= others @ (others.conj().T @ g)
                norm = np.linalg.norm(g)
                if norm < 1e-14:
                    continue
                candidate = g / norm
            delta = np.dot(eigenvalues, abs(candidate)**2 - abs(old)**2).real
            accept = np.log(rng.random()) < min(0.0, delta)
            if accept:
                X[:, i] = candidate
            if sweep >= burnin:
                proposed += 1
                accepted += int(accept)
            elif local:
                local_num += 1
                local_acc += int(accept)
        if (sweep + 1) % 20 == 0:
            # QR controls floating point drift; right rotations leave XX* fixed.
            X = np.linalg.qr(X)[0]
            if k > 1:
                Q = np.linalg.qr(gaussian((k, k)))[0]
                X = X @ Q
        if adapt and sweep < burnin and (sweep + 1) % 25 == 0 and local_num:
            gain = min(0.5, 1 / np.sqrt((sweep + 1) / 25))
            step_size *= np.exp(gain * (local_acc / local_num - 0.44))
            step_size = float(np.clip(step_size, 1e-5, 1.4))
            local_acc = local_num = 0
        if sweep >= burnin and (sweep - burnin + 1) % thin == 0:
            retained[count] = U @ X
            row_norms[count] = np.sum(abs(X)**2, axis=1)
            count += 1

    m = row_norms.mean(axis=0)
    spectral_mcse = _batch_mcse(row_norms)
    # Exact commutant averaging: all rows in a repeated eigenvalue block agree.
    start = 0
    for end in range(1, n + 1):
        if end == n or not np.isclose(eigenvalues[end], eigenvalues[start],
                                      rtol=1e-11, atol=1e-12):
            if end - start > 1:
                m[start:end] = m[start:end].mean()
            start = end
    mean = (U * m) @ U.conj().T
    raw_mean = np.einsum('bik,bjk->ij', retained, retained.conj()) / draws
    varying = np.std(row_norms, axis=0) > 1e-12
    esses = [effective_sample_size(row_norms[:, j]) for j in np.flatnonzero(varying)]
    return MomentResult(retained, mean, raw_mean,
                        accepted / max(proposed, 1), min(esses, default=float(draws)),
                        float(np.max(spectral_mcse)), step_size, burnin, thin,
                        spectral_mcse)


@dataclass
class QuadratureResult:
    mean: np.ndarray
    log_normalizer: float
    bases: np.ndarray
    weights: np.ndarray
    order: int

    def weighted_quantile(self, values, probability):
        return weighted_quantile(values, probability, self.weights)


def weighted_quantile(values, probability, weights=None):
    """Left inverse of a discrete weighted CDF (weights need not be normalized)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not 0 <= probability <= 1 or len(values) == 0:
        raise ValueError("require nonempty 1-D values and probability in [0,1]")
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != values.shape or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and have positive sum")
    indices = np.argsort(values)
    j = np.searchsorted(np.cumsum(w[indices]), probability * w.sum(), side='left')
    return float(values[indices[min(j, len(values) - 1)]])


def _rank_one_result(A, eigenvalues, U, local_bases, prior_weights, order):
    energies = np.einsum('bi,i,bi->b', local_bases, eigenvalues, local_bases)
    lw = energies + np.log(prior_weights)
    log_normalizer = float(logsumexp(lw))
    weights = np.exp(lw - log_normalizer)
    # Symmetric quadrature gives a diagonal mean in the eigenbasis; restore
    # that exact symmetry rather than retaining quadrature roundoff off diagonal.
    m = np.einsum('b,bi,bi->i', weights, local_bases, local_bases)
    mean = (U * m) @ U.T
    bases = (local_bases @ U.T)[:, :, None]
    return QuadratureResult(mean, log_normalizer, bases, weights, order)


def quadrature_gr13(A, order=96):
    """Tensor Gauss–Legendre/periodic quadrature on real Gr(1,3).

    Uses the normalized sphere area measure dq dphi/(4*pi), q in [-1,1].
    Antipodal symmetry makes it exactly the normalized Grassmann measure.
    Compare orders to assess numerical error, particularly for tail quantiles.
    """
    A = _hermitian(A)
    if A.shape != (3, 3) or np.iscomplexobj(A) or order < 8:
        raise ValueError("real 3x3 A and order >= 8 are required")
    eigenvalues, U = np.linalg.eigh(A)
    # Put the highest-concentration axis at the poles for efficient integration.
    eigenvalues, U = eigenvalues[::-1], U[:, ::-1]
    q, w = leggauss(order)
    phi = 2 * np.pi * np.arange(2 * order) / (2 * order)
    q = np.repeat(q, len(phi))
    phi = np.tile(phi, order)
    radius = np.sqrt(np.maximum(0, 1 - q*q))
    local = np.column_stack((q, radius * np.cos(phi), radius * np.sin(phi)))
    prior_weights = np.repeat(w / (4 * order), 2 * order)
    return _rank_one_result(A, eigenvalues, U, local, prior_weights, order)


def quadrature_circle(A, order=1024):
    """Periodic quadrature on real Gr(1,2), with normalized invariant measure."""
    A = _hermitian(A)
    if A.shape != (2, 2) or np.iscomplexobj(A) or order < 8:
        raise ValueError("real 2x2 A and order >= 8 are required")
    eigenvalues, U = np.linalg.eigh(A)
    angle = 2 * np.pi * np.arange(order) / order
    local = np.column_stack((np.cos(angle), np.sin(angle)))
    return _rank_one_result(A, eigenvalues, U, local,
                            np.full(order, 1 / order), order)


def quadrature(A, k=1, order=96):
    """Dispatch deterministic small real rank-one quadrature."""
    if k != 1:
        raise ValueError("deterministic quadrature supports k=1 only")
    n = np.asarray(A).shape[0]
    if n == 2:
        return quadrature_circle(A, order=max(256, 8 * order))
    if n == 3:
        return quadrature_gr13(A, order=order)
    raise ValueError("deterministic quadrature supports n=2 or n=3 only")


def mean_bingham(A, k, rng=None, **kwargs):
    """Convenience mean evaluator: exact uniform case, otherwise Monte Carlo.

    Use ``sample_bingham`` directly whenever diagnostics or credible draws are
    needed. No automatic Laplace approximation is used at large concentration.
    """
    A = _hermitian(A)
    if not 0 < k < len(A):
        raise ValueError("k must satisfy 0 < k < n")
    if np.linalg.norm(A - np.trace(A) * np.eye(len(A)) / len(A)) < 1e-12:
        return k * np.eye(len(A), dtype=A.dtype) / len(A)
    return sample_bingham(A, k, rng=rng, **kwargs).mean


def chain_diagnostics(results):
    """Independent-chain comparison for the spectral moment estimates.

    Return the largest elementwise discrepancy, standardized by conservative
    batch MCSEs, and split R-hat of diagonal projector statistics in a common
    coordinate system. Start chains independently and inspect both diagnostics;
    neither is a convergence theorem. R-hat uses raw samples, not symmetry-
    averaged means, and so also detects poor orientation mixing at repeated
    eigenvalues. Use at least two chains and four draws per chain.
    """
    results = list(results)
    if len(results) < 2 or min(r.draws for r in results) < 4:
        raise ValueError("at least two chains with at least four draws are required")
    if any(r.bases.shape[1:] != results[0].bases.shape[1:] for r in results):
        raise ValueError("chains must have matching dimensions and rank")
    length = min(r.draws for r in results) // 2
    splits = []
    for result in results:
        diagonal = np.sum(abs(result.bases)**2, axis=2)
        splits.extend([diagonal[:length], diagonal[-length:]])
    values = np.asarray(splits)
    W = values.var(axis=1, ddof=1).mean(axis=0)
    B = length * values.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (length - 1) * W / length + B / length
    rhat = np.sqrt(np.divide(var_hat, W, out=np.ones_like(W), where=W > 1e-15))
    discrepancy = max(float(np.max(abs(a.mean - b.mean)))
                      for i, a in enumerate(results) for b in results[i + 1:])
    z = max(float(np.max(abs(a.mean - b.mean))) /
            max(np.hypot(a.mcse, b.mcse), 1e-15)
            for i, a in enumerate(results) for b in results[i + 1:])
    return {"max_rhat": float(max(1.0, np.max(rhat))),
            "max_mean_difference": discrepancy, "max_standardized_difference": z,
            "min_ess": min(r.ess for r in results),
            "max_mcse": max(r.mcse for r in results)}
