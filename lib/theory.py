"""Excitation, fixed-time tracking, and certified regional task bounds.

The tracking certificate uses deterministic inputs and the true fixed variance;
it is an evaluation tool, not an observable posterior radius.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space


def _hermitian(A):
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("matrix entries must be finite")
    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError("matrix must be symmetric or Hermitian")
    return (A + A.conj().T) / 2


def _basis(X):
    X = np.asarray(X)
    if X.ndim != 2 or not 0 < X.shape[1] < X.shape[0]:
        raise ValueError("basis must have 0 < rank < ambient dimension")
    if not np.allclose(X.conj().T @ X, np.eye(X.shape[1]), atol=1e-9):
        raise ValueError("basis columns must be orthonormal")
    return X


def fisher_information(X, V, s):
    """Real-model Fisher information in column-major chart coordinates.

    ``V`` has shape (n, observations). The complementary basis is the one
    returned by scipy.linalg.null_space(X.T). The function is deliberately
    restricted to real inputs.
    """
    X = _basis(X)
    V = np.asarray(V)
    if np.iscomplexobj(X) or np.iscomplexobj(V):
        raise ValueError("Fisher information requires real observations")
    if s <= 0 or not np.isfinite(s):
        raise ValueError("s must be a positive finite variance")
    if V.ndim != 2 or V.shape[0] != X.shape[0]:
        raise ValueError("V must have shape (n, observations)")
    Q = null_space(X.T)
    C = V @ V.T
    k = X.shape[1]
    r = Q.shape[1]
    return (np.kron(X.T @ C @ X, np.eye(r)) +
            np.kron(np.eye(k), Q.T @ C @ Q)) / s


@dataclass(frozen=True)
class TrackingBound:
    C: np.ndarray
    D: np.ndarray
    gamma: float
    drift: float
    initialization: float
    noise: float
    epsilon: float
    valid: bool
    bound: float
    ceiling: float
    transports: np.ndarray


def tracking_bound(X_history, V, s, beta=1., G=None, A0=None, delta=.05):
    """Evaluate the tracking bound at the last observation time.

    X_history has shape (t,n,k), V has shape (t,n). G has shape (t,n,n),
    with G[j] the supplied transformation into observation j; G[0] transports
    the initialization into the first observation. All past observations are
    retained here, independently of any EM fitting window. ``bound`` is NaN
    whenever the strict validity condition fails; ``ceiling`` always applies.
    """
    Xs = np.asarray(X_history)
    vs = np.asarray(V)
    if Xs.ndim != 3 or not len(Xs):
        raise ValueError("X_history must have shape (t,n,k), t >= 1")
    t, n, k = Xs.shape
    for X in Xs:
        _basis(X)
    if np.iscomplexobj(Xs) or np.iscomplexobj(vs):
        raise ValueError("the tracking bound requires real observations")
    if vs.shape != (t, n) or not np.all(np.isfinite(vs)):
        raise ValueError("V must have finite entries and shape (t,n)")
    if not 0 < beta <= 1 or not 0 < delta < 1 or s <= 0:
        raise ValueError("require 0 < beta <= 1, 0 < delta < 1, s > 0")
    if G is None:
        G = np.broadcast_to(np.eye(n), (t, n, n))
    G = np.asarray(G)
    if G.shape != (t, n, n):
        raise ValueError("G must have shape (t,n,n)")
    if np.iscomplexobj(G) or not np.allclose(G.transpose(0, 2, 1) @ G, np.eye(n), atol=1e-9):
        raise ValueError("G must contain orthogonal transformations")
    A0 = np.zeros((n, n)) if A0 is None else _hermitian(A0)
    if A0.shape != (n, n) or np.iscomplexobj(A0):
        raise ValueError("A0 must be real symmetric with shape (n,n)")
    transports = np.empty((t, n, n))
    T = np.eye(n)
    for j in range(t-1, -1, -1):
        transports[j] = T
        T = T @ G[j]
    P = Xs[-1] @ Xs[-1].T
    C = np.zeros((n, n))
    D = np.zeros_like(C)
    drift_matrix = np.zeros_like(C)
    for j in range(t):
        w = beta ** (t-1-j)
        v = transports[j] @ vs[j]
        outer = np.outer(v, v)
        C += w * outer
        D += w*w * outer
        Y = transports[j] @ Xs[j]
        difference = Y @ Y.T - P
        drift_matrix += w * (difference @ outer + outer @ difference) / (2*s)
    Q = null_space(Xs[-1].T)
    gamma = float((np.linalg.eigvalsh(Xs[-1].T @ C @ Xs[-1])[0] +
                   np.linalg.eigvalsh(Q.T @ C @ Q)[0]) / (2*s))
    drift = float(np.linalg.norm(drift_matrix, 2))
    eigenvalues = np.linalg.eigvalsh(A0)
    initialization = float(beta**t * (eigenvalues[-1] - eigenvalues[0]) / 2)
    noise = float((2*np.sqrt(n) + np.sqrt(2*np.log(1/delta))) *
                  np.sqrt(max(0., np.linalg.eigvalsh(D)[-1]) / s))
    epsilon = initialization + drift + noise
    ceiling = float(np.sqrt(min(k, n-k)))
    valid = bool(gamma > 0 and epsilon < gamma/2)
    bound = ceiling * epsilon / (gamma-epsilon) if valid else np.nan
    return TrackingBound(C, D, gamma, drift, initialization, noise, epsilon,
                         valid, float(bound), ceiling, transports)


@dataclass(frozen=True)
class RegionalBound:
    """Primal/dual certificate for a specified deterministic region.

    This gap excludes error in a sampled credible threshold. ``basis`` is a
    feasible rank-k witness, ``lower`` its objective, and ``upper`` a dual bound.
    """
    basis: np.ndarray
    lower: float
    upper: float
    gap: float
    mu: float
    constraint: float
    feasible: bool


def _trace(A, X):
    return float(np.trace(X.conj().T @ A @ X).real)


def _leading_family(M, C, k, tolerance):
    """Strict leading vectors, cutoff block, C-spectrum and trace interval."""
    values, vectors = np.linalg.eigh(M)
    cutoff = values[-k]
    tied = np.abs(values-cutoff) <= tolerance
    above = values > cutoff+tolerance
    H, E = vectors[:, above], vectors[:, tied]
    needed = k-H.shape[1]
    d, U = np.linalg.eigh(E.conj().T @ C @ E)
    base = _trace(C, H)
    low = base + float(np.sum(d[:needed]))
    high = base + float(np.sum(d[-needed:]))
    return H, E @ U, needed, base, d, low, high


def _select_family(H, E, rank, base, eigenvalues, target):
    """Continuous rotation between the minimum and maximum C trace bases."""
    m = len(eigenvalues)
    if rank == 0:
        return H
    low = base + float(np.sum(eigenvalues[:rank]))
    high = base + float(np.sum(eigenvalues[-rank:]))
    fraction = np.clip((target-low)/(high-low), 0, 1) if high > low else 0.
    p = min(rank, m-rank)
    columns = []
    for j in range(p):
        columns.append(np.sqrt(1-fraction)*E[:, j] + np.sqrt(fraction)*E[:, m-p+j])
    for j in range(m-rank, rank):
        columns.append(E[:, j])
    return np.column_stack([H] + columns) if columns else H


def regional_bound(B, C, a, k, tol=1e-9, maxiter=100):
    """Maximize tr(BP) subject to tr(CP)>=a, rank(P)=k, P²=P.

    Implements the one-dimensional convex dual, including tied eigenspaces,
    complex Hermitian data, and the endpoint
    a=sum(top-k eigenvalues(C)). Empty regions raise ValueError. The returned
    objective gap certifies the numerical optimization for this region.
    """
    B, C = _hermitian(B), _hermitian(C)
    if B.shape != C.shape or not 0 < k < len(B):
        raise ValueError("require matching shapes and 0 < k < n")
    if not np.isfinite(a) or tol <= 0:
        raise ValueError("require finite threshold and positive tolerance")
    scale = max(1., np.linalg.norm(B, 2), np.linalg.norm(C, 2), abs(a))
    absolute = tol*scale
    cv, U = np.linalg.eigh(C)
    maximum = float(np.sum(cv[-k:]))
    if a > maximum + absolute:
        raise ValueError("empty region: threshold exceeds the Ky Fan maximum")

    def finish(X, upper, mu):
        lower, constraint = _trace(B, X), _trace(C, X)
        # Roundoff cushion for the Hermitian eigensolver; never hide a primal
        # infeasibility or a substantive negative primal/dual gap.
        cushion = 20*np.finfo(float).eps*scale*(1+abs(mu) if np.isfinite(mu) else 1.)
        upper = float(upper + cushion)
        if lower > upper + absolute:
            raise RuntimeError("negative primal-dual gap exceeds numerical tolerance")
        return RegionalBound(X, lower, upper, max(0., upper-lower), float(mu),
                             constraint, bool(constraint >= a-absolute))

    if abs(a-maximum) <= absolute:
        # The feasible set consists exactly of leading subspaces of C. Choose
        # the best B-subspace inside C's cutoff block; no finite dual optimum
        # need exist at this endpoint.
        cut = cv[-k]
        above = cv > cut+absolute
        equal = np.abs(cv-cut) <= absolute
        H, E = U[:, above], U[:, equal]
        q = k-H.shape[1]
        _, W = np.linalg.eigh(E.conj().T @ B @ E)
        X = np.column_stack((H, E @ W[:, -q:])) if q else H
        return finish(X, _trace(B, X), np.inf if np.ptp(cv) > absolute else 0.)

    def family(mu):
        M = B+mu*C
        # A small relative tie tolerance allows stable resolution of a dual
        # cusp; the actual primal-dual gap is still computed from B and C.
        etol = tol*max(1., np.linalg.norm(M, 2))
        result = _leading_family(M, C, k, etol)
        upper = float(np.sum(np.linalg.eigvalsh(M)[-k:])-mu*a)
        return result, upper

    f, upper = family(0.)
    if f[-1] >= a-absolute:
        X = _select_family(*f[:5], max(a, f[-2]))
        return finish(X, upper, 0.)
    lo, hi = 0., 1.
    for _ in range(maxiter):
        f, upper = family(hi)
        if f[-1] >= a:
            break
        hi *= 2
    else:
        raise RuntimeError("failed to bracket regional dual minimizer")
    best = None
    for _ in range(maxiter):
        mu = (lo+hi)/2
        f, upper = family(mu)
        cmin, cmax = f[-2:]
        if cmin <= a <= cmax:
            X = _select_family(*f[:5], a)
            return finish(X, upper, mu)
        if cmax < a:
            lo = mu
        else:
            hi = mu
            X = _select_family(*f[:5], cmin)
            best = finish(X, upper, mu)
            if best.gap <= absolute:
                return best
        if hi-lo <= np.finfo(float).eps*max(1., hi):
            break
    if best is None:
        raise RuntimeError("no feasible regional witness found")
    return best
