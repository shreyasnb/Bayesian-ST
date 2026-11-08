"""Passive proper-complex information and fixed-time modal certificates.

These statements concern u ~ CN(0, s I + q P). They are not the real known-input
measurement theorem from lib.theory. The covariance certificate requires fixed
known powers, a uniform initial prior, deterministic weights and transports,
and an independently generated proper-complex observation stream.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space


def _powers(s, q, *, positive_q=False):
    s, q = float(s), float(q)
    if not np.isfinite(s) or s <= 0 or not np.isfinite(q) or q < 0:
        raise ValueError('s is a positive variance and q is a nonnegative power')
    if positive_q and q == 0:
        raise ValueError('The tracking certificate requires a positive eigengap q')
    return s, q


def _basis(X):
    X = np.asarray(X)
    if X.ndim != 2 or not 0 < X.shape[1] < X.shape[0] or not np.isfinite(X).all():
        raise ValueError('Expected a finite n by k basis with 0 < k < n')
    if not np.allclose(X.conj().T @ X, np.eye(X.shape[1]), atol=1e-10, rtol=0):
        raise ValueError('Basis columns must be orthonormal')
    return X


def _projectors(path):
    path = np.asarray(path)
    if path.ndim == 2:
        if not len(path) or path.shape[1] < 2 or not np.isfinite(path).all():
            raise ValueError('Signature history must have shape (T,n), T >= 1')
        if not np.allclose(np.sum(abs(path)**2, axis=1), 1, atol=1e-9, rtol=0):
            raise ValueError('Every spatial signature must be unit length')
        path = np.einsum('ti,tj->tij', path, path.conj())
    if path.ndim != 3 or not len(path) or path.shape[1] != path.shape[2]:
        raise ValueError('Expected signatures (T,n) or projectors (T,n,n)')
    n = path.shape[1]
    if not np.isfinite(path).all() or not np.allclose(
            path, path.conj().transpose(0, 2, 1), atol=1e-9, rtol=0):
        raise ValueError('Projector history must be finite and Hermitian')
    if not np.allclose(path @ path, path, atol=1e-8, rtol=0):
        raise ValueError('History matrices must be orthogonal projectors')
    traces = np.trace(path, axis1=1, axis2=2).real
    k = int(round(float(traces[0])))
    if not 0 < k < n or not np.allclose(traces, k, atol=1e-8, rtol=0):
        raise ValueError('All projectors must have the same rank between zero and n')
    return path, n, k


def passive_covariance(P, s, q):
    """Covariance at a supplied projector; s and q are powers, not deviations."""
    s, q = _powers(s, q)
    matrices, n, _ = _projectors(np.asarray(P)[None])
    return s*np.eye(n) + q*matrices[0]


def passive_log_likelihood(u, P, s, q, complex_data=True):
    """Normalized passive Gaussian likelihood by a direct covariance solve."""
    R = passive_covariance(P, s, q)
    if not complex_data and np.iscomplexobj(R) and np.max(abs(R.imag)) > 1e-12:
        raise ValueError('The real model needs a real covariance')
    u = np.asarray(u)
    if u.shape != (len(R),) or not np.isfinite(u).all():
        raise ValueError('Observation dimension or finiteness')
    if not complex_data and (np.max(abs(u.imag)) if np.iscomplexobj(u) else 0) > 1e-12:
        raise ValueError('The real model needs real observations')
    f = 1. if complex_data else .5
    constant = len(R)*np.log(np.pi if complex_data else 2*np.pi)
    return -f*float(constant + np.linalg.slogdet(R)[1] + np.vdot(u, np.linalg.solve(R, u)).real)


def passive_tangent_basis(X, complex_data=True, complement=None):
    """Projector derivatives for real coordinates [Re vec_F Z, Im vec_F Z].

    The real model uses vec_F Z only. The returned derivative matrices are
    Hermitian; a complex tangent coordinate is two real parameters, not one.
    """
    X = _basis(X)
    if not complex_data and np.iscomplexobj(X) and np.max(abs(X.imag)) > 1e-12:
        raise ValueError('Real-model tangent coordinates need a real basis')
    Q = null_space(X.conj().T) if complement is None else _basis(complement)
    n, k = X.shape
    if Q.shape != (n, n-k) or not np.allclose(Q.conj().T @ X, 0, atol=1e-10, rtol=0):
        raise ValueError('Complement must span the orthogonal complement of X')
    real, imaginary = [], []
    for i in range(k):
        for a in range(n-k):
            outer = np.outer(Q[:, a], X[:, i].conj())
            real.append(outer + outer.conj().T)
            imaginary.append(1j*(outer - outer.conj().T))
    result = real + imaginary if complex_data else real
    return np.asarray(result)


def passive_fisher_information(n, k, s, q, observations=1, complex_data=True):
    """Measurement Fisher matrix in real local Grassmann coordinates.

    This is the Gaussian score information, not a Bingham-normalizer Hessian,
    a posterior precision or a discounted effective sample size.
    """
    if not isinstance(n, (int, np.integer)) or not isinstance(k, (int, np.integer)) or not 0 < k < n:
        raise ValueError('Expected integer dimensions with 0 < k < n')
    if not np.isfinite(observations) or observations <= 0:
        raise ValueError('Observation count must be positive')
    s, q = _powers(s, q)
    factor = 2. if complex_data else 1.
    d = (2 if complex_data else 1)*k*(n-k)
    return observations*factor*q*q/(s*(s+q))*np.eye(d)


def passive_scores(U, X, s, q, complex_data=True, complement=None):
    """Real score vectors using direct Gaussian covariance derivatives."""
    X = _basis(X)
    U = np.asarray(U)
    if U.ndim != 2 or U.shape[1] != len(X) or not np.isfinite(U).all():
        raise ValueError('U must contain finite row-wise snapshots')
    if not complex_data and np.iscomplexobj(U) and np.max(abs(U.imag)) > 1e-12:
        raise ValueError('Real-model scores need real observations')
    derivatives = q*passive_tangent_basis(X, complex_data, complement)
    R = passive_covariance(X @ X.conj().T, s, q)
    inverse = np.linalg.inv(R)
    whitened = U @ inverse.T
    values = np.einsum('bi,dij,bj->bd', whitened.conj(), derivatives, whitened, optimize=True).real
    traces = np.einsum('ij,dji->d', inverse, derivatives).real
    return (1. if complex_data else .5)*(values-traces)


def ula_signature(n, theta):
    """Unit half-wavelength ULA signature; theta is always in radians."""
    if not isinstance(n, (int, np.integer)) or n < 2 or not np.isfinite(theta):
        raise ValueError('Expected n >= 2 and a finite radian angle')
    return np.exp(1j*np.pi*np.arange(n)*np.sin(theta))/np.sqrt(n)


def ula_derivative(n, theta):
    return 1j*np.pi*np.arange(n)*np.cos(theta)*ula_signature(n, theta)


def ula_angle_information(n, theta, s, q, observations=1, per_degree=False):
    """Pullback Fisher information; input theta remains radians in both modes."""
    ula_signature(n, theta)  # Validate the radian angle and array dimension.
    coefficient = passive_fisher_information(n, 1, s, q, observations)[0, 0]
    tangent_norm_squared = np.pi**2*np.cos(theta)**2*(n*n-1)/12
    return float(coefficient*tangent_norm_squared*((np.pi/180)**2 if per_degree else 1.))


def exponential_weights(T, beta=1.):
    """Normalized observation weights and the unnormalized likelihood sum."""
    if not isinstance(T, (int, np.integer)) or T < 1 or not np.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError('Expected T >= 1 and 0 < beta <= 1')
    weights = np.exp(np.log(beta)*np.arange(T-1, -1, -1))
    total = float(weights.sum())
    return weights/total, total


@dataclass(frozen=True)
class PassiveCovarianceCertificate:
    eta: float
    drift: float
    epsilon: float
    valid: bool
    bound: float
    neff: float
    weight_sum: float
    alpha: np.ndarray
    x: float
    ceiling: float
    transports: np.ndarray
    delta: float
    s: float
    q: float
    n: int
    k: int
    mean_projector: np.ndarray
    probability_scope: str = 'Pointwise at one predeclared time; fixed known powers and deterministic weights/transports'


def passive_covariance_certificate(projectors, s, q, weights=None, beta=1., G=None, delta=.05):
    """Passive quadratic-Gaussian modal bound under its stated assumptions.

    ``projectors`` accepts row-wise unit signatures (T,n) or projector matrices
    (T,n,n). Explicit weights may be unnormalized and are normalized internally.
    G[j] transports state j-1 to j; G[0] is immaterial for a uniform prior.
    ``drift`` uses the true path and is an evaluation quantity, never an estimator
    input. A learned transport, inferred powers, stochastic moment projection,
    or a nonuniform initialization is outside this certificate.
    """
    s, q = _powers(s, q, positive_q=True)
    Ps, n, k = _projectors(projectors)
    T = len(Ps)
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError('Expected 0 < delta < 1')
    if weights is None:
        alpha, weight_sum = exponential_weights(T, beta)
    else:
        if beta != 1:
            raise ValueError('Supply explicit weights or beta, not both')
        weights = np.asarray(weights, float)
        if weights.shape != (T,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Weights must be finite, nonnegative and have positive sum')
        weight_sum = float(weights.sum())
        alpha = weights/weight_sum
    if G is None:
        transports = np.broadcast_to(np.eye(n), (T, n, n))
        aligned = Ps
    else:
        G = np.asarray(G)
        if G.shape != (T, n, n) or not np.isfinite(G).all() or not np.allclose(
                G.conj().transpose(0, 2, 1) @ G, np.eye(n), atol=1e-10, rtol=0):
            raise ValueError('G must contain deterministic supplied unitary transports')
        transports = np.empty_like(G, dtype=np.result_type(G.dtype, complex))
        product = np.eye(n, dtype=transports.dtype)
        for j in range(T-1, -1, -1):
            transports[j] = product
            product = product @ G[j]
        aligned = transports @ Ps @ transports.conj().transpose(0, 2, 1)
    mean_projector = np.einsum('t,tij->ij', alpha, aligned)
    drift = float(q*np.linalg.norm(mean_projector-Ps[-1], 2))
    sum_squared = float(alpha @ alpha)
    x = float(2*n*np.log(9)+np.log(2/delta))
    eta = float(2*(s+q)*(np.sqrt(2*x*sum_squared)+x*alpha.max()))
    epsilon = drift+eta
    ceiling = float(np.sqrt(min(k, n-k)))
    valid = bool(epsilon < q/2)
    bound = ceiling*epsilon/(q-epsilon) if valid else np.nan
    return PassiveCovarianceCertificate(eta, drift, epsilon, valid, float(bound),
        1/sum_squared, weight_sum, alpha, x, ceiling, transports, float(delta),
        s, q, n, k, mean_projector)


def modal_scanning_bound(certificate):
    """Uniform unit-direction scanning-power bound on the same modal event."""
    return float(certificate.q*certificate.bound) if certificate.valid else np.nan
