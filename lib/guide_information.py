"""Independent low-dimensional algebra and measurement-information checks.

All information coordinates are real: vec_F(Z) for real subspaces and
[Re vec_F(Z), Im vec_F(Z)] for complex subspaces.  Recorded inputs are supplied
explicitly.  Truth is used here only to evaluate a mathematical reference.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from .beamforming_theory import passive_fisher_information, passive_scores, passive_tangent_basis
from .filter import BinghamFilter
from .geometry import center
from .models import constant, increment, log_likelihood
from .spectral_series import rank_one_bingham
from .theory import fisher_information


def chart_basis(X, coordinate, complex_data=False, complement=None):
    """Graph chart with symmetric orthonormalization and column-major ordering."""
    X = np.asarray(X)
    n, k = X.shape
    Q = null_space(X.conj().T) if complement is None else np.asarray(complement)
    coordinate = np.asarray(coordinate, float)
    dimension = k*(n-k)
    if coordinate.shape != ((2 if complex_data else 1)*dimension,):
        raise ValueError('Incorrect real tangent-coordinate dimension')
    if not np.allclose(X.conj().T@X, np.eye(k), atol=1e-11):
        raise ValueError('X must have orthonormal columns')
    if Q.shape != (n, n-k) or not np.allclose(Q.conj().T@X, 0, atol=1e-11):
        raise ValueError('Q must be an aligned orthonormal complement')
    if not np.allclose(Q.conj().T@Q, np.eye(n-k), atol=1e-11):
        raise ValueError('Q must be an aligned orthonormal complement')
    z = coordinate[:dimension]
    if complex_data:
        z = z + 1j*coordinate[dimension:]
    Z = z.reshape((n-k, k), order='F')
    Y = X + Q@Z
    eigenvalues, U = np.linalg.eigh(Y.conj().T@Y)
    return Y@((U/eigenvalues**.5)@U.conj().T)


def gaussian_kl(X, coordinate, s, tau2=1., complex_data=False, V=None, complement=None):
    """Exact Gaussian KL from the reference state to a chart candidate.

    V contains row-wise, independently prescribed recorded inputs. For this
    channel the KL sums its independent observations; no hidden input is used.
    """
    if s <= 0 or tau2 < 0:
        raise ValueError('Use a positive sensor variance and nonnegative signal variance')
    X = np.asarray(X)
    Y = chart_basis(X, coordinate, complex_data, complement)
    P, P0 = Y@Y.conj().T, X@X.conj().T
    f = 1. if complex_data else .5
    if V is not None:
        difference = np.asarray(V)@(P-P0).T
        return float(f*np.sum(abs(difference)**2)/s)
    n = len(X)
    R, R0 = s*np.eye(n)+tau2*P, s*np.eye(n)+tau2*P0
    # Equal-rank projectors make the two determinants exactly equal. Keeping
    # direct determinants also checks the actual Gaussian covariance convention.
    return float(f*(np.linalg.slogdet(R)[1]-np.linalg.slogdet(R0)[1]
                    + np.trace(np.linalg.solve(R, R0)).real-n))


def expected_nll(X, coordinate, s, tau2=1., complex_data=False, V=None, complement=None):
    """Direct Gaussian expected negative log likelihood, including constants."""
    X = np.asarray(X)
    Y = chart_basis(X, coordinate, complex_data, complement)
    P, P0 = Y@Y.conj().T, X@X.conj().T
    f, n = (1. if complex_data else .5), len(X)
    base = n*np.log(np.pi if complex_data else 2*np.pi)
    if V is not None:
        V = np.asarray(V)
        difference = V@(P-P0).T
        return float(f*(len(V)*(base+n*np.log(s)+n)+np.sum(abs(difference)**2)/s))
    R, R0 = s*np.eye(n)+tau2*P, s*np.eye(n)+tau2*P0
    return float(f*(base+np.linalg.slogdet(R)[1]+np.trace(np.linalg.solve(R, R0)).real))


def finite_hessian(function, dimension, step):
    """Central differences at zero; independent of analytic score formulas."""
    if not 0 < step < .2:
        raise ValueError('Use a finite positive small step')
    E = step*np.eye(dimension)
    base = function(np.zeros(dimension))
    H = np.empty((dimension, dimension))
    for i in range(dimension):
        H[i, i] = (function(E[i])-2*base+function(-E[i]))/step**2
        for j in range(i):
            H[i, j] = H[j, i] = (function(E[i]+E[j])-function(E[i]-E[j])
                -function(-E[i]+E[j])+function(-E[i]-E[j]))/(4*step**2)
    return H


def conditional_fisher(X, V, s, complex_data=False, complement=None):
    """Direct Gaussian mean-derivative information for known row-wise inputs."""
    D = passive_tangent_basis(X, complex_data, complement)
    J = np.einsum('dij,tj->tdi', D, np.asarray(V))
    return ((2. if complex_data else 1.)/s)*np.einsum('tdi,tei->de', J.conj(), J).real


def conditional_scores(noise, X, V, s, complex_data=False, complement=None):
    """Per-episode scores for independent Gaussian response errors.

    noise has shape (episodes, observations, n); V is recorded design, not
    inferred from noisy observations. A score is averaged only across episodes.
    """
    D = passive_tangent_basis(X, complex_data, complement)
    J = np.einsum('dij,tj->tdi', D, np.asarray(V))
    return ((2. if complex_data else 1.)/s)*np.einsum('bti,tdi->bd', np.asarray(noise).conj(), J).real


def score_statistics(scores):
    scores = np.asarray(scores, float)
    count = len(scores)
    mean = scores.mean(axis=0)
    empirical = scores.T@scores/count
    squared = scores*scores
    mcse = np.sqrt(np.maximum(0., squared.T@squared/count-empirical*empirical)/count)
    return empirical, mean, mcse


def independent_moments(A, k=1, complex_data=False, order=64):
    """Independent continuous quadrature references at N=3.

    Real integration is normalized sphere area. Complex integration uses the
    uniform simplex law of squared eigen-coordinates; it is a different measure.
    Complementation supplies rank two, including the trace shift in log Z.
    """
    A = np.asarray(A)
    if A.shape != (3, 3) or k not in (1, 2):
        raise ValueError('This independent reference is restricted to N=3, k=1 or 2')
    if k == 2:
        M, psi = independent_moments(-A, 1, complex_data, order)
        return np.eye(3)-M, float(np.trace(A).real+psi)
    values, U = np.linalg.eigh(A)
    g, w = np.polynomial.legendre.leggauss(order)
    if complex_data:
        x = (g+1)/2
        xx, yy = np.meshgrid(x, x, indexing='ij')
        squared = np.stack([xx, (1-xx)*yy, (1-xx)*(1-yy)], axis=-1).reshape(-1, 3)
        prior = ((w[:, None]*w[None, :])/4*2*(1-xx)).ravel()
    else:
        phi = 2*np.pi*np.arange(4*order)/(4*order)
        radial = np.sqrt(1-g*g)
        coordinates = np.stack(np.broadcast_arrays(radial[:, None]*np.cos(phi),
            radial[:, None]*np.sin(phi), g[:, None]), axis=-1).reshape(-1, 3)
        squared = coordinates*coordinates
        prior = np.broadcast_to(w[:, None]/(8*order), (order, 4*order)).ravel()
    logits = squared@values+np.log(prior)
    psi = float(logsumexp(logits))
    means = np.exp(logits-psi)@squared
    return (U*means)@U.conj().T, psi


def native_moments(A, k=1, complex_data=False, tol=1e-11):
    if k == 1:
        result = rank_one_bingham(A, complex_data=complex_data, tol=tol)
        return result.mean, result.log_normalizer
    if k == len(A)-1:
        M, psi = native_moments(-A, 1, complex_data, tol)
        return np.eye(len(A))-M, float(np.trace(A).real+psi)
    raise ValueError('Native rank/codimension-one reference only')


def algebra_checks(rng, complex_data=False, k=1):
    """Independent likelihood, gauge, sequential and evidence checks at N=3."""
    n, T, s, q = 3, 5, .7, 1.2
    def normal(shape):
        value = rng.normal(size=shape)
        return (value+1j*rng.normal(size=shape))/np.sqrt(2) if complex_data else value
    X = np.linalg.qr(normal((n, k)))[0]
    P = X@X.conj().T
    U, V = normal((T, n)), normal((T, n))
    raw = normal((n, n)); A0 = .15*(raw+raw.conj().T)
    rows = []
    def keep(check, error, tolerance=2e-9):
        rows.append(dict(check=check, error=float(error), tolerance=tolerance,
            status='passed' if error <= tolerance else 'failed'))
    for conditional in (False, True):
        label = 'conditional' if conditional else 'marginal'
        Bs, constants = [], []
        direct_error = 0.
        for u, v in zip(U, V):
            vv = v if conditional else None
            R = s*np.eye(n) if conditional else s*np.eye(n)+q*P
            mu = P@v if conditional else np.zeros(n)
            if complex_data:
                Rreal = .5*np.block([[R.real, -R.imag], [R.imag, R.real]])
                direct = multivariate_normal.logpdf(np.r_[u.real, u.imag], mean=np.r_[mu.real, mu.imag], cov=Rreal)
            else:
                direct = multivariate_normal.logpdf(u, mean=mu, cov=R)
            actual = log_likelihood(u, P, s, q, vv, complex_data)
            direct_error = max(direct_error, abs(actual-direct))
            Bs.append(increment(u, s, q, vv, complex_data))
            constants.append(constant(u, k, s, q, vv, complex_data))
        B = np.sum(Bs, axis=0)
        keep(label+'_normalized_likelihood', direct_error)
        tracker = BinghamFilter(n, k, s, tau2=q, A0=A0, complex_data=complex_data)
        for u, v in zip(U, V):
            tracker.update(u, v=v if conditional else None)
        keep(label+'_one_assimilation', np.linalg.norm(tracker.A-center(A0+B), 2))
        _, prior_z = independent_moments(A0, k, complex_data)
        _, posterior_z = independent_moments(A0+B, k, complex_data)
        block = sum(constants)+posterior_z-prior_z
        current, sequential = A0.copy(), 0.
        for const, add in zip(constants, Bs):
            _, z0 = independent_moments(current, k, complex_data)
            _, z1 = independent_moments(current+add, k, complex_data)
            sequential += const+z1-z0
            current += add
        keep(label+'_block_sequential_evidence', abs(block-sequential))
        c = float(np.trace(A0+B).real/n)
        mean, centered_z = independent_moments(center(A0+B), k, complex_data)
        mean2, shifted_z = independent_moments(A0+B+2.3*np.eye(n), k, complex_data)
        keep(label+'_trace_restoration', abs(centered_z+k*c-posterior_z))
        keep(label+'_gauge_evidence', abs(shifted_z-posterior_z-k*2.3))
        keep(label+'_gauge_mean', np.linalg.norm(mean-mean2, 2))
    # The marginal mode must agree with EW PCA at precisely matching memory.
    beta = .84
    tracker = BinghamFilter(n, k, s, tau2=q, beta=beta, complex_data=complex_data)
    scatter = np.zeros((n, n), dtype=U.dtype)
    for u in U:
        tracker.update(u)
        scatter = beta*scatter+np.outer(u, u.conj())
    E = np.linalg.eigh(scatter)[1][:, -k:]
    keep('marginal_mode_EW_PCA', np.linalg.norm(E@E.conj().T-tracker.basis@tracker.basis.conj().T, 2))
    return rows
