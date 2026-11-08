"""Independent normalized Gaussian integration and stationary-block EM checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.optimize import minimize

from lib.block_em import (BlockData, block_posterior, fit_block_em,
                          expected_residual, ordered_variance_update)
from lib.models import increment


def real_reference(data, s, tau2, order=56):
    """Direct Gaussian likelihoods integrated on S²; no Bingham increments."""
    z, w = leggauss(order)
    phi = 2 * np.pi * (np.arange(2 * order) + .5) / (2 * order)
    x = np.stack(np.broadcast_arrays(np.sqrt(1-z[:, None]**2)*np.cos(phi),
        np.sqrt(1-z[:, None]**2)*np.sin(phi), z[:, None]), axis=-1).reshape(-1, 3)
    weights = np.broadcast_to(w[:, None] / (4 * order), (order, 2*order)).ravel()
    P = x[:, :, None] * x[:, None, :]
    if data.k == 2:
        P = np.eye(3) - P
    energy = np.einsum('ij,bji->b', data.boundary_prior, P)
    numerator = np.log(weights) + energy
    for j, u in enumerate(data.U):
        if data.V is None:
            R = s*np.eye(3) + tau2*P
            exponent = np.einsum('i,bij,j->b', u, np.linalg.inv(R), u)
            numerator += -.5*(3*np.log(2*np.pi) + np.linalg.slogdet(R)[1] + exponent)
        else:
            residual = u - P @ data.V[j]
            numerator += -.5*(3*np.log(2*np.pi*s) + np.sum(residual**2, axis=1)/s)
    log_evidence = logsumexp(numerator) - logsumexp(np.log(weights) + energy)
    posterior_weights = np.exp(numerator-logsumexp(numerator))
    return float(log_evidence), np.einsum('b,bij->ij', posterior_weights, P)


def complex_reference(data, s, tau2, order=72):
    """Native CP¹ product quadrature and proper-complex Gaussian density."""
    r, w = leggauss(order)
    r, w = (r+1)/2, w/2
    phi = 2*np.pi*(np.arange(2*order)+.5)/(2*order)
    x = np.stack(np.broadcast_arrays(np.sqrt(r[:, None]),
        np.sqrt(1-r[:, None])*np.exp(1j*phi)), axis=-1).reshape(-1, 2)
    weights = np.broadcast_to(w[:, None]/(2*order), (order, 2*order)).ravel()
    P = x[:, :, None]*x[:, None, :].conj()
    energy = np.einsum('ij,bji->b', data.boundary_prior, P).real
    numerator = np.log(weights)+energy
    for u in data.U:
        R = s*np.eye(2)+tau2*P
        quadratic = np.einsum('i,bij,j->b', u.conj(), np.linalg.inv(R), u).real
        numerator += -(2*np.log(np.pi)+np.linalg.slogdet(R)[1]+quadratic)
    evidence = logsumexp(numerator)-logsumexp(np.log(weights)+energy)
    weights = np.exp(numerator-logsumexp(numerator))
    return float(evidence), np.einsum('b,bij->ij', weights, P)


def test_real_fixed_boundary_evidence_and_mean_direct_gaussian_quadrature():
    rng = np.random.default_rng(891)
    A0 = np.array([[.5, .2, .1], [.2, -.3, .0], [.1, .0, .1]])
    for k in (1, 2):
        for conditional in (False, True):
            V = rng.normal(size=(5, 3))*.5 if conditional else None
            data = BlockData(rng.normal(size=(5, 3))*.5, k, V, boundary_prior=A0)
            signal = None if conditional else .8
            result = block_posterior(data, .7, signal)
            evidence, mean = real_reference(data, .7, signal)
            tight_evidence, tight_mean = real_reference(data, .7, signal, 80)
            assert_allclose(evidence, tight_evidence, atol=2e-11)
            assert_allclose(mean, tight_mean, atol=2e-11)
            assert_allclose(result.log_evidence, evidence, atol=2e-10)
            assert_allclose(result.mean, mean, atol=2e-10)
            assert result.numerical_ok


def test_native_complex_evidence_not_real_sphere_measure():
    U = np.array([[.6+.4j, -.1+.3j], [.2-.1j, .5+.2j], [-.1+.7j, .2-.4j]])
    A0 = np.array([[.4, .2+.1j], [.2-.1j, -.1]])
    data = BlockData(U, 1, complex_data=True, boundary_prior=A0)
    native = block_posterior(data, .4, .9)
    evidence, mean = complex_reference(data, .4, .9)
    assert_allclose(native.log_evidence, evidence, atol=2e-10)
    assert_allclose(native.mean, mean, atol=2e-10)


def test_block_once_only_assimilation_and_prior_trace_gauge():
    rng = np.random.default_rng(892)
    U = rng.normal(size=(8, 3)); A0 = np.diag([.3, -.2, .1])
    data = BlockData(U, 1, boundary_prior=A0)
    original = data.boundary_prior.copy()
    result = block_posterior(data, .4, .8)
    sequential = A0.copy()
    for u in U:
        sequential += increment(u, .4, .8)
    sequential -= np.trace(sequential)/3*np.eye(3)
    assert_allclose(result.A, sequential, atol=1e-12)
    shifted = block_posterior(BlockData(U, 1, boundary_prior=A0+19*np.eye(3)), .4, .8)
    assert_allclose(shifted.A, result.A, atol=1e-12)
    assert_allclose(shifted.mean, result.mean, atol=1e-11)
    assert_allclose(shifted.log_evidence, result.log_evidence, atol=1e-11)
    fit = fit_block_em(data, .6, .4, iterations=6)
    rebuilt = block_posterior(data, fit.s, fit.tau2)
    assert_allclose(rebuilt.A, fit.posterior.A, atol=1e-12)
    assert_allclose(data.boundary_prior, original, atol=0, rtol=0)


def test_ordered_variance_update_matches_constrained_numerical_maximization():
    for hp, hn, floor in [(1., .2, .01), (.1, .9, .01), (.08, .02, .1), (2., .02, .1)]:
        n, k = 5, 2
        s, tau2 = ordered_variance_update(hp, hn, n, k, floor)
        def objective(z):
            return k*(np.log(z[0]+z[1])+hp/(z[0]+z[1]))+(n-k)*(np.log(z[0])+hn/z[0])
        numerical = minimize(objective, [.4, .4], bounds=[(floor, None), (0., None)],
                             method='L-BFGS-B', options={'ftol': 1e-14, 'gtol': 1e-9})
        assert objective([s, tau2]) <= numerical.fun + 2e-8
    assert ordered_variance_update(0., 0., 3, 1, .2) == (.2, 0.)


def test_uncertainty_residual_and_zero_signal_absorbing_boundary():
    V = np.array([[1., 2., -.5], [.2, -.3, .7]])
    U = np.array([[.1, -.1, .2], [.2, .1, -.1]])
    data = BlockData(U, 1, V)
    M = np.diag([.6, .3, .1])
    direct = sum(np.linalg.norm(u)**2 - 2*u@M@v + v@M@v for u, v in zip(U, V))
    assert_allclose(expected_residual(data, M), direct, atol=1e-13)
    assert direct > np.linalg.norm(U-V@M.T)**2
    marginal = BlockData(U, 1)
    fit = fit_block_em(marginal, 1., 0., iterations=10)
    assert fit.tau2 == 0.
    assert all(r['tau2'] == 0 for r in fit.history)
    assert_allclose(fit.s, np.mean(U**2), atol=1e-13)


def test_exact_block_em_ascent_real_complex_and_rank_two():
    rng = np.random.default_rng(893)
    for complex_data, k in [(False, 1), (False, 2), (True, 1)]:
        X = np.linalg.qr(rng.normal(size=(3, 3)) + (1j*rng.normal(size=(3, 3)) if complex_data else 0))[0][:, :k]
        z = rng.normal(size=(18, k))
        e = rng.normal(size=(18, 3))
        if complex_data:
            z = (z+1j*rng.normal(size=z.shape))/np.sqrt(2)
            e = (e+1j*rng.normal(size=e.shape))/np.sqrt(2)
        U = z@X.T + np.sqrt(.2)*e
        data = BlockData(U, k, complex_data=complex_data)
        fit = fit_block_em(data, .8, .3, iterations=12)
        assert fit.accepted > 0 and fit.rejected == 0
        assert np.min(np.diff(fit.objectives)) >= -1e-8
        assert fit.s > 0 and fit.tau2 >= 0
    V = rng.normal(size=(16, 3))
    U = V@np.diag([1., 1., 0.]) + .2*rng.normal(size=V.shape)
    conditional = BlockData(U, 2, V)
    fit = fit_block_em(conditional, .5, iterations=12)
    assert fit.accepted > 0 and fit.rejected == 0
    assert np.min(np.diff(fit.objectives)) >= -1e-8


def test_block_rejects_unsupported_ranks_and_complex_casts():
    with pytest.raises(ValueError):
        BlockData(np.ones((2, 4)), 2)
    with pytest.raises(ValueError):
        BlockData(np.ones((2, 3), complex), 1)
    with pytest.raises(ValueError):
        block_posterior(BlockData(np.ones((2, 3)), 1, np.ones((2, 3))), .3, .4)
