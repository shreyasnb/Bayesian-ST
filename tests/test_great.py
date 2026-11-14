"""Independent geometric and streaming checks for the GREAT reference."""
import numpy as np
import pytest
from scipy.linalg import expm

from benchmarks.common.great import (
    GREAT, grassmann_exp, residual_objective, riemannian_gradient)


def normal(rng, shape, complex_data=False):
    z = rng.normal(size=shape)
    return (z + 1j * rng.normal(size=shape)) / np.sqrt(2) if complex_data else z


@pytest.mark.parametrize('complex_data', [False, True])
def test_gradient_matches_direct_residual_directional_derivative(complex_data):
    rng = np.random.default_rng(11001)
    X = np.linalg.qr(normal(rng, (7, 2), complex_data))[0]
    W = normal(rng, (7, 11), complex_data)
    C = W @ W.conj().T
    D = normal(rng, X.shape, complex_data)
    D -= X @ (X.conj().T @ D)
    D /= np.linalg.norm(D)
    def direct(Y):
        return float(np.linalg.norm(W - Y @ (Y.conj().T @ W), 'fro') ** 2)
    h = 1e-6
    numerical = (direct(grassmann_exp(X, h * D))
                 - direct(grassmann_exp(X, -h * D))) / (2 * h)
    analytic = np.vdot(riemannian_gradient(X, C), D).real
    assert abs(numerical - analytic) < 2e-7
    assert abs(residual_objective(X, C) - direct(X)) < 2e-12


@pytest.mark.parametrize('complex_data', [False, True])
@pytest.mark.parametrize('n,k', [(8, 3), (3, 2)])
def test_exponential_matches_ambient_generator_and_basis_invariance(complex_data, n, k):
    rng = np.random.default_rng(11002)
    X = np.linalg.qr(normal(rng, (n, k), complex_data))[0]
    D = .1 * normal(rng, X.shape, complex_data)
    D -= X @ (X.conj().T @ D)
    # Ambient skew-Hermitian generator gives an independent exponential path.
    K = D @ X.conj().T - X @ D.conj().T
    expected = expm(K) @ X
    actual = grassmann_exp(X, D)
    assert np.allclose(actual, expected, atol=2e-13)
    Q = np.linalg.qr(normal(rng, (k, k), complex_data))[0]
    changed = grassmann_exp(X @ Q, D @ Q)
    assert np.allclose(actual @ actual.conj().T, changed @ changed.conj().T, atol=3e-13)
    assert np.allclose(actual.conj().T @ actual, np.eye(k), atol=3e-13)
    assert np.allclose(grassmann_exp(X, np.zeros_like(X)), X, atol=2e-13)


def test_real_restriction_and_finite_gradient_descent_without_eigensolve(monkeypatch):
    rng = np.random.default_rng(11003)
    U = .2 * rng.normal(size=(16, 6))
    X = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    real = GREAT(6, 2, stepsize=.01, inner_steps=3, window=8,
                 initial_basis=X, record_objective=True)
    complex_tracker = GREAT(6, 2, stepsize=.01, inner_steps=3, window=8,
                            complex_data=True, initial_basis=X, record_objective=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('GREAT must not optimize by covariance eigensolve')
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    monkeypatch.setattr(np.linalg, 'eig', forbidden)
    for u in U:
        real.update(u)
        complex_tracker.update(u)
        assert np.allclose(real.projector, complex_tracker.projector, atol=2e-12)
        assert np.all(np.diff(real.step_losses) <= 2e-12)
    assert real.total_gradient_steps == 3 * len(U)
    assert real.descent_violations == 0
    # Finite steps need not solve the new objective exactly at each snapshot.
    C = real.covariance
    assert np.linalg.norm(riemannian_gradient(real.basis, C)) > 1e-3


@pytest.mark.parametrize('complex_data', [False, True])
def test_sliding_window_rank_two_update_transports_all_history(complex_data):
    rng = np.random.default_rng(11004)
    n, T, window = 5, 13, 4
    X = np.linalg.qr(normal(rng, (n, 2), complex_data))[0]
    tracker = GREAT(n, 2, stepsize=.001, inner_steps=2, window=window,
                    complex_data=complex_data, initial_basis=X)
    explicit = []
    for t in range(T):
        # Noncommuting coordinate changes; the test reconstructs the window
        # independently, including old measurements which leave the window.
        G = np.linalg.qr(normal(rng, (n, n), complex_data))[0]
        u = normal(rng, n, complex_data)
        explicit = [G @ old for old in explicit] + [u]
        explicit = explicit[-window:]
        tracker.update(u, G=G)
        W = np.array(explicit).T
        expected = W @ W.conj().T
        assert np.allclose(tracker.covariance, expected, atol=5e-12)
        assert tracker.effective_mass == min(t + 1, window)
        assert np.allclose(tracker.basis.conj().T @ tracker.basis, np.eye(2), atol=3e-12)


@pytest.mark.parametrize('complex_data', [False, True])
def test_ew_weights_match_gamma_squared_and_transported_objective(complex_data):
    rng = np.random.default_rng(11005)
    n, T, beta = 4, 9, .73
    X = np.linalg.qr(normal(rng, (n, 1), complex_data))[0]
    tracker = GREAT(n, 1, stepsize=.001, inner_steps=1, beta=beta,
                    complex_data=complex_data, initial_basis=X)
    explicit = []
    for t in range(T):
        G = np.linalg.qr(normal(rng, (n, n), complex_data))[0]
        u = normal(rng, n, complex_data)
        explicit = [G @ old for old in explicit] + [u]
        tracker.update(u, G)
        ages = np.arange(t, -1, -1)
        W = np.array(explicit).T * tracker.gamma ** ages
        assert np.allclose(tracker.covariance, W @ W.conj().T, atol=4e-12)
        assert abs(tracker.effective_mass - np.sum(beta ** ages)) < 2e-13
        # gamma=beta would square the desired covariance discount.
        wrong = np.array(explicit).T * beta ** ages
        if t > 1:
            assert np.linalg.norm(tracker.covariance - wrong @ wrong.conj().T) > .01
    assert tracker.diagnostics()['descent_monitored'] is False
    assert tracker.diagnostics()['descent_violations'] is None


def test_invalid_controls_and_fields_rejected():
    with pytest.raises(ValueError):
        GREAT(3, 1, window=4, beta=.9)
    with pytest.raises(ValueError):
        GREAT(3, 1, beta=0.)
    with pytest.raises(ValueError):
        GREAT(3, 1, window=0)
    with pytest.raises(ValueError):
        GREAT(3, 1, inner_steps=0)
    tracker = GREAT(3, 1, beta=.9)
    with pytest.raises(ValueError):
        tracker.update(np.array([1., 1j, 0.]))
    with pytest.raises(ValueError):
        tracker.update(np.ones(3), G=2 * np.eye(3))
    with pytest.raises(ValueError):
        grassmann_exp(np.eye(3)[:, :1], np.ones((3, 1)))
