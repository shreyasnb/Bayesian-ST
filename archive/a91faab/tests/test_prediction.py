"""Geometry, causality and likelihood bookkeeping for prediction controls."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from lib.filter import BinghamFilter
from lib.geometry import center, projector
from lib.models import increment
from lib.prediction import (PredictiveBinghamFilter, geodesic_generator,
                            geodesic_forecast, mean_observation_age)
from benchmarks.case2 import EWCovariancePCA


def orthogonal(rng, shape, complex_data=False):
    A = rng.normal(size=shape)
    if complex_data:
        A = A + 1j * rng.normal(size=shape)
    return np.linalg.qr(A)[0]


@pytest.mark.parametrize('complex_data', [False, True])
def test_geodesic_maps_subspaces_and_is_basis_invariant(complex_data):
    rng = np.random.default_rng(402)
    Y = orthogonal(rng, (8, 3), complex_data)
    # Generate a horizontal tangent with safely sub-cut-locus angles.
    Q = orthogonal(rng, (8, 8), complex_data)
    R = (np.eye(8) - projector(Y)) @ Q[:, :3]
    Q = np.linalg.qr(R)[0]
    theta = np.array([.04, .2, .45])
    L = (Q * theta) @ Y.conj().T - (Y * theta) @ Q.conj().T
    Z = expm(L) @ Y
    calculated, angles = geodesic_generator(Y, Z)
    assert_allclose(calculated, L, atol=2e-13)
    assert_allclose(expm(calculated) @ projector(Y) @ expm(calculated).conj().T,
                    projector(Z), atol=2e-13)
    U = orthogonal(rng, (3, 3), complex_data)
    V = orthogonal(rng, (3, 3), complex_data)
    transformed, _ = geodesic_generator(Y @ U, Z @ V)
    assert_allclose(transformed, calculated, atol=3e-13)
    f = geodesic_forecast(Y @ U, Z @ V, lag=4, damping=.5)
    assert_allclose(f.rotation.conj().T @ f.rotation, np.eye(8), atol=2e-13)
    assert_allclose(f.rotation @ projector(Z) @ f.rotation.conj().T,
                    expm(1.125 * L) @ projector(Y) @ expm(1.125 * L).conj().T,
                    atol=3e-13)


def test_geodesic_small_and_zero_angles_and_noiseless_extrapolation():
    Y = np.eye(6, 2)
    L = np.zeros((6, 6));L[2, 0] = .06;L[0, 2] = -.06
    Z = expm(4 * L) @ Y
    forecast = geodesic_forecast(Y, Z, lag=4, damping=1)
    assert_allclose(forecast.rotation @ projector(Z) @ forecast.rotation.T,
                    expm(5 * L) @ projector(Y) @ expm(5 * L).T, atol=1e-13)
    very_small, _ = geodesic_generator(Y, expm(1e-9 * L) @ Y)
    assert_allclose(very_small, 1e-9 * L, atol=1e-15)
    stationary, _ = geodesic_generator(Y, Y @ np.array([[0., -1.], [1., 0.]]))
    assert_allclose(stationary, np.zeros((6, 6)), atol=1e-15)


def test_cut_locus_zero_damping_and_explicit_speed_cap():
    Y = np.eye(4, 1);Z = np.eye(4)[:, 1:2]
    with pytest.raises(ValueError, match='cut locus'):
        geodesic_generator(Y, Z)
    assert_allclose(geodesic_forecast(Y, Z, damping=0).rotation, np.eye(4), atol=0)
    L = np.zeros((4, 4));L[1, 0] = .4;L[0, 1] = -.4
    forecast = geodesic_forecast(Y, expm(L) @ Y, speed_cap=.05)
    assert forecast.capped and forecast.speed == .05
    assert_allclose(forecast.rotation, expm(L / 8), atol=1e-14)
    with pytest.raises(ValueError):
        geodesic_forecast(Y, Z, lag=0)


@pytest.mark.parametrize('conditional', [False, True])
@pytest.mark.parametrize('complex_data', [False, True])
def test_rolling_queue_matches_full_recomputation_and_stored_prior(conditional, complex_data):
    rng = np.random.default_rng(403)
    f = PredictiveBinghamFilter(6, 2, .3, None if conditional else 1.5,
                                prediction='rolling', likelihood_window=4,
                                window=16, complex_data=complex_data)
    increments = []
    for _ in range(12):
        u = rng.normal(size=6)
        v = rng.normal(size=6) if conditional else None
        if complex_data:
            u = u + 1j * rng.normal(size=6)
            if v is not None:
                v = v + 1j * rng.normal(size=6)
        prior_expected = center(sum(increments[-3:], np.zeros((6, 6))))
        prior = f.predict()
        assert_allclose(prior, prior_expected, atol=2e-13)
        assert_allclose(f.predict(), prior, atol=0)
        f.update(u, v=v)
        increments.append(increment(u, .3, None if conditional else 1.5, v, complex_data))
        assert_allclose(f.A, center(sum(increments[-4:], np.zeros((6, 6)))), atol=3e-13)
        assert_allclose(f.A, f.recompute_window(), atol=3e-13)
        assert_allclose(f.records[-1].prior, prior, atol=0)
        saved = f.records[-1].prior.copy()
        f.A[0, 0] += 1.
        assert_allclose(f.records[-1].prior, saved, atol=0)
        f.A[0, 0] -= 1.
    assert len(f.increments) == 4
    with pytest.raises(ValueError):
        f.predict(G=np.eye(6))
    with pytest.raises(ValueError):
        PredictiveBinghamFilter(6, 2, .3, 1.5, beta=.9, prediction='rolling')
    f.s = .4
    with pytest.raises(ValueError, match='fixed noise'):
        f.update(np.ones(6), v=np.ones(6) if conditional else None)


@pytest.mark.parametrize('conditional', [False, True])
def test_exponential_recursion_and_zero_damping_reproduce_original(conditional):
    rng = np.random.default_rng(404)
    reference = BinghamFilter(6, 2, .3, None if conditional else 1.5, beta=.97)
    identity = PredictiveBinghamFilter(6, 2, .3, None if conditional else 1.5, beta=.97)
    velocity = PredictiveBinghamFilter(6, 2, .3, None if conditional else 1.5,
                                       beta=.97, prediction='velocity', lag=4, damping=0)
    for _ in range(20):
        u = rng.normal(size=6);v = rng.normal(size=6) if conditional else None
        reference.update(u, v=v);identity.update(u, v=v);velocity.update(u, v=v)
        assert_allclose(identity.A, reference.A, atol=0)
        assert_allclose(velocity.A, reference.A, atol=0)
        assert_allclose(identity.records[-1].prior, reference.records[-1].prior, atol=0)
    assert_allclose(mean_observation_age(.97), 32.3333333333333)
    assert np.isinf(mean_observation_age(1.))


def test_velocity_prediction_is_causal_and_prior_is_idempotent():
    rng = np.random.default_rng(405)
    f = PredictiveBinghamFilter(6, 2, .3, 1.5, beta=.9,
                               prediction='velocity', lag=4, damping=.5)
    for _ in range(10):
        f.update(rng.normal(size=6))
    a, b = deepcopy(f), deepcopy(f)
    pa, pb = a.predict(), b.predict()
    assert_allclose(pa, pb, atol=0)
    assert_allclose(pa, a.predict(), atol=0)
    assert a.motion_diagnostics['reason'] == 'forecast'
    a.update(np.ones(6));b.update(np.arange(6.))
    assert_allclose(a.records[-1].prior, b.records[-1].prior, atol=0)
    assert not np.allclose(a.A, b.A)
    assert_allclose(a.last_motion.conj().T @ a.last_motion, np.eye(6), atol=1e-13)
    assert a.last_update_seconds > 0 and a.prediction_seconds > 0


@pytest.mark.parametrize('complex_data', [False, True])
def test_matching_marginal_ew_pca_shares_causal_or_supplied_mode(complex_data):
    rng = np.random.default_rng(406)
    n, k = 6, 2
    f = PredictiveBinghamFilter(n, k, .3, 1.5, beta=.9,
                               prediction='velocity', lag=4, damping=.5,
                               complex_data=complex_data)
    ew = EWCovariancePCA(n, k, beta=.9)
    for t in range(18):
        u = rng.normal(size=n)
        if complex_data:
            u = u + 1j * rng.normal(size=n)
        f.predict();G = f.last_motion.copy()
        ew.transform(G);ew.update(u);f.update(u)
        if t >= k:
            assert_allclose(projector(f.basis), projector(ew.basis), atol=1e-11)


def test_supplied_rotation_is_used_before_one_correction():
    rng = np.random.default_rng(407)
    f = PredictiveBinghamFilter(6, 2, .3, 1.5, beta=.95)
    original = BinghamFilter(6, 2, .3, 1.5, beta=.95)
    for _ in range(8):
        G = orthogonal(rng, (6, 6));u = rng.normal(size=6)
        f.predict(G=G);f.update(u, G=G);original.update(u, G=G)
        assert_allclose(f.A, original.A, atol=0)
    assert f.t == 8 and len(f.records) == 8
