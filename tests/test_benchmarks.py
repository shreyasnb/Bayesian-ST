"""Functional checks of reference optimization and reconstruction behavior."""
import numpy as np
import pytest

from benchmarks.case1 import LoadedCovariance, mvdr, output_power, sample_output_power
from benchmarks.case2 import EWCovariancePCA, RollingPCA, GROUSE, reconstruct


def test_mvdr_is_distortionless_and_minimizes_feasible_power():
    rng = np.random.default_rng(23)
    z = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    covariance = z @ z.conj().T + np.eye(8)
    steering = rng.normal(size=8) + 1j * rng.normal(size=8)
    weight = mvdr(covariance, steering)
    assert abs(np.vdot(weight, steering) - 1) < 1e-12
    minimum = output_power(weight, covariance)
    # Perturb only in directions preserving the gain constraint.
    for _ in range(20):
        tangent = rng.normal(size=8) + 1j * rng.normal(size=8)
        tangent -= steering * np.vdot(steering, tangent) / np.vdot(steering, steering)
        assert output_power(weight + tangent, covariance) >= minimum - 1e-10
    evaluation = (rng.normal(size=(20000, 8)) + 1j * rng.normal(size=(20000, 8))) / np.sqrt(2)
    evaluation = evaluation @ np.linalg.cholesky(covariance).T
    assert abs(sample_output_power(weight, evaluation) / minimum - 1) < 0.04


def test_covariance_loading_and_frame_transport():
    rng = np.random.default_rng(24)
    observations = rng.normal(size=(30, 6))
    motion = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    original = LoadedCovariance(6, beta=.93, loading=.2)
    transported = LoadedCovariance(6, beta=.93, loading=.2)
    for observation in observations:
        original.update(observation)
        transported.update(motion @ observation)
    original.transform(motion)
    np.testing.assert_allclose(original.covariance, transported.covariance, atol=1e-12)
    assert np.linalg.eigvalsh(original.covariance)[0] > 0
    assert np.linalg.eigvalsh(LoadedCovariance(6).covariance)[0] > 0


@pytest.mark.parametrize("factory", [lambda: EWCovariancePCA(6, 2, beta=1),
                                    lambda: RollingPCA(6, 2, window=64)])
def test_covariance_references_recover_noiseless_span(factory):
    rng = np.random.default_rng(25)
    truth = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    tracker = factory()
    for coefficients in rng.normal(size=(64, 2)):
        basis = tracker.update(truth @ coefficients)
    np.testing.assert_allclose(basis @ basis.T, truth @ truth.T, atol=1e-11)
    observation = truth @ np.array([1.3, -.4])
    np.testing.assert_allclose(reconstruct(observation, basis, .2, .8), .8 * observation, atol=1e-12)


def test_grouse_geodesic_reduces_loss_and_tracks_fixed_span():
    rng = np.random.default_rng(26)
    tracker = GROUSE(6, 1, step=.08, rng=27)
    truth = rng.normal(size=6)
    truth /= np.linalg.norm(truth)
    initial_error = 1 - abs(np.vdot(tracker.basis[:, 0], truth)) ** 2
    for _ in range(200):
        before = np.linalg.norm(truth - tracker.basis @ (tracker.basis.T @ truth)) ** 2
        basis = tracker.update(truth)
        after = np.linalg.norm(truth - basis @ (basis.T @ truth)) ** 2
        assert after <= before + 1e-13
        np.testing.assert_allclose(basis.T @ basis, np.eye(1), atol=1e-12)
    assert 1 - abs(np.vdot(basis[:, 0], truth)) ** 2 < initial_error * 1e-6


def test_rolling_pca_forgets_outdated_span():
    tracker = RollingPCA(4, 1, window=3)
    for _ in range(8):
        tracker.update(np.eye(4)[0])
    for _ in range(3):
        basis = tracker.update(np.eye(4)[1])
    np.testing.assert_allclose(basis @ basis.T, np.diag([0, 1, 0, 0]), atol=1e-12)
