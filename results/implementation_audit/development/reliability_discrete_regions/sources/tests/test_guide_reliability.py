import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from lib.guide_reliability import (configuration, episode_seed, logscore,
    normal_mixture_quantile, scaled_contact, response_values, binomial_interval)
from lib.contact_spherical_reference import sphere_grid


def test_real_preobservation_mixture_score_matches_direct_gaussian_integration():
    nodes, weights = sphere_grid(80, 160)
    u = np.array([.4, -.2, .1])
    v = np.array([.3, .8, -.1])
    s = .4
    A = np.diag([.4, -.2, -.2])
    # Direct density of the plane projector, without likelihood increments.
    planes = np.eye(3)[None] - np.einsum('bi,bj->bij', nodes, nodes)
    energies = np.einsum('ij,bji->b', A, planes)
    means = np.einsum('bij,j->bi', planes, v)
    likelihood = np.array([multivariate_normal.logpdf(u, mean=m, cov=s * np.eye(3)) for m in means])
    expected = logsumexp(np.log(weights) + energies + likelihood) - logsumexp(np.log(weights) + energies)
    before = A.copy()
    score, ok = logscore(A, u, s, v=v, k=2)
    assert ok
    np.testing.assert_allclose(score, expected, atol=2e-10, rtol=0)
    np.testing.assert_array_equal(A, before)
    shifted, _ = logscore(A + 7 * np.eye(3), u, s, v=v, k=2)
    np.testing.assert_allclose(shifted, score, atol=2e-10, rtol=0)


def test_complex_preobservation_score_matches_direct_covariance_integral():
    u = np.array([.2 + .4j, -.1 + .3j, .7 - .2j])
    s, tau2 = .3, .8
    x, w = leggauss(128)
    r, weights = (x + 1) / 2, w * (1 - (x + 1) / 2)
    # Haar complex N=3 gives |h^H u/||u|||² ~ Beta(1,2).
    # Rotate u to e1, then evaluate full proper-complex Gaussian covariances.
    rotated_u = np.array([np.linalg.norm(u), 0, 0], complex)
    logs = []
    for overlap in r:
        h = np.array([np.sqrt(overlap), np.sqrt(1 - overlap), 0.], complex)
        R = s * np.eye(3) + tau2 * np.outer(h, h.conj())
        logs.append(-3 * np.log(np.pi) - np.linalg.slogdet(R)[1]
                    - np.vdot(rotated_u, np.linalg.solve(R, rotated_u)).real)
    expected = logsumexp(np.log(weights) + logs)
    score, ok = logscore(np.zeros((3, 3), complex), u, s, tau2, k=1, complex_data=True)
    assert ok
    np.testing.assert_allclose(score, expected, atol=2e-11, rtol=0)


def test_known_damping_changes_sensor_law_as_well_as_coordinates():
    D = np.array([[2., .2, .1], [.2, 3., -.1], [.1, -.1, 1.]])
    n = np.array([1., 2., -1.]) / np.sqrt(6)
    force = np.array([.3, -.5, .8])
    isotropic = scaled_contact(force, n, D, .2 * np.eye(3))
    conjugate = scaled_contact(force, n, D, .2 * np.linalg.inv(D))
    assert isotropic['projection_residual'] < 2e-14
    assert conjugate['projection_residual'] < 2e-14
    np.testing.assert_allclose(isotropic['scaled_sensor_covariance'], .2 * D, atol=1e-14)
    np.testing.assert_allclose(conjugate['scaled_sensor_covariance'], .2 * np.eye(3), atol=1e-14)
    assert not np.allclose(isotropic['scaled_sensor_covariance'], .2 * np.eye(3))


def test_predictive_intervals_are_continuous_mixtures_and_responses_use_hermitian_projection():
    from scipy.stats import norm
    probabilities = [.025, .5, .975]
    result = normal_mixture_quantile(np.array([.7, .7]), np.array([.4, .4]), [.2, .8], probabilities)
    np.testing.assert_allclose(result, .7 + np.sqrt(.4) * norm.ppf(probabilities), atol=2e-10)
    nodes = np.array([[1., 1j, 0.], [0., 1., 1j]]) / np.sqrt(2)
    v = np.array([.2 + .1j, .3 - .4j, -.5 + .2j])
    result = response_values(nodes, v, 'beamforming')
    direct = np.array([np.outer(h, h.conj()) @ v for h in nodes])
    np.testing.assert_allclose(result, direct, atol=1e-14)


def test_stage_and_profile_seed_ranges_do_not_overlap():
    all_profiles = []
    for profile in ['smoke', 'paper']:
        cfg = configuration(profile)
        sets = []
        for stage in ['validation', 'test']:
            base = cfg[stage + '_seed']
            sets.append({episode_seed(base, cfg, model, regime, episode)
                for model in cfg['models'] for regime in range(3)
                for episode in range(cfg[stage + '_episodes'])})
        assert sets[0].isdisjoint(sets[1])
        all_profiles.append(sets[0] | sets[1])
    assert all_profiles[0].isdisjoint(all_profiles[1])
    lower, upper = binomial_interval(0, 128)
    assert lower == 0 and 0 < upper < .03
