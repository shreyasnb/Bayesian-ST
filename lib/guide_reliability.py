"""Calibration and scale-learning checks for the implementation guide.

These are deliberately small, reference-quality three-coordinate experiments.
Prior-predictive calibration, fixed-state inclusion and drifting-path inclusion
have different probability interpretations.  Scores are always computed for a
fresh observation excluded from both parameter fitting and state correction.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import beta as beta_distribution, t as student_t

from .block_em import BlockData, block_posterior, fit_block_em, moments
from .models import constant, increment
from .geometry import center
from .contact_mechanics import solve_contact_velocity
from .contact_posterior import normal_quadrature, _angular_geometry, _polar_integral, _ResponseCDF
from .beamforming_numerics import sample_complex_bingham


def configuration(profile):
    if profile not in ('smoke', 'paper'):
        raise ValueError('Use smoke or paper')
    paper = profile == 'paper'
    return dict(profile=profile, source_commit='6e78cc9911e7e78452633f13dd947afa669c3470',
        n=3, horizon=16, models=['beamforming', 'contact'],
        regimes=['prior_predictive', 'fixed_state', 'drifting'],
        validation_episodes=32 if paper else 3, test_episodes=128 if paper else 3,
        adaptation_episodes=24 if paper else 3, mismatch_episodes=32 if paper else 3,
        nominal=[.5, .8, .95], target_coverage=.95,
        width_multipliers=[.6, .8, 1., 1.2, 1.5, 2., 3.],
        posterior_draws=4096 if paper else 1024,
        real_orders=[[48, 96], [64, 192]],
        validation_seed=2026110901, test_seed=2026110902,
        adaptation_seed=2026110903, mismatch_seed=2026110904,
        seed_offset=0 if paper else 100000,
        s=.2, tau2=1., drift_angle=.8, block_length=24,
        em_iterations=25, em_starts=[[.15, .4], [.6, 1.5]], variance_floor=1e-5,
        moment_tolerance=1e-10, state_mass_tolerance=.035,
        quantile_resolution_tolerance=.006,
        mismatch=dict(desired_contamination_power=.5, secondary_interference_power=.35,
            friction_coefficient=.35, anisotropic_damping=[1., 2., 4.],
            position_noise_sigma=.001, position_dt=.05,
            position_error_samples=30000 if paper else 6000),
        precision='128 independent test episodes give binomial MCSE <=0.0193 at coverage .95; exact binomial intervals and achieved widths are retained. Three smoke episodes check execution only.',
        variance_convention='s is variance per real coordinate, or proper-complex covariance; real/imaginary beam noise has variance s/2.',
        scope='N=3 native-complex marginal lines and real conditional codimension-one constraints; no general-rank or physical controller claim.')


def unit(x):
    x = np.asarray(x)
    return x / np.linalg.norm(x)


def gaussian(rng, shape, complex_data=False):
    if complex_data:
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    return rng.normal(size=shape)


def episode_seed(base, cfg, model, regime_index, episode):
    """Allocate disjoint integer ranges to stages, models and regimes."""
    return ((int(base) + cfg['seed_offset']) * 100000
            + 10000 * cfg['models'].index(model) + 1000 * regime_index + episode)


def projector(h, model):
    P = np.outer(h, h.conj())
    return P if model == 'beamforming' else np.eye(len(h)) - P


def path(h, length, angle):
    # A deterministic path conditional on its prescribed start; signal/noise
    # repetitions never alter this path.  Calibration's prior is Haar only in
    # the separately named prior_predictive regime.
    axis = np.eye(len(h), dtype=h.dtype)[:, np.argmin(abs(h))]
    tangent = unit(axis - h * np.vdot(h, axis))
    theta = np.linspace(0., angle, length)
    return np.cos(theta)[:, None] * h + np.sin(theta)[:, None] * tangent


def data_episode(model, regime, rng, horizon, s=.2, tau2=1., drift_angle=.8):
    complex_data = model == 'beamforming'
    h = (unit(gaussian(rng, 3, complex_data)) if regime == 'prior_predictive'
         else unit(np.array([1., .5, -.3], dtype=complex if complex_data else float)))
    H = path(h, horizon + 1, drift_angle if regime == 'drifting' else 0.)
    P = np.array([projector(row, model) for row in H])
    if complex_data:
        coefficients = np.sqrt(tau2) * gaussian(rng, horizon + 1, True)
        U = H * coefficients[:, None] + np.sqrt(s) * gaussian(rng, (horizon + 1, 3), True)
        V = None
        k = 1
    else:
        # Requests are recorded, independently prescribed physical inputs in
        # the ideal unit-damping linear-constraint observation control.
        V = gaussian(rng, (horizon + 1, 3))
        V /= np.linalg.norm(V, axis=1)[:, None]
        U = np.einsum('bij,bj->bi', P, V) + np.sqrt(s) * gaussian(rng, (horizon + 1, 3))
        k = 2
    block = BlockData(U[:-1], k, None if V is None else V[:-1], complex_data)
    return block, U[-1], None if V is None else V[-1], P[-1], H[-1]


def logscore(A, u, s, tau2=None, v=None, k=1, complex_data=False):
    """Normalized mixture score before the supplied observation is assimilated."""
    A = np.asarray(A)
    B = increment(u, s, tau2, v, complex_data)
    _, psi0, d0 = moments(A, k, complex_data)
    _, psi1, d1 = moments(A + B, k, complex_data)
    return float(constant(u, k, s, tau2, v, complex_data) + psi1 - psi0), bool(d0['numerical_ok'] and d1['numerical_ok'])


def gaussian_logscore(u, mean, covariance, complex_data=False):
    delta = np.asarray(u) - mean
    sign, ld = np.linalg.slogdet(covariance)
    if sign.real <= 0 or abs(sign.imag if np.iscomplexobj(sign) else 0) > 1e-10:
        raise ArithmeticError('Predictive covariance is not positive definite')
    quadratic = np.vdot(delta, np.linalg.solve(covariance, delta)).real
    f = 1. if complex_data else .5
    return float(-f * (len(delta) * np.log(np.pi if complex_data else 2 * np.pi) + ld.real + quadratic))


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    x = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cdf = (np.cumsum(w) - .5 * w) / np.sum(w)
    return np.interp(probabilities, cdf, x, left=x[0], right=x[-1])


def normal_mixture_quantile(means, variances, weights, probabilities):
    """Continuous scalar Gaussian-mixture quantiles; node states are integrated."""
    means = np.asarray(means, float)
    sd = np.sqrt(np.broadcast_to(variances, means.shape))
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    lo, hi = float(np.min(means - 11 * sd)), float(np.max(means + 11 * sd))
    return np.array([brentq(lambda x: float(weights @ ndtr((x - means) / sd) - p),
                           lo, hi, xtol=1e-11) for p in probabilities])


def real_region_thresholds(C, masses, angular_order):
    """Continuous radial integration for real RP² balls and HPD regions.

    Thresholds of discrete quadrature-node distances are deliberately avoided:
    radial rings can otherwise produce large spurious credible-mass errors.
    """
    values = np.linalg.eigvalsh(C)[::-1]
    gaps = np.maximum(0., values[0] - values[1:])
    _, _, g, weights = _angular_geometry(gaps, angular_order)
    normalizer = float(weights @ _polar_integral(g, 0.))
    def ball_cdf(distance):
        return float(weights @ _polar_integral(g, np.sqrt(max(0., 1 - distance))) / normalizer)
    def hpd_cdf(loss):
        cutoff = np.sqrt(np.maximum(0., 1 - np.divide(loss, g, out=np.full_like(g, np.inf), where=g > 0)))
        return float(weights @ _polar_integral(g, cutoff) / normalizer)
    radius = np.array([brentq(lambda x: ball_cdf(x) - p, 0., 1., xtol=1e-12) for p in masses])
    if gaps[-1] == 0:
        # Uniform-density ties use the same auxiliary modal ball.
        return radius, radius.copy(), ball_cdf, ball_cdf
    else:
        threshold = np.array([brentq(lambda x: hpd_cdf(x) - p, 0., float(gaps[-1]), xtol=1e-12) for p in masses])
    return radius, threshold, ball_cdf, hpd_cdf


def posterior_nodes(A, model, cfg, rng):
    """Independent complex batches or two declared real quadrature resolutions."""
    if model == 'beamforming':
        first = sample_complex_bingham(A, cfg['posterior_draws'], rng)
        second = sample_complex_bingham(A, cfg['posterior_draws'], rng)
        nodes, check = first.vectors, second.vectors
        weights = np.full(len(nodes), 1 / len(nodes))
        check_weights = np.full(len(check), 1 / len(check))
        numerical = first.diagnostics['numerical_ok'] and second.diagnostics['numerical_ok']
    else:
        low = normal_quadrature(-A, *cfg['real_orders'][0])
        high = normal_quadrature(-A, *cfg['real_orders'][1])
        nodes, weights = high['normal_nodes'], high['weights']
        check, check_weights = low['normal_nodes'], low['weights']
        numerical = (np.linalg.norm(high['mean'] - low['mean'], 2) < 1e-7
                     and abs(high['log_normalizer'] - low['log_normalizer']) < 1e-7)
    return nodes, weights, check, check_weights, bool(numerical)


def response_values(nodes, v, model):
    rank_one = nodes * (nodes.conj() @ v)[:, None]
    return rank_one if model == 'beamforming' else v - rank_one


def summarize_episode(model, regime, stage, episode, cfg):
    seed_base = cfg['validation_seed'] if stage == 'validation' else cfg['test_seed']
    seed = episode_seed(seed_base, cfg, model, cfg['regimes'].index(regime), episode)
    rng = np.random.default_rng(seed)
    block, future, request, Ptrue, htrue = data_episode(model, regime, rng, cfg['horizon'], cfg['s'], cfg['tau2'], cfg['drift_angle'])
    tau2 = cfg['tau2'] if model == 'beamforming' else None
    post = block_posterior(block, cfg['s'], tau2)
    A, M = post.A, post.mean
    nodes, weights, check, check_weights, numerical = posterior_nodes(A, model, cfg, rng)
    normal_A = A if model == 'beamforming' else -A
    values, vectors = np.linalg.eigh(normal_A)
    mode = vectors[:, -1]
    distance = np.maximum(0., 1 - abs(nodes.conj() @ mode) ** 2)
    check_distance = np.maximum(0., 1 - abs(check.conj() @ mode) ** 2)
    loss = values[-1] - np.einsum('bi,ij,bj->b', nodes.conj(), normal_A, nodes).real
    check_loss = values[-1] - np.einsum('bi,ij,bj->b', check.conj(), normal_A, check).real
    actual_distance = max(0., 1 - abs(np.vdot(mode, htrue)) ** 2)
    actual_loss = float(values[-1] - np.vdot(htrue, normal_A @ htrue).real)
    if np.ptp(values) == 0:
        actual_loss = actual_distance
        loss, check_loss = distance.copy(), check_distance.copy()
    query = unit(gaussian(rng, 3, model == 'beamforming'))
    responses = response_values(nodes, query, model)
    check_responses = response_values(check, query, model)
    true_response = Ptrue @ query
    if model == 'beamforming':
        clean = abs(nodes.conj() @ query) ** 2
        clean_check = abs(check.conj() @ query) ** 2
        clean_truth = float(np.vdot(query, Ptrue @ query).real)
        predictive_means = np.zeros(len(nodes))
        predictive_variances = (cfg['s'] + cfg['tau2'] * abs(nodes[:, 0]) ** 2) / 2
        check_predictive_means = np.zeros(len(check))
        check_predictive_variances = (cfg['s'] + cfg['tau2'] * abs(check[:, 0]) ** 2) / 2
        noisy_truth = float(future[0].real)
        covariance = cfg['s'] * np.eye(3) + cfg['tau2'] * M
        surrogate = gaussian_logscore(future, np.zeros(3), covariance, True)
        predicted_clean_mean = float(np.vdot(query, M @ query).real)
    else:
        clean = responses[:, 0]
        clean_check = check_responses[:, 0]
        clean_truth = float(true_response[0])
        # Fresh noisy observation has its own recorded request; do not use the
        # reporting query in its likelihood or predictive mean.
        future_responses = response_values(nodes, request, model)
        predictive_means = future_responses[:, 0]
        predictive_variances = np.full(len(nodes), cfg['s'])
        check_predictive_means = response_values(check, request, model)[:, 0]
        check_predictive_variances = np.full(len(check), cfg['s'])
        noisy_truth = float(future[0])
        predictive_mean = M @ request
        residual = future_responses - predictive_mean
        covariance = cfg['s'] * np.eye(3) + np.einsum('b,bi,bj->ij', weights, residual, residual)
        surrogate = gaussian_logscore(future, predictive_mean, covariance)
        predicted_clean_mean = float((M @ query)[0])
    mixture, score_ok = logscore(A, future, cfg['s'], tau2, request, block.k, block.complex_data)
    predicted_risk = float(np.vdot(query, (M - M @ M) @ query).real)
    realized_risk = float(np.linalg.norm(M @ query - true_response) ** 2)
    raw = []
    if model == 'contact':
        regions = real_region_thresholds(normal_A, cfg['nominal'], 192)
        reference_regions = real_region_thresholds(normal_A, cfg['nominal'], 384)
        probabilities = np.ravel([[(1 - p) / 2, (1 + p) / 2] for p in cfg['nominal']])
        continuous_clean = _ResponseCDF(normal_A, 128).quantiles(query[None], probabilities)[0, 0].reshape(-1, 2)
        continuous_clean_ref = _ResponseCDF(normal_A, 256).quantiles(query[None], probabilities)[0, 0].reshape(-1, 2)
    for nominal_index, nominal in enumerate(cfg['nominal']):
        q = [(1 - nominal) / 2, (1 + nominal) / 2]
        if model == 'contact':
            radius, threshold = float(regions[0][nominal_index]), float(regions[1][nominal_index])
            mass_error = max(abs(reference_regions[2](radius) - nominal),
                             abs(reference_regions[3](threshold) - nominal))
            clean_quantiles, clean_low = continuous_clean[nominal_index], continuous_clean_ref[nominal_index]
        else:
            radius = float(weighted_quantile(distance, weights, [nominal])[0])
            threshold = float(weighted_quantile(loss, weights, [nominal])[0])
            mass_error = max(abs(float(check_weights @ (check_distance <= radius)) - nominal),
                             abs(float(check_weights @ (check_loss <= threshold)) - nominal))
            clean_quantiles = weighted_quantile(clean, weights, q)
            clean_low = weighted_quantile(clean_check, check_weights, q)
        clean_error = float(np.max(abs(clean_quantiles - clean_low)))
        clean_cdf_error = float(np.max(abs(np.array([check_weights @ (clean_check <= value) for value in clean_quantiles]) - q)))
        noisy_quantiles = normal_mixture_quantile(predictive_means, predictive_variances, weights, q)
        noisy_cdf_error = float(np.max(abs(np.array([check_weights @ ndtr((value - check_predictive_means) / np.sqrt(check_predictive_variances)) for value in noisy_quantiles]) - q)))
        sd_surrogate = float(np.sqrt(covariance[0, 0].real / (2 if model == 'beamforming' else 1)))
        mean_surrogate = 0. if model == 'beamforming' else float((M @ request)[0])
        from scipy.special import ndtri
        gaussian_quantiles = mean_surrogate + sd_surrogate * ndtri(q)
        # Finite independent MC precision is separate from mean resolution.
        mass_allowance = cfg['state_mass_tolerance'] if model == 'contact' else max(cfg['state_mass_tolerance'], 5 * np.sqrt(2 * nominal * (1 - nominal) / len(nodes)))
        clean_allowance = (cfg['quantile_resolution_tolerance'] if model == 'contact'
                           else max(5 * np.sqrt(2 * np.asarray(q) * (1 - np.asarray(q)) / len(nodes))) + 2 / len(nodes))
        common = dict(model=model, regime=regime, stage=stage, episode=episode, seed=seed,
                      t=cfg['horizon'], nominal=nominal, mass_error=mass_error,
                      quantile_error=clean_error, clean_cdf_error=clean_cdf_error,
                      predictive_cdf_error=noisy_cdf_error, model_correct=regime != 'drifting')
        for kind, truth, lower, upper, midpoint in [
            ('state_ball', actual_distance, 0., radius, 0.),
            ('state_hpd', actual_loss, 0., threshold, 0.),
            ('clean_response', clean_truth, *clean_quantiles, predicted_clean_mean),
            ('noisy_observation', noisy_truth, *noisy_quantiles,
             float(weights @ predictive_means))]:
            resolved = numerical and post.numerical_ok and score_ok
            if kind.startswith('state'):
                resolved &= mass_error <= mass_allowance
            elif kind == 'clean_response':
                resolved &= (clean_error if model == 'contact' else clean_cdf_error) <= clean_allowance
            elif kind == 'noisy_observation':
                resolved &= noisy_cdf_error <= cfg['quantile_resolution_tolerance']
            raw.append(dict(**common, method='posterior', kind=kind, covered=bool(lower <= truth <= upper),
                width=float(upper - lower), lower=float(lower), upper=float(upper),
                truth=float(truth), midpoint=float(midpoint), numerical_ok=bool(resolved)))
        raw.append(dict(**common, method='Gaussian predictive surrogate', kind='noisy_observation',
            covered=bool(gaussian_quantiles[0] <= noisy_truth <= gaussian_quantiles[1]),
            width=float(np.diff(gaussian_quantiles)[0]), lower=float(gaussian_quantiles[0]),
            upper=float(gaussian_quantiles[1]), truth=noisy_truth, midpoint=mean_surrogate,
            numerical_ok=bool(numerical and post.numerical_ok)))
    score = dict(model=model, regime=regime, stage=stage, episode=episode, seed=seed,
        t=cfg['horizon'], mixture_log_score=mixture, gaussian_log_score=surrogate,
        risk_predicted=predicted_risk, risk_realized=realized_risk,
        true_s=cfg['s'], true_tau2=tau2, numerical_ok=bool(numerical and post.numerical_ok and score_ok),
        model_correct=regime != 'drifting')
    return raw, score


def interval(values, confidence=.95):
    x = np.asarray(values, float)
    finite = x[np.isfinite(x)]
    if len(finite) < 2:
        return float(np.mean(finite)) if len(finite) else np.nan, np.nan, np.nan
    mean = float(finite.mean())
    half = float(student_t.ppf((1 + confidence) / 2, len(finite) - 1) * finite.std(ddof=1) / np.sqrt(len(finite)))
    return mean, mean - half, mean + half


def binomial_interval(successes, trials, confidence=.95):
    alpha = (1 - confidence) / 2
    return (0. if not successes else float(beta_distribution.ppf(alpha, successes, trials - successes + 1)),
            1. if successes == trials else float(beta_distribution.ppf(1 - alpha, successes + 1, trials - successes)))


def adaptation_episode(model, change, episode, cfg):
    seed = episode_seed(cfg['adaptation_seed'], cfg, model, int(change == 'motion_only'), episode)
    rng = np.random.default_rng(seed)
    complex_data = model == 'beamforming'
    h = unit(gaussian(rng, 3, complex_data))
    rows = []
    for block_index in range(2):
        s = .8 if change == 'noise_only' and block_index else cfg['s']
        angle = cfg['drift_angle'] if change == 'motion_only' and block_index else 0.
        L = cfg['block_length']
        H = path(h, L + 1, angle)
        P = np.array([projector(x, model) for x in H])
        tau2 = cfg['tau2'] if complex_data else None
        if complex_data:
            U = H * gaussian(rng, L + 1, True)[:, None] + np.sqrt(s) * gaussian(rng, (L + 1, 3), True)
            V = None
        else:
            V = gaussian(rng, (L + 1, 3))
            V /= np.linalg.norm(V, axis=1)[:, None]
            U = np.einsum('bij,bj->bi', P, V) + np.sqrt(s) * gaussian(rng, (L + 1, 3))
        data = BlockData(U[:-1], 1 if complex_data else 2, None if V is None else V[:-1], complex_data)
        fitted = []
        starts = []
        for initial_s, initial_tau in cfg['em_starts']:
            fit = fit_block_em(data, initial_s, initial_tau if complex_data else None,
                iterations=cfg['em_iterations'], floor=cfg['variance_floor'])
            fitted.append(fit)
            starts.append(dict(initial_s=initial_s, initial_tau2=initial_tau if complex_data else None,
                               objective=fit.objectives[-1], accepted=fit.accepted, rejected=fit.rejected))
        fit = max(fitted, key=lambda value: value.objectives[-1])
        initial = block_posterior(data, cfg['s'], tau2)
        ev, Q = np.linalg.eigh(initial.A)
        modal = Q[:, -data.k:] @ Q[:, -data.k:].conj().T
        if complex_data:
            S = U[:-1].T @ U[:-1].conj() / L
            sn = max(float(np.trace((np.eye(3) - modal) @ S).real / 2), cfg['variance_floor'])
            qn = max(0., float(np.trace(modal @ S).real - sn))
        else:
            sn = max(float(np.linalg.norm(U[:-1] - V[:-1] @ modal.T) ** 2 / (3 * L)), cfg['variance_floor'])
            qn = None
        policies = [('known', s, tau2, None), ('misspecified', .05, .2 if complex_data else None, None),
                    ('block_em', fit.s, fit.tau2, fit), ('residual', sn, qn, None)]
        for method, estimated_s, estimated_tau, selected in policies:
            post = fit.posterior if selected is not None else block_posterior(data, estimated_s, estimated_tau)
            score, ok = logscore(post.A, U[-1], estimated_s, estimated_tau,
                                 None if V is None else V[-1], data.k, complex_data)
            query = unit(gaussian(np.random.default_rng(seed + block_index + 1100000), 3, complex_data))
            loss = float(np.linalg.norm((post.mean - P[-1]) @ query) ** 2)
            rows.append(dict(model=model, change=change, episode=episode, seed=seed, block=block_index,
                method=method, true_s=s, fitted_s=estimated_s, true_tau2=tau2,
                fitted_tau2=estimated_tau, prequential_score=score, response_loss=loss,
                accepted=0 if selected is None else selected.accepted,
                rejected=0 if selected is None else selected.rejected,
                converged=None if selected is None else selected.converged,
                log_evidence=post.log_evidence, numerical_ok=bool(post.numerical_ok and ok),
                model_correct=not bool(angle), starts=starts if selected is not None else None,
                scoring_scope='Fresh observation excluded from block fitting and state correction; intervals, if reported, condition on fitted scales.'))
    return rows


def scaled_contact(force, normal, damping, sensor_covariance):
    """Independent KKT mechanics and exact scaled-coordinate/noise identities."""
    D = np.asarray(damping, float)
    values, U = np.linalg.eigh(D)
    if values.min() <= 0:
        raise ValueError('Damping must be positive definite')
    root = (U * np.sqrt(values)) @ U.T
    inverse_root = (U / np.sqrt(values)) @ U.T
    velocity, multiplier = solve_contact_velocity(force, normal, D)
    scaled_normal = unit(inverse_root @ normal)
    projected = (np.eye(3) - np.outer(scaled_normal, scaled_normal)) @ (inverse_root @ force)
    return dict(velocity=velocity, scaled_velocity=root @ velocity, projected_velocity=projected,
        scaled_normal=scaled_normal, scaled_sensor_covariance=root @ sensor_covariance @ root,
        projection_residual=float(np.linalg.norm(root @ velocity - projected)), multiplier=multiplier)


def friction_force(force, normal, coefficient):
    """Unit-damping bilateral Coulomb slip/stick force, opposing actual motion.

    The tangential friction magnitude cannot exceed the available tangential
    force.  The sticking branch prevents an opposing-force correction from
    reversing the velocity and injecting energy.
    """
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Nonnegative friction coefficient required')
    force = np.asarray(force, float)
    normal = np.asarray(normal, float)
    if (force.shape != (3,) or normal.shape != (3,) or not np.isfinite(force).all()
            or not np.isfinite(normal).all() or not np.isclose(normal @ normal, 1.)):
        raise ValueError('Finite three-vector force and unit normal required')
    tangent = force - normal * (normal @ force)
    length = float(np.linalg.norm(tangent))
    if length == 0:
        return force.copy()
    magnitude = min(length, coefficient * abs(normal @ force))
    return force - (magnitude / length) * tangent


def mismatch_episode(model, episode, cfg):
    seed = episode_seed(cfg['mismatch_seed'], cfg, model, 0, episode)
    rng = np.random.default_rng(seed)
    L = cfg['block_length']
    h = unit(gaussian(rng, 3, model == 'beamforming'))
    P = projector(h, model)
    rows = []
    if model == 'beamforming':
        other = unit(gaussian(rng, 3, True))
        a = unit(np.array([1., 1., 1.], complex))
        common = (h * gaussian(rng, L + 1, True)[:, None]
                  + np.sqrt(cfg['s']) * gaussian(rng, (L + 1, 3), True))
        extra = gaussian(rng, L + 1, True)
        for stress, vector, power in [('matched', other, 0.),
            ('desired_contamination', a, cfg['mismatch']['desired_contamination_power']),
            ('unequal_rank_two_powers', other, cfg['mismatch']['secondary_interference_power'])]:
            U = common + np.sqrt(power) * extra[:, None] * vector
            data = BlockData(U[:-1], 1, complex_data=True)
            post = block_posterior(data, cfg['s'], cfg['tau2'])
            score, ok = logscore(post.A, U[-1], cfg['s'], cfg['tau2'], k=1, complex_data=True)
            rows.append(dict(model=model, stress=stress, episode=episode, seed=seed,
                response_loss=float(np.linalg.norm(post.mean - P, 'fro') ** 2), logscore=score,
                numerical_ok=bool(post.numerical_ok and ok), model_correct=not bool(power),
                mechanics_residual=0., sensor_covariance_error=0., lag_one_covariance=np.nan))
    else:
        F = gaussian(rng, (L + 1, 3))
        noise = np.sqrt(cfg['s']) * gaussian(rng, (L + 1, 3))
        for stress in ['matched', 'anisotropic_damping', 'friction', 'contact_loss']:
            D = np.diag(cfg['mismatch']['anisotropic_damping']) if stress == 'anisotropic_damping' else np.eye(3)
            clean = []
            maximum = 0.
            for force in F:
                if stress == 'contact_loss':
                    velocity = force
                else:
                    applied = force.copy()
                    if stress == 'friction':
                        applied = friction_force(force, h, cfg['mismatch']['friction_coefficient'])
                    velocity, reaction = solve_contact_velocity(applied, h, D)
                    residual = max(np.linalg.norm(D @ velocity - applied - h * reaction), abs(h @ velocity))
                    maximum = max(maximum, float(residual))
                clean.append(velocity)
            U = np.asarray(clean) + noise
            data = BlockData(U[:-1], 2, V=F[:-1])
            post = block_posterior(data, cfg['s'])
            score, ok = logscore(post.A, U[-1], cfg['s'], v=F[-1], k=2)
            rows.append(dict(model=model, stress=stress, episode=episode, seed=seed,
                response_loss=float(np.linalg.norm(post.mean @ F[-1] - clean[-1]) ** 2),
                logscore=score, numerical_ok=bool(post.numerical_ok and ok),
                model_correct=stress == 'matched', mechanics_residual=maximum,
                sensor_covariance_error=0., lag_one_covariance=np.nan))
    return rows


def sensor_diagnostics(cfg):
    rng = np.random.default_rng(cfg['mismatch_seed'] + cfg['seed_offset'] + 50000)
    D = np.diag(cfg['mismatch']['anisotropic_damping'])
    n = unit(np.array([1., 2., -1.]))
    force = np.array([.4, -.2, .8])
    original_isotropic = scaled_contact(force, n, D, cfg['s'] * np.eye(3))
    transformed_isotropic = scaled_contact(force, n, D, cfg['s'] * np.linalg.inv(D))
    T = cfg['mismatch']['position_error_samples']
    sigma, dt = cfg['mismatch']['position_noise_sigma'], cfg['mismatch']['position_dt']
    positions = sigma * rng.normal(size=(T + 1, 3))
    errors = np.diff(positions, axis=0) / dt
    empirical_variance = float(np.mean(errors ** 2))
    empirical_lag = float(np.mean(errors[:-1] * errors[1:]))
    return dict(scaled_projection_residual=original_isotropic['projection_residual'],
        original_isotropic_transformed_covariance=original_isotropic['scaled_sensor_covariance'].tolist(),
        original_isotropic_expected_transformed_covariance=(cfg['s'] * D).tolist(),
        physical_inverse_damping_transformed_covariance=transformed_isotropic['scaled_sensor_covariance'].tolist(),
        physical_inverse_damping_expected_transformed_covariance=(cfg['s'] * np.eye(3)).tolist(),
        position_difference_samples=T, position_noise_sigma=sigma, dt=dt,
        differenced_noise_variance=empirical_variance, theoretical_variance=2 * sigma ** 2 / dt ** 2,
        lag_one_covariance=empirical_lag, theoretical_lag_one_covariance=-sigma ** 2 / dt ** 2,
        warning='Independent isotropic position errors create temporally correlated velocity errors. Known anisotropic damping with isotropic physical velocity noise produces anisotropic scaled noise; it is not the isotropic conjugate conditional likelihood.')
