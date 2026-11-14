from dataclasses import asdict
from time import perf_counter
import numpy as np
from .causal_variance import CausalVarianceTracker
from .block_em import moments
from .models import increment, constant

def gaussian_logscore(u, mean, covariance, complex_data=False):
    delta = np.asarray(u) - mean
    sign, ld = np.linalg.slogdet(covariance)
    if sign.real <= 0 or abs(sign.imag if np.iscomplexobj(sign) else 0) > 1e-10:
        raise ArithmeticError('Predictive covariance is not positive definite')
    quadratic = np.vdot(delta, np.linalg.solve(covariance, delta)).real
    f = 1. if complex_data else .5
    return float(-f * (len(delta) * np.log(np.pi if complex_data else 2 * np.pi) + ld.real + quadratic))


def seed(cfg, base, example, episode, condition_index=0):
    return np.random.SeedSequence([int(cfg[base]) + cfg['seed_offset'], 0, int(condition_index), int(episode)])

def tracker_kwargs(example, n, policy, cfg, *, learn, calibration=False):
    if calibration:
        sigma2, tau2, floor = (0.05, 0.2, cfg['calibration_floor'])
    else:
        sigma2, tau2, floor = (0.05, 0.2, cfg['beam_floor'])
    return dict(n=n, k=1, sigma2=sigma2, tau2=tau2, beta=policy['beta'], block_length=policy['block_length'], em_iterations=policy['em_iterations'], starts=None, floor=floor, complex_data=example == 'beamforming', learn=learn, moment_tolerance=cfg['moment_tolerance'], max_order=cfg['max_order'], ascent_tolerance=cfg['ascent_tolerance'])

def same_mode_controls(A, M, k):
    values, Q = np.linalg.eigh(A)
    X = Q[:, -k:]
    P = X @ X.conj().T
    n = len(A)
    alpha = float(np.trace(P @ M).real / k)
    xi = float((k - k * alpha) / (n - k))
    Mi = alpha * P + xi * (np.eye(n) - P)
    return (P, Mi, float(values[-k] - values[-k - 1]))

def accuracy_fields(d, cfg):
    quadrature = 'resolution_error' in d
    return dict(moment_backend=d.get('backend', 'unknown'), mean_accuracy_estimate=float(d.get('resolution_error', d.get('moment_error_bound', np.nan))), normalizer_accuracy_estimate=float(d.get('resolution_error', d.get('log_normalizer_error_bound', np.nan))), accuracy_tolerance=cfg['real_quadrature_resolution_tolerance'] if quadrature else cfg['moment_tolerance'], accuracy_scope='Quadrature resolution difference' if quadrature else 'Exact-arithmetic series truncation')

def mvdr(R, a):
    z = np.linalg.solve(R, a)
    return z / np.vdot(a, z)

def axial_mvdr(P, a, sigma2, coefficient):
    """Closed-form inverse for sigma2 I + coefficient P."""
    z = a / sigma2 - coefficient / (sigma2 * (sigma2 + coefficient)) * (P @ a)
    return z / np.vdot(a, z)

def beam_actions(M, P, Mi, a, sigma2, tau2):
    n = len(M)
    alpha = float(np.trace(P @ Mi).real)
    xi = float((np.trace(Mi).real - alpha) / (n - 1))
    return dict(mean=mvdr(sigma2 * np.eye(n) + tau2 * M, a), mode=axial_mvdr(P, a, sigma2, tau2), iso=axial_mvdr(P, a, sigma2 + tau2 * xi, tau2 * (alpha - xi)))

def physical_task_rows(example, stream, t, tr, cfg, rng, common):
    """Future-only task rules use the pre-correction parameter at time t."""
    G = stream.get('G', None)
    g = None if G is None else G[t]
    A = tr.predict_parameter(g)
    start = perf_counter()
    M, psi0, diag = moments(A, tr.k, tr.complex_data, tr.moment_tolerance, tr.max_order)
    mean_seconds = perf_counter() - start
    P, Mi, gap = same_mode_controls(A, M, tr.k)
    truth = stream['P'][t]
    sigma2 = float(np.asarray(stream['sigma2'])[t])
    tau2 = float(np.asarray(stream['tau2'])[t])
    rows = []
    a = stream['a']
    n = len(a)
    h = stream['h'][t]
    held = common
    oracle = axial_mvdr(truth, a, sigma2, tau2)
    oracle_loss = float(sigma2 * np.vdot(oracle, oracle).real + tau2 * abs(np.vdot(oracle, h)) ** 2)
    start = perf_counter()
    actions = beam_actions(M, P, Mi, a, tr.sigma2, tr.tau2)
    action_seconds = perf_counter() - start
    for control, w in actions.items():
        true = float(sigma2 * np.vdot(w, w).real + tau2 * abs(np.vdot(w, h)) ** 2)
        heldloss = float(np.mean(abs(held @ w.conj()) ** 2))
        rows.append(dict(control=control, future_task_loss=heldloss, true_task_loss=true, oracle_loss=oracle_loss, output_oracle_db=float(10 * np.log10(true / oracle_loss)), heldout_output_oracle_db=float(10 * np.log10(heldloss / oracle_loss)), chordal_error=float(np.linalg.norm(P - truth, 'fro') / np.sqrt(2)), distortionless_error=float(abs(np.vdot(w, a) - 1)), response_mse=np.nan, command_cost=np.nan, action_seconds=action_seconds))
    for row in rows:
        row.update(mean_ok=bool(diag['numerical_ok']), mode_gap=gap, mean_seconds=mean_seconds, **accuracy_fields(diag, cfg))
    return (rows, dict(A=A, M=M, P=P, Mi=Mi, truth=truth, log_normalizer=psi0, mean_numerical_ok=bool(diag['numerical_ok'])))

def adaptation_trial(example, stream, policies, cfg, episode, split, condition, *, story=False, method_subset=None):
    """Run fixed/learned variances on identical measurements and query draws."""
    if example != 'beamforming':
        raise ValueError('This release runner supports beamforming only')
    U = stream['U']
    T, n = U.shape
    methods = ['known', 'misspecified', 'learned'] if method_subset is None else list(method_subset)
    trackers = {name: CausalVarianceTracker(**tracker_kwargs(example, n, policies[name], cfg, learn=name == 'learned')) for name in methods}
    rows, events, arrays = ([], [], {})
    rng = np.random.default_rng(seed(cfg, 'test_seed' if split == 'test' else 'validation_seed', example, episode, 100 + cfg['beam_conditions'].index(condition)))
    report_times = set(range(cfg['primary_report_every'] - 1, T, cfg['primary_report_every']))
    for t, u in enumerate(U):
        g = None if stream.get('G') is None else stream['G'][t]
        v = None
        common = None
        if t in report_times:
            common = stream['heldout'][t]
        for method, tr in trackers.items():
            if method == 'known':
                tr.sigma2 = float(stream['sigma2'][t])
                tr.tau2 = float(stream['tau2'][t])
            pending = []
            if t in report_times:
                try:
                    pending, state = physical_task_rows(example, stream, t, tr, cfg, rng, common)
                    started = perf_counter()
                    if split == 'validation':
                        score, score_ok = (np.nan, True)
                    else:
                        try:
                            add = increment(u, tr.sigma2, tr.tau2, v, tr.complex_data)
                            _, psi1, score_diag = moments(state['A'] + add, tr.k, tr.complex_data, tr.moment_tolerance, tr.max_order)
                            score = float(constant(u, tr.k, tr.sigma2, tr.tau2, v, tr.complex_data) + psi1 - state['log_normalizer'])
                            score_ok = bool(score_diag['numerical_ok'] and state['mean_numerical_ok'])
                        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                            score, score_ok = (np.nan, False)
                    score_seconds = perf_counter() - started
                    surrogate = gaussian_logscore(u, np.zeros(n), tr.sigma2 * np.eye(n) + tr.tau2 * state['M'], True)
                    for row in pending:
                        row.update(example=example, condition=condition, episode=episode, split=split, method=method, t=t + 1, time_s=t * stream['dt'], sigma2_true=float(stream['sigma2'][t]), tau2_true=float(stream['tau2'][t]), sigma2_estimate=tr.sigma2, tau2_estimate=tr.tau2, prequential_score=score, gaussian_score=surrogate, score_numerical_ok=score_ok, score_seconds=score_seconds, beta=tr.beta, block_length=tr.block_length, em_iterations=tr.em_iterations, prediction_scope='Before assimilation of the scored observation', variance_information='True scales supplied' if method == 'known' else 'Recorded past observations only')
                    if story and method == 'learned':
                        prefix = f'{example}_{condition}_t{t + 1}_'
                        arrays.update({prefix + key: value for key, value in state.items()})
                except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
                    pending = [dict(example=example, condition=condition, episode=episode, split=split, method=method, control='mean', t=t + 1, time_s=t * stream['dt'], sigma2_true=float(stream['sigma2'][t]), sigma2_estimate=tr.sigma2, tau2_true=float(stream['tau2'][t]), tau2_estimate=tr.tau2, mean_ok=False, score_numerical_ok=False, future_task_loss=np.nan, true_task_loss=np.nan, prequential_score=np.nan, error=type(error).__name__ + ': ' + str(error))]
            record = tr.update(u, v, g)
            if pending:
                for row in pending:
                    row.update(sigma2_next=record.sigma2_next, tau2_next=record.tau2_next, fit_performed=record.fit_performed, fit_numerical_ok=record.fit_numerical_ok, fitting_seconds=record.fitting_seconds, correction_seconds=record.correction_seconds)
                rows.extend(pending)
            if record.fit_performed:
                event = dict(example=example, condition=condition, episode=episode, split=split, method=method, t=t + 1, time_s=t * stream['dt'], **asdict(record), history=tr.fit_history[-1])
                events.append(event)
        if story:
            prefix = f'{example}_{condition}_'
            arrays[prefix + 'U'] = U
            arrays[prefix + 'sigma2_true'] = np.asarray(stream['sigma2'])
            arrays[prefix + 'tau2_true'] = np.asarray(stream['tau2'])
            arrays[prefix + 'h'] = stream['h']
            arrays[prefix + 'a'] = stream['a']
    if story:
        prefix = f'{example}_{condition}_'
        for method, tr in trackers.items():
            arrays[prefix + method + '_sigma2_used'] = np.array([x.sigma2_used for x in tr.records])
            arrays[prefix + method + '_sigma2_next'] = np.array([x.sigma2_next for x in tr.records])
            arrays[prefix + method + '_fit_performed'] = np.array([x.fit_performed for x in tr.records])
            arrays[prefix + method + '_tau2_used'] = np.array([x.tau2_used for x in tr.records])
            arrays[prefix + method + '_tau2_next'] = np.array([x.tau2_next for x in tr.records])
    return (rows, events, arrays)
