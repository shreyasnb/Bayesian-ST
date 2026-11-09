"""Final snapshot-only beamforming cohorts and saved decision records.

Variance parameters are sigma2 and tau2.  Training snapshots are row-wise.
The spatial signatures are simulation/scoring records, never estimator inputs.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from scipy.stats import t as student_t

from .geometry import center
from .beamforming_protocol import proper_complex, haar_line, ula
from .beamforming_numerics import complex_moments
from .beamforming_decisions import (isotropized_mean, scalar_statistics,
    scalar_validation_loss)
from .array_geometry_study import (
    distortionless_weight, covariance_shrinkage, ledoit_wolf,
    scm_statistics, scm_validation_loss, continuous_minimum)
from benchmarks.common.great import GREAT
from benchmarks.case1.baselines import output_power, sample_output_power
from benchmarks.case2.baselines import GROUSE

SOURCE_COMMIT = 'a91faabeb10a63b1d1c613d75cc32503cfc0cac6'
CONDITIONS = ('static', 'smooth', 'abrupt', 'noise_change', 'power_change')
METHODS = ('mean', 'mode', 'isotropic', 'modal_loading', 'covariance_loading',
           'great_window', 'great_ew', 'grouse', 'ledoit_wolf', 'oracle')
METHOD_INFORMATION = {
    'mean': 'Snapshots, stated variances, desired steering only for the action',
    'mode': 'Same parameter, variance, memory and data as the posterior mean; coefficient-weighted PCA, equal to ordinary EW PCA at fixed scales',
    'isotropic': 'Same mode and mean squared chordal distance as the posterior mean; no invented tails',
    'modal_loading': 'Snapshot modal estimate with loading independently selected on validation',
    'covariance_loading': 'Loaded sample covariance with independent validation and the isotropic endpoint',
    'great_window': 'GREAT complex extension; response snapshots only, sliding optimization window',
    'great_ew': 'GREAT complex extension; matched exponential covariance weights, gamma=sqrt(beta)',
    'grouse': 'GROUSE complex complete-observation update, separately named',
    'ledoit_wolf': 'Known-zero-mean Hermitian covariance shrinkage',
    'oracle': 'True interference covariance; evaluation-only oracle',
}


def configuration(stage='verify'):
    if stage not in ('verify', 'pilot', 'paper'):
        raise ValueError('Use verify, pilot or paper')
    paper = stage == 'paper'
    verify = stage == 'verify'
    offset = 20000 if verify else 10000 if stage == 'pilot' else 0
    return dict(schema=1, stage=stage, source_commit=SOURCE_COMMIT, rank=1,
        dimensions=[32, 64, 128], stationary_ratios=[.25, .5, 1., 2.],
        validation_episodes=3 if verify else 8 if stage == 'pilot' else 64,
        test_episodes=3 if verify else 12 if stage == 'pilot' else 256,
        validation_seed=2026111001 + offset, test_seed=2026111002 + offset,
        story_seed=2026111003 + offset, sigma2=.2, tau2=1., overlap=.5,
        moving_n=32, moving_T=64 if verify else 256,
        moving_conditions=list(CONDITIONS), action_cadence=8,
        primary_burnin=16 if verify else 32, dt=.01,
        betas=[1., .99, .97, .9],
        great_step_multipliers=[.02, .08, .24],
        great_inner_steps=[1, 3], great_window_ratios=[.5, 1., 2.],
        grouse_steps=[.001, .01, .1, 1.],
        loading_bracket_points=33,
        heldout_snapshots=64 if verify else 128 if stage == 'pilot' else 256,
        moment_tolerance=1e-10, moment_max_order=4096, moment_fallback_max_order=8192,
        block_lengths=[1, 4, 16, 32], block_test_episodes=3 if verify else 8 if stage == 'pilot' else 64,
        block_n=32, block_T=64, block_beta=.97,
        block_conditions=['static', 'smooth'],
        story=dict(stationary_n=64, stationary_ratio=1., episode=0,
                   moving_episode=0, moving_time=32 if verify else 128,
                   moving_times=[16,32,64] if verify else [32,128,256]),
        outcome='Independent held-out disturbance power divided by true-covariance oracle power; 10 log10 of the episode-averaged ratio for displayed dB outcomes.',
        repetition_unit='Independent episode. Held-out snapshots average inside a checkpoint; moving primary outcomes average checkpoints inside the episode.',
        selection='Independent validation episodes only. Static beta=1. Moving mean/mode/isotropic share a mode-selected beta. Modal loading and loaded covariance select their own memory/loading. Sliding GREAT selects window/step/inner count; EW GREAT matches the shared Bingham beta and selects step/inner count.',
        variance_convention='Proper-complex covariance sigma2*I; real/imaginary component variances sigma2/2. Unit steering signatures give total interference power tau2.',
        motion='Identity transport for all primary unknown-motion comparisons. Baseline half-power desired/interference overlap is held exactly fixed across dimensions.',
        numerical_scope='Native complex rank-one means and normalizers only; no higher-rank, tail, interval or second-moment accuracy is inferred from these mean flags.',
        scale_information='This performance cohort conditions on the generating variances at correction/action time as a calibrated reference. The separate causal variance-learning cohort never receives future generating scales.',
        block_budget='Supplementary cadence/constant-state-block experiment uses 64 independent test episodes per motion at n=32,T=64. Primary stationary/moving performance cohorts retain 256. This smaller block cohort was selected from operation counts/native-mean preflight before any block task outcomes.',
        precision='Paper plan: 64 validation and 256 independent test episodes per primary cohort; all paired pointwise 95% intervals and achieved SEs retained. Broad grids are exploratory without multiplicity correction.')


def matched_overlap_angle(n, overlap=.5):
    """First-lobe ULA angle with a declared squared desired overlap."""
    if n < 2 or not 0 < overlap < 1:
        raise ValueError('Require n>=2 and overlap strictly between zero and one')
    desired = ula(n, 0.)
    return float(brentq(lambda theta: abs(np.vdot(desired, ula(n, theta)))**2-overlap,
                        0., np.arcsin(min(1., 2./n)), xtol=1e-14))


def generate_stream(n, length, condition, seed, sigma2=.2, tau2=1., overlap=.5, dt=.01):
    """Generate interference-only snapshots and separate evaluation records."""
    if condition not in CONDITIONS or n < 2 or length < 1:
        raise ValueError('Unknown condition or invalid dimension/length')
    if sigma2 <= 0 or tau2 <= 0 or dt <= 0:
        raise ValueError('Positive variances and sampling interval required')
    children = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(children[0])
    theta = matched_overlap_angle(n, overlap)
    angles = np.full(length, theta)
    if condition == 'smooth':
        angles = np.linspace(theta, 3*theta, length)
    elif condition == 'abrupt':
        angles[length//2:] = 3*theta
    sigma = np.full(length, sigma2)
    power = np.full(length, tau2)
    if condition == 'noise_change':
        sigma[length//2:] *= 4.
    elif condition == 'power_change':
        power[length//2:] *= 4.
    h = np.array([ula(n, value) for value in angles])
    z = proper_complex(rng, length)*np.sqrt(power)
    e = proper_complex(rng, (length, n))*np.sqrt(sigma)[:, None]
    U = h*z[:, None] + e
    initial = haar_line(np.random.default_rng(children[1]), n)
    return dict(U=U, h=h, a=ula(n, 0.), sigma2=sigma, tau2=power,
        angles=angles, initial=initial, time=np.arange(1, length+1)*dt,
        heldout_seed=children[2], condition=condition, baseline_angle=theta,
        estimator_inputs='U, stated variances and desired steering for decisions; h and true covariance are scoring only')


def beam_episode(cfg, n, condition, trial, split='test', stationary=False, length=None):
    if split not in ('validation', 'test'):
        raise ValueError('Use a disjoint validation or test split')
    seed = cfg['validation_seed'] if split == 'validation' else cfg['test_seed']
    stream_seed = np.random.SeedSequence([seed, n, CONDITIONS.index(condition), int(trial)])
    # SeedSequence is accepted by generate_stream through a reproducible state.
    seed_words = stream_seed.generate_state(4).tolist()
    length = (2*n if stationary else cfg['moving_T']) if length is None else int(length)
    return generate_stream(n, length, condition, seed_words, cfg['sigma2'],
                           cfg['tau2'], cfg['overlap'], cfg['dt'])


def heldout_snapshots(stream, t, draws):
    """Independent fresh disturbances at the present checkpoint."""
    parent = stream['heldout_seed']
    words = parent.generate_state(4).tolist()
    rng = np.random.default_rng(np.random.SeedSequence(words+[int(t)]))
    return (proper_complex(rng, draws, stream['tau2'][t-1])[:, None]*stream['h'][t-1]
            + proper_complex(rng, (draws, stream['U'].shape[1]), stream['sigma2'][t-1]))


def projector_mode(A):
    values, E = np.linalg.eigh(A)
    h = E[:, -1]
    gap = float(values[-1]-values[-2])
    return np.outer(h, h.conj()), h, gap


def modal_weight(h, a, sigma2, tau2, rho=1.):
    """Efficient rank-one modal covariance action, including infinite loading."""
    if not 0 <= rho <= 1:
        raise ValueError('rho must be in [0,1]')
    n = len(a)
    c = sigma2+tau2*(1-rho)/n
    shrink = tau2*rho/(c+tau2*rho)
    overlap = np.vdot(h, a)
    raw = a-shrink*h*overlap
    denominator = np.vdot(a, raw).real
    if denominator <= 0:
        raise ArithmeticError('Invalid positive-definite modal decision')
    w = raw/denominator
    return w/np.conj(np.vdot(w, a))


def covariance_power(w, h, sigma2, tau2):
    return float(sigma2*np.vdot(w, w).real+tau2*abs(np.vdot(w, h))**2)


def checked_moments(A,tol=1e-10,max_order=4096,fallback_max_order=8192):
    """Declared cap extension without relaxing the numerical tolerance.

    The legacy positive-series backend chooses its safe arithmetic branch from
    the cap. Trying its regular cap first avoids unnecessarily using logarithmic
    convolution for moderate n=128 states that already resolve below that cap.
    """
    try:
        result=complex_moments(A,tol,max_order)
        return result,False
    except ArithmeticError:
        if fallback_max_order<=max_order:
            raise
        return complex_moments(A,tol,fallback_max_order),True


def action_from_parameter(A,a,sigma2,tau2,method='mean',tol=1e-10,max_order=4096,
                          fallback_max_order=8192):
    """One production action for cadence timing, without any scoring work.

    Native mean evaluation already includes its eigensystem. A mode-only
    action uses the efficient rank-one inverse identity. Optional uncertainty
    reports and variance fitting are separate operations.
    """
    if method=='mode':
        P,h,gap=projector_mode(A)
        return dict(weight=modal_weight(h,a,sigma2,tau2),projector=P,
                    numerical_ok=True,mode_gap=gap)
    if method not in ('mean','isotropic'):
        raise ValueError('Use mean, mode or isotropic')
    posterior,fallback=checked_moments(A,tol,max_order,fallback_max_order)
    M=posterior.mean
    if method=='isotropic':
        P,h,gap=projector_mode(A)
        M,_=isotropized_mean(M,P)
    return dict(weight=distortionless_weight(sigma2*np.eye(len(A))+tau2*M,a),
                projector=M,numerical_ok=posterior.numerical_ok,
                moment_diagnostics=dict(posterior.diagnostics,cap_extension=fallback),
                log_normalizer=posterior.log_normalizer)


def oracle_power(a, h, sigma2, tau2):
    return covariance_power(modal_weight(h, a, sigma2, tau2), h, sigma2, tau2)


def weighted_statistics(U, sigma2, tau2, beta, times):
    """One assimilation per observation; unnormalized likelihood statistics."""
    n = U.shape[1]
    S = np.zeros((n, n), complex)
    A = np.zeros_like(S)
    mass = 0.
    output = {}
    for t, u in enumerate(U, 1):
        outer = np.outer(u, u.conj())
        s, q = float(sigma2[t-1]), float(tau2[t-1])
        S = beta*S+outer
        A = beta*A+q/(s*(s+q))*outer
        mass = beta*mass+1.
        if t in times:
            output[t] = dict(A=center(A), S=S.copy(), mass=mass)
    return output


def great_grid(cfg, n, sigma2, tau2, beta=1., exponential=False):
    """Fixed step values; objective covariance is never secretly normalized."""
    windows = [None] if exponential else sorted(set(max(1, int(round(n*r)))
                                                   for r in cfg['great_window_ratios']))
    memory = 2*n if beta == 1 else min(2*n, 1/(1-beta))
    output = []
    for window in windows:
        scale = memory if exponential else window
        for multiplier in cfg['great_step_multipliers']:
            for inner in cfg['great_inner_steps']:
                output.append(dict(window=window, beta=beta if exponential else None,
                    stepsize=float(multiplier/(max(1., scale)*(sigma2+tau2))),
                    inner_steps=int(inner), step_multiplier=float(multiplier)))
    return output


def point_paths(U, initial, times, great_parameters=None, grouse_step=None):
    n = U.shape[1]
    if great_parameters is not None:
        tracker = GREAT(n, 1, complex_data=True, initial_basis=initial[:, None],
            **{k: great_parameters[k] for k in ('window', 'beta', 'stepsize', 'inner_steps')})
    elif grouse_step is not None:
        tracker = GROUSE(n, 1, step=grouse_step, rng=0)
        tracker.basis = initial[:, None].copy()
    else:
        raise ValueError('Specify a point-reference algorithm')
    records = {}
    for t, u in enumerate(U, 1):
        X = tracker.update(u)
        if t in times:
            records[t] = np.asarray(X)[:, 0].copy()
    return records


def _candidate_score(lines, data, times):
    losses = []
    for t in times:
        s, q = data['sigma2'][t-1], data['tau2'][t-1]
        w = modal_weight(lines[t], data['a'], s, q)
        losses.append(covariance_power(w, data['h'][t-1], s, q)
                      /oracle_power(data['a'], data['h'][t-1], s, q))
    return float(np.mean(losses))


def validation_case(cfg, n, condition, stationary):
    """Independent validation; no posterior test outcome enters policy selection."""
    times = [int(round(n*r)) for r in cfg['stationary_ratios']] if stationary else list(
        range(cfg['action_cadence'], cfg['moving_T']+1, cfg['action_cadence']))
    scored_times = times if stationary else [t for t in times if t >= cfg['primary_burnin']]
    betas = [1.] if stationary else cfg['betas']
    data = [beam_episode(cfg, n, condition, trial, 'validation', stationary)
            for trial in range(cfg['validation_episodes'])]
    statistics = {beta: {t: [] for t in times} for beta in betas}
    scalar = {beta: {t: [] for t in times} for beta in betas}
    modal_losses = {beta: [] for beta in betas}
    curves = []
    for episode in data:
        for beta in betas:
            snapshots = weighted_statistics(episode['U'], episode['sigma2'], episode['tau2'], beta, times)
            losses = []
            for t, state in snapshots.items():
                P, h, gap = projector_mode(state['A'])
                s, q = episode['sigma2'][t-1], episode['tau2'][t-1]
                R = s*np.eye(n)+q*np.outer(episode['h'][t-1], episode['h'][t-1].conj())
                row = scalar_statistics(P, episode['a'], R)
                scalar[beta][t].append(row)
                statistics[beta][t].append(scm_statistics(state['S']/state['mass'],
                                                         episode['a'], episode['h'][t-1]))
                if t in scored_times:
                    w = modal_weight(h, episode['a'], s, q)
                    losses.append(output_power(w, R)/oracle_power(episode['a'], episode['h'][t-1], s, q))
            modal_losses[beta].append(float(np.mean(losses)))
    shared_beta = min(betas, key=lambda beta: np.mean(modal_losses[beta]))
    for beta in betas:
        curves.append(dict(n=n, condition=condition, stationary=stationary,
            family='shared_mode_memory', beta=beta, t=0, parameter=beta,
            loss=float(np.mean(modal_losses[beta])), validation_episodes=len(data)))
    # Static policies may use each declared count. Moving policies choose one
    # coefficient and memory across all predeclared action times.
    groups = [[t] for t in times] if stationary else [scored_times]
    policies = []
    for group in groups:
        choices_scalar, choices_scm = [], []
        for beta in betas:
            def scalar_loss(rho):
                losses = []
                for t in group:
                    s, q = data[0]['sigma2'][t-1], data[0]['tau2'][t-1]
                    reference=oracle_power(data[0]['a'],data[0]['h'][t-1],s,q)
                    losses.append(scalar_validation_loss(rho, scalar[beta][t], s, q, n)/reference)
                return float(np.mean(losses))
            def covariance_loss(alpha):
                losses = []
                for t in group:
                    s, q = data[0]['sigma2'][t-1], data[0]['tau2'][t-1]
                    reference=oracle_power(data[0]['a'],data[0]['h'][t-1],s,q)
                    losses.append(scm_validation_loss(alpha, statistics[beta][t], s, q)/reference)
                return float(np.mean(losses))
            sf = continuous_minimum(scalar_loss, cfg['loading_bracket_points'])
            cf = continuous_minimum(covariance_loss, cfg['loading_bracket_points'])
            choices_scalar.append((sf['loss'], beta, sf['value']))
            choices_scm.append((cf['loss'], beta, cf['value']))
            for family, fit in [('modal_loading', sf), ('covariance_loading', cf)]:
                curves.extend(dict(n=n, condition=condition, stationary=stationary,
                    family=family, beta=beta, t=group[0] if stationary else 0,
                    parameter=value, loss=loss, validation_episodes=len(data))
                    for value, loss in zip(fit['grid'], fit['losses']))
        _, scalar_beta, rho = min(choices_scalar)
        _, covariance_beta, alpha = min(choices_scm)
        policies.append(dict(times=group if stationary else times,
            scalar_beta=scalar_beta, rho=rho, covariance_beta=covariance_beta,
            alpha=alpha, shared_beta=shared_beta))
    great_selected = {}
    for family, exponential in [('great_window', False), ('great_ew', True)]:
        candidates = great_grid(cfg, n, cfg['sigma2'], cfg['tau2'], shared_beta, exponential)
        candidates_losses = []
        for candidate_id, candidate in enumerate(candidates):
            values = [_candidate_score(point_paths(episode['U'], episode['initial'], times,
                great_parameters=candidate), episode, scored_times) for episode in data]
            loss = float(np.mean(values))
            candidates_losses.append(loss)
            curves.append(dict(n=n, condition=condition, stationary=stationary,
                family=family, beta=candidate['beta'], t=0,
                parameter=candidate['stepsize'], inner_steps=candidate['inner_steps'],
                window=candidate['window'], candidate_id=candidate_id,
                loss=loss, validation_episodes=len(data)))
        index = int(np.argmin(candidates_losses))
        great_selected[family] = candidates[index]
    grouse_losses = []
    for step in cfg['grouse_steps']:
        values = [_candidate_score(point_paths(episode['U'], episode['initial'], times,
            grouse_step=step), episode, scored_times) for episode in data]
        grouse_losses.append(float(np.mean(values)))
        curves.append(dict(n=n, condition=condition, stationary=stationary,
            family='grouse', beta=np.nan, t=0, parameter=step,
            loss=grouse_losses[-1], validation_episodes=len(data)))
    return dict(n=n, condition=condition, stationary=stationary, times=times,
        primary_times=scored_times, policies=policies, shared_beta=shared_beta,
        **great_selected, grouse_step=cfg['grouse_steps'][int(np.argmin(grouse_losses))]), curves


def decision_records(U, sigma2, tau2, a, initial, policy, cfg, return_states=False):
    """Estimate from snapshots only and form common distortionless decisions."""
    n = U.shape[1]
    times = policy['times']
    betas = {policy['shared_beta']}
    for item in policy['policies']:
        betas.update([item['scalar_beta'], item['covariance_beta']])
    snapshots = {beta: weighted_statistics(U, sigma2, tau2, beta, times) for beta in betas}
    great = {name: point_paths(U, initial, times, great_parameters=policy[name])
             for name in ('great_window', 'great_ew')}
    grouse = point_paths(U, initial, times, grouse_step=policy['grouse_step'])
    output = []
    for t in times:
        selected = next(item for item in policy['policies'] if t in item['times'])
        state = snapshots[policy['shared_beta']][t]
        P, h, gap = projector_mode(state['A'])
        s, q = float(sigma2[t-1]), float(tau2[t-1])
        begin = perf_counter()
        try:
            posterior,fallback = checked_moments(state['A'], cfg['moment_tolerance'],
                cfg['moment_max_order'],cfg['moment_fallback_max_order'])
            M = posterior.mean
            Mi, rho = isotropized_mean(M, P)
            mean_ok = bool(posterior.numerical_ok)
            diagnostics = dict(posterior.diagnostics,cap_extension=fallback)
        except ArithmeticError as error:
            M = Mi = np.full_like(P, np.nan)
            mean_ok = False
            diagnostics = dict(numerical_ok=False, error=str(error))
        elapsed = perf_counter()-begin
        Pscalar, hscalar, _ = projector_mode(snapshots[selected['scalar_beta']][t]['A'])
        Sstate = snapshots[selected['covariance_beta']][t]
        Rloaded = covariance_shrinkage(Sstate['S']/Sstate['mass'], selected['alpha'])
        Rlw, lw_alpha = ledoit_wolf(U[:t])
        weights = dict(mode=modal_weight(h, a, s, q),
            modal_loading=modal_weight(hscalar, a, s, q, selected['rho']),
            covariance_loading=distortionless_weight(Rloaded, a),
            great_window=modal_weight(great['great_window'][t], a, s, q),
            great_ew=modal_weight(great['great_ew'][t], a, s, q),
            grouse=modal_weight(grouse[t], a, s, q),
            ledoit_wolf=distortionless_weight(Rlw, a))
        if mean_ok:
            weights.update(mean=distortionless_weight(s*np.eye(n)+q*M, a),
                           isotropic=distortionless_weight(s*np.eye(n)+q*Mi, a))
        else:
            weights.update(mean=np.full(n, np.nan+0j), isotropic=np.full(n, np.nan+0j))
        projectors = dict(mean=M, mode=P, isotropic=Mi,
            modal_loading=selected['rho']*Pscalar+(1-selected['rho'])*np.eye(n)/n,
            great_window=np.outer(great['great_window'][t], great['great_window'][t].conj()),
            great_ew=np.outer(great['great_ew'][t], great['great_ew'][t].conj()),
            grouse=np.outer(grouse[t], grouse[t].conj()))
        point_projectors = dict(mean=P,mode=P,isotropic=P,modal_loading=Pscalar,
            great_window=projectors['great_window'],great_ew=projectors['great_ew'],
            grouse=projectors['grouse'])
        posterior_powers = {method: float(s*np.vdot(w,w).real+q*np.vdot(w,M@w).real)
                            if mean_ok and np.isfinite(w).all() else np.nan
                            for method,w in weights.items()}
        record = dict(t=t, weights=weights, projectors=projectors,
            point_projectors=point_projectors,posterior_powers=posterior_powers,mean_ok=mean_ok,
            mean_diagnostics=diagnostics, mean_seconds=elapsed, mode_gap=gap,
            unique_mode=bool(gap > 1e-10*max(1., np.linalg.norm(state['A'], 'fro'))),
            lw_alpha=lw_alpha, policy=selected)
        if return_states:
            record.update(A=state['A'], M=M, P=P, M_iso=Mi,
                log_normalizer=posterior.log_normalizer if mean_ok else np.nan)
        output.append(record)
    return output


def evaluate_episode(cfg, policy, trial, stationary=False, save_story=False):
    n, condition = policy['n'], policy['condition']
    data = beam_episode(cfg, n, condition, trial, 'test', stationary)
    estimated = decision_records(data['U'], data['sigma2'], data['tau2'], data['a'],
        data['initial'], policy, cfg, save_story)
    rows, stories = [], {}
    for record in estimated:
        t = record['t']
        s, q, h = data['sigma2'][t-1], data['tau2'][t-1], data['h'][t-1]
        Poracle = np.outer(h, h.conj())
        w_oracle = modal_weight(h, data['a'], s, q)
        p_oracle = covariance_power(w_oracle, h, s, q)
        heldout = heldout_snapshots(data, t, cfg['heldout_snapshots'])
        record['weights']['oracle'] = w_oracle
        for method in METHODS:
            w = record['weights'][method]
            ok = bool(np.isfinite(w).all())
            true_power = covariance_power(w, h, s, q) if ok else np.nan
            heldout_power = sample_output_power(w, heldout) if ok else np.nan
            D = Poracle if method == 'oracle' else record['projectors'].get(method)
            point = Poracle if method == 'oracle' else record['point_projectors'].get(method)
            chordal = (float(np.linalg.norm(point-Poracle,'fro')/np.sqrt(2))
                       if point is not None else np.nan)
            response_error = (float(np.linalg.norm(D-Poracle,'fro')**2)
                              if D is not None and np.isfinite(D).all() else np.nan)
            posterior_power = record['posterior_powers'].get(method,np.nan)
            rows.append(dict(n=n, condition=condition, stationary=stationary, trial=trial,
                t=t, time_s=t*cfg['dt'], method=method, sigma2=s, tau2=q,
                beta=(record['policy']['scalar_beta'] if method == 'modal_loading'
                      else record['policy']['covariance_beta'] if method == 'covariance_loading'
                      else policy['shared_beta'] if method in ('mean','mode','isotropic','great_ew')
                      else 1. if method == 'ledoit_wolf' else np.nan),
                optimization_window=(policy['great_window']['window'] if method=='great_window' else np.nan),
                gradient_steps=(policy[method]['inner_steps'] if method in ('great_window','great_ew') else np.nan),
                gradient_stepsize=(policy[method]['stepsize'] if method in ('great_window','great_ew') else np.nan),
                true_power=true_power, heldout_power=heldout_power, oracle_power=p_oracle,
                true_ratio=true_power/p_oracle, heldout_ratio=heldout_power/p_oracle,
                true_ratio_db=10*np.log10(true_power/p_oracle) if ok else np.nan,
                heldout_ratio_db=10*np.log10(heldout_power/p_oracle) if ok else np.nan,
                chordal_error=chordal, response_map_error=response_error,
                posterior_power=posterior_power, numerical_ok=ok,
                mean_ok=record['mean_ok'], mean_seconds=record['mean_seconds'] if method == 'mean' else 0.,
                mean_order=record['mean_diagnostics'].get('order', np.nan),
                mean_error_bound=record['mean_diagnostics'].get('moment_error_bound', np.nan),
                mean_cap_extension=record['mean_diagnostics'].get('cap_extension',False),
                unique_mode=record['unique_mode'], unit_gain_error=float(abs(np.vdot(w,data['a'])-1)) if ok else np.nan,
                same_parameter=method in ('mean','mode','isotropic'),
                oracle_information=method == 'oracle'))
        display_time = stationary or t in cfg['story']['moving_times']
        if save_story and display_time:
            prefix = f'{condition}_n{n}_t{t}_'
            for name in ['A','M','P','M_iso']:
                stories[prefix+name] = record[name]
            stories[prefix+'weights'] = np.array([record['weights'][name] for name in METHODS])
            values,E=np.linalg.eigh(record['A'])
            frame=E[:,[-1,-2,0]]
            stories.update({prefix+'eigenvalues':values,prefix+'section_frame':frame,
                prefix+'section_parameter':frame.conj().T@record['A']@frame,
                prefix+'log_normalizer':record['log_normalizer']})
    if save_story:
        prefix = f'{condition}_n{n}_'
        for name in ('U','h','a','angles','sigma2','tau2','time'):
            stories[prefix+name] = data[name]
        if stationary and n == cfg['story']['stationary_n']:
            chosen_t = int(round(n*cfg['story']['stationary_ratio']))
            record = next(item for item in estimated if item['t'] == chosen_t)
            scan = np.linspace(-4/n, 4/n, 161)
            signatures = np.array([ula(n, value) for value in scan])
            stories.update(setup_U=data['U'][:chosen_t], setup_h=data['h'][chosen_t-1],
                setup_a=data['a'], setup_interferer_angle=data['angles'][chosen_t-1],
                setup_sigma2=data['sigma2'][chosen_t-1], setup_tau2=data['tau2'][chosen_t-1],
                scan_angles=scan, scan_signatures=signatures,
                setup_weights=np.array([record['weights'][name] for name in METHODS]),
                setup_methods=np.array(METHODS),setup_M=record['M'],setup_Miso=record['M_iso'],
                setup_A=record['A'],setup_log_normalizer=record['log_normalizer'])
            values,E=np.linalg.eigh(record['A']);frame=E[:,[-1,-2,0]]
            stories.update(setup_eigenvalues=values,setup_section_frame=frame,
                setup_section_parameter=frame.conj().T@record['A']@frame,
                setup_mean_eigenvalues=np.diag(E[:,::-1].conj().T@record['M']@E[:,::-1]).real,
                setup_iso_mean_eigenvalues=np.diag(E[:,::-1].conj().T@record['M_iso']@E[:,::-1]).real,
                section_iso_density_fitted=np.array(False))
    return rows, stories


def episode_average(rows, primary_burnin):
    """Reduce moving observations to the independent episode repetition unit."""
    import pandas as pd
    frame = pd.DataFrame(rows)
    selected = frame[frame.t >= primary_burnin]
    result = []
    for keys, group in selected.groupby(['n','condition','trial','method'], sort=False):
        finite = np.isfinite(group.heldout_ratio)
        ok = bool(group.numerical_ok.astype(bool).all() and finite.all())
        ratio = float(group.heldout_ratio.mean()) if finite.any() else np.nan
        true = float(group.true_ratio.mean()) if finite.any() else np.nan
        result.append(dict(zip(['n','condition','trial','method'],keys),
            checkpoints=len(group), resolved_checkpoints=int(finite.sum()),
            heldout_ratio=ratio, true_ratio=true,
            heldout_ratio_db=10*np.log10(ratio) if ratio>0 else np.nan,
            true_ratio_db=10*np.log10(true) if true>0 else np.nan,
            response_map_error=float(group.response_map_error.mean()),
            numerical_ok=ok, mean_ok=bool(group.mean_ok.all())))
    return result


def paired_summary(frame, groups):
    """Paired Student intervals across episodes, preserving unresolved pairs."""
    import pandas as pd
    rows = []
    for keys, group in frame.groupby(groups, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        own = group[group.method.eq('mean')].set_index('trial')
        for method, other in group[~group.method.eq('mean')].groupby('method', sort=False):
            other = other.set_index('trial')
            index = own.index.intersection(other.index)
            for endpoint in ('heldout_ratio_db','true_ratio_db'):
                delta = other.loc[index, endpoint]-own.loc[index, endpoint]
                finite = np.isfinite(delta)
                values = delta[finite].to_numpy()
                error = float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else np.nan
                mean = float(np.mean(values)) if len(values) else np.nan
                rows.append(dict(zip(groups,keys),reference=method,endpoint=endpoint,
                    episodes=len(index), resolved_pairs=int(finite.sum()),
                    gain=mean, lower=mean-error, upper=mean+error,
                    numerical_ok=bool(len(values)==len(index) and own.loc[index].numerical_ok.all()
                                      and other.loc[index].numerical_ok.all()),
                    interval='Pointwise paired Student 95%; independent episodes; positive means comparator minus posterior-mean oracle-normalized loss in dB'))
    return pd.DataFrame(rows)


def block_episode(cfg, condition, trial):
    """Exact aggregation, delayed actions and common-state window approximations."""
    from .final_theory import exact_aggregate, sequential_parameter
    n, T, beta = cfg['block_n'], cfg['block_T'], cfg['block_beta']
    data = beam_episode(cfg,n,condition,trial,'test',False,length=T)
    # A nonuniform prior detects accidental covariance normalization. It is
    # independent of the generating signature and shared by every cadence.
    A0 = center(.3*np.outer(data['initial'],data['initial'].conj()))
    sigma, power, U, a = data['sigma2'], data['tau2'], data['U'], data['a']
    singleton = []
    A = A0.copy()
    for t,u in enumerate(U,1):
        A = center(beta*A+power[t-1]/(sigma[t-1]*(sigma[t-1]+power[t-1]))*np.outer(u,u.conj()))
        singleton.append(A.copy())
    rows = []
    for length in cfg['block_lengths']:
        endpoint_A = A0.copy()
        last_weight = a.copy()
        losses, true_losses, delays = [], [], []
        maximum_error = 0.
        begin = perf_counter()
        corrections = reports = 0
        for t in range(1,T+1):
            if t % length == 0:
                start = t-length
                endpoint_A = exact_aggregate(endpoint_A,U[start:t],sigma[start:t],
                    power[start:t],beta=beta,complex_data=True)
                maximum_error = max(maximum_error,float(np.linalg.norm(endpoint_A-singleton[t-1],'fro')))
                M = checked_moments(endpoint_A,cfg['moment_tolerance'],cfg['moment_max_order'],
                    cfg['moment_fallback_max_order'])[0].mean
                last_weight = distortionless_weight(sigma[t-1]*np.eye(n)+power[t-1]*M,a)
                corrections += length
                reports += 1
            heldout = heldout_snapshots(data,t,cfg['heldout_snapshots'])
            reference = oracle_power(a,data['h'][t-1],sigma[t-1],power[t-1])
            losses.append(sample_output_power(last_weight,heldout)/reference)
            true_losses.append(covariance_power(last_weight,data['h'][t-1],sigma[t-1],power[t-1])/reference)
            delays.append(t % length)
        elapsed = perf_counter()-begin
        rows.append(dict(kind='delayed_exact',condition=condition,trial=trial,n=n,block_length=length,
            observations=T,outputs=reports,assimilated_observations=corrections,
            heldout_ratio=float(np.mean(losses)),true_ratio=float(np.mean(true_losses)),
            mean_delay_observations=float(np.mean(delays)),maximum_delay_observations=max(delays),
            physical_sampling_interval_s=cfg['dt'],maximum_delay_s=max(delays)*cfg['dt'],
            maximum_parameter_error=maximum_error,numerical_ok=True,
            execution_seconds=elapsed,timing_scope='Concurrent study elapsed time only; isolated cadence study supplies comparative timing',
            available_from_observation=length))
        # A separate last-L constant-state likelihood at the SAME final time,
        # uniform boundary prior, compares information versus physical drift.
        window_A = exact_aggregate(np.zeros((n,n),complex),U[-length:],
            sigma[-length:],power[-length:],beta=1.,complex_data=True)
        M = checked_moments(window_A,cfg['moment_tolerance'],cfg['moment_max_order'],
            cfg['moment_fallback_max_order'])[0].mean
        weight = distortionless_weight(sigma[-1]*np.eye(n)+power[-1]*M,a)
        reference = oracle_power(a,data['h'][-1],sigma[-1],power[-1])
        rows.append(dict(kind='constant_state_window',condition=condition,trial=trial,n=n,
            block_length=length,observations=length,outputs=1,assimilated_observations=length,
            heldout_ratio=sample_output_power(weight,heldout_snapshots(data,T,cfg['heldout_snapshots']))/reference,
            true_ratio=covariance_power(weight,data['h'][-1],sigma[-1],power[-1])/reference,
            mean_delay_observations=0.,maximum_delay_observations=0.,
            physical_sampling_interval_s=cfg['dt'],maximum_delay_s=0.,
            maximum_parameter_error=np.nan,numerical_ok=True,execution_seconds=np.nan,
            timing_scope='Different constant-state block likelihood; no equivalence to discounted recursion asserted',
            available_from_observation=T))
    return rows
