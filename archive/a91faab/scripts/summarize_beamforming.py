"""Summarize verified beamforming artifacts without running any simulation.

Paper mode requires the current general gate and all six completed scientific
components. Quick mode is engineering-only and can explicitly report incomplete
components. Statistical comparisons use saved paired trajectory intervals;
no displayed regime, tuning parameter or simulation budget is selected here.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from lib.beamforming_validation import require_beamforming_gate, verify_component, beamforming_hash

COMPONENTS = ('numerics', 'information', 'decisions', 'tracking', 'process', 'inference')
REFERENCES = ('mode', 'pca', 'isotropic', 'scalar', 'loaded', 'grouse', 'oracle')
LABELS = dict(mode='Posterior mode', pca='EW PCA', isotropic='Matched isotropization',
              scalar='Regularized PCA', loaded='Loaded covariance', grouse='GROUSE',
              oracle='True-covariance oracle', mean='Posterior mean')
ENDPOINTS = dict(true_power='True-covariance power', heldout_power='Held-out power',
                 projection_loss='Spatial-response loss', posterior_power='Posterior expected power')
CONDITIONS = dict(haar='Haar stationary', ula_static='ULA stationary', ula_moving='ULA moving')
LAWS = dict(stationary='Stationary', matched_motion='Matched process', ula_motion='Moving ULA')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(values):
    """Preserve missing diagnostics as unresolved, rather than truthy strings."""
    return values.map({True: True, False: False, 'True': True, 'False': False,
                       'true': True, 'false': False}).fillna(False).astype(bool)


def good_count(frame, column):
    return int(boolean(frame[column]).sum()) if column in frame else 0


def unresolved(frame, column):
    return len(frame) - good_count(frame, column)


def number(value, digits=3):
    return f'{float(value):.{digits}g}' if np.isfinite(value) else '—'


def maximum(frame, column):
    return float(pd.to_numeric(frame[column], errors='coerce').max()) if column in frame else np.nan


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', r'\|').replace('\n', ' ')
    return ['| ' + ' | '.join(map(cell, headers)) + ' |',
            '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
                '| ' + ' | '.join(map(cell, row)) + ' |' for row in rows] + ['']


def proportion(values):
    values = boolean(values)
    n, k = len(values), int(values.sum())
    if not n:
        return '—'
    ci = binomtest(k, n).proportion_ci(confidence_level=.95, method='wilson')
    return f'{k}/{n} = {k/n:.3f} [{ci.low:.3f}, {ci.high:.3f}]'


class Evidence:
    def __init__(self, profile, allow_incomplete=False):
        self.profile = profile
        self.base = ROOT / 'results/beamforming_unified' / profile
        self.meta = {}
        self.config = {}
        self.pending = {}
        self.inputs = {}
        self.presentation_components = set()
        self.gate = require_beamforming_gate(ROOT) if profile == 'paper' else None
        for component in COMPONENTS:
            try:
                meta = verify_component(ROOT, profile, component)
                directory = self.base / component
                manifest = next(p for p in [directory/'manifest.json', directory/'config.json'] if p.exists())
                self.inputs[str(manifest.relative_to(ROOT))] = sha(manifest)
                preregistered = directory / 'preregistered.json'
                declaration = json.loads(preregistered.read_text())
                config = declaration.get('config', meta.get('config', {}))
                if config.get('name', config.get('profile')) != profile:
                    raise RuntimeError(f'Profile declaration does not match {profile}/{component}')
                if profile == 'paper' and declaration.get('development_only', False):
                    raise RuntimeError('Development-only output cannot serve as paper evidence')
                self.meta[component], self.config[component] = meta, config
            except (OSError, ValueError, KeyError, StopIteration, RuntimeError) as error:
                if not allow_incomplete:
                    raise RuntimeError(f'Cannot summarize {profile}/{component}: {error}') from error
                self.pending[component] = str(error)
        self.cache = {}

    def frame(self, component, name):
        key = component, name
        if key not in self.cache:
            self.cache[key] = pd.read_csv(self.base / component / f'{name}.csv')
        return self.cache[key]

    def checks(self, component):
        return self.meta[component].get('checks', self.meta[component])

    def verify_again(self):
        for component in set(self.meta) | self.presentation_components:
            verify_component(ROOT, self.profile, component)
        if self.profile == 'paper':
            require_beamforming_gate(ROOT)
        for name, digest in self.inputs.items():
            if sha(ROOT/name) != digest:
                raise RuntimeError(f'Evidence changed during summary: {name}')


def decisions(e):
    if 'decisions' not in e.meta:
        return []
    paired = e.frame('decisions', 'paired')
    config = e.config['decisions']
    keys = ['condition', 's', 't', 'endpoint', 'reference']
    if paired.duplicated(keys).any():
        raise RuntimeError('Duplicate paired decision cells')
    expected = set(itertools.product(config['conditions'], config['noises'],
                   config['snapshots'], ENDPOINTS, REFERENCES))
    actual = set(paired[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise RuntimeError(f'Paired decision map is incomplete: missing {len(expected-actual)}, extra {len(actual-expected)}')
    counts = []
    for endpoint in ENDPOINTS:
        for reference in REFERENCES:
            g = paired[(paired.endpoint == endpoint) & (paired.reference == reference)].copy()
            numeric = np.isfinite(g[['gain', 'low', 'high']]).all(axis=1)
            if ((g.low > g.high) & numeric).any():
                raise RuntimeError('Reversed paired confidence interval')
            not_defined = pd.Series(endpoint == 'projection_loss' and reference == 'loaded', index=g.index)
            valid = boolean(g.resolved) & numeric & (g.trials == config['test_trials']) & ~not_defined
            exact = boolean(g.analytic_tie) & valid
            if (abs(g.loc[exact, ['gain', 'low', 'high']]) > 1e-10).any().any():
                raise RuntimeError('Analytic equality disagrees with its saved interval')
            gain = valid & ~exact & (g.low > 0)
            loss = valid & ~exact & (g.high < 0)
            crossing = valid & ~exact & ~gain & ~loss
            counts.append([ENDPOINTS[endpoint], LABELS[reference], int(gain.sum()),
                           int(crossing.sum()), int(loss.sum()), int(exact.sum()),
                           int((~valid & ~not_defined).sum()), int(not_defined.sum()), len(g)])
    out = ['## Decision comparisons', '',
           f"The complete map contains {len(expected)//(len(ENDPOINTS)*len(REFERENCES))} predeclared condition/noise/horizon cells per comparator, "
           f"with {config['test_trials']} independent test trajectories and {config['validation_trials']} independent validation trajectories per condition/noise. "
           'A saved paired 95% Student interval for baseline minus posterior-mean loss lies above zero for a gain and below zero for a loss. '
           'Crossing zero means insufficient evidence of a difference, not proven equality. Exact ties are the analytic one-snapshot control. '
           'The intervals are pointwise exploratory comparisons; these cell counts are not a multiple-testing-adjusted discovery claim. '
           'Queries, snapshots and posterior draws within one trajectory are not extra replications.', '']
    primary = {row[1]: row for row in counts if row[0] == ENDPOINTS['true_power']}
    mode, grouse, scalar, isotropic = [primary[LABELS[r]] for r in ['mode','grouse','scalar','isotropic']]
    out += [f"For true-covariance power, the posterior mean records {mode[2]}/{mode[-1]} gains and {mode[4]} losses against the mode; "
            f"{grouse[2]}/{grouse[-1]} gains and {grouse[4]} losses against GROUSE; "
            f"{scalar[2]}/{scalar[-1]} gains and {scalar[4]} losses against validated regularized PCA; "
            f"and {isotropic[2]}/{isotropic[-1]} gains and {isotropic[4]} losses against matched isotropization. "
            'These distinctions matter: improvement over a point mode does not by itself demonstrate an advantage over regularization or prove a benefit from anisotropy.', '']
    scalar_rows=paired[(paired.endpoint=='true_power') & (paired.reference=='scalar')]
    bylaw=[]
    for condition,g in scalar_rows.groupby('condition',sort=False):
        valid=boolean(g.resolved)&np.isfinite(g[['low','high']]).all(axis=1)
        bylaw.append(f"{CONDITIONS.get(condition,condition)} {int((valid&(g.low>0)).sum())} gains/{int((valid&(g.high<0)).sum())} losses in {len(g)} cells")
    iso_rows=paired[(paired.endpoint=='true_power') & (paired.reference=='isotropic')]
    max_anisotropy_gain=100*iso_rows.relative_gain.max()
    out += ['The regularized-PCA comparison separates by generation law: '+ '; '.join(bylaw)+'. '
            f"The largest observed relative power gain over matched isotropization is {max_anisotropy_gain:.4f}%; anisotropy contributes little and its signs are mixed. "
            'The isotropized posterior retains data-dependent spread, whereas validated regularized PCA uses one scalar per validation condition/horizon shared across test trajectories. '
            'The static ULA likelihood remains correct: its fixed steering geometry and decision-risk weighting differ from Haar-prior averaging. '
            'This decision difference does not contradict the stationary state-invariance argument. Moving ULA motion introduces a separate process mismatch.', '']
    out += table(['Endpoint', 'Comparator', 'Gain', 'Crosses zero', 'Loss', 'Exact tie', 'Unresolved', 'Not defined', 'Cells'], counts)
    out += ['Loaded covariance supplies no point-subspace action in this implementation, so its spatial-response loss is not defined. '
            'Those saved NaN pairs are not numerical failures. Posterior expected-power rows evaluate the model objective under the same full posterior; '
            'true-covariance and held-out power are separate empirical endpoints.', '']
    shape = e.frame('decisions', 'shape')
    one = shape[shape.t == 1]
    metric = e.frame('decisions', 'metrics')
    checks = e.checks('decisions')
    out += [f"One snapshot: {good_count(one, 'one_snapshot_equality')}/{len(one)} trajectory-level equalities; "
            f"maximum `||M-M_iso||_F` {number(maximum(one, 'anisotropy'))}. "
            f"Maximum mode/PCA projector difference {number(checks.get('maximum_mode_pca_error', np.nan))}; "
            f"conditional expected-power identity residual {number(checks.get('maximum_expected_power_identity_error', np.nan))}. "
            'A one-snapshot gain over the mode therefore cannot demonstrate anisotropy. The true-covariance oracle uses scoring truth and is a lower reference, not an implementable method.', '']
    # Every condition is retained, rather than reporting only a favorable regime.
    detail = paired[paired.endpoint == 'true_power']
    rows = []
    for condition in config['conditions']:
        for reference in REFERENCES[:-1]:
            g = detail[(detail.condition == condition) & (detail.reference == reference)]
            valid = boolean(g.resolved) & np.isfinite(g[['low', 'high']]).all(axis=1)
            exact = boolean(g.analytic_tie) & valid
            rows.append([CONDITIONS.get(condition, condition), LABELS[reference],
                int((valid & ~exact & (g.low > 0)).sum()),
                int((valid & ~exact & (g.low <= 0) & (g.high >= 0)).sum()),
                int((valid & ~exact & (g.high < 0)).sum()), int(exact.sum()),
                int((~valid).sum())])
    out += ['True-covariance power by generation law, combining both declared noise levels and all seven horizons:', '']
    out += table(['Law', 'Comparator', 'Gain', 'Crosses zero', 'Loss', 'Exact tie', 'Unresolved'], rows)
    coverage = e.frame('decisions', 'coverage')
    powers = e.frame('decisions', 'power_components')
    out += [f"Decision uncertainty reports: {len(coverage)}, with {unresolved(coverage, 'mean_ok')} unresolved means, "
            f"{unresolved(coverage, 'state_ok')} unresolved state-mass checks and {unresolved(coverage, 'response_ok')} unresolved scalar checks. "
            f"The {len(powers)} queried scalar intervals belong to {len(coverage)} trajectory/time reports and are not independent trajectory repeats. All reports remain in the tables.", '']
    n = max(config['snapshots'])
    terminal = metric[metric.t == n]
    cost = []
    for method in ['mean', 'pca', 'grouse', 'loaded']:
        g = terminal[terminal.method == method]
        cost.append([LABELS[method], number(1000*g.update_seconds.median()),
                     number(1000*g.mode_seconds.median()), number(1000*g.moment_seconds.median()),
                     number(1000*g.decision_seconds.median())])
    out += [f'Recorded decision costs at `N={n}`, median milliseconds across declared conditions and trajectories. '
            'Update/eigensystem clocks are cumulative through that horizon; the mean evaluation and decision solve are calls at the reporting horizon. '
            'They do not represent continuously reporting every posterior tail, and are not a portable speed guarantee.', '']
    out += table(['Method', 'Update', 'Eigensystem', 'Current mean call', 'Decision'], cost)
    return out


def information(e):
    if 'information' not in e.meta:
        return []
    frame = e.frame('information', 'case_checks')
    out = ['## Information and deterministic checks', '',
        f"Passive Fisher verification uses {len(frame)} fixed-state/noise cases with {int(frame.repeats.min()):,} independent proper-complex snapshots per case. "
        f"Maximum score-covariance relative matrix error is {number(maximum(frame, 'relative_matrix_error'))}; "
        f"maximum normalized off-diagonal {number(maximum(frame, 'maximum_normalized_offdiagonal'))}; "
        f"maximum expected-likelihood curvature relative error {number(maximum(frame, 'expected_curvature_relative_error'))}. "
        'Score-covariance discrepancies are Monte Carlo error; curvature error is a separate finite-difference check. '
        'The analytic information is isotropic in the stated real tangent coordinates. Physical-angle information varies with the steering-curve tangent.', '']
    return out


def numerics(e):
    if 'numerics' not in e.meta:
        return []
    mean = e.frame('numerics', 'mean_feasibility')
    negative = boolean(mean.negative_control)
    reports = e.frame('numerics', 'posterior_reports')
    tails = reports[reports.tail_ok.notna()]
    inverse = reports[reports.inverse_numerical_ok.notna()]
    transition = e.frame('numerics', 'transition_checks')
    risk = e.frame('numerics', 'risk_checks')
    out = [f"Native numerical checks: {unresolved(mean[~negative], 'numerical_ok')}/{int((~negative).sum())} unresolved ordinary mean targets; "
           f"{int(negative.sum())} deliberately out-of-range series targets retained; "
           f"{unresolved(transition, 'numerical_ok')}/{len(transition)} unresolved transition-moment checks; "
           f"{unresolved(risk, 'numerical_ok')}/{len(risk)} unresolved vector-risk checks. "
           f"Inverse moment fits: {unresolved(inverse, 'inverse_numerical_ok')}/{len(inverse)} unresolved, "
           f"maximum residual {number(maximum(inverse, 'inverse_residual'))}. "
           f"Posterior scalar precision: {unresolved(tails, 'tail_ok')}/{len(tails)} unresolved fixed-budget reports. "
           'Accurate means and exact iid sampling laws do not remove empirical-quantile uncertainty.', '']
    timing = e.frame('numerics', 'timings')
    rows = []
    for (n, operation), group in timing[timing.n.isin([8,16])].groupby(['n', 'operation'], sort=False):
        rows.append([int(n), operation, len(group), number(1000*group.seconds.median()),
                     number(1000*group.cpu_seconds.median())])
    out += ['Whole-call native numerical costs, measured on a shared host. Contention can separate CPU time from wall time; these are not dedicated latency measurements or a complexity law. The full posterior report contains a mean evaluation, two iid batches and uncertainty calculations; inverse fits contain forward evaluations. These nested categories must not be added as disjoint costs.', '']
    out += table(['Complex dimension', 'Operation', 'Repeated timings', 'Median wall ms', 'Median CPU ms'], rows)
    return out


def tracking(e):
    if 'tracking' not in e.meta:
        return []
    bounds = e.frame('tracking', 'bounds')
    rows = []
    for condition, g in bounds.groupby('condition', sort=False):
        valid = boolean(g.valid)
        rows.append([condition, len(g), int(valid.sum()), good_count(g, 'certificate_violated'),
                     number(g.loc[valid, 'bound'].min()), number(g.loc[valid, 'bound'].max())])
    out = ['## Fixed-time certificates and state coverage', '',
           'The certificate uses known fixed powers, deterministic weights/transports and a truth-dependent drift for evaluation. '
           'Invalid sufficient conditions remain invalid; no posterior radius is substituted. The observations at each predeclared reporting time give pointwise guarantees.', '']
    out += table(['Cohort', 'Fixed-time reports', 'Valid', 'Valid bound violated', 'Smallest valid bound', 'Largest valid bound'], rows)
    cover = e.frame('tracking', 'coverage')
    rows = []
    for (state, t), g in cover.groupby(['state', 't'], sort=False):
        rows.append([state, int(t), proportion(g.hpd_covered), proportion(g.ball_covered),
                     unresolved(g, 'state_ok'), unresolved(g, 'tail_ok')])
    out += ['Fixed-state inclusion, with pointwise 95% Wilson intervals across independent trajectories. Numerical failures remain included and are separately counted.', '']
    out += table(['State', 'N', 'HPD inclusion [CI]', 'Ball inclusion [CI]', 'State unresolved', 'Scalar unresolved'], rows)
    inclusion = []
    for (state,t), g in cover.groupby(['state','t'],sort=False):
        for column,label in [('hpd_covered','HPD'),('ball_covered','ball')]:
            values=boolean(g[column]);k=int(values.sum());n=len(values)
            ci=binomtest(k,n).proportion_ci(confidence_level=.95,method='wilson')
            inclusion.append(dict(state=state,t=int(t),region=label,k=k,n=n,p=k/n,low=ci.low,high=ci.high))
    low=min(inclusion,key=lambda x:x['p']);high=max(x['p'] for x in inclusion)
    out += [f"Across the {len(inclusion)} predeclared state/time/region entries, observed inclusion ranges from {low['p']:.3f} to {high:.3f}. "
            f"The lowest is {low['region']} at {low['state']}, N={low['t']}: {low['k']}/{low['n']}, with pointwise Wilson interval [{low['low']:.3f}, {low['high']:.3f}]. "
            'This deviation remains visible. These finite, dependent comparisons do not establish exact 95% coverage at every state and are not a blanket empirical confirmation. '
            'The theorem concerns ideal equivariant regions of exact posterior mass; finite training/check batches and trajectory counts are separate qualifications.', '']
    tails = e.frame('tracking', 'tails')
    physical_rows=[]
    for (s,policy,t),g in tails.groupby(['s','policy','t'],sort=False):
        physical_rows.append([s,policy,int(t),proportion(g.hpd_covered),proportion(g.ball_covered),
                              f"{good_count(g,'numerical_ok')}/{len(g)}"])
    out += ['Physical moving-ULA state inclusion is a separate calibration check. These are fixed-time, independent-trajectory intervals within each cell; policies and times share trajectories and cannot be pooled as independent repetitions.', '']
    out += table(['s','Memory policy','Time','HPD inclusion [CI]','Ball inclusion [CI]','Numerically resolved'],physical_rows)
    low_noise=tails[tails.s==tails.s.min()]
    low_cells=[(good_count(g,'hpd_covered'),len(g)) for _,g in low_noise.groupby(['policy','t'])]
    if all(k==0 for k,n in low_cells):
        denominators=sorted(set(n for k,n in low_cells))
        out += [f"At s={low_noise.s.iloc[0]}, every one of {len(low_cells)} predeclared memory/time cells has HPD inclusion 0/{'/'.join(map(str,denominators))}. "
                f"Numerical reports resolve {good_count(low_noise,'numerical_ok')}/{len(low_noise)} of these evaluations. "
                'Accurate posterior masses do not ensure frequentist state coverage under unmodeled drift. The selected memory minimizes independent-validation decision power, not a coverage loss. '
                'This observed calibration failure is substantive and must remain visible when interpreting the uncertainty plots.', '']
    out += [f"Moving-ULA uncertainty: {unresolved(tails, 'state_ok')}/{len(tails)} unresolved state checks and "
            f"{unresolved(tails, 'tail_ok')}/{len(tails)} unresolved scalar checks. "
            'The ideal stationary state-inclusion argument applies to known powers, uniform initialization, beta=1 and exact mass. '
            'Forgetting, unmodeled ULA drift, learned powers and assumed-density prediction do not inherit it. '
            'Central scalar intervals have a further support-boundary limitation: an exactly aligned query has truth one, above every finite nondegenerate 97.5% posterior quantile. '
            'A correct state-coverage result therefore does not imply uniform central scalar inclusion. A common state-region image and separate marginal intervals answer different questions.', '']
    return out


def process(e):
    if 'process' not in e.meta:
        return []
    reference = e.frame('process', 'reference_checks')
    shapes = e.frame('process', 'shape_checks')
    metrics = e.frame('process', 'metrics')
    native = metrics[metrics.method != 'Full-density CP1 reference']
    evaluated = native[boolean(native.tail_evaluated)] if 'tail_evaluated' in native else native[native.tail_ok.notna()]
    out = ['## Process projection and parameter inference', '',
           f"The complex dimension-two full-density reference has {unresolved(reference, 'reference_numerical_ok')}/{len(reference)} unresolved full-density reference reports. "
           f"Maximum reference refinement total variation is {number(maximum(reference, 'density_tv'))}, "
           f"maximum scalar-CDF discrepancy {number(maximum(reference, 'scalar_cdf_error'))}. "
           f"The single-Bingham filter differs from that reference by total variation "
           f"{number(shapes.density_tv.min())}–{number(shapes.density_tv.max())} and mean operator error "
           f"{number(shapes.mean_error.min())}–{number(shapes.mean_error.max())} at declared checkpoints. "
           'These are density-approximation discrepancies, separate from a small inverse-fit residual and separate from reference discretization error. '
           'The joint reference status includes density-validity and region-mass checks; small refinement discrepancies alone do not resolve a failed flag. '
           'The CP1 reference is not an eight-channel exact filtering result.', '',
           f"Native process reports: {unresolved(native, 'mean_ok')}/{len(native)} unresolved means. "
           f"Tails were evaluated for {len(evaluated)}/{len(native)} reports on a preassigned trajectory subset; "
           f"{unresolved(evaluated, 'state_ok')} state checks and {unresolved(evaluated, 'tail_ok')} scalar checks are unresolved. "
           'Unevaluated tails are not counted as resolved. Exact transition first moments do not establish exact dynamic regions.', '']
    horizon=max(e.config['process']['process_reports'])
    actual=metrics[(metrics.method=='Process Bingham') & (metrics.t==horizon)]
    costs=[]
    for (n,kappa),g in actual.groupby(['n','kappa'],sort=False):
        costs.append([int(n),kappa,horizon,g.trial.nunique(),good_count(g,'mean_ok'),
                      number(g.prediction_seconds.median()),number(g.prediction_seconds.mean()),number(g.prediction_seconds.max())])
    out += ['Actual matched-process trajectory costs can vary substantially from the representative inverse-fit fixture. '
            'These are cumulative prediction-call wall times through the final reporting horizon, not complete pipeline times: score evaluation, posterior reporting and simulation are separate. '
            'Shared-host contention also affects the wall distribution; no dedicated-latency or end-to-end complexity claim follows.', '']
    out += table(['Complex dimension','kappa','Final T','Trajectories','Resolved means','Median prediction s','Mean prediction s','Maximum prediction s'],costs)
    return out


def inference(e):
    if 'inference' not in e.meta:
        return []
    metric = e.frame('inference', 'metrics')
    joint = metric[metric.method == 'Joint inference']
    rows = []
    for (law, prior), g in joint.groupby(['law', 'prior'], sort=False):
        rows.append([LAWS.get(law,law), prior, len(g), unresolved(g, 'numerical_ok'),
                     good_count(g, 'parameter_mixing_ok'), good_count(g, 'state_mean_mixing_ok'),
                     good_count(g, 'covariance_mean_mixing_ok'), good_count(g, 'full_state_tail_ok'),
                     good_count(g, 'mixing_ok'), number(g.fit_seconds.median())])
    out = ['Joint inference uses proper ordered variance priors and paired power/state/concentration draws. '
           'The following are counts passing finite-chain diagnostics, not proofs of convergence. All fits, including unresolved ones, remain in performance tables.', '']
    out += table(['Law', 'Prior', 'Fits', 'Calculation failure', 'Parameter flag', 'State-mean flag', 'Covariance-mean flag', 'State-tail agreement', 'All flags', 'Median fit s'], rows)
    detailed = e.frame('inference', 'mixing')
    kappa = detailed[(detailed.quantity == 'kappa') & (detailed.prior != 'Known full-path control')]
    if len(kappa):
        ok = (kappa.min_chain_ess >= 50) & (kappa.rhat < 1.1) & (kappa.tail_difference_fraction < .15)
        out += [f"Concentration diagnostics: {int(ok.sum())}/{len(kappa)} kappa traces meet the declared scalar ESS/R-hat/tail-agreement thresholds. "
                'There is no ground-truth kappa for deterministic ULA motion. Failure of a state-tail or scalar check must not be described as a resolved full posterior, even when the decision covariance mean has better diagnostics.', '']
    causal = metric.trained_through.notna() & metric.scored_observation.notna()
    if not ((metric.loc[causal, 'scored_observation'] - metric.loc[causal, 'trained_through']) == 1).all():
        raise RuntimeError('A prospective inference row does not score the next observation')
    em = e.frame('inference', 'em_objectives')
    bad_em = 0
    for _, g in em.groupby(['law','s_true','N','trial'], sort=False):
        if np.any(np.diff(g.sort_values('iteration').objective) < -1e-8):
            bad_em += 1
    out += [f"Prospective metadata identify {int(causal.sum())} prefix/next-observation scores; {bad_em} recorded-prior EM traces decrease beyond tolerance. "
            'EM ascent concerns its frozen-prior objective only. Predictive scoring uses the held-out next observation without assimilation. '
            'Joint covariance averaging preserves E[qP] and its concentration/state dependence; replacing it by a product of separate means defines a different control.', '']
    rows = []
    for (law, method), g in metric.groupby(['law', 'method'], sort=False):
        rows.append([LAWS.get(law,law), method, len(g), unresolved(g,'numerical_ok'),
                     number(g.fit_seconds.median())])
    out += table(['Law', 'Method', 'Recorded fits', 'Calculation failure', 'Median fit s'], rows)
    out += [f"These inference cohorts and Monte Carlo budgets differ from the main fixed-power {e.config.get('decisions',{}).get('test_trials','declared')}-trajectory comparison. "
            'Conditional sampler correctness does not establish full-path Gibbs mixing, calibrated kappa intervals, or a universal predictive gain over validated fixed parameters.', '']
    return out


def execution(e):
    rows = []
    if e.profile == 'paper':
        rows.append(['General theory/regression gate', 'Verified passed',
                     'Current original-core and passive-science fingerprints match the saved gate'])
    else:
        rows.append(['Paper gate', 'Not used for engineering summary', 'Quick outputs are not paper evidence'])
    for component in COMPONENTS:
        if component in e.meta:
            rows.append([component, 'Completed and source/artifact verified',
                         'Numerical and approximation qualifications are reported separately'])
        else:
            rows.append([component, 'Pending or unverified', e.pending[component]])
    # Verify presentation artifacts independently, without requiring them for science summaries.
    try:
        figure = verify_component(ROOT, e.profile, 'figures')
        if figure.get('profile')!=e.profile:
            raise RuntimeError('Figure manifest profile does not match summary profile')
        figure_manifest=e.base/'figures/manifest.json'
        e.inputs[str(figure_manifest.relative_to(ROOT))]=sha(figure_manifest)
        e.presentation_components.add('figures')
        rows.append(['Figures', 'Source/artifact verified', f"{len(figure.get('files',{}))} PDF/PNG exports; presentation manifest checked"])
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        rows.append(['Figures', 'Pending or unverified', str(error)])
    from lib.beamforming_history import verify_historical
    try:
        verify_historical(ROOT)
        rows.append(['Historical appendices', 'Archive verified', 'Original rank-three and real low-data artifacts retain their source/certificate snapshot'])
    except (OSError,ValueError,KeyError,RuntimeError) as error:
        rows.append(['Historical appendices', 'Pending or unverified', str(error)])
    rows.append(['Array notebook', 'Separate execution manifest',
                 'See results/beamforming_unified/execution.json for the executed notebook, figure and findings hashes, complete cell execution and saved error checks'])
    return ['## Execution status', ''] + table(['Item', 'Status', 'Qualification'], rows)



def main_summary(e):
    needed={'decisions','tracking','process','inference','numerics'}
    if not needed.issubset(e.meta):
        return ('Engineering summary pending: complete the source-verified scientific components before drawing an empirical conclusion.\n\n'
                'The detailed findings retain current execution status; partial checkpoints are not paper evidence.\n')
    paired=e.frame('decisions','paired')
    items=[]
    for reference,label in [('mode','mode'),('isotropic','matched isotropization'),('scalar','regularized PCA'),('loaded','loaded covariance'),('grouse','GROUSE')]:
        g=paired[(paired.endpoint=='true_power') & (paired.reference==reference)]
        valid=boolean(g.resolved) & np.isfinite(g[['low','high']]).all(axis=1)
        exact=boolean(g.analytic_tie)
        win=int((valid & ~exact & (g.low>0)).sum());loss=int((valid & ~exact & (g.high<0)).sum())
        items.append(f'{label} {win} gains/{loss} losses')
    cells=len(g)
    profile='paper' if e.profile=='paper' else 'engineering quick'
    first=(f"The n=8 passive rank-one {profile} study has {cells} predeclared cells. Posterior-mean true-covariance power comparisons are: "
           +'; '.join(items)+'. '
           'Gains and losses use paired 95% Student intervals excluding zero; crossing intervals remain inconclusive. '
           'These counts provide no multiple-comparison guarantee. One snapshot gives exact mean/isotropization equality, so improvement over the mode there cannot demonstrate anisotropy. '
           'The conditional expected-power identity is not a universal empirical advantage.')
    scalar=paired[(paired.endpoint=='true_power') & (paired.reference=='scalar')]
    matched=scalar[scalar.condition=='haar'];physical=scalar[scalar.condition!='haar']
    first+=f" Against regularized PCA, Haar gives {int((matched.low>0).sum())} gains/{int((matched.high<0).sum())} losses; physical ULA gives {int((physical.low>0).sum())} gains/{int((physical.high<0).sum())} losses."
    mean=e.frame('numerics','mean_feasibility');ordinary=mean[~boolean(mean.negative_control)]
    process=e.frame('process','metrics');native=process[process.method!='Full-density CP1 reference']
    evaluated=native[boolean(native.tail_evaluated)]
    bounds=e.frame('tracking','bounds')
    cover=e.frame('tracking','coverage');entries=[]
    for (state,t),g in cover.groupby(['state','t'],sort=False):
        for column,label in [('hpd_covered','HPD'),('ball_covered','ball')]:
            k=good_count(g,column);n=len(g);entries.append((k/n,label,state,int(t),k,n))
    low=min(entries,key=lambda x:x[0]);ci=binomtest(low[4],low[5]).proportion_ci(.95,method='wilson')
    detail=e.frame('inference','mixing');kappa=detail[(detail.quantity=='kappa') & (detail.prior!='Known full-path control')]
    kok=(kappa.min_chain_ess>=50)&(kappa.rhat<1.1)&(kappa.tail_difference_fraction<.15)
    physical=e.frame('tracking','tails');physical=physical[physical.s==physical.s.min()]
    cells_physical=[(good_count(g,'hpd_covered'),len(g)) for _,g in physical.groupby(['policy','t'])]
    if all(k==0 for k,n in cells_physical) and len({n for k,n in cells_physical})==1:
        drift_rate=f"0/{cells_physical[0][1]}"
    else:
        rates=[k/n for k,n in cells_physical];drift_rate=f"{min(rates):.2f}–{max(rates):.2f}"
    precision='despite resolved numerics' if good_count(physical,'numerical_ok')==len(physical) else 'with unresolved numerical precision'
    drift_sentence=f"Physical-drift HPD inclusion at s={physical.s.iloc[0]} is {drift_rate} per policy/time cell, {precision}. "
    second=(f"Ordinary native means resolve {good_count(ordinary,'numerical_ok')}/{len(ordinary)} targets; {unresolved(evaluated,'tail_ok')}/{len(evaluated)} evaluated process-tail reports remain unresolved. "
            f"The known-power certificate is valid in {good_count(bounds,'valid')}/{len(bounds)} fixed-time reports, with {good_count(bounds,'certificate_violated')} violations. "
            f"Lowest fixed-state inclusion is {low[1]}, {low[2]}, N={low[3]}: {low[4]}/{low[5]} [{ci.low:.3f}, {ci.high:.3f}]. This deviation remains visible. "
            'Exact stationary state coverage assumes known powers, a uniform prior, no discounting and exact posterior mass; it does not imply uniform central scalar coverage. '
            +drift_sentence+f"Inferred-kappa checks remain unresolved in {len(kappa)-int(kok.sum())}/{len(kappa)} fits. "
            f"Maximum cumulative process prediction: {number(native[(native.n==8)&(native.method=='Process Bingham')].prediction_seconds.max())}s/trajectory. "
            'Shared-host CPU/wall timings reflect contention, not dedicated latency or a complexity law. '
            'GROUSE suits cheap point estimates; posterior decisions and uncertainty justify extra work when their assumptions hold and those outputs matter.')
    text=first+'\n\n'+second+'\n'
    words=len(text.split())
    if e.profile=='paper' and not 180<=words<=220:
        raise RuntimeError(f'Main summary has {words} words; require 180–220')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['quick','paper'], default='paper')
    parser.add_argument('--allow-incomplete', action='store_true', help='Engineering quick summary only')
    parser.add_argument('--output', type=Path, default=ROOT/'results/beamforming_unified/findings.md')
    parser.add_argument('--main-output', type=Path, default=ROOT/'results/beamforming_unified/main_summary.md')
    parser.add_argument('--manifest-output', type=Path, default=ROOT/'results/beamforming_unified/summary_manifest.json')
    args = parser.parse_args()
    if args.profile == 'paper' and args.allow_incomplete:
        parser.error('A paper summary requires every completed verified scientific component')
    evidence = Evidence(args.profile, args.allow_incomplete)
    title = '# Unified passive beamforming results' if args.profile == 'paper' else '# Engineering summary: paper results pending'
    lines = [title, '', 'This report is generated from source- and artifact-verified saved tables. No simulations, fitting, posterior sampling or tuning occur in the summary command.', '']
    if args.profile == 'quick':
        lines += ['**Engineering only.** These quick-profile outcomes are not completed paper comparisons. Pending components are not read or inferred from partial caches.', '']
    lines += execution(evidence)
    if args.profile=='paper' and 'figures' not in evidence.presentation_components:
        raise RuntimeError('Complete and verify the paper figure manifest before final summary generation')
    lines += decisions(evidence)
    lines += information(evidence)
    lines += numerics(evidence)
    lines += tracking(evidence)
    lines += process(evidence)
    lines += inference(evidence)
    lines += ['## Interpretation and paper placement', '',
        'The framework offers posterior decisions and uncertainty diagnostics at an extra computational cost. Its mode is weighted covariance PCA under matching assumptions. '
        'Any empirical advantage of the full mean must be judged against matched isotropization, validated scalar regularization, broad covariance loading and geometric tracking; '
        'the complete comparison map above retains gains, intervals crossing zero, losses, analytic equalities and unresolved calculations. '
        'Conditional posterior expected-power optimality does not guarantee lower true-covariance power on every trajectory or in every regime.', '',
        'GROUSE is a reasonable choice when a cheap point subspace estimate is the main requirement. The posterior framework is useful when downstream decisions, directional risk, '
        'state regions or joint parameter uncertainty justify the additional work. Exact likelihood correction and exact first-moment prediction remain valid mathematical ingredients, '
        'while density projection, quantile Monte Carlo and Gibbs mixing have separate limitations. Identity transport permits drift to be modeled or bounded; it does not make unmodeled motion disappear.', '',
        'Main-paper candidates are the setup, complete decision/anisotropy comparison and a carefully explained uncertainty/response example (`01_setup`, `04_decisions`, '
        '`05_uncertainty`, `06_prediction_and_risk`). A compact tracking validity panel can accompany a tracking claim. Full information matrices, detailed bounds, all-regime comparison panels, '
        'process/reference discrepancies, inference diagnostics and cost tables belong in the supplement. Unresolved concentration or state-tail results support a documented limitation, '
        'not a claim of calibrated full dynamic uncertainty.', '',
        'The earlier real conditional theorem remains in its original theory/contact or field fixtures. Historical rank-three and real low-data figures are separately archived supplementary experiments, '
        'not new rank-one passive results. See `docs/beamforming_claims.md` for the claim-by-claim scope map.', '',
        '## Provenance', '',
        f"Profile: `{args.profile}`. Summary generator SHA-256: `{sha(Path(__file__))}`. "
        'Component manifests below were rechecked after reading the tables. Their artifact hashes cover the input CSVs and declared scientific sources.', '']
    lines += table(['Verified manifest/config', 'SHA-256'], evidence.inputs.items())
    brief=main_summary(evidence)
    evidence.verify_again()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text('\n'.join(lines).rstrip()+'\n')
    temporary.replace(args.output)
    args.main_output.parent.mkdir(parents=True,exist_ok=True)
    args.main_output.write_text(brief)
    def relative(path):
        try:return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:return str(Path(path).resolve())
    gate_path=ROOT/'results/beamforming_unified/validation/validation.json'
    sources={}
    for meta in evidence.meta.values():
        for name,digest in meta['sources'].items():
            sources[name]=digest.get('sha256') if isinstance(digest,dict) else digest
    summary_manifest=dict(schema=1,profile=args.profile,
        status='completed' if not evidence.pending else 'engineering incomplete',
        paper_evidence=bool(args.profile=='paper' and not evidence.pending),
        created_utc=datetime.now(timezone.utc).isoformat(),
        generator=dict(path=relative(Path(__file__)),sha256=sha(Path(__file__))),
        science_fingerprint=beamforming_hash(ROOT),source_hashes=sources,
        verified_input_hashes=evidence.inputs,verified_components=list(evidence.meta),
        pending=evidence.pending,figures_verified='figures' in evidence.presentation_components,
        gate=dict(path=relative(gate_path),sha256=sha(gate_path),status=evidence.gate['status'],
            core_hash=evidence.gate['core_hash'],beamforming_hash=evidence.gate['beamforming_hash']) if evidence.gate else None,
        outputs={relative(args.output):sha(args.output),relative(args.main_output):sha(args.main_output)},
        main_summary_words=len(brief.split()),main_summary_paragraphs=len(brief.strip().split('\n\n')),
        statistics='Saved paired independent-trajectory pointwise 95% Student intervals; no multiple-comparison guarantee; exact analytic equalities and undefined endpoints are separate')
    args.manifest_output.parent.mkdir(parents=True,exist_ok=True)
    args.manifest_output.write_text(json.dumps(summary_manifest,indent=2)+'\n')
    print(f'Wrote {args.output}; verified components: {", ".join(evidence.meta)}; pending: {", ".join(evidence.pending) or "none"}')


if __name__ == '__main__':
    main()
