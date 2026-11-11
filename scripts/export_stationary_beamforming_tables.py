"""Export fixed-T stationary array tables from saved per-simulation records.

This script never generates observations, fits an estimator or selects settings.
It requires the completed T=200 cohort, checks every method and simulation ID,
and summarizes the individual exact-covariance power scores in dB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
COHORT = Path('results/beamforming/data/paper/performance/stationary_T200')
TABLES = Path('results/beamforming/tables/paper/performance/stationary_T200')
METHODS = (
    'mean', 'mode', 'isotropic', 'modal_loading', 'covariance_loading',
    'great_window', 'great_ew', 'grouse', 'ledoit_wolf', 'oracle')
TABLE_METHODS = ('mean', 'mode', 'great_window', 'covariance_loading')
HEADERS = (
    r'\shortstack{Posterior\\mean}', 'Mode',
    r'\shortstack{GREAT\\(sliding window)}',
    r'\shortstack{Regularized sample\\covariance}')
DIMENSIONS = (32, 64, 128)
REPORT_TIMES = (25, 50, 100, 200)
SIMULATIONS = 256
DOF = SIMULATIONS - 1
ENDPOINT = 'true_ratio_db'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config_hash(config):
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _truth(values, name):
    """Accept CSV booleans while rejecting absent or unresolved flags."""
    if values.isna().any():
        raise ValueError('Missing numerical status: ' + name)
    normalized = values.astype(str).str.lower()
    if not normalized.isin(['true', '1', '1.0']).all():
        raise ValueError('Unresolved numerical records: ' + name)


def _declared_raw_hash(manifest, raw, root):
    relative = raw.relative_to(root).as_posix()
    declarations = []
    for key in ('artifacts', 'files', 'data_hashes'):
        mapping = manifest.get(key, {})
        if not isinstance(mapping, dict):
            continue
        for name, record in mapping.items():
            if name in (relative, raw.name):
                declarations.append(record.get('sha256') if isinstance(record, dict) else record)
    actual = sha(raw)
    if not declarations or any(digest != actual for digest in declarations):
        raise ValueError('The completed cohort manifest must bind the raw stationary.csv hash')
    return actual


def read_cohort(root=ROOT):
    """Validate all saved checkpoints and return the primary t=200 records."""
    root = Path(root).resolve()
    raw, manifest_path = root/COHORT/'stationary.csv', root/COHORT/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'completed':
        raise ValueError('Complete the fixed-T stationary cohort before exporting tables')
    if (manifest.get('example') != 'beamforming' or manifest.get('stage') != 'paper'
            or manifest.get('component') != 'stationary_T200'):
        raise ValueError('Expected the completed paper stationary_T200 beamforming component')
    config = manifest.get('config', {})
    expected = {'stationary_T': 200, 'stationary_primary_t': 200,
                'validation_episodes': 64, 'test_episodes': SIMULATIONS}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('The table requires paper settings with T=t=200 and 256 test simulations')
    if tuple(config.get('dimensions', ())) != DIMENSIONS:
        raise ValueError('The fixed-T table requires dimensions 32, 64 and 128')
    if tuple(config.get('stationary_times', ())) != REPORT_TIMES:
        raise ValueError('The fixed-T cohort must retain checkpoints 25, 50, 100 and 200')
    for key, value in [('sigma2', .2), ('tau2', 1.), ('overlap', .5)]:
        if not np.isclose(config.get(key, np.nan), value, rtol=0., atol=1e-14):
            raise ValueError('Unexpected stationary parameter: ' + key)
    raw_hash = _declared_raw_hash(manifest, raw, root)
    frame = pd.read_csv(raw)
    required = {'n', 'condition', 't', 'trial', 'method', ENDPOINT,
                'true_power', 'oracle_power', 'numerical_ok', 'mean_ok'}
    if not required <= set(frame):
        raise ValueError('Missing raw stationary fields: ' + ', '.join(sorted(required-set(frame))))
    if set(frame.condition) != {'static'}:
        raise ValueError('A stationary table cannot pool other motion conditions')
    if 'split' in frame and set(frame.split.astype(str)) != {'test'}:
        raise ValueError('The table requires independent test simulations only')
    if set(frame.n) != set(DIMENSIONS) or set(frame.t) != set(REPORT_TIMES):
        raise ValueError('The raw cohort does not contain the complete fixed-T checkpoint grid')
    if set(frame.method) != set(METHODS):
        raise ValueError('The raw cohort must retain all ten implemented methods')
    if frame.duplicated(['n', 't', 'method', 'trial']).any():
        raise ValueError('Duplicated per-simulation records cannot be silently averaged')
    if not np.isfinite(frame[[ENDPOINT, 'true_power', 'oracle_power']].to_numpy(float)).all():
        raise ValueError('Nonfinite exact-covariance power record')
    if not (frame.true_power.gt(0).all() and frame.oracle_power.gt(0).all()):
        raise ValueError('Exact-covariance powers must be positive')
    reconstructed = 10*np.log10(frame.true_power/frame.oracle_power)
    if not np.allclose(frame[ENDPOINT], reconstructed, rtol=1e-11, atol=1e-11):
        raise ValueError('true_ratio_db disagrees with true_power/oracle_power')
    _truth(frame.numerical_ok, 'numerical_ok')
    _truth(frame.mean_ok, 'mean_ok')
    expected_ids = set(range(SIMULATIONS))
    expected_groups = len(DIMENSIONS)*len(REPORT_TIMES)*len(METHODS)
    groups = list(frame.groupby(['n', 't', 'method'], sort=False))
    if len(groups) != expected_groups:
        raise ValueError('The raw checkpoint/method grid is incomplete')
    for (n, time, method), group in groups:
        if len(group) != SIMULATIONS or set(group.trial) != expected_ids:
            raise ValueError(f'Expected the same 256 simulation IDs at n={n}, t={time}, method={method}')
    if 'beta' in frame:
        tracked = frame[frame.method.isin(['mean', 'mode', 'isotropic'])]
        if not np.allclose(tracked.beta.to_numpy(float), 1., rtol=0., atol=1e-14):
            raise ValueError('Stationary Bingham comparisons require beta=1')
    primary = frame[frame.condition.eq('static') & frame.t.eq(200)].copy()
    provenance = dict(source_csv=raw.relative_to(root).as_posix(),
        source_csv_sha256=raw_hash, source_manifest=manifest_path.relative_to(root).as_posix(),
        source_config_sha256=_config_hash(config), config=config,
        raw_checkpoint_records=len(frame), primary_records=len(primary),
        endpoint=ENDPOINT, t=200, simulations=SIMULATIONS,
        degrees_of_freedom=DOF,
        averaging='Arithmetic mean of the 256 individual true_ratio_db values; not 10 log10 of an averaged power ratio.',
        validation='Every checkpoint/method/dimension retains simulation IDs 0,...,255, finite exact powers and successful numerical flags.')
    return primary, provenance


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (SIMULATIONS,) or not np.isfinite(values).all():
        raise ValueError('A reported mean must use all 256 finite simulations')
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = standard_deviation/np.sqrt(SIMULATIONS)
    half = float(student_t.ppf(.975, DOF)*standard_error)
    return dict(mean=mean, lower=mean-half, upper=mean+half,
                standard_deviation=standard_deviation, standard_error=standard_error)


def summarize(primary):
    summaries, paired_summaries, paired_trials = [], [], []
    for n in DIMENSIONS:
        frame = primary[primary.n.eq(n)]
        samples = {method: frame[frame.method.eq(method)].set_index('trial')[ENDPOINT].sort_index()
                   for method in METHODS}
        baseline = samples['mean']
        for method in METHODS:
            sample = samples[method]
            if not sample.index.equals(baseline.index):
                raise ValueError('Paired simulation identifiers differ for ' + method)
            statistic = mean_interval(sample.to_numpy())
            summaries.append(dict(n=n, t=200, method=method, endpoint=ENDPOINT,
                mean_relative_power_db=statistic['mean'],
                ci_lower_db=statistic['lower'], ci_upper_db=statistic['upper'],
                standard_deviation_db=statistic['standard_deviation'],
                standard_error_db=statistic['standard_error'],
                simulations=SIMULATIONS, degrees_of_freedom=DOF))
            if method == 'mean':
                continue
            differences = sample.to_numpy()-baseline.to_numpy()
            paired = mean_interval(differences)
            paired_summaries.append(dict(n=n, t=200, reference=method, baseline='mean',
                endpoint=ENDPOINT, mean_difference_db=paired['mean'],
                ci_lower_db=paired['lower'], ci_upper_db=paired['upper'],
                standard_deviation_db=paired['standard_deviation'],
                standard_error_db=paired['standard_error'], simulations=SIMULATIONS,
                degrees_of_freedom=DOF, paired_ids_identical=True,
                sign='reference minus posterior mean; positive favors posterior mean'))
            paired_trials.extend(dict(n=n, t=200, trial=int(trial), reference=method,
                baseline='mean', reference_relative_power_db=float(reference),
                posterior_mean_relative_power_db=float(mean), difference_db=float(difference))
                for trial, reference, mean, difference in zip(
                    baseline.index, sample.to_numpy(), baseline.to_numpy(), differences))
    return pd.DataFrame(summaries), pd.DataFrame(paired_summaries), pd.DataFrame(paired_trials)


def _number(value, digits):
    result = f'{float(value):.{digits}f}'
    # Normalize only a displayed rounded negative zero; CSV/JSON retain precision.
    return result[1:] if result.startswith('-') and float(result) == 0 else result


def latex_table(summary, dimensions, *, main=False, digits=3):
    dimensions = tuple(dimensions)
    sensor_text = (f'$n={dimensions[0]}$' if len(dimensions) == 1
                   else r'$n\in\{'+','.join(map(str, dimensions))+r'\}$')
    caption = (r'Stationary beamforming after $T=200$ measurements with '+sensor_text+
        r' sensors, $\sigma^2=0.2$, $\tau^2=1$, and $|a^{\mathrm H}h|^2=1/2$. '
        r'The Bingham tracker uses the uniform prior $A_0=0$ and $\beta=1$. '
        r'Relative power is $\ell_{200}=10\log_{10}((w_{200}^{\mathrm H}R_{200}w_{200})/p_{200}^{\star})$, '
        r'where $p_{200}^{\star}$ is the minimum output power under the unit-gain constraint. '
        r'The mean and two-sided Student 95\% confidence interval use the 256 individual '
        r'Monte Carlo simulation scores in dB (255 degrees of freedom). '
        r'Settings are selected using 64 separate validation simulations. Smaller values are better.')
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{'+caption+'}',
             r'\label{tab:beamforming-stationary-T200'+('' if main else '-supp')+'}',
             r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             ' & '+' & '.join(HEADERS)+r' \\', r'\midrule']
    for index, n in enumerate(dimensions):
        if len(dimensions) > 1:
            if index:
                lines.append(r'\midrule')
            lines.append(r'\multicolumn{5}{l}{$n='+str(n)+r',\ T=200$} \\')
        rows = summary[summary.n.eq(n)].set_index('method')
        if not set(TABLE_METHODS) <= set(rows.index):
            raise ValueError('Missing a displayed method in the fixed-T table')
        means = [_number(rows.loc[method, 'mean_relative_power_db'], digits) for method in TABLE_METHODS]
        intervals = [r'$['+_number(rows.loc[method, 'ci_lower_db'], digits)+r',\ '
                     +_number(rows.loc[method, 'ci_upper_db'], digits)+']$' for method in TABLE_METHODS]
        lines.append('Mean relative power (dB) & '+' & '.join(means)+r' \\')
        lines.append(r'95\% confidence interval (dB) & '+' & '.join(intervals)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def export(root=ROOT, *, digits=3):
    root = Path(root).resolve()
    primary, provenance = read_cohort(root)
    summary, paired, trials = summarize(primary)
    table_dir = root/TABLES
    main_path = root/'results/main/beamforming/stationary_T200_table.tex'
    supp_path = root/'results/supplementary/beamforming/stationary_T200_table.tex'
    table_dir.mkdir(parents=True, exist_ok=True)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    supp_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {'summary.csv': summary, 'paired_summary.csv': paired, 'paired_trials.csv': trials}
    files = {}
    for name, frame in outputs.items():
        path = table_dir/name
        frame.to_csv(path, index=False, float_format='%.17g')
        files[path.relative_to(root).as_posix()] = dict(sha256=sha(path), bytes=path.stat().st_size)
    for path, dimensions, is_main in [(main_path, (64,), True), (supp_path, DIMENSIONS, False)]:
        path.write_text(latex_table(summary, dimensions, main=is_main, digits=digits))
        files[path.relative_to(root).as_posix()] = dict(sha256=sha(path), bytes=path.stat().st_size)
    report = dict(schema=1, status='completed', producer='scripts/export_stationary_beamforming_tables.py',
        producer_sha256=sha(Path(__file__)), source=provenance,
        displayed_methods=list(TABLE_METHODS), display_decimals=digits, all_methods=list(METHODS),
        summaries=summary.to_dict(orient='records'),
        paired_summaries=paired.to_dict(orient='records'), files=files,
        scope='Read-only presentation summaries from the completed fixed-T=200 raw cohort. '
              'All ten methods and nine paired reference-minus-posterior-mean comparisons are retained. '
              'No simulation, inference, fitting or setting selection occurs during export.')
    (table_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return dict(status='completed', mean_summary_rows=len(summary),
        paired_summary_rows=len(paired), paired_simulation_rows=len(trials),
        primary_simulations_per_method=SIMULATIONS, source_csv_sha256=provenance['source_csv_sha256'],
        outputs=[*files, (TABLES/'summary.json').as_posix()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Repository root (use a temporary fixture root for export checks).')
    parser.add_argument('--digits', type=int, default=3, help='Displayed table decimals; CSV/JSON retain precision.')
    args = parser.parse_args()
    if not 1 <= args.digits <= 12:
        parser.error('--digits must be between 1 and 12')
    print(json.dumps(export(args.root, digits=args.digits), indent=2))


if __name__ == '__main__':
    main()
