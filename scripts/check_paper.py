"""Check the paper's numerical claims directly from individual simulations."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
SIMULATIONS = 256
PDFS = {
    'contact/contact_comparison.pdf': (504., 199.48),
    'beamforming/moving_interference_comparison.pdf': (504., 252.),
    'contact/information_rotation.pdf': (1080., 360.),
    'contact/posterior_anisotropy_spheres.pdf': (1080., 360.),
    'beamforming/noise_adaptation.pdf': (501.181, 319.47),
}
STATIONARY = {
    32: ((.119, .115, .122), (.114, .111, .118), (.363, .351, .375), (.379, .369, .389)),
    64: ((.237, .231, .242), (.223, .218, .228), (.350, .342, .358), (.576, .565, .588)),
    128: ((.482, .473, .490), (.440, .433, .448), (.677, .666, .689), (.856, .844, .868)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rounded(actual, expected, digits, name):
    require(f'{float(actual):.{digits}f}' == f'{expected:.{digits}f}',
            f'{name}: got {actual:.12g}; expected {expected:.{digits}f} after rounding')


def interval(values):
    values = np.asarray(values, dtype=float)
    require(values.shape == (SIMULATIONS,) and np.isfinite(values).all(),
            'Each interval requires 256 finite independent simulation outcomes')
    mean = float(values.mean())
    half = float(student_t.ppf(.975, SIMULATIONS - 1) * values.std(ddof=1) / 16.)
    return np.array([mean, mean - half, mean + half])


def truth(frame, *columns):
    for name in columns:
        require(name in frame and frame[name].astype(str).str.lower().isin(['true', '1', '1.0']).all(),
                'Unresolved numerical status: ' + name)


def grid(frame, keys, expected, *, identity='trial'):
    require(not frame.duplicated([*keys, identity]).any(), 'Duplicate simulation records')
    groups = list(frame.groupby(keys, sort=False))
    require(len(groups) == expected, 'Incomplete experiment/method/reporting-time grid')
    for key, group in groups:
        require(len(group) == SIMULATIONS and set(group[identity]) == set(range(SIMULATIONS)),
                f'{key}: expected exactly simulation IDs 0,...,255')


def powers(frame, numerator='true_power', denominator='oracle_power', loss='true_ratio_db'):
    values = frame[[numerator, denominator, loss]].to_numpy(float)
    require(np.isfinite(values).all() and (values[:, :2] > 0).all(), 'Invalid exact-covariance powers')
    np.testing.assert_allclose(frame[loss], 10*np.log10(frame[numerator]/frame[denominator]),
                               rtol=1e-11, atol=1e-11, err_msg='dB losses must be formed per observation')


def check_contact(results):
    frame = pd.read_csv(results/'contact/data/metrics.csv.gz')
    require(set(frame.method) == {'mean', 'mode', 'isotropic', 'mean_identity'}, 'Wrong contact methods')
    truth(frame, 'numerical_ok')
    require(np.isfinite(frame.response_mse).all() and (frame.response_mse >= 0).all(),
            'Invalid contact response losses')
    require((frame.moment_resolution_error < 5e-8).all(), 'Contact moment accuracy check failed')
    checkpoints = frame[frame.endpoint.eq('checkpoint')]
    require(set(checkpoints.t) == set(range(4, 193, 4)), 'Contact needs all 48 dense reporting times')
    grid(checkpoints, ['method', 't'], 4*48)
    averages = checkpoints.groupby(['trial', 'method']).response_mse.mean().unstack()*1e6
    saved = frame[frame.endpoint.eq('episode_average')]
    grid(saved, ['method'], 4)
    np.testing.assert_allclose(saved.pivot(index='trial', columns='method', values='response_mse')*1e6,
                               averages, rtol=1e-12, atol=1e-12,
                               err_msg='Contact averages must use all 48 times within each simulation')
    expected = {'mode': (1.73, 1.22, 2.25), 'isotropic': (.34, .23, .46),
                'mean_identity': (2.65, 2.10, 3.21)}
    reductions = {}
    for method, target in expected.items():
        summary = interval(averages[method] - averages['mean'])
        for value, wanted in zip(summary, target):
            rounded(value, wanted, 2, 'Contact paired reduction ' + method)
        reductions[method] = summary.tolist()
    for filename, source in [('checkpoint_summary.csv', checkpoints),
                             ('episode_average_summary.csv', saved)]:
        summaries = pd.read_csv(results/'contact/data'/filename)
        require(len(summaries) == source.groupby(['method', 't']).ngroups, 'Incomplete contact summaries')
        for row in summaries.itertuples():
            method = 'isotropic' if row.method == 'iso' else row.method
            values = source[source.method.eq(method) & source.t.eq(row.t)].response_mse*1e6
            np.testing.assert_allclose([row.mean, row.lower, row.upper], interval(values),
                                       rtol=1e-12, atol=1e-12)
    return reductions


def check_stationary(results):
    frame = pd.read_csv(results/'beamforming/data/stationary.csv.gz')
    require(set(frame.n) == {32, 64, 128} and set(frame.t) == {200}, 'Stationary data must use T=t=200')
    require(set(frame.condition) == {'static'}, 'Wrong stationary experiment')
    methods = ('mean', 'mode', 'great_window', 'covariance_loading')
    require(set(frame.method) == set(methods), 'Wrong stationary table methods')
    truth(frame, 'numerical_ok', 'mean_ok')
    powers(frame)
    grid(frame, ['n', 'method'], 12)
    tracked = frame[frame.method.isin(['mean', 'mode'])]
    require((tracked.beta == 1.).all(), 'Stationary Bingham comparisons require beta=1')
    summaries = {}
    for n, targets in STATIONARY.items():
        summaries[str(n)] = {}
        for method, target in zip(methods, targets):
            summary = interval(frame[frame.n.eq(n) & frame.method.eq(method)].true_ratio_db)
            for value, wanted in zip(summary, target):
                rounded(value, wanted, 3, f'Stationary n={n}, {method}')
            summaries[str(n)][method] = summary.tolist()
    return summaries


def check_moving(results):
    frame = pd.read_csv(results/'beamforming/data/moving.csv.gz')
    methods = ('mean', 'mode', 'isotropic', 'great_window')
    require(set(frame.condition) == {'smooth', 'abrupt'} and set(frame.n) == {32}, 'Wrong moving experiment')
    require(set(frame.method) == set(methods), 'Wrong moving methods')
    require(set(frame.t) == set(range(8, 257, 8)), 'Moving averages require all times 8,...,256')
    grid(frame, ['condition', 'method', 't'], 2*4*32)
    truth(frame, 'numerical_ok', 'mean_ok')
    powers(frame)
    tracked = frame[frame.method.isin(['mean', 'mode', 'isotropic'])]
    require((tracked.beta == .97).all(), 'Moving Bingham comparisons require beta=.97')
    averages = frame.groupby(['condition', 'trial', 'method']).true_ratio_db.mean().unstack()
    summaries = {}
    for condition, targets in [('smooth', (.330, .331, .332, .296)),
                               ('abrupt', (.680, .665, .695, .694))]:
        summaries[condition] = {}
        for method, target in zip(methods, targets):
            summary = interval(averages.loc[condition, method])
            rounded(summary[0], target, 3, f'Moving {condition}, {method}')
            summaries[condition][method] = summary.tolist()
    for method, target in zip(methods[:3], (2.63, 3.17, 2.82)):
        value = frame[frame.condition.eq('abrupt') & frame.t.eq(144) & frame.method.eq(method)].true_ratio_db.mean()
        rounded(value, target, 2, 'Abrupt t=144 ' + method)
    difference = interval(averages.loc['abrupt', 'great_window'] - averages.loc['abrupt', 'mean'])
    for value, wanted in zip(difference, (.013, .006, .021)):
        rounded(value, wanted, 3, 'Moving paired GREAT minus mean')
    summaries['abrupt_paired_reduction'] = difference.tolist()
    return summaries


def check_noise(results):
    frame = pd.read_csv(results/'beamforming/data/noise.csv.gz')
    require(set(frame.condition) == {'noise_change'} and set(frame.split) == {'test'}, 'Wrong noise cohort')
    require(set(frame.t) == set(range(32, 257, 32)), 'Noise experiment needs all eight reporting times')
    require(set(frame.control) == {'mean', 'mode', 'iso'}, 'Wrong noise controls')
    require(set(frame.method) == {'known', 'misspecified', 'learned'}, 'Wrong noise policies')
    grid(frame, ['control', 'method', 't'], 3*3*8, identity='episode')
    truth(frame, 'mean_ok', 'score_numerical_ok', 'fit_numerical_ok')
    powers(frame, 'true_task_loss', 'oracle_loss', 'output_oracle_db')
    require((frame.block_length == 32).all() and (frame.em_iterations == 2).all(), 'Noise EM schedule changed')
    require((frame.accuracy_tolerance == 1e-9).all(), 'Noise moment tolerance changed')
    require((frame.mean_accuracy_estimate < 1e-9).all()
            and (frame.normalizer_accuracy_estimate < 1e-9).all(), 'Noise moment certificate failed')
    np.testing.assert_allclose(frame.sigma2_true, np.where(frame.t <= 128, .2, .8), rtol=0, atol=1e-14)
    require((frame.tau2_true == 1.).all(), 'Noise experiment signal variance changed')
    require(np.isfinite(frame[['prequential_score', 'sigma2_estimate', 'tau2_estimate']]).all().all(),
            'Nonfinite noise figure values')
    require((frame.sigma2_estimate >= 1e-5).all() and (frame.tau2_estimate >= 0).all(), 'Variance floor violated')
    summaries = {}
    for method, target in [('learned', (.435, .425, .445)),
                           ('misspecified', (1.056, 1.044, 1.069)), ('known', (.378, .371, .385))]:
        rows = frame[frame.method.eq(method)]
        require((rows.beta == (.9 if method == 'misspecified' else .98)).all(), 'Noise beta changed')
        mean_rows = rows[rows.control.eq('mean')]
        averages = mean_rows.groupby('episode').output_oracle_db.mean()
        summary = interval(averages)
        for value, wanted in zip(summary, target):
            rounded(value, wanted, 3, 'Noise power ' + method)
        summaries[method] = summary.tolist()
        if method == 'misspecified':
            np.testing.assert_allclose(rows.sigma2_estimate, .05, rtol=0, atol=1e-14)
            np.testing.assert_allclose(rows.tau2_estimate, .2, rtol=0, atol=1e-14)
        elif method == 'known':
            np.testing.assert_allclose(rows.sigma2_estimate, rows.sigma2_true, rtol=0, atol=1e-14)
            np.testing.assert_allclose(rows.tau2_estimate, rows.tau2_true, rtol=0, atol=1e-14)
    return summaries


def check_pdfs(results):
    sizes = {}
    for name, target in PDFS.items():
        output = subprocess.run(['pdfinfo', str(results/name)], check=True, capture_output=True, text=True).stdout
        pages = re.search(r'^Pages:\s+(\d+)', output, re.MULTILINE)
        size = re.search(r'^Page size:\s+([\d.]+) x ([\d.]+) pts', output, re.MULTILINE)
        require(pages is not None and int(pages[1]) == 1, name + ' must contain exactly one page')
        require(size is not None, name + ' has no readable page dimensions')
        actual = (float(size[1]), float(size[2]))
        np.testing.assert_allclose(actual, target, rtol=0, atol=.03, err_msg=name + ' canvas differs from paper')
        sizes[name] = actual
    return sizes


def check_tables(results, contact, stationary, moving, noise):
    tables = results/'beamforming/tables'
    definitions = [('stationary_results', ('n', 'method'), stationary),
                   ('moving_averages', ('condition', 'method'), moving),
                   ('noise_averages', ('method',), noise)]
    for name, keys, reference in definitions:
        frame = pd.read_csv(tables/(name+'.csv'))
        expected_count = {'stationary_results': 12, 'moving_averages': 8, 'noise_averages': 3}[name]
        require(len(frame) == expected_count and not frame.duplicated(list(keys)).any(),
                'Incomplete or duplicate exported table: ' + name)
        for row in frame.itertuples():
            values = reference
            for key in keys:
                values = values[str(getattr(row, key))]
            np.testing.assert_allclose([row.mean, row.lower, row.upper], values,
                                       rtol=1e-12, atol=1e-12, err_msg=name)
            require(row.simulations == 256 and row.degrees_of_freedom == 255, 'Wrong table interval unit')
    settings = pd.read_csv(tables/'stationary_settings.csv').set_index('n')
    require(set(settings.index) == {32, 64, 128} and len(settings) == 3, 'Wrong stationary settings table')
    for n, window, alpha in [(32, 64, .3538007162976233), (64, 128, .45923178071496473),
                              (128, 128, .5304028814675519)]:
        row = settings.loc[n]
        require(row['T'] == 200 and row.beta == 1 and row.great_steps == 3
                and row.great_window == window, 'Stationary baseline settings changed')
        np.testing.assert_allclose([row.great_stepsize, row.covariance_alpha],
                                   [1/(5*window), alpha], rtol=0, atol=1e-15)
    pairs = pd.read_csv(results/'contact/data/paired.csv')
    require(len(pairs) == 3 and not pairs.control.duplicated().any(), 'Incomplete paired contact table')
    for row in pairs.itertuples():
        np.testing.assert_allclose([row.reduction, row.lower, row.upper], contact[row.control],
                                   rtol=1e-11, atol=1e-11)
    for name in ('stationary_settings', 'stationary_results', 'moving_averages'):
        require((tables/(name+'.tex')).is_file(), 'Missing TeX table: ' + name)
    return 'passed'


def check_plot_statistics(results):
    from lib.paper_plots import moving_interference_comparison, noise_adaptation
    from lib.plotting import plt

    def band(collection, expected):
        vertices = np.concatenate([path.vertices for path in collection.get_paths()])
        for time, lower, upper in expected:
            values = vertices[vertices[:, 0] == time, 1]
            require(len(values) > 0, 'Missing shaded-band reporting time')
            np.testing.assert_allclose([values.min(), values.max()], [lower, upper], rtol=1e-12, atol=1e-12)

    moving = pd.read_csv(results/'beamforming/data/moving.csv.gz')
    figure = moving_interference_comparison(ROOT, results)
    try:
        for ax, condition in zip(figure.axes, ('smooth', 'abrupt')):
            for i, method in enumerate(('mean', 'mode', 'isotropic', 'great_window')):
                groups = moving[moving.condition.eq(condition) & moving.method.eq(method)].groupby('t').true_ratio_db
                np.testing.assert_allclose(ax.lines[i].get_ydata(), groups.mean(), rtol=1e-12, atol=1e-12)
                band(ax.collections[i], [(time, group.quantile(.25), group.quantile(.75))
                                         for time, group in groups])
    finally:
        plt.close(figure)
    noise = pd.read_csv(results/'beamforming/data/noise.csv.gz')
    noise = noise[noise.control.eq('mean')]
    figure = noise_adaptation(ROOT, results)
    try:
        for ax, endpoint in zip(figure.axes[:2], ('sigma2_estimate', 'tau2_estimate')):
            groups = noise[noise.method.eq('learned')].groupby('t')[endpoint]
            np.testing.assert_allclose(ax.lines[1].get_ydata(), groups.median(), rtol=1e-12, atol=1e-12)
            band(ax.collections[0], [(time, group.quantile(.25), group.quantile(.75))
                                     for time, group in groups])
        for ax, endpoint in zip(figure.axes[2:], ('output_oracle_db', 'prequential_score')):
            for i, method in enumerate(('known', 'misspecified', 'learned')):
                groups = noise[noise.method.eq(method)].groupby('t')[endpoint]
                points = np.array([(time, *interval(group)) for time, group in groups])
                np.testing.assert_allclose(ax.lines[i].get_ydata(), points[:, 1], rtol=1e-12, atol=1e-12)
                band(ax.collections[i], points[:, [0, 2, 3]])
    finally:
        plt.close(figure)
    return 'passed'


def check_manifest(results):
    path = results/'manifest.json'
    if not path.exists():
        return 'absent; numerical cohort checked independently'
    manifest = json.loads(path.read_text())
    artifacts = manifest.get('artifacts')
    require(isinstance(artifacts, dict) and artifacts, 'Release manifest has no artifact hashes')
    for name, digest in artifacts.items():
        target = (results/name).resolve()
        require(target.is_relative_to(results.resolve()), 'Artifact path escapes results directory')
        if target.suffix == '.pdf':
            continue
        require(target.is_file(), 'Missing release artifact: ' + name)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        require(actual == digest, 'Release artifact checksum mismatch: ' + name)
    return 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    results = args.results.resolve()
    report = dict(contact=check_contact(results), stationary=check_stationary(results),
                  moving=check_moving(results), noise=check_noise(results))
    report['tables'] = check_tables(results, **report)
    report['plot_statistics'] = check_plot_statistics(results)
    report['figures'] = check_pdfs(results)
    report['manifest'] = check_manifest(results)
    print(json.dumps(dict(status='passed', independent_simulations=SIMULATIONS, **report), indent=2))


if __name__ == '__main__':
    main()
