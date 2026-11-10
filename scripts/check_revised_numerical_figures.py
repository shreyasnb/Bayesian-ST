"""Read-only audit of the revised numerical-section figure quantities.

Run from the repository root, or pass --root. It reads existing records only;
no simulation, fitting, posterior integration or inference is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

def audit(root, tex=None):
    root = Path(root).resolve()
    inputs = {}
    gate_name = 'results/beamforming/data/validation/validation.json'
    gate = json.loads((root / gate_name).read_text())
    assert gate['status'] == 'passed', 'Scientific validation did not pass'
    digest = hashlib.sha256()
    for name, expected in sorted(gate['sources'].items()):
        payload = (root / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected, ('Scientific source changed', name)
        digest.update(name.encode())
        digest.update(payload)
    assert digest.hexdigest() == gate['final_science_hash'], 'Scientific fingerprint changed'
    frozen_gate = {'status': 'passed', 'scientific_sha256': digest.hexdigest(), 'sources_checked': len(gate['sources'])}
    inputs[gate_name] = hashlib.sha256((root / gate_name).read_bytes()).hexdigest()
    manifests = {}
    for app, component in [('beamforming', 'performance'), ('beamforming', 'prediction'), ('contact', 'performance')]:
        name = f'results/{app}/data/paper/{component}/manifest.json'
        path = root / name
        manifests[app, component] = json.loads(path.read_text())
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read(name, app, component, kind='csv'):
        expected = manifests[app, component]['artifacts'][name]
        if isinstance(expected, dict):
            expected = expected['sha256']
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, (name, 'raw hash differs')
        inputs[name] = actual
        if kind == 'csv':
            return pd.read_csv(root / name)
        with np.load(root / name, allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}

    def interval(values):
        v = np.asarray(values, float)
        assert np.isfinite(v).all() and len(v) == 256
        mean = float(v.mean())
        half = float(student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
        return dict(mean=mean, lower=mean - half, upper=mean + half, simulations=len(v))

    def close(a, b, atol=1e-11):
        np.testing.assert_allclose(a, b, atol=atol, rtol=1e-11)
    raw = read('results/beamforming/data/paper/performance/stationary.csv', 'beamforming', 'performance')
    paired = read('results/beamforming/tables/paper/performance/paired_stationary.csv', 'beamforming', 'performance')
    selected = raw[raw.condition.eq('static') & raw.n.isin([32, 64, 128]) & raw.t.eq(raw.n)]
    assert not selected.duplicated(['n', 'method', 'trial']).any()
    close(selected.true_ratio_db, 10 * np.log10(selected.true_power / selected.oracle_power))
    close(selected.oracle_power, 0.2 / (1 - 0.5 / (0.2 + 1.0)))
    assert selected[selected.method.isin(['mean', 'mode', 'isotropic'])].beta.eq(1).all()
    expected = {'mean': 0.763263643767, 'mode': 0.689371667838, 'great_window': 0.690951042886, 'covariance_loading': 1.077038079841}
    stationary = {}
    for n in [32, 64, 128]:
        group = selected[selected.n.eq(n)]
        base = group[group.method.eq('mean')].set_index('trial').true_ratio_db
        target = paired[paired.condition.eq('static') & paired.n.eq(n) & paired.t.eq(n) & paired.endpoint.eq('true_ratio_db')]
        means = {}
        for method in expected:
            values = group[group.method.eq(method)].set_index('trial').true_ratio_db
            summary = interval(values)
            means[method] = summary
            if method != 'mean':
                difference = interval(values.loc[base.index] - base)
                saved = target[target.reference.eq(method)].iloc[0]
                close([difference['mean'], difference['lower'], difference['upper']], [saved.gain, saved.lower, saved.upper])
            if n == 64:
                close(summary['mean'], expected[method], atol=5e-12)
        oracle = target[target.reference.eq('oracle')].iloc[0]
        close(-oracle.gain, means['mean']['mean'])
        for method in ['mode', 'great_window', 'covariance_loading']:
            close(-oracle.gain + target[target.reference.eq(method)].gain.iloc[0], means[method]['mean'])
        stationary[str(n)] = means
    adaptation = read('results/beamforming/tables/paper/prediction/adaptation.csv', 'beamforming', 'prediction')
    adaptation_summary = read('results/beamforming/tables/paper/prediction/adaptation_summary.csv', 'beamforming', 'prediction')
    frame = adaptation[adaptation.condition.eq('noise_change') & adaptation.split.eq('test') & adaptation.control.eq('mean')].copy()
    report_times = list(range(32, 257, 32))
    assert sorted(frame.t.unique()) == report_times
    assert not frame.duplicated(['method', 'episode', 't']).any()
    assert frame.groupby(['method', 't']).episode.nunique().eq(256).all()
    close(frame.output_oracle_db, 10 * np.log10(frame.true_task_loss / frame.oracle_loss))
    close(frame.sigma2_true, np.where(frame.t <= 128, 0.2, 0.8))
    close(frame.tau2_true, 1.0)
    close(frame.time_s, (frame.t - 1) * 0.01)
    assert frame.prediction_scope.eq('Before assimilation of the scored observation').all()
    adaptation_checks = {}
    causality = {}
    for method, beta, expected_power, expected_log in [('known', 0.98, 0.37837206047412786, -40.86334812738405), ('misspecified', 0.9, 1.05629496634431, -237.44519598295508), ('learned', 0.98, 0.43463276974461973, -53.25556493147081)]:
        group = frame[frame.method.eq(method)]
        assert group.beta.eq(beta).all()
        assert group.block_length.eq(32).all() and group.em_iterations.eq(2).all()
        summaries = {}
        for endpoint, expected_mean in [('output_oracle_db', expected_power), ('prequential_score', expected_log)]:
            per_simulation = group.groupby('episode')[endpoint].mean()
            summary = interval(per_simulation)
            close(summary['mean'], expected_mean)
            old = adaptation_summary[adaptation_summary.condition.eq('noise_change') & adaptation_summary.control.eq('mean') & adaptation_summary.method.eq(method) & adaptation_summary.endpoint.eq(endpoint)].iloc[0]
            close([summary['mean'], summary['lower'], summary['upper']], [old['mean'], old.lower, old.upper])
            summaries[endpoint] = summary
        pointwise = {str(int(t)): interval(g.output_oracle_db) for t, g in group.groupby('t')}
        summaries.update(beta=beta, pointwise_exact_power=pointwise)
        adaptation_checks[method] = summaries
        if method == 'known':
            continue
        comparisons = 0
        max_difference = 0.0
        for _, g in group.groupby('episode'):
            g = g.sort_values('t')
            for field in ['sigma2', 'tau2']:
                used = g[field + '_estimate'].to_numpy()
                future = g[field + '_next'].to_numpy()
                close(used[1:], future[:-1])
                dense = np.full(256, used[0])
                for _, row in g.iterrows():
                    dense[int(row.t):] = row[field + '_next']
                max_difference = max(max_difference, float(np.max(np.abs(dense[g.t.to_numpy(int) - 1] - used))))
                comparisons += len(used)
                close(dense[g.t.to_numpy(int) - 1], used)
        causality[method] = dict(pre_update_vs_dense_reconstruction_comparisons=comparisons, maximum_absolute_difference=max_difference, fit_effective_index='t+1', last_fit_after_t256_outside_display=True)
    assert frame[frame.method.eq('learned')].fit_performed.eq(True).all()
    assert not frame[frame.method.eq('misspecified')].fit_performed.any()
    contact = read('results/contact/tables/paper/performance/metrics.csv', 'contact', 'performance')
    contact_paired = read('results/contact/tables/paper/performance/paired.csv', 'contact', 'performance')
    config = manifests['contact', 'performance']['config']
    v0 = float(config['v0'])
    assert v0 == 0.02
    identity = v0 * v0 * contact.response_map_frobenius / 3.0
    close(contact.response_mse, identity, atol=2e-16)
    counts = {}
    contact_means = {}
    gains = {}
    for condition, endpoint, times in [('static_concentrated', 'checkpoint', [1, 4, 16, 32]), ('sphere', 'episode_average', [192])]:
        group = contact[contact.condition.eq(condition) & contact.endpoint.eq(endpoint) & contact.t.isin(times)]
        assert group.groupby(['method', 't']).trial.nunique().eq(256).all()
        assert group[group.method.isin(['mean', 'mode', 'isotropic'])].beta.eq(1.0 if condition == 'static_concentrated' else 0.9).all()
        counts[condition] = dict(simulations=256, reported_indices=times)
        contact_means[condition] = {}
        for (method, t), g in group.groupby(['method', 't']):
            contact_means[condition][f'{method}:t{int(t)}'] = interval(g.response_mse * 1000000.0)
        terminal = group[group.t.eq(16 if condition == 'static_concentrated' else 192)]
        mean = terminal[terminal.method.eq('mean')].set_index('trial').response_mse
        gains[condition] = {}
        for method in ['mode', 'isotropic']:
            comparison = terminal[terminal.method.eq(method)].set_index('trial').response_mse
            result = interval((comparison.loc[mean.index] - mean) * 1000000.0)
            stored = contact_paired[contact_paired.condition.eq(condition) & contact_paired.endpoint.eq(endpoint) & contact_paired.t.eq(16 if condition == 'static_concentrated' else 192) & contact_paired.reference.eq(method) & contact_paired.metric.eq('response_mse')].iloc[0]
            close([result['mean'], result['lower'], result['upper']], 1000000.0 * np.array([stored.gain, stored.lower, stored.upper]), atol=1e-09)
            gains[condition][method] = result
    anisotropy = read('results/contact/data/paper/performance/anisotropy.npz', 'contact', 'performance', kind='npz')
    close(anisotropy['true_normal'], [0, 0, 1])
    assert int(anisotropy['report_time']) == 16
    V = anisotropy['input_vectors']
    F = (V[:, :2].T @ V[:, :2] + np.sum(V[:, 2] ** 2) * np.eye(2)) / 0.01 ** 2
    close(np.linalg.eigvalsh(F), anisotropy['fisher_eigenvalues'])
    divergences = []
    for angle, z in [(5.0, [np.tan(np.deg2rad(5.0)), 0.0]), (30.0, [0.0, np.tan(np.deg2rad(30.0))])]:
        z = np.asarray(z)
        normal = np.r_[-z, 1.0] / np.sqrt(1 + z @ z)
        P = np.eye(3) - np.outer(normal, normal)
        P0 = np.diag([1.0, 1.0, 0.0])
        direct = float(np.sum((V @ (P - P0).T) ** 2) / (2 * 0.01 ** 2))
        formula = float(z @ F @ z / (2 * (1 + z @ z)))
        close(direct, formula)
        divergences.append(dict(angle_deg=angle, divergence=formula))
    q = anisotropy['unit_queries']
    physical = anisotropy['physical_queries']
    close(physical, v0 * q)
    risk_errors = {}
    for label, mean_key, risk_key in [('full', 'plane_mean', 'risks_full'), ('isotropic', 'iso_plane_mean', 'risks_isotropic')]:
        M = anisotropy[mean_key]
        risk = np.einsum('qi,ij,qj->q', q, M - M @ M, q)
        normalized = np.einsum('qi,ij,qj->q', physical, M - M @ M, physical) / (v0 * v0)
        close(risk, anisotropy[risk_key])
        close(normalized, anisotropy[risk_key])
        risk_errors[label] = float(np.max(abs(normalized - anisotropy[risk_key])))
    close(np.round(anisotropy['fisher_eigenvalues'], 2), [1.85, 63.08])
    close(np.round(anisotropy['curvature_eigenvalues'], 2), [6.74, 74.42])
    close(np.round([x['divergence'] for x in divergences], 3), [0.24, 0.231])
    information = dict(fisher_eigenvalues=anisotropy['fisher_eigenvalues'].tolist(), curvature_eigenvalues=anisotropy['curvature_eigenvalues'].tolist(), marked_KL=divergences, normalized_physical_risk_maximum_disagreements=risk_errors)
    for condition, expected_values in [('static_concentrated', {'mode': [7.12, 4.33, 9.91], 'isotropic': [1.37, 0.73, 2.02]}), ('sphere', {'mode': [1.73, 1.22, 2.25], 'isotropic': [0.34, 0.23, 0.46]})]:
        for method, expected_values_for_method in expected_values.items():
            actual = gains[condition][method]
            close(np.round([actual['mean'], actual['lower'], actual['upper']], 2), expected_values_for_method)
    for method, expected_power, expected_log in [('learned', 0.435, -53.3), ('misspecified', 1.056, -237.4), ('known', 0.378, -40.9)]:
        close(round(adaptation_checks[method]['output_oracle_db']['mean'], 3), expected_power)
        close(round(adaptation_checks[method]['prequential_score']['mean'], 1), expected_log)
    basis = np.linalg.qr(np.array([[1.0, 0.2, 0.3], [-0.4, 1.0, 0.1], [0.1, -0.2, 1.0]]))[0]
    close(basis.T @ basis, np.eye(3))
    test_difference = np.array([[0.7, 0.2, -0.1], [0.2, -0.4, 0.3], [-0.1, 0.3, 0.1]])
    rotation = np.linalg.qr(np.array([[0.3, 0.4, 1.0], [1.0, -0.2, 0.5], [0.1, 1.0, -0.3]]))[0]
    query_basis = v0 * basis.T @ rotation
    mean_squared_error = float(np.mean(np.sum((query_basis @ test_difference.T) ** 2, axis=1)))
    close(mean_squared_error, v0 * v0 * np.linalg.norm(test_difference, 'fro') ** 2 / 3)
    source_names = ['lib/final_beamforming.py', 'lib/final_prediction.py', 'lib/causal_variance.py', 'lib/final_contact.py']
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names}
    manuscript = {'path': str(tex), 'present': bool(tex and Path(tex).is_file())}
    if manuscript['present']:
        manuscript['sha256'] = hashlib.sha256(Path(tex).read_bytes()).hexdigest()
    return dict(status='passed', scope='Independent raw-record and algebraic verification only; no new simulation, selection, fit or posterior integration.', manuscript=manuscript, frozen_scientific_gate=frozen_gate, stationary_exact_power=stationary, adaptation_exact_power_and_log_density=adaptation_checks, causality=causality, true_schedule=dict(t_start=1, T=256, L=32, sigma2_before_inclusive_t128=0.2, sigma2_from_t129=0.8, tau2=1.0, event_line_index=128.5), contact=dict(counts=counts, mean_squared_velocity_error_mm2_s2=contact_means, paired_reductions_mm2_s2=gains, unit_factor=1000000.0, coordinate_sum_not_extra_coordinate_average=True, orthonormal_query_basis_invariance_checked=True, response_loss_vs_frobenius_identity_maximum_error=float(np.max(abs(contact.response_mse - identity))), v0_m_s=v0, information_and_risk=information), raw_input_sha256=inputs, producer_source_sha256=sources, limitations=['The full per-time contact results at all 48 action times are not retained for every simulation; the saved episode_average records and producer source establish the requested time averaging. These aggregate records independently satisfy the Frobenius identity and paired-summary checks.', 'Contact evaluation uses a fixed rotated orthonormal query basis rather than the literal canonical basis; the three-query average equals v0^2||D-P||_F^2/3 and therefore equals the revised definition for any measured rotation. No metric rescaling is needed.', 'The audit does not recompute Bingham normalizers or predictive densities; it verifies the saved normalized prequential_score and its source definition.'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tex', type=Path, help='Optional revised numerical-section TeX source, recorded for provenance.')
    parser.add_argument('--report', '--output', dest='output', type=Path, help='Optional JSON report path; without this option no file is written.')
    args = parser.parse_args()
    report = audit(args.root, args.tex)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'stationary_n64': report['stationary_exact_power']['64'], 'adaptation_means': {m: {k: v['mean'] for k, v in d.items() if isinstance(v, dict) and 'mean' in v} for m, d in report['adaptation_exact_power_and_log_density'].items()}, 'contact_gains': report['contact']['paired_reductions_mm2_s2']}, indent=2))
if __name__ == '__main__':
    main()
