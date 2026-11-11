"""Independently audit dense contact diagnostics without replaying simulations.

Only sealed tables, native accuracy certificates, and the original saved
simulation-zero matrices are read. No tracker, plant, selection, or benchmark
routine is invoked. The optional report is written only to a new location.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, t as student_t


OUTPUT = 'results/contact_transport_diagnostics'
TIMES = np.arange(4, 193, 4)
METHODS = ('mean', 'mode', 'isotropic', 'regularized', 'great_window',
           'great_ew', 'mean_identity', 'mode_identity')
METRICS = ('response_mse', 'response_map_frobenius', 'command_cost',
           'posterior_response_excess', 'posterior_command_excess',
           'command_excess_identity_error')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert(value, message):
    if not value:
        raise AssertionError(message)


def _close(actual, expected, *, atol=1e-10, rtol=1e-10):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _interval(values):
    values = np.asarray(values, float)
    _assert(len(values) == 256 and np.isfinite(values).all(), 'Incomplete simulation cohort')
    mean = float(values.mean())
    half = float(student_t.ppf(.975, 255)*np.std(values, ddof=1)/np.sqrt(256))
    return {'simulations': 256, 'mean': mean, 'lower': mean-half, 'upper': mean+half}


def _distribution(values):
    values = np.asarray(values, float)
    result = _interval(values)
    result.update(fraction_positive=float(np.mean(values > 0)), median=float(np.median(values)),
                  minimum=float(values.min()), maximum=float(values.max()),
                  sd=float(np.std(values, ddof=1)), skewness=float(skew(values, bias=False)))
    for name, probability in [('q01', .01), ('q05', .05), ('q25', .25),
                              ('q75', .75), ('q95', .95), ('q99', .99)]:
        result[name] = float(np.quantile(values, probability))
    absolute = np.sort(np.abs(values))[::-1]
    positive = np.sort(np.maximum(values, 0))[::-1]
    result['top_1pct_absolute_contribution'] = float(absolute[:3].sum()/absolute.sum())
    result['top_5pct_absolute_contribution'] = float(absolute[:13].sum()/absolute.sum())
    result['top_5pct_positive_contribution'] = float(positive[:13].sum()/positive.sum()) if positive.sum() else 0.
    return result


def _check_statistics(row, statistics):
    for name, value in statistics.items():
        _close(float(row[name]), value, atol=2e-10, rtol=2e-10)


def _alias(method):
    return 'iso' if method == 'isotropic' else method


def _long_summary(frame, table, metrics):
    count = 0
    for measurement, group in frame.groupby('t'):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            selected = table[table.t.eq(measurement) & table.metric.eq(metric)]
            if not np.isfinite(values).all():
                _assert(selected.empty, 'A nonfinite initial increment was summarized')
                continue
            _assert(len(selected) == 1, 'Missing diagnostic summary')
            _check_statistics(selected.iloc[0], _interval(values))
            _close(selected.iloc[0][['q25', 'q75']].to_numpy(float), np.quantile(values, [.25, .75]))
            _assert(bool(selected.iloc[0].deterministic) == bool(np.ptp(values) == 0),
                    'Incorrect deterministic/Monte Carlo distinction')
            count += 1
    _assert(count == len(table), 'Unexpected diagnostic summary rows')
    return count


def _story_audit(root, physical, diagnostics, dense):
    """Independent frame, likelihood-increment, Gram, and curvature algebra."""
    path = root/'results/contact/data/paper/performance/story.npz'
    with np.load(path, allow_pickle=False) as saved:
        story = {name: saved[name] for name in saved.files}
    sphere_R = story['sphere_R']
    np.testing.assert_array_equal(sphere_R, story['plane_R'])
    np.testing.assert_array_equal(story['sphere_G'], story['plane_G'])
    geometry = {}
    for condition in ('plane', 'sphere'):
        R, G, P = [story[condition+'_'+key] for key in ('R', 'G', 'P')]
        normals = story[condition+'_world_normal']
        world_P = np.eye(3)-np.einsum('ti,tj->tij', normals, normals)
        previous_R = np.concatenate([np.eye(3)[None], R[:-1]])
        _close(G, np.einsum('tji,tjk->tik', R, previous_R), atol=1e-12, rtol=0.)
        _close(P, np.einsum('tji,tjk,tkl->til', R, world_P, R), atol=1e-12, rtol=0.)
        _close(story[condition+'_normal'], np.einsum('tji,tj->ti', R, normals), atol=1e-12, rtol=0.)
        for key in ('U', 'V', 'clean'):
            _close(story[condition+'_'+key], np.einsum('tji,tj->ti', R, story[condition+'_world_'+key]), atol=1e-12, rtol=0.)
        _close(story[condition+'_forces'], 10*story[condition+'_world_V'], atol=1e-12, rtol=0.)
        transported = np.einsum('tij,tjk,tlk->til', G[1:], P[:-1], G[1:])
        uncorrected = np.linalg.norm(np.diff(P, axis=0), axis=(1, 2))
        compensated = np.linalg.norm(P[1:]-transported, axis=(1, 2))
        world = np.linalg.norm(np.diff(world_P, axis=0), axis=(1, 2))
        _close(compensated, world, atol=2e-14, rtol=0.)
        if condition == 'plane':
            _assert(compensated.max() < 2e-14, 'Transported rotating-plane state should be constant')
        geometry[condition] = {
            'mean_sensor_increment_fro': float(uncorrected.mean()),
            'mean_compensated_increment_fro': float(compensated.mean()),
            'maximum_compensated_vs_world_error': float(np.max(abs(compensated-world))),
        }
        if condition == 'sphere':
            rows = physical[physical.trial.eq(0)].sort_values('t')
            _close(rows.sensor_increment_fro.iloc[1:], uncorrected)
            _close(rows.transported_increment_fro.iloc[1:], compensated)
            _close(rows.world_increment_fro.iloc[1:], world)
    R, G = sphere_R, story['sphere_G']
    speed = np.arccos(np.clip((np.trace(G, axis1=1, axis2=2)-1)/2, -1., 1.))/.05
    for _, group in physical.groupby('trial'):
        _close(group.sort_values('t').frame_angular_speed_rad_s, speed, atol=1e-11, rtol=0.)
    # Reconstruct algebra only; no posterior integrator or simulation is called.
    A = np.zeros((3, 3)); identity_A = A.copy(); world_A = A.copy()
    S = A.copy(); identity_S = A.copy(); mass = 0.
    sigma2 = .004**2
    rows = diagnostics[diagnostics.trial.eq(0)].set_index('t')
    for j, (u, v, g) in enumerate(zip(story['sphere_U'], story['sphere_V'], G)):
        increment = (np.outer(u, v)+np.outer(v, u)-np.outer(v, v))/(2*sigma2)
        A = .9*g@A@g.T+increment
        A -= np.trace(A)*np.eye(3)/3
        identity_A = .9*identity_A+increment
        identity_A -= np.trace(identity_A)*np.eye(3)/3
        world_u, world_v = story['sphere_world_U'][j], story['sphere_world_V'][j]
        world_A = .9*world_A+(np.outer(world_u, world_v)+np.outer(world_v, world_u)
                             -np.outer(world_v, world_v))/(2*sigma2)
        world_A -= np.trace(world_A)*np.eye(3)/3
        _close(A, story['sphere_A'][j], atol=1e-10, rtol=1e-10)
        _close(A, R[j].T@world_A@R[j], atol=1e-10, rtol=1e-10)
        S = .9*g@S@g.T+np.outer(v, v)
        identity_S = .9*identity_S+np.outer(v, v)
        mass = .9*mass+1.
        normal = story['sphere_normal'][j]
        row = physical[physical.trial.eq(0) & physical.t.eq(j+1)].iloc[0]
        _close(row.normal_input_fraction, normal@S@normal/np.trace(S))
        if j+1 not in TIMES:
            continue
        row = rows.loc[j+1]
        _close(row.effective_input_mass, mass)
        for matrix, prefix in ((S, ''), (identity_S, 'identity_')):
            spectrum = np.linalg.eigvalsh((matrix+matrix.T)/(2*sigma2*mass))
            _close([row['input_gram_'+prefix+'lambda_'+str(k)] for k in (1, 2, 3)], spectrum)
        for matrix, prefix in ((A, ''), (identity_A, 'identity_')):
            spectrum = np.linalg.eigvalsh(matrix)
            _close([row['curvature_'+prefix+'min'], row['curvature_'+prefix+'max']],
                   2*(spectrum[1:]-spectrum[0]))
    # The original saved means/modes have all 48 action times. Verify their
    # response losses directly using the common orthonormal query identity.
    times = story['sphere_report_times']
    np.testing.assert_array_equal(times, TIMES)
    for method, key in [('mean', 'M'), ('mode', 'Pmode'), ('isotropic', 'isotropic')]:
        matrices = story['sphere_'+key]
        errors = np.linalg.norm(matrices-story['sphere_P'][times-1], axis=(1, 2))**2*.02**2/3
        recorded = dense[dense.trial.eq(0) & dense.method.eq(method) & dense.endpoint.eq('checkpoint')].sort_values('t')
        _close(recorded.response_mse, errors, atol=2e-16, rtol=1e-10)
    return {'geometry': geometry, 'plane_and_sphere_same_measured_frame_schedule': True,
            'trial_zero_all_192_likelihood_corrections_reconstructed': True,
            'fixed_world_frame_identity_transport_equals_sensor_known_transport_all_192_times': True,
            'trial_zero_all_48_input_gram_and_curvature_records_verified': True,
            'trial_zero_all_48_main_method_response_losses_verified': True,
            'frame_speed_degrees_per_step_mean_max': np.rad2deg([speed[1:].mean()*.05, speed[1:].max()*.05]).tolist()}


def fullcheck(root):
    root = Path(root).resolve(); base = root/OUTPUT
    manifest_path = base/'data/manifest.json'
    meta = json.loads(manifest_path.read_text())
    _assert(meta['status'] == 'completed', 'Dense recovery has not completed')
    for group in ('input_hashes', 'source_hashes', 'artifacts'):
        for name, digest in meta[group].items():
            _assert(_sha(root/name) == digest, 'Changed sealed evidence: '+name)
    config = meta['config']
    expected = {'n': 3, 'k': 2, 'T': 192, 'dt': .05, 'v0': .02, 'resistance': 10.,
                'radius': .05, 'sigma2': .004**2, 'beta': .9, 'test_simulations': 256,
                'original_validation_simulations': 64, 'test_seed': 2026111002,
                'validation_seed': 2026111001, 'stage_seed_offset': 0,
                'moment_tolerance': 5e-8, 'algebra_tolerance': 1e-10}
    for name, value in expected.items():
        _assert(config[name] == value, 'Changed original setting: '+name)
    np.testing.assert_array_equal(config['action_times'], TIMES)
    _assert(config['methods'] == list(METHODS), 'Changed method cohort')
    original = pd.read_csv(root/'results/contact/tables/paper/performance/metrics.csv')
    selected = json.loads((root/'results/contact/data/paper/performance/selected.json').read_text())
    _assert(config['selected_settings'] == next(row for row in selected if row['condition'] == 'sphere'),
            'Changed selected settings')
    dense = pd.read_csv(base/'data/dense_metrics.csv.gz')
    physical = pd.read_csv(base/'data/physical_diagnostics.csv.gz')
    diagnostics = pd.read_csv(base/'data/trial_diagnostics.csv.gz')
    trial_checks = json.loads((base/'data/trial_checks.json').read_text())
    _assert(len(dense) == 256*49*8 and len(physical) == 256*192 and len(diagnostics) == 256*48,
            'Incomplete dense records')
    _assert(dense.numerical_ok.astype(bool).all(), 'Unresolved simulation was retained')
    _assert(dense.moment_resolution_error.max() < 5e-8, 'Relaxed moment accuracy')
    _close(dense.beta, .9, rtol=0., atol=0.); _close(dense.sigma2, .004**2, rtol=0., atol=1e-20)
    _close(dense.time_s, (dense.t-1)*.05, rtol=0., atol=1e-14)
    for (trial, method), group in dense.groupby(['trial', 'method']):
        np.testing.assert_array_equal(group[group.endpoint.eq('checkpoint')].sort_values('t').t, TIMES)
        _assert(len(group[group.endpoint.eq('episode_average')]) == 1, 'Duplicate run average')
    _close(dense.response_mse, dense.response_map_frobenius*.02**2/3, atol=2e-16, rtol=1e-10)
    old = original[original.condition.eq('sphere')]
    keys = ['condition', 'trial', 'method', 't', 'endpoint']
    joined = old.merge(dense, on=keys, suffixes=('_old', '_dense'), validate='one_to_one')
    _assert(len(joined) == len(old) == 256*56, 'Missing original checkpoints or full-run endpoints')
    errors = {}
    for field in (*METRICS, 'time_s', 'beta', 'sigma2', 'moment_resolution_error'):
        a, b = joined[field+'_old'].to_numpy(float), joined[field+'_dense'].to_numpy(float)
        _close(b, a)
        errors[field] = float(np.max(abs(a-b)))
    for field in ('numerical_ok', 'mode_unique'):
        np.testing.assert_array_equal(joined[field+'_old'], joined[field+'_dense'])
    checkpoints = dense[dense.endpoint.eq('checkpoint')]
    averages = dense[dense.endpoint.eq('episode_average')].set_index(['trial', 'method'])
    recomputed = checkpoints.groupby(['trial', 'method'])[list(METRICS)].mean()
    _close(averages.loc[recomputed.index, list(METRICS)], recomputed, atol=1e-10, rtol=1e-10)
    tables = {path.stem: pd.read_csv(path) for path in (base/'tables').glob('*.csv')}
    for (endpoint, t, method), group in dense.groupby(['endpoint', 't', 'method']):
        np.testing.assert_array_equal(np.sort(group.trial), np.arange(256))
        table = tables['checkpoint_summary' if endpoint == 'checkpoint' else 'episode_average_summary']
        row = table[table.t.eq(t) & table.method.eq(_alias(method))]
        _assert(len(row) == 1, 'Missing response-error mean/CI')
        _check_statistics(row.iloc[0], _interval(group.response_mse.to_numpy()*1e6))
    late = {}; late_arrays = {}
    for condition in ('sphere', 'plane'):
        frame = checkpoints if condition == 'sphere' else original[original.condition.eq('plane') & original.endpoint.eq('checkpoint')]
        table = tables['paired_rise' if condition == 'sphere' else 'plane_control_late_rise']
        late[condition] = {}
        for method in METHODS:
            wide = frame[frame.method.eq(method)].pivot(index='trial', columns='t', values='response_mse')*1e6
            np.testing.assert_array_equal(wide.index, np.arange(256))
            values = (wide[192]-wide[128]).to_numpy()
            stats = _distribution(values)
            row = table[table.method.eq(_alias(method))]
            _assert(len(row) == 1, 'Missing paired late-change summary')
            _check_statistics(row.iloc[0], stats)
            late[condition][_alias(method)] = stats
            if condition == 'sphere':
                late_arrays[_alias(method)] = values
                raw = tables['late_rise_trials']; raw = raw[raw.method.eq(_alias(method))].sort_values('trial')
                _close(raw.difference_mm2_s2, values)
                _close(raw.error128_mm2_s2, wide[128]); _close(raw.error192_mm2_s2, wide[192])
    contrast = late_arrays['mean']-late_arrays['mean_identity']
    _check_statistics(tables['paired_rise_control'].iloc[0], _distribution(contrast))
    _close(tables['paired_rise_control_trials'].sort_values('trial').difference_mm2_s2, contrast)
    for prefix in ('transported', 'identity'):
        for suffix in ('mean_resolution_error', 'log_normalizer_resolution_error'):
            _assert(np.isfinite(diagnostics[prefix+'_'+suffix]).all()
                    and diagnostics[prefix+'_'+suffix].max() < 5e-8, 'Unresolved native moments')
    _assert(len(trial_checks) == 256, 'Missing per-trial native certificates')
    for trial, check in enumerate(trial_checks):
        _assert(check['trial'] == trial and check['status'] == 'passed'
                and check['native_checked_moment_calls'] == 96
                and check['original_tolerance'] == 5e-8, 'Changed native accuracy certificate')
        _assert(check['maximum_mean_resolution_error'] < 5e-8
                and check['maximum_log_normalizer_resolution_error'] < 5e-8, 'Unresolved native certificate')
    for table, frame, metrics in [
        ('projector_changes', physical, ['sensor_increment_fro', 'transported_increment_fro', 'world_increment_fro']),
        ('frame_rotation', physical, ['frame_angular_speed_rad_s']),
        ('excitation', diagnostics, ['input_gram_lambda_1', 'input_gram_lambda_2', 'input_gram_lambda_3']),
        ('curvature', diagnostics, ['curvature_min', 'curvature_max', 'curvature_identity_min', 'curvature_identity_max']),
        ('numerical_accuracy', diagnostics, [prefix+'_'+suffix for prefix in ('transported', 'identity')
                                           for suffix in ('mean_resolution_error', 'log_normalizer_resolution_error', 'angular_order')])]:
        _long_summary(frame, tables[table], metrics)
    _close(physical.transported_increment_fro.iloc[np.flatnonzero(physical.t.to_numpy() > 1)],
           physical.world_increment_fro.iloc[np.flatnonzero(physical.t.to_numpy() > 1)], atol=1e-10, rtol=1e-10)
    story = _story_audit(root, physical, diagnostics, dense)
    return {
        'status': 'passed', 'checker_sha256': _sha(Path(__file__)),
        'manifest_sha256': _sha(manifest_path), 'raw_rows': len(dense),
        'simulations': 256, 'action_times_per_run': 48, 'original_parity_rows': len(joined),
        'maximum_original_absolute_errors': errors,
        'maximum_mean_resolution_error': float(max(diagnostics.transported_mean_resolution_error.max(), diagnostics.identity_mean_resolution_error.max())),
        'maximum_log_normalizer_resolution_error': float(max(diagnostics.transported_log_normalizer_resolution_error.max(), diagnostics.identity_log_normalizer_resolution_error.max())),
        'original_tolerance': 5e-8, 'paired_late_change_mm2_s2': late,
        'paired_late_change_mean_minus_identity_mm2_s2': _distribution(contrast),
        'saved_simulation_zero_algebra': story,
        'input_hashes': meta['input_hashes'], 'source_hashes': meta['source_hashes'],
        'interpretation': [
            'Positive paired late change means E_192 exceeds E_128 within the same simulation; its CI uses all 256 paired differences.',
            'Known-G and identity-G controls use the same observations, requested inputs, variance, beta, and correction times; only prior transport differs.',
            'Identity transport in a fixed world frame is algebraically equivalent to known transport in the moving sensor frame. Identity transport applied directly to rotating sensor data is a different, misspecified control.',
            'Plane and sphere share the same measured-frame schedule. Cross-condition differences cannot be attributed to faster wrist rotation.',
            'Measured-frame transport does not remove physical motion of the world contact normal. Forgetting remains an approximation for that changing physical state, so these diagnostics do not prove posterior calibration under arbitrary contact drift.',
            'Compensated projector increments isolate physical plane changes. Untransported increments mix wrist rotation with physical motion and can cancel.',
            'The observable normalized input Gram matrix is an excitation diagnostic, not tangent Fisher information. Posterior curvature is the actual Bingham mode Hessian.',
            'The full-run endpoint averages all 48 action-time losses within each simulation; it is not a six-checkpoint average.',
        ],
        'scope_limits': [
            'All 256 native certificates and original saved endpoints are independently checked. Complete matrices are saved only for simulation zero; its frame, likelihood, Gram, curvature and main-method loss algebra are independently reconstructed.',
            'The checker verifies native resolution bounds without rerunning the plant, posterior integration, parameter selection, or benchmark algorithms.',
            'Student intervals describe means across independent episodes and do not establish deterministic robustness to arbitrary unmodeled dynamics.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path)
    arguments = parser.parse_args()
    result = fullcheck(arguments.root)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
