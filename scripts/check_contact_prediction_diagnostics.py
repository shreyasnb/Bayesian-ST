"""Read-only audit of the optional contact-prediction diagnostic notebook.

Route discovery imports no plotting or numerical packages. The full audit
checks existing observations, derived statistics, posterior geometry and saved
figures; it does not execute a notebook or change scientific evidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_moving_beamforming import _source, _sha, _verify_hashes, _exports_check

NOTEBOOK = 'notebooks/contact_prediction_diagnostics.ipynb'
OUTPUT = Path('results/contact_prediction_diagnostics')
LOCAL_DIRECTORY = OUTPUT / 'figures'
FILENAMES = {
    'prediction_error_stationary.pdf': 'main',
    'prediction_error_sphere.pdf': 'main',
    'prediction_error_references.pdf': 'supplementary',
    'moving_sphere_posterior_spheres.pdf': 'supplementary',
}


def optional_contact_diagnostic_routes(root=ROOT):
    """Return manuscript destinations in the notebook PDF-export schema."""
    from lib.notebook_pdf_exports import _path_parts, _qualified
    root = Path(root).resolve()
    path = root / NOTEBOOK
    if not path.is_file():
        return {}
    notebook = json.loads(path.read_text())
    meta = notebook.get('metadata', {}).get('editable_plotting', {})
    if meta.get('output_directory') != LOCAL_DIRECTORY.as_posix():
        raise ValueError('Contact diagnostic exports must use ' + LOCAL_DIRECTORY.as_posix())
    routes = {}
    for index, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code':
            continue
        filename = section = None
        declared = False
        for node in ast.parse(_source(cell)).body:
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if 'PDF_PATH' in names:
                    parts = _path_parts(node.value)
                    if (len(parts) != 2 or parts[0] != ('variable', 'EDITABLE_OUTPUT')
                            or parts[1][0] != 'literal' or parts[1][1] not in FILENAMES):
                        raise ValueError(f'{NOTEBOOK}, cell {index}: unexpected PDF_PATH')
                    filename, section, declared = parts[1][1], None, False
                if 'DRAFT_SECTION' in names:
                    section = ast.literal_eval(node.value)
                    if section not in ('main', 'supplementary'):
                        raise ValueError('Unexpected contact diagnostic section')
                if 'DRAFT_PATH' in names:
                    expected = [('variable', 'ROOT'), ('literal', 'results'),
                        ('variable', 'DRAFT_SECTION'), ('literal', 'contact'),
                        ('variable', 'PDF_PATH.name')]
                    if (filename is None or section != FILENAMES[filename]
                            or _path_parts(node.value) != expected):
                        raise ValueError(f'{NOTEBOOK}, cell {index}: inconsistent DRAFT_PATH')
                    destination = (Path('results') / section / 'contact' / filename).as_posix()
                    if destination in routes:
                        raise ValueError('Duplicate contact diagnostic export: ' + destination)
                    routes[destination] = dict(notebook=NOTEBOOK, cell=index,
                        example='contact', section=section, filename=filename,
                        local_pdf=(LOCAL_DIRECTORY / filename).as_posix())
                    declared = True
            value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
            if isinstance(value, ast.Call) and _qualified(value.func) in ('plt.figure', 'plt.subplots'):
                if not declared:
                    raise ValueError(f'{NOTEBOOK}, cell {index}: undeclared figure')
                declared = False
        if declared:
            raise ValueError(f'{NOTEBOOK}, cell {index}: PDF declaration without a figure')
    if len(routes) != 4 or {row['filename'] for row in routes.values()} != set(FILENAMES):
        raise ValueError('Expected all four contact diagnostic figure declarations')
    return routes


def _notebook_check(root, notebook, routes, required_inputs):
    from lib.notebook_pdf_exports import _qualified
    meta = notebook['metadata']['editable_plotting']
    inputs = dict(meta.get('input_hashes', {}))
    inputs.update(meta.get('data_hashes', {}))
    sources = dict(meta.get('loader_sources', {}))
    if meta.get('loader'):
        sources[meta['loader']] = meta['loader_sha256']
    _verify_hashes(root, inputs, 'contact diagnostic input')
    _verify_hashes(root, sources, 'contact diagnostic producer')
    for name, expected in required_inputs.items():
        assert inputs.get(name) == expected, 'Unrecorded contact input: ' + name
    for name in ('lib/contact_prediction_diagnostics.py', 'lib/contact_moving_snapshot.py'):
        assert sources.get(name) == _sha(root/name), 'Missing current loader source: ' + name
    by_id = {cell.get('id'): cell for cell in notebook['cells']}
    for cell_id, expected in meta.get('numerical_cell_hashes', {}).items():
        assert hashlib.sha256(_source(by_id[cell_id]).encode()).hexdigest() == expected
    count = images = figures = 0
    tex_true = False
    figure_indices = {row['cell'] for row in routes.values()}
    for index, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code' or not _source(cell).strip():
            continue
        count += 1
        assert cell.get('execution_count') is not None, f'Unexecuted contact diagnostic cell {index}'
        tree = ast.parse(_source(cell))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        constructors = [node for node in calls if _qualified(node.func) in ('plt.figure', 'plt.subplots')]
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == 'text.usetex':
                        assert ast.literal_eval(value) is True
                        tex_true = True
        if constructors:
            assert index in figure_indices and len(constructors) == 1
            assigned = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                        for target in node.targets if isinstance(target, ast.Name)}
            assert ast.literal_eval(assigned['DPI']) == 300
            assert tuple(ast.literal_eval(assigned['FIGSIZE'])) == (15, 5)
            assert ast.literal_eval(assigned['SAVE_PDF']) is True
            for node in constructors:
                kwargs = {key.arg: key.value for key in node.keywords}
                assert isinstance(kwargs.get('dpi'), ast.Name) and kwargs['dpi'].id == 'DPI'
                assert isinstance(kwargs.get('figsize'), ast.Name) and kwargs['figsize'].id == 'FIGSIZE'
            assert sum(_qualified(call.func) == 'plt.show' for call in calls) == 1
            assert any(isinstance(call.func, ast.Attribute) and call.func.attr == 'savefig' for call in calls)
            figures += 1
        for call in calls:
            assert _qualified(call.func) not in ('exec', 'eval', 'show', 'show_saved')
        for output in cell.get('outputs', []):
            assert output.get('output_type') != 'error'
            images += int('image/png' in output.get('data', {}))
    assert tex_true and figures == images == 4
    return dict(code_cells=count, visible_figures=figures, embedded_images=images,
                text_usetex=True, dpi=300, figure_size_in=[15, 5],
                numerical_inputs=len(inputs), producer_sources=len(sources))


def _numeric_check(root, data):
    """Recompute displayed summaries, unit conversion and paired intervals."""
    import numpy as np
    import pandas as pd
    from scipy.stats import t as student_t

    def interval(values):
        values = np.asarray(values, float)
        assert len(values) == 256 and np.isfinite(values).all()
        mean = float(values.mean())
        half = float(student_t.ppf(.975, 255)*values.std(ddof=1)/16)
        return dict(simulations=256, mean=mean, lower=mean-half, upper=mean+half)

    def compare(actual, expected, keys):
        actual = actual.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        assert set(actual.columns) == set(expected.columns)
        pd.testing.assert_frame_equal(actual, expected[actual.columns], check_dtype=False,
                                      check_exact=False, rtol=1e-11, atol=1e-10)

    original = pd.read_csv(root/'results/contact/tables/paper/performance/metrics.csv')
    old_paired = pd.read_csv(root/'results/contact/tables/paper/performance/paired.csv')
    assert not original.duplicated(['condition', 'endpoint', 't', 'method', 'trial']).any()
    assert original.numerical_ok.astype(bool).all()
    np.testing.assert_allclose(original.response_mse,
        .02**2/3*original.response_map_frobenius, rtol=1e-11, atol=1e-15)
    for row in old_paired.itertuples():
        group = original[original.condition.eq(row.condition) & original.endpoint.eq(row.endpoint)
                         & original.t.eq(row.t)]
        own = group[group.method.eq('mean')].set_index('trial')[row.metric]
        other = group[group.method.eq(row.reference)].set_index('trial')[row.metric]
        stats = interval(other.loc[own.index]-own)
        np.testing.assert_allclose([stats['mean'], stats['lower'], stats['upper']],
                                   [row.gain, row.lower, row.upper], rtol=1e-11, atol=1e-15)
    selected = original[original.condition.isin(['static_concentrated', 'plane', 'sphere'])].copy()
    selected['method'] = selected.method.replace({'isotropic': 'iso'})
    selected['response_mse_mm2_s2'] = selected.response_mse*1e6
    compare(data['raw_stats'], selected, ['condition', 'endpoint', 't', 'method', 'trial'])
    summaries, pairs, differences = [], [], []
    for (condition, endpoint, t), group in selected.groupby(['condition', 'endpoint', 't']):
        wide = group.pivot(index='trial', columns='method', values='response_mse_mm2_s2')
        np.testing.assert_array_equal(wide.index, np.arange(256))
        for method in wide:
            summaries.append(dict(condition=condition, endpoint=endpoint, t=int(t),
                                  method=method, **interval(wide[method])))
            if method == 'mean':
                continue
            delta = wide[method]-wide['mean']
            stats = interval(delta)
            pairs.append(dict(condition=condition, endpoint=endpoint, t=int(t), reference=method,
                              gain=stats['mean'], **stats))
            differences.extend(dict(condition=condition, endpoint=endpoint, t=int(t), trial=int(trial),
                reference=method, difference_mm2_s2=float(value)) for trial, value in delta.items())
    summary = pd.DataFrame(summaries)
    pairs = pd.DataFrame(pairs)
    differences = pd.DataFrame(differences)
    expected_frames = {'raw_stats': (selected, ['condition','endpoint','t','method','trial'])}
    for endpoint, stem in [('checkpoint', 'checkpoint'), ('episode_average', 'episode_average')]:
        expected_frames[stem+'_summary'] = (
            summary[summary.endpoint.eq(endpoint)], ['condition','endpoint','t','method'])
        expected_frames['paired_'+stem+'_summary'] = (
            pairs[pairs.endpoint.eq(endpoint)], ['condition','endpoint','t','reference'])
        expected_frames['paired_'+stem+'_differences'] = (
            differences[differences.endpoint.eq(endpoint)], ['condition','endpoint','t','reference','trial'])
    data_manifest = root/OUTPUT/'data/manifest.json'
    manifest = json.loads(data_manifest.read_text())
    assert manifest['status'] == 'completed' and manifest['example'] == 'contact_prediction_diagnostics'
    assert manifest['config'] == data['config'] and manifest['inputs'] == data['input_hashes']
    assert manifest['original_sources'] == data['source_hashes']
    _verify_hashes(root, manifest['sources'], 'contact-summary producer')
    _verify_hashes(root, manifest['inputs'], 'contact-summary input')
    _verify_hashes(root, manifest['artifacts'], 'contact-summary output')
    for key, (expected, keys) in expected_frames.items():
        compare(data[key], expected, keys)
        names = [name for name in manifest['artifacts'] if Path(name).name in (key+'.csv', key+'.csv.gz')]
        assert len(names) == 1
        compare(pd.read_csv(root/names[0]), expected, keys)
    for condition, record in data['conditions'].items():
        times = [1,4,16,32] if condition=='static_concentrated' else [4,16,32,64,128,192]
        np.testing.assert_array_equal(record['time'], times)
        assert record['T'] == (32 if condition=='static_concentrated' else 192)
        assert record['beta'] == (.9 if condition=='sphere' else 1.)
        assert {row['method'] for row in record['curves']} == {'mean','mode','iso'}
        for curve in record['all_curves']:
            rows = summary[summary.condition.eq(condition)&summary.endpoint.eq('checkpoint')
                           &summary.method.eq(curve['method'])].sort_values('t')
            np.testing.assert_allclose(curve['points'], rows[['t','mean','lower','upper']],
                                       rtol=1e-11, atol=1e-10)
    for key, condition, endpoint, t in [('static_concentrated_t16','static_concentrated','checkpoint',16),
                                      ('sphere_episode_average','sphere','episode_average',192)]:
        group = selected[selected.condition.eq(condition)&selected.endpoint.eq(endpoint)&selected.t.eq(t)]
        for method, values in data['histogram_samples'][key].items():
            expected = group[group.method.eq(method)].sort_values('trial').response_mse_mm2_s2
            np.testing.assert_allclose(values, expected, rtol=1e-11, atol=1e-10)
    # Independently reconstruct all action errors for the retained story,
    # including the regularized predictor, without updating any estimator.
    with np.load(root/'results/contact/data/paper/performance/story.npz', allow_pickle=False) as saved:
        story = {name:saved[name] for name in saved.files}
    story_checks = {}
    for condition in ('static_concentrated','plane','sphere'):
        times = story[condition+'_report_times'].astype(int)
        expected = [1,4,16,32] if condition=='static_concentrated' else list(range(4,193,4))
        np.testing.assert_array_equal(times, expected)
        rho = data['selected_settings'][condition]['regularization_rho']
        matrices = {'mean': story[condition+'_M'], 'mode': story[condition+'_Pmode'],
                    'iso': story[condition+'_isotropic']}
        matrices['regularized'] = rho*matrices['mode']+(1-rho)*2*np.eye(3)/3
        target = story[condition+'_P'][times-1]
        trial = selected[selected.condition.eq(condition)&selected.trial.eq(0)]
        errors = {}
        for method, values in matrices.items():
            loss = .02**2/3*np.sum((values-target)**2, axis=(1,2))*1e6
            recorded = trial[trial.method.eq(method)&trial.endpoint.eq('episode_average')]
            assert len(recorded) == 1
            np.testing.assert_allclose(loss.mean(), recorded.response_mse_mm2_s2.iloc[0],
                                       rtol=1e-11, atol=1e-10)
            for row in trial[trial.method.eq(method)&trial.endpoint.eq('checkpoint')].itertuples():
                np.testing.assert_allclose(loss[np.flatnonzero(times==row.t)[0]], row.response_mse_mm2_s2,
                                           rtol=1e-11, atol=1e-10)
            errors[method] = float(loss.mean())
        story_checks[condition] = dict(internal_action_count=len(times),
            full_run_errors_independently_reconstructed=errors, regularization_rho=rho)
    sphere = selected[selected.condition.eq('sphere')&selected.method.eq('mean')]
    sparse = sphere[sphere.endpoint.eq('checkpoint')].groupby('trial').response_mse_mm2_s2.mean()
    whole = sphere[sphere.endpoint.eq('episode_average')].set_index('trial').response_mse_mm2_s2
    discrepancy = float(np.max(abs(sparse-whole)))
    assert discrepancy > 1.
    assert data['checks']['new_simulations'] == data['checks']['new_parameter_selections'] == data['checks']['new_posterior_fits'] == 0
    return dict(raw_records=len(selected), checkpoint_intervals=int(summary.endpoint.eq('checkpoint').sum()),
        episode_average_intervals=int(summary.endpoint.eq('episode_average').sum()),
        paired_checkpoint_intervals=int(pairs.endpoint.eq('checkpoint').sum()),
        paired_episode_average_intervals=int(pairs.endpoint.eq('episode_average').sum()),
        paired_simulation_differences=len(differences), original_paired_table_rows_verified=len(old_paired),
        exported_dataframes_roundtripped=len(expected_frames), units='(mm/s)^2', unit_factor=1e6,
        error_equals_v0_squared_frobenius_error_over_three=True, no_extra_division_by_three=True,
        confidence_level=.95, degrees_of_freedom=255, test_simulations_per_group=256,
        full_run_story_checks=story_checks, full_sphere_action_count=48,
        sparse_checkpoints_not_averaged=True, maximum_sparse_average_discrepancy_mm2_s2=discrepancy,
        physical_checks=data['physical_checks'], data_manifest_sha256=_sha(data_manifest),
        main_comparisons=summary[(summary.condition.eq('static_concentrated')&summary.endpoint.eq('checkpoint')&summary.t.eq(16))|
                                 (summary.condition.eq('sphere')&summary.endpoint.eq('episode_average'))].to_dict('records'),
        limitations=data['provenance']['limitations'])


def _snapshot_check(root, snapshot):
    """Check saved-frame geometry and integrate each displayed probability law."""
    import numpy as np
    from scipy.special import logsumexp

    t = int(snapshot['t'])
    assert t == 96 and snapshot['simulation'] == 0 and snapshot['condition'] == 'sphere'
    _verify_hashes(root, snapshot['input_hashes'], 'moving contact snapshot input')
    _verify_hashes(root, snapshot['source_hashes'], 'moving contact snapshot producer')
    with np.load(root/'results/contact/data/paper/performance/story.npz', allow_pickle=False) as saved:
        A = saved['sphere_A'][t-1]
        R = saved['sphere_R'][t-1]
        native = saved['sphere_normal'][t-1]
        world = saved['sphere_world_normal'][t-1]
        previous = saved['sphere_R'][t-2]
        true_projector = saved['sphere_P'][t-1]
        input_vector = saved['sphere_V'][t-1]
    np.testing.assert_allclose(snapshot['A'], A, rtol=0., atol=1e-12)
    np.testing.assert_allclose(snapshot['B'], -A, rtol=0., atol=1e-12)
    np.testing.assert_allclose(snapshot['R'], R, rtol=0., atol=1e-12)
    np.testing.assert_allclose(R.T@world, native, rtol=0., atol=1e-12)
    np.testing.assert_allclose(snapshot['G'], R.T@previous, rtol=0., atol=1e-12)
    np.testing.assert_allclose(true_projector, np.eye(3)-np.outer(snapshot['true_normal'], snapshot['true_normal']),
                               rtol=0., atol=1e-12)
    np.testing.assert_allclose(snapshot['true_normal_native'], native, rtol=0., atol=1e-12)
    mode = snapshot['mode']
    assert mode[np.argmax(abs(mode))] > 0 and mode@snapshot['true_normal'] >= 0
    nodes = snapshot['nodes']
    assert nodes.shape == (161,321,3)
    np.testing.assert_allclose(np.sum(nodes**2,axis=-1),1.,rtol=0.,atol=2e-15)
    np.testing.assert_allclose(snapshot['checks']['display_rotation'],np.eye(3),rtol=0.,atol=0.)
    assert snapshot['checks']['display_mesh_used_for_integration'] is False
    # Independent quadrature uses its own spectral frame and nodes; the
    # adaptive rendering mesh is never assigned integration weights.
    x, wx = np.polynomial.legendre.leggauss(256)
    phi = np.arange(512)*2*np.pi/512
    radius = np.sqrt(1-x*x)
    local = np.stack(np.broadcast_arrays(x[:,None],radius[:,None]*np.cos(phi),
                                        radius[:,None]*np.sin(phi)),axis=-1)
    log_weight = np.broadcast_to(np.log(wx[:,None]/(2*512)), local.shape[:2])
    laws = {}
    for suffix, parameter_key, logz_key, normal_key, plane_key, logdensity_key, risk_key in [
        ('full','B','log_normalizer_full','normal_mean','plane_mean','log_density_full','response_variance_full'),
        ('isotropic','B_iso','log_normalizer_isotropic','normal_mean_isotropic','plane_mean_isotropic',
         'log_density_isotropic','response_variance_isotropic')]:
        B = snapshot[parameter_key]
        logz = float(snapshot[logz_key])
        eig, Q = np.linalg.eigh(B)
        eig, Q = eig[::-1], Q[:,::-1]
        quadrature_nodes = local@Q.T
        exponent = np.einsum('...i,ij,...j->...',quadrature_nodes,B,quadrature_nodes)
        independent_logz = float(logsumexp(exponent+log_weight))
        qweight = np.exp(exponent-logz+log_weight)
        mass = float(qweight.sum())
        flat = quadrature_nodes.reshape(-1,3)
        N = flat.T@(qweight.reshape(-1,1)*flat)
        np.testing.assert_allclose(independent_logz,logz,rtol=0.,atol=5e-8)
        np.testing.assert_allclose(mass,1.,rtol=0.,atol=5e-8)
        np.testing.assert_allclose(N,snapshot[normal_key],rtol=0.,atol=5e-8)
        np.testing.assert_allclose(snapshot[plane_key],np.eye(3)-snapshot[normal_key],rtol=0.,atol=1e-12)
        direct = np.einsum('...i,ij,...j->...',nodes,B,nodes)-logz
        antipodal = np.einsum('...i,ij,...j->...',-nodes,B,-nodes)-logz
        np.testing.assert_allclose(snapshot[logdensity_key],direct,rtol=0.,atol=1e-12)
        np.testing.assert_allclose(antipodal,direct,rtol=0.,atol=1e-12)
        curvature = 2*(eig[0]-eig[1:])
        saved_curvature = snapshot['curvature_eigenvalues' if suffix=='full' else 'isotropic_curvature_eigenvalues']
        np.testing.assert_allclose(curvature,saved_curvature,rtol=1e-12,atol=1e-12)
        projector = np.outer(Q[:,0],Q[:,0])
        np.testing.assert_allclose(projector,np.outer(mode,mode),rtol=0.,atol=1e-12)
        M = snapshot[plane_key]
        risk = float(input_vector@(M-M@M)@input_vector)
        np.testing.assert_allclose(snapshot[risk_key],risk,rtol=1e-12,atol=1e-16)
        laws[suffix] = dict(independent_log_normalizer_error=abs(independent_logz-logz),
            independent_mass=mass,independent_mean_error=float(np.linalg.norm(N-snapshot[normal_key],2)),
            curvature_eigenvalues=curvature.tolist(),response_variance_m2_s2=risk,
            response_variance_uses_own_law_mean=True)
    overlap = float(mode@snapshot['normal_mean']@mode)
    overlap_iso = float(mode@snapshot['normal_mean_isotropic']@mode)
    np.testing.assert_allclose(overlap,overlap_iso,rtol=0.,atol=1e-8)
    expected_iso = (1-overlap)*np.outer(mode,mode)+(1-(1-overlap)/2)*(np.eye(3)-np.outer(mode,mode))
    np.testing.assert_allclose(snapshot['plane_mean_isotropic'],expected_iso,rtol=0.,atol=5e-8)
    angle = float(np.rad2deg(np.arccos(np.clip(abs(mode@native),0.,1.))))
    np.testing.assert_allclose(angle,snapshot['checks']['antipodal_mode_true_angle_deg'],rtol=0.,atol=1e-10)
    metadata_path = root/OUTPUT/'data/moving_sphere_t96.json'
    meta = json.loads(metadata_path.read_text())
    assert meta['status']=='completed' and meta['t']==96 and meta['simulation']==0
    assert meta['array_sha256']==_sha(root/meta['array_path'])
    assert meta['input_hashes']==snapshot['input_hashes'] and meta['source_hashes']==snapshot['source_hashes']
    with np.load(root/meta['array_path'],allow_pickle=False) as saved:
        for name in saved.files:
            np.testing.assert_allclose(saved[name],snapshot[name],rtol=0.,atol=0.)
    return dict(t=t,simulation=0,sensor_frame_and_measured_transport_verified=True,
        same_native_mesh_and_identity_display_rotation=True,antipodal_sign_geometry_verified=True,
        modal_true_normal_angle_deg=angle,matching_overlap_error=abs(overlap-overlap_iso),
        expected_squared_chordal_distance=1-overlap,
        integration_rule='Independent 256-point Gauss-Legendre by 512-angle spectral-frame sphere quadrature; normalized invariant measure',
        laws=laws,metadata_sha256=_sha(metadata_path))


def _static_response_variance_check(root):
    import numpy as np
    from lib.notebook_contact_data import load_contact_plot_data
    path = root/'results/contact/data/paper/performance/anisotropy.npz'
    with np.load(path,allow_pickle=False) as saved:
        a = {name:saved[name] for name in saved.files}
    plotted = load_contact_plot_data(root,'posterior_anisotropy')
    selected = a['query_family']==1
    v0 = .02
    errors = {}
    for label, mean_key, risk_key, plot_key in [
        ('full','plane_mean','risks_full','risk_full'),
        ('isotropic','iso_plane_mean','risks_isotropic','risk_iso')]:
        M = a[mean_key]
        v = a['physical_queries']
        normalized = np.einsum('qi,ij,qj->q',v,M-M@M,v)/(v0*v0)
        np.testing.assert_allclose(normalized,a[risk_key],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(plotted[plot_key],normalized[selected],rtol=1e-12,atol=1e-12)
        errors[label] = float(np.max(abs(plotted[plot_key]-normalized[selected])))
    return dict(status='passed',original_arrays_sha256=_sha(path),
        each_curve_uses_its_own_law_mean=True,normalization='r_t(v(theta))/v0^2',
        maximum_absolute_errors=errors,
        interpretation='Posterior response variance under each displayed law; not an empirical loss under one shared law.')


def _preservation_check(root):
    proof_path = root/OUTPUT/'preserved_files.json'
    proof = json.loads(proof_path.read_text())
    mapping = proof['tracked_files']
    _verify_hashes(root,mapping,'protected original result/library/notebook')
    baseline_path = Path('/tmp/contact-prediction-preservation-baseline.json')
    local = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        _verify_hashes(root,baseline,'protected original local file')
        local = dict(files=len(baseline),sha256=_sha(baseline_path),all_files_unchanged=True)
    return dict(status='passed',tracked_files=len(mapping),tracked_proof_sha256=_sha(proof_path),
                all_original_tracked_files_unchanged=True,local_snapshot=local)


def _figure_manifest_check(root, notebook, routes, required_inputs):
    path = root/OUTPUT/'figure_manifest.json'
    meta = json.loads(path.read_text())
    assert meta['status']=='completed' and meta['notebook']==NOTEBOOK
    assert meta['notebook_sha256']==_sha(root/NOTEBOOK)
    assert meta['producer_sha256']==_sha(root/meta['producer'])
    _verify_hashes(root,meta['source_hashes'],'contact diagnostic figure producer')
    _verify_hashes(root,meta['input_hashes'],'contact diagnostic figure input')
    for name,expected in required_inputs.items():
        assert meta['input_hashes'].get(name)==expected,'Missing figure numerical input: '+name
    assert meta['snapshot_time']==96 and set(meta['figures'])==set(FILENAMES)
    expected_outputs=set()
    for destination,route in routes.items():
        record=meta['figures'][route['filename']]
        assert record['section']==route['section']
        assert record['figure_size_in']==[15,5] and record['dpi']==300
        assert record['text_usetex'] is True
        assert record['local_pdf']==route['local_pdf'] and record['draft_pdf']==destination
        assert record['local_png']==str(Path(route['local_pdf']).with_suffix('.png'))
        assert record['draft_png']==str(Path(destination).with_suffix('.png'))
        outputs={record[key] for key in ['local_pdf','local_png','draft_pdf','draft_png']}
        assert set(record['hashes'])==outputs
        _verify_hashes(root,record['hashes'],'contact diagnostic figure output')
        expected_outputs.update(outputs)
    assert expected_outputs<=set(meta['files'])
    assert str(OUTPUT/'data/manifest.json') in meta['files']
    _verify_hashes(root,meta['files'],'contact diagnostic manifest file')
    return dict(manifest=str(path.relative_to(root)),sha256=_sha(path),
                figure_copies=len(expected_outputs),all_bound_files=len(meta['files']))


def audit_new_pdfs(root=ROOT):
    """Use the unchanged PDF auditor; requires PyMuPDF in this interpreter."""
    import re
    from scripts.check_manuscript_plot_exports import audit_pdf,figure_declarations,fitz
    root=Path(root).resolve();routes=optional_contact_diagnostic_routes(root)
    if not routes:
        return dict(status='skipped',reason='Contact diagnostics notebook is absent')
    notebook=json.loads((root/NOTEBOOK).read_text());reports={}
    for destination,route in routes.items():
        declarations=figure_declarations(_source(notebook['cells'][route['cell']]))
        assert len(declarations)==1 and declarations[0]['filename']==route['filename']
        declaration=declarations[0]
        report=audit_pdf(root/destination,[declaration['figure_size_in']],declaration['dpi'],
            primary_fonts=declaration['primary_font_sizes_pt'],require_primary_sizes=True,
            main_panel_policy=[] if route['section']=='main' else None)
        assert not report['errors'],(destination,report['errors'])
        reports[destination]=report
    information=root/'results/main/contact/information_rotation.pdf'
    with fitz.open(information) as doc:
        text='\n'.join(page.get_text() for page in doc)
    letters=re.findall(r'\([a-z]\)',text)
    assert not letters,'Retained standalone information plot has an inherited panel letter'
    return dict(status='passed',figures=reports,
        audit_source_sha256=_sha(root/'scripts/check_manuscript_plot_exports.py'),
        retained_information=dict(path=str(information.relative_to(root)),sha256=_sha(information),
            inherited_panel_letters=letters,unchanged_lowercase_chart_notation=True,
            interpretation='The retained p_z and the new p_Z denote the same candidate-plane conditional measurement law, with z=vec(Z).'))


def _pdf_audit(root):
    import importlib.util
    if importlib.util.find_spec('pymupdf') or importlib.util.find_spec('fitz'):
        return audit_new_pdfs(root)
    # This workspace keeps the independent PyMuPDF auditor in its runtime
    # interpreter. Use an argument list; no notebook or rendering is executed.
    import shutil
    import subprocess
    candidates=list(dict.fromkeys(path for path in [shutil.which('python3'),shutil.which('python')]
                                  if path and Path(path).absolute()!=Path(sys.executable).absolute()))
    errors=[]
    for interpreter in candidates:
        result=subprocess.run([interpreter,str(Path(__file__).resolve()),'--pdf-only','--root',str(root)],
                              capture_output=True,text=True,check=False)
        if result.returncode==0:
            return json.loads(result.stdout)
        errors.append(result.stderr.strip())
    raise RuntimeError('PyMuPDF is required for the independent PDF audit: '+'; '.join(errors))


def fullcheck(root=ROOT):
    """Verify the complete optional addendum without changing its files."""
    root = Path(root).resolve()
    routes = optional_contact_diagnostic_routes(root)
    if not routes:
        return dict(status='skipped', reason='Contact prediction diagnostics notebook is absent', routes={})
    from lib.contact_prediction_diagnostics import load_contact_prediction_diagnostics
    from lib.contact_moving_snapshot import load_moving_sphere_snapshot
    from lib.final_validation import require_current_gate
    gate=require_current_gate(root)
    data=load_contact_prediction_diagnostics(root)
    snapshot=load_moving_sphere_snapshot(root,t=96)
    inputs=dict(data['input_hashes'])
    for name,expected in snapshot['input_hashes'].items():
        assert name not in inputs or inputs[name]==expected
        inputs[name]=expected
    data_manifest=root/OUTPUT/'data/manifest.json'
    inputs[str(data_manifest.relative_to(root))]=_sha(data_manifest)
    notebook=json.loads((root/NOTEBOOK).read_text())
    notebook_checks=_notebook_check(root,notebook,routes,inputs)
    files=_exports_check(root,routes)
    numerical=_numeric_check(root,data)
    snapshot_checks=_snapshot_check(root,snapshot)
    static_variance=_static_response_variance_check(root)
    manifest=_figure_manifest_check(root,notebook,routes,inputs)
    preservation=_preservation_check(root)
    pdf_audit=_pdf_audit(root)
    return dict(status='passed',notebook=NOTEBOOK,notebook_sha256=_sha(root/NOTEBOOK),routes=routes,
        notebook_checks=notebook_checks,files=files,numerical=numerical,snapshot=snapshot_checks,
        retained_static_response_variance=static_variance,figure_manifest=manifest,
        preservation=preservation,pdf_audit=pdf_audit,scientific_sha256=gate['final_science_hash'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--routes-only', action='store_true')
    parser.add_argument('--pdf-only',action='store_true',help='Run only the independent PyMuPDF audit.')
    parser.add_argument('--report', type=Path, help='Optional JSON report; result files are read-only.')
    args = parser.parse_args()
    report = (optional_contact_diagnostic_routes(args.root) if args.routes_only else
              audit_new_pdfs(args.root) if args.pdf_only else fullcheck(args.root))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
