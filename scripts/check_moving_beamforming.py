"""Read-only checks for the optional moving-interference beamforming notebook.

The route declaration works without importing numerical or plotting packages.
The full audit checks saved raw records and exports; it performs no simulation,
parameter selection, posterior integration, or notebook execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NOTEBOOK = 'notebooks/beamforming_moving_interference.ipynb'
LOCAL_DIRECTORY = Path('results/beamforming_moving/figures')
FILENAMES = {
    f'moving_interference_{condition}_output_power.pdf': condition
    for condition in ('smooth', 'abrupt')
}
FILENAMES['moving_interference_comparison.pdf']='comparison'
FIGURE_SIZES={name:([7,3.5] if condition=='comparison' else [15,5]) for name,condition in FILENAMES.items()}


def _source(cell):
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optional_moving_routes(root=ROOT):
    """Return the three manuscript-PDF declarations, or {} before creation.

    Each record has the same schema and zero-based notebook cell position as
    ``lib.notebook_pdf_exports.declared_notebook_pdf_routes``.
    """
    from lib.notebook_pdf_exports import _path_parts, _qualified
    root = Path(root).resolve()
    path = root / NOTEBOOK
    if not path.is_file():
        return {}
    notebook = json.loads(path.read_text())
    meta = notebook.get('metadata', {}).get('editable_plotting', {})
    if meta.get('output_directory') != LOCAL_DIRECTORY.as_posix():
        raise ValueError('Moving notebook local exports must use ' + LOCAL_DIRECTORY.as_posix())
    routes = {}
    for index, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code':
            continue
        tree = ast.parse(_source(cell))
        filename = section = None
        declared = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if 'PDF_PATH' in names:
                    parts = _path_parts(node.value)
                    if (len(parts) != 2 or parts[0] != ('variable', 'EDITABLE_OUTPUT')
                            or parts[1][0] != 'literal' or parts[1][1] not in FILENAMES):
                        raise ValueError(f'{NOTEBOOK}, cell {index}: unexpected PDF_PATH')
                    filename = parts[1][1]
                    section = None
                    declared = False
                if 'DRAFT_SECTION' in names:
                    section = ast.literal_eval(node.value)
                    if section != 'main':
                        raise ValueError('Moving notebook PDFs must be declared for main/')
                if 'DRAFT_PATH' in names:
                    expected = [('variable', 'ROOT'), ('literal', 'results'),
                        ('variable', 'DRAFT_SECTION'), ('literal', 'beamforming'),
                        ('variable', 'PDF_PATH.name')]
                    if filename is None or section != 'main' or _path_parts(node.value) != expected:
                        raise ValueError(f'{NOTEBOOK}, cell {index}: inconsistent DRAFT_PATH')
                    destination = (Path('results/main/beamforming') / filename).as_posix()
                    if destination in routes:
                        raise ValueError('Duplicate moving notebook destination: ' + destination)
                    routes[destination] = dict(notebook=NOTEBOOK, cell=index,
                        example='beamforming', section='main', filename=filename,
                        local_pdf=(LOCAL_DIRECTORY / filename).as_posix())
                    declared = True
            value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
            if isinstance(value, ast.Call) and _qualified(value.func) in ('plt.figure', 'plt.subplots'):
                if not declared:
                    raise ValueError(f'{NOTEBOOK}, cell {index}: figure has no PDF declaration')
                declared = False
        if declared:
            raise ValueError(f'{NOTEBOOK}, cell {index}: PDF declaration has no figure')
    if {record['filename'] for record in routes.values()} != set(FILENAMES) or len(routes) != 3:
        raise ValueError('The moving notebook must declare smooth, abrupt and combined figures')
    return routes


def _verify_hashes(root, mapping, category):
    if not isinstance(mapping, dict) or not mapping:
        raise AssertionError('Missing ' + category + ' hashes')
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        assert (root / name).is_file() and _sha(root / name) == expected, category + ': ' + name


def _notebook_check(root, notebook, routes, data):
    from lib.notebook_pdf_exports import _qualified
    meta = notebook['metadata']['editable_plotting']
    inputs = dict(meta.get('input_hashes', {}))
    inputs.update(meta.get('data_hashes', {}))
    sources = dict(meta.get('loader_sources', {}))
    if meta.get('loader'):
        sources[meta['loader']] = meta['loader_sha256']
    _verify_hashes(root, inputs, 'notebook input')
    _verify_hashes(root, sources, 'notebook producer')
    for name, expected in data['input_hashes'].items():
        assert inputs.get(name) == expected, 'Unrecorded notebook numerical input: ' + name
    loader = 'lib/beamforming_moving_evidence.py'
    assert sources.get(loader) == _sha(root / loader), 'Missing current moving data-loader hash'
    original_sources = {record['path']: record['sha256']
                        for record in data['provenance']['verified_source_files'].values()}
    assert notebook['metadata']['moving_beamforming']['source_hashes'] == original_sources
    by_id = {cell.get('id'): cell for cell in notebook['cells']}
    for cell_id, expected in meta.get('numerical_cell_hashes', {}).items():
        assert hashlib.sha256(_source(by_id[cell_id]).encode()).hexdigest() == expected
    code_count = images = figures = 0
    tex_true = False
    figure_indices = {record['cell'] for record in routes.values()}
    for index, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code' or not _source(cell).strip():
            continue
        code_count += 1
        assert cell.get('execution_count') is not None, f'Unexecuted moving cell {index}'
        tree = ast.parse(_source(cell))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        constructors = [node for node in calls if _qualified(node.func) in ('plt.figure', 'plt.subplots')]
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value == 'text.usetex':
                        assert ast.literal_eval(value) is True
                        tex_true = True
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript)
                            and _qualified(target.value).endswith('rcParams')
                            and isinstance(target.slice, ast.Constant)
                            and target.slice.value == 'text.usetex'):
                        assert ast.literal_eval(node.value) is True
                        tex_true = True
        if constructors:
            assert index in figure_indices and len(constructors) == 1
            assigned = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                        for target in node.targets if isinstance(target, ast.Name)}
            assert ast.literal_eval(assigned['DPI']) == 300
            route=next(record for record in routes.values() if record['cell']==index)
            expected_size=FIGURE_SIZES[route['filename']]
            assert tuple(ast.literal_eval(assigned['FIGSIZE'])) == tuple(expected_size)
            combined=route['filename']=='moving_interference_comparison.pdf'
            assert ast.literal_eval(assigned['LABEL_FONT_SIZE'])==(12. if combined else 25.)
            assert ast.literal_eval(assigned['TICK_FONT_SIZE'])==(11. if combined else 23.)
            assert ast.literal_eval(assigned['LEGEND_FONT_SIZE'])==(10.5 if combined else 22.)
            assert ast.literal_eval(assigned['IQR_ALPHA'])==.22
            assert ast.literal_eval(assigned['SPINE_WIDTH'])==.95
            assert all(ast.literal_eval(assigned['SHOW_IQR']).values())
            assert 'MARKER_OFFSETS' not in assigned and 'MARKER_SIZE' not in assigned
            styles=ast.literal_eval(assigned['STYLE'])
            assert {key:value[2:] for key,value in styles.items()}=={
                method:('-',) for method in ('mean','mode','iso','great_window')}
            trajectories=[call for call in calls if _qualified(call.func)=='ax.plot']
            assert len(trajectories)==1
            keywords={item.arg:item.value for item in trajectories[0].keywords}
            assert not {'marker','markersize','markevery'} & set(keywords)
            assert isinstance(keywords['linestyle'],ast.Name) and keywords['linestyle'].id=='linestyle_'
            assert not any(isinstance(call.func,ast.Attribute) and call.func.attr in ('set_visible','annotate','text') for call in calls)
            quantiles=[call for call in calls if isinstance(call.func,ast.Attribute) and call.func.attr=='quantile']
            assert len(quantiles)==1 and ast.literal_eval(quantiles[0].args[0])==[.25,.75]
            ticks=[call for call in calls if isinstance(call.func,ast.Attribute) and call.func.attr=='tick_params']
            assert len(ticks)==1
            options={item.arg:ast.literal_eval(item.value) for item in ticks[0].keywords if isinstance(item.value,ast.Constant)}
            assert options['direction']=='in' and all(options[name] is True for name in ['top','right','bottom','left','labelbottom','labelleft'])
            assert ast.literal_eval(assigned['SAVE_PDF']) is True
            for node in constructors:
                kwargs = {item.arg: item.value for item in node.keywords}
                assert isinstance(kwargs.get('dpi'), ast.Name) and kwargs['dpi'].id == 'DPI'
                assert isinstance(kwargs.get('figsize'), ast.Name) and kwargs['figsize'].id == 'FIGSIZE'
            assert sum(_qualified(call.func) == 'plt.show' for call in calls) == 1
            assert sum(isinstance(call.func, ast.Attribute) and call.func.attr == 'savefig' for call in calls) >= 1
            assert any(_qualified(call.func).endswith('.plot') for call in calls)
            assert any(_qualified(call.func).endswith('.fill_between') for call in calls)
            figures += 1
        for call in calls:
            assert _qualified(call.func) not in ('exec', 'eval', 'show', 'show_saved')
        for output in cell.get('outputs', []):
            assert output.get('output_type') != 'error'
            images += int('image/png' in output.get('data', {}))
    assert tex_true and figures == images == 3
    return dict(code_cells=code_count, visible_figures=figures, embedded_images=images,
                text_usetex=True, dpi=300, figure_size_in=[15, 5],
                native_combined_size_in=[7,3.5],all_bands_empirical_iqr=True,
                method_trajectories=dict(linestyle='-',markers=False,marker_staggering=False),
                numerical_inputs=len(inputs), producer_sources=len(sources))


def _exports_check(root, routes, figure_sizes=None):
    """Check declared canvases; other notebooks retain the 15-by-5 default."""
    from PIL import Image
    files = {}
    for destination, route in routes.items():
        size=tuple((figure_sizes or {}).get(destination,route.get('figure_size_in',(15,5))))
        local=root/route['local_pdf'];main=root/destination
        assert local.is_file() and main.is_file() and _sha(local)==_sha(main)
        content=local.read_bytes();assert content.startswith(b'%PDF-')
        boxes=re.findall(rb'/MediaBox\s*\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*([-+0-9.eE]+)\s*\]',content)
        assert len(boxes)==1,'Expected a single-page Matplotlib PDF'
        x0,y0,x1,y1=map(float,boxes[0])
        assert abs((x1-x0)/72-size[0])<1e-6 and abs((y1-y0)/72-size[1])<1e-6
        local_png,main_png=local.with_suffix('.png'),main.with_suffix('.png')
        assert local_png.is_file() and main_png.is_file() and _sha(local_png)==_sha(main_png)
        pixels=tuple(round(value*300) for value in size)
        with Image.open(local_png) as image:
            assert image.size==pixels
            assert len(image.info.get('dpi',[]))==2
            assert all(abs(float(value)-300)<.1 for value in image.info['dpi'])
        files[destination]=dict(sha256=_sha(local),local_pdf=route['local_pdf'],
            png_sha256=_sha(local_png),png_size_px=list(pixels),figure_size_in=list(size))
    return files


def _numeric_check(root, data):
    """Recompute every plotted mean and paired interval from original rows."""
    import numpy as np
    import pandas as pd
    from scipy.stats import t as student_t

    methods = ('mean', 'mode', 'iso', 'great_window')
    references = methods[1:]
    times = list(range(8, 257, 8))
    conditions = ('smooth', 'abrupt')
    columns = ['condition', 'n', 'trial', 't', 'method', 'beta', 'true_power',
               'oracle_power', 'true_ratio_db', 'numerical_ok', 'mean_ok']
    original = pd.read_csv(root / 'results/beamforming/data/paper/performance/moving_checkpoints.csv.gz',
                           usecols=columns)
    original = original[original.condition.isin(conditions) & original.n.eq(32)
        & original.method.isin(('mean', 'mode', 'isotropic', 'great_window'))].copy()
    original.rename(columns={'trial': 'episode'}, inplace=True)
    original['method'] = original.method.replace({'isotropic': 'iso'})
    original = original[data['checkpoint_losses'].columns]

    def compare(actual, expected, keys):
        actual = actual.sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        assert set(actual.columns) == set(expected.columns)
        pd.testing.assert_frame_equal(actual, expected[actual.columns],
            check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)

    def interval(values):
        values = np.asarray(values, float)
        assert len(values) == 256 and np.isfinite(values).all()
        mean = float(values.mean())
        half = float(student_t.ppf(.975, 255) * values.std(ddof=1) / 16)
        return dict(simulations=256, mean=mean, lower=mean-half, upper=mean+half)

    compare(original, data['checkpoint_losses'], ['condition', 'method', 'episode', 't'])
    assert len(original) == 65536
    np.testing.assert_allclose(original.true_ratio_db,
        10*np.log10(original.true_power/original.oracle_power), rtol=1e-12, atol=1e-12)
    summaries, paired, differences, time_summaries, time_pairs, time_differences = [], [], [], [], [], []
    simulation_means = original.groupby(['condition', 'episode', 'method']).true_ratio_db.agg(
        time_average_db='mean', reporting_points='size').reset_index()
    assert simulation_means.reporting_points.eq(32).all()
    for condition in conditions:
        raw = original[original.condition.eq(condition)]
        expected_power = {}
        n = 32
        a = np.ones(n, complex) / np.sqrt(n)
        angles = data['conditions'][condition]['angles']
        for t in times:
            h = np.exp(1j*np.pi*np.arange(n)*np.sin(angles[t-1])) / np.sqrt(n)
            expected_power[t] = .2 / (1 - abs(np.vdot(a, h))**2 / 1.2)
        np.testing.assert_allclose(raw.oracle_power, raw.t.map(expected_power), rtol=1e-12, atol=1e-12)
        for method in methods:
            curve = []
            for t in times:
                rows = raw[raw.method.eq(method) & raw.t.eq(t)].sort_values('episode')
                np.testing.assert_array_equal(rows.episode, np.arange(256))
                stats = interval(rows.true_ratio_db)
                summaries.append(dict(condition=condition, n=32, t=t, method=method, **stats))
                curve.append([t, stats['mean'], stats['lower'], stats['upper']])
            plotted = [row for row in data['conditions'][condition]['curves'] if row['method'] == method]
            assert len(plotted) == 1
            np.testing.assert_allclose(plotted[0]['points'], curve, rtol=1e-12, atol=1e-12)
        pivot = raw.pivot(index=['episode', 't'], columns='method', values='true_ratio_db')
        for reference in references:
            delta = pivot[reference] - pivot['mean']
            for (episode, t), value in delta.items():
                differences.append(dict(condition=condition, n=32, episode=int(episode),
                    t=int(t), reference=reference, difference_db=float(value)))
            for t in times:
                paired.append(dict(condition=condition, n=32, t=t, reference=reference,
                                   **interval(delta.xs(t, level='t'))))
        average = simulation_means[simulation_means.condition.eq(condition)]
        for method in methods:
            rows = average[average.method.eq(method)].sort_values('episode')
            time_summaries.append(dict(condition=condition, method=method,
                                       **interval(rows.time_average_db)))
        pivot = average.pivot(index='episode', columns='method', values='time_average_db')
        for reference in references:
            delta = pivot[reference] - pivot['mean']
            time_pairs.append(dict(condition=condition, reference=reference, **interval(delta)))
            time_differences.extend(dict(condition=condition, episode=int(episode),
                reference=reference, difference_db=float(value)) for episode, value in delta.items())
    expected_frames = {
        'checkpoint_losses': (original, ['condition', 'method', 'episode', 't']),
        'checkpoint_summary': (pd.DataFrame(summaries), ['condition', 'method', 't']),
        'paired_checkpoint_summary': (pd.DataFrame(paired), ['condition', 'reference', 't']),
        'paired_checkpoint_differences': (pd.DataFrame(differences), ['condition', 'reference', 'episode', 't']),
        'simulation_time_averages': (simulation_means, ['condition', 'method', 'episode']),
        'time_average_summary': (pd.DataFrame(time_summaries), ['condition', 'method']),
        'paired_time_average_summary': (pd.DataFrame(time_pairs), ['condition', 'reference']),
        'paired_time_average_differences': (pd.DataFrame(time_differences), ['condition', 'reference', 'episode']),
    }
    manifest_path = root / 'results/beamforming_moving/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'completed' and manifest['component'] == 'original_moving_interference'
    assert manifest['config'] == data['config'] and manifest['inputs'] == data['input_hashes']
    assert manifest['original_sources'] == data['source_hashes']
    _verify_hashes(root, manifest['sources'], 'moving-summary producer')
    _verify_hashes(root, manifest['inputs'], 'moving-summary input')
    _verify_hashes(root, manifest['artifacts'], 'moving-summary artifact')
    for key, (expected, keys) in expected_frames.items():
        compare(data[key], expected, keys)
        matches = [name for name in manifest['artifacts']
                   if Path(name).name in (key+'.csv', key+'.csv.gz')]
        assert len(matches) == 1
        compare(pd.read_csv(root / matches[0]), expected, keys)
    assert data['checks']['new_simulations'] == data['checks']['new_parameter_selections'] == data['checks']['new_posterior_fits'] == 0
    assert data['config']['n'] == 32 and data['config']['T'] == 256
    assert data['config']['initial_parameter'] == 'A_0=0' and data['config']['transport'] == 'G_t=I_n'
    assert data['config']['beta'] == .97 and data['config']['sigma2'] == .2 and data['config']['tau2'] == 1.
    return dict(raw_simulation_rows=len(original), checkpoint_intervals=len(summaries),
        paired_checkpoint_intervals=len(paired), paired_checkpoint_differences=len(differences),
        simulation_time_averages=len(simulation_means), time_average_intervals=len(time_summaries),
        paired_time_average_intervals=len(time_pairs), paired_time_average_differences=len(time_differences),
        exported_dataframes_roundtripped=len(expected_frames),
        arithmetic_mean_of_individual_db_values=True, confidence_level=.95,
        interval_degrees_of_freedom=255, exact_covariance_power_and_ideal_power_checked=True,
        model_and_shared_posterior_checks=data['checks']['saved_geometry'],
        time_average_summary=time_summaries, paired_time_average_summary=time_pairs,
        data_manifest_sha256=_sha(manifest_path))


def _preservation_check(root):
    """Check the session's independent snapshot when it is available."""
    baseline_path = Path('/tmp/independent-moving-preservation-baseline.json')
    if not baseline_path.is_file():
        return dict(status='not_available', scope='Optional session snapshot is absent')
    historical = json.loads(baseline_path.read_text())
    from lib.presentation_update_provenance import verify_presentation_preservation
    checked = verify_presentation_preservation(root, historical)
    return dict(status='passed', protected_files=len(historical),
                baseline_sha256=_sha(baseline_path),
                exact_preservation=checked,
                scientific_data_and_settings_unchanged=True)


def _iqr_check(root,data):
    """Verify display quartiles directly, separately from unchanged table CIs."""
    import numpy as np
    import pandas as pd
    raw=pd.read_csv(root/'results/beamforming/data/paper/performance/moving_checkpoints.csv.gz',
                    usecols=['condition','n','trial','t','method','true_ratio_db'])
    raw=raw[raw.condition.isin(['smooth','abrupt'])&raw.n.eq(32)
            &raw.method.isin(['mean','mode','isotropic','great_window'])].copy()
    raw['method']=raw.method.replace({'isotropic':'iso'})
    expected=[]
    for (condition,method,t),group in raw.groupby(['condition','method','t'],sort=True):
        np.testing.assert_array_equal(np.sort(group.trial),np.arange(256))
        q25,q75=np.quantile(group.true_ratio_db.to_numpy(),[.25,.75])
        expected.append(dict(condition=condition,method=method,t=t,simulations=256,
            mean=float(group.true_ratio_db.mean()),q25=float(q25),q75=float(q75)))
    folder=root/'results/beamforming_moving/plotting'
    actual=pd.read_csv(folder/'pointwise_iqr.csv')
    expected=pd.DataFrame(expected)
    keys=['condition','method','t']
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-12)
    original=data['checkpoint_summary'][['condition','method','t','mean']]
    checked=actual.merge(original,on=keys,suffixes=('_display','_original'),validate='one_to_one')
    np.testing.assert_allclose(checked.mean_display,checked.mean_original,rtol=1e-13,atol=1e-13)
    meta=json.loads((folder/'manifest.json').read_text())
    assert meta['status']=='completed' and meta['points']==256 and meta['means_unchanged'] is True
    assert meta['inputs']==data['input_hashes'] and meta['old_confidence_intervals_unchanged'] is True
    assert meta['producer_sha256']==_sha(root/'scripts/make_moving_beamforming_notebook.py')
    _verify_hashes(root,meta['artifacts'],'empirical-IQR display artifacts')
    guard=json.loads((folder/'run_guard.json').read_text())
    assert guard['status']=='passed' and guard['all_protected_files_unchanged'] is True
    assert guard['protected_files']==guard['after_hashes']
    for name,digest in guard['protected_files'].items():
        if not name.endswith('.ipynb'):
            assert _sha(root/name)==digest,'Protected numerical/settings input changed: '+name
    return dict(points=len(expected),simulations_per_point=256,all_four_methods=True,
        quantiles=[.25,.75],computed_from_original_trial_records=True,
        means_and_original_table_confidence_intervals_unchanged=True,
        empirical_spread_not_posterior_credible_interval=True,
        exclusive_render_guard_passed=True,guarded_files=guard['protected_count'],
        plotting_manifest_sha256=_sha(folder/'manifest.json'))


def _figure_manifest_check(root, notebook, routes, data):
    path = root / 'results/beamforming_moving/figure_manifest.json'
    meta = json.loads(path.read_text())
    assert meta['status'] == 'completed' and meta['example'] == 'beamforming_moving'
    assert meta['notebook'] == NOTEBOOK and meta['notebook_sha256'] == _sha(root / NOTEBOOK)
    assert meta['config'] == meta['configuration'] == data['config']
    assert meta['optional_main_candidate'] is True and meta['figure_count'] == 3
    assert meta['dpi'] == 300 and meta['figure_size_in'] == [15, 5] and meta['text_usetex'] is True
    data_manifest = root / 'results/beamforming_moving/data/manifest.json'
    assert meta['data_manifest'] == str(data_manifest.relative_to(root))
    assert meta['data_manifest_sha256'] == _sha(data_manifest)
    plotting_manifest=root/'results/beamforming_moving/plotting/manifest.json'
    assert meta['plotting_manifest']==plotting_manifest.relative_to(root).as_posix()
    assert meta['plotting_manifest_sha256']==_sha(plotting_manifest)
    assert meta['input_hashes'] == data['input_hashes']
    original_sources = {record['path']: record['sha256']
                        for record in data['provenance']['verified_source_files'].values()}
    assert meta['original_source_hashes'] == original_sources
    _verify_hashes(root, original_sources, 'archived original moving producer')
    assert meta['source_hashes'] == notebook['metadata']['editable_plotting']['loader_sources']
    assert meta['numerical_cell_hashes'] == notebook['metadata']['editable_plotting']['numerical_cell_hashes']
    _verify_hashes(root, meta['source_hashes'], 'moving-figure producer')
    _verify_hashes(root, meta['input_hashes'], 'moving-figure input')
    assert set(meta['figures']) == set(FILENAMES)
    expected_outputs = set()
    for destination, route in routes.items():
        record = meta['figures'][route['filename']]
        condition = FILENAMES[route['filename']]
        assert record['status'] == 'completed' and record['condition'] == condition
        assert record['figure_size_in'] == FIGURE_SIZES[route['filename']] and record['dpi'] == 300
        assert record['text_usetex'] is True and record['optional_main_candidate'] is True
        assert record['event_t'] == (129 if condition=='comparison' else data['conditions'][condition]['event'])
        assert record['iqr_alpha']==.22 and record['mean_curves_unchanged'] is True
        assert '25th–75th' in record['uncertainty']
        assert record['caption'].strip() and record['formula_or_claim'].strip()
        cell = notebook['cells'][route['cell']]
        assert record['source_cell_id'] == cell['id']
        assert record['source_cell_sha256'] == hashlib.sha256(_source(cell).encode()).hexdigest()
        outputs = {destination, route['local_pdf'],
            str(Path(destination).with_suffix('.png')),
            str(Path(route['local_pdf']).with_suffix('.png'))}
        assert set(record['output_hashes']) == outputs
        _verify_hashes(root, record['output_hashes'], 'moving-figure output')
        expected_outputs.update(outputs)
    assert set(meta['files']) == expected_outputs
    _verify_hashes(root, meta['files'], 'moving-figure files')
    if meta.get('retained_previous_main'):
        from lib.presentation_update_provenance import verify_presentation_preservation
        verify_presentation_preservation(root, meta['retained_previous_main'])
    return dict(manifest=str(path.relative_to(root)), sha256=_sha(path),
                exported_files=len(meta['files']), optional_main_candidates=True)


def fullcheck(root=ROOT):
    """Verify the optional notebook, saved summary arithmetic and paired CIs."""
    root = Path(root).resolve()
    routes = optional_moving_routes(root)
    if not routes:
        return dict(status='skipped', reason='Moving-interference notebook is absent', routes={})
    from lib.beamforming_moving_evidence import load_moving_evidence
    from lib.final_validation import require_current_gate
    gate = require_current_gate(root)
    data = load_moving_evidence(root)
    notebook = json.loads((root / NOTEBOOK).read_text())
    notebook_checks = _notebook_check(root, notebook, routes, data)
    files = _exports_check(root, routes, {destination:FIGURE_SIZES[route['filename']] for destination,route in routes.items()})
    numerical = _numeric_check(root, data)
    display_spread = _iqr_check(root,data)
    preservation = _preservation_check(root)
    figure_manifest = _figure_manifest_check(root, notebook, routes, data)
    return dict(status='passed', notebook=NOTEBOOK, notebook_sha256=_sha(root / NOTEBOOK),
        routes=routes, notebook_checks=notebook_checks, files=files, numerical=numerical,
        preservation=preservation, figure_manifest=figure_manifest,display_spread=display_spread,
        scientific_sha256=gate['final_science_hash'], input_hashes=data['input_hashes'],
        original_producer_hashes={record['path']: record['sha256']
            for record in data['provenance']['verified_source_files'].values()},
        current_loader_sha256=_sha(root / 'lib/beamforming_moving_evidence.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, help='Optional JSON report; no result files are changed by default.')
    parser.add_argument('--routes-only', action='store_true')
    args = parser.parse_args()
    report = optional_moving_routes(args.root) if args.routes_only else fullcheck(args.root)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
