"""Read-only audit of the optional contact-prediction diagnostic notebook.

Route discovery imports no plotting or numerical packages. The full audit
checks existing observations, derived statistics, posterior geometry and saved
figures; it does not execute a notebook or change scientific evidence. An
explicit maintenance flag refreshes only the two presentation metadata files.
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
ORIGINAL_FILENAMES = {
    'prediction_error_stationary.pdf': 'main',
    'prediction_error_sphere.pdf': 'main',
    'prediction_error_references.pdf': 'supplementary',
    'moving_sphere_posterior_spheres.pdf': 'supplementary',
}
TRANSPORT_OUTPUT=Path('results/contact_transport_diagnostics')
TRANSPORT_DIRECTORY=TRANSPORT_OUTPUT/'figures'
TRANSPORT_FILENAMES={name:'supplementary' for name in (
    'transport_paired_rise.pdf','transport_projector_change.pdf',
    'transport_excitation.pdf','transport_curvature.pdf')}
FILENAMES=dict(ORIGINAL_FILENAMES,**TRANSPORT_FILENAMES)


def _local_directory(filename):
    return TRANSPORT_DIRECTORY if filename in TRANSPORT_FILENAMES else LOCAL_DIRECTORY


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
                    if (len(parts) != 2 or parts[1][0] != 'literal' or parts[1][1] not in FILENAMES):
                        raise ValueError(f'{NOTEBOOK}, cell {index}: unexpected PDF_PATH')
                    expected_variable='TRANSPORT_OUTPUT' if parts[1][1] in TRANSPORT_FILENAMES else 'EDITABLE_OUTPUT'
                    if parts[0]!=('variable',expected_variable):
                        raise ValueError(f'{NOTEBOOK}, cell {index}: unexpected output directory')
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
                        local_pdf=(_local_directory(filename) / filename).as_posix())
                    declared = True
            value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
            if isinstance(value, ast.Call) and _qualified(value.func) in ('plt.figure', 'plt.subplots'):
                if not declared:
                    raise ValueError(f'{NOTEBOOK}, cell {index}: undeclared figure')
                declared = False
        if declared:
            raise ValueError(f'{NOTEBOOK}, cell {index}: PDF declaration without a figure')
    found={row['filename'] for row in routes.values()}
    expected=set(FILENAMES) if found & set(TRANSPORT_FILENAMES) else set(ORIGINAL_FILENAMES)
    if found!=expected or len(routes)!=len(expected):
        raise ValueError('Expected the original four figures and a complete optional transport set')
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
    for name in ('lib/contact_prediction_diagnostics.py', 'lib/contact_moving_snapshot.py',
                 'lib/contact_transport_diagnostics.py'):
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
    assert tex_true and figures == images == len(routes)
    # The three ordinary plots use full borders; the 3D density cell keeps its
    # original source and is deliberately excluded from this requirement.
    for identifier in ('prediction-error-stationary','prediction-error-sphere',
                       'prediction-error-references'):
        tree=ast.parse(_source(by_id[identifier]))
        widths=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='BORDER_WIDTH' for target in node.targets)]
        assert widths==[.9], 'Missing visible .9-point border control: '+identifier
        ticks=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=='tick_params'
               and any(key.arg=='direction' and isinstance(key.value,ast.Constant)
                       and key.value.value=='in' for key in node.keywords)]
        assert len(ticks)==1,'Expected one four-sided inward-tick instruction'
        keywords={key.arg:key.value for key in ticks[0].keywords}
        for name in ('top','right','bottom','left','labelbottom','labelleft'):
            assert ast.literal_eval(keywords[name]) is True
        for name in ('labeltop','labelright'):
            assert ast.literal_eval(keywords[name]) is False
    return dict(code_cells=count, visible_figures=figures, embedded_images=images,
                text_usetex=True, dpi=300, figure_size_in=[15, 5],
                numerical_inputs=len(inputs), producer_sources=len(sources),
                ordinary_plot_borders=dict(width_points=.9,sides=4,ticks='inward',
                                           labels='bottom and left'))


def _identity_curve_check(root,notebook,data,transport):
    """Check the fourth saved curve and its full-run paired comparison."""
    import numpy as np
    by_id={cell.get('id'):cell for cell in notebook['cells']}
    text=_source(by_id['prediction-error-sphere'])
    tree=ast.parse(text)
    assigned={}
    for node in tree.body:
        if not isinstance(node,ast.Assign):
            continue
        for target in node.targets:
            assigned[ast.unparse(target)]=node.value
    assert ast.literal_eval(assigned["LABELS['mean_identity']"])==r'$\overline P_t^+,\ G_t=I_3$'
    assert ast.literal_eval(assigned["COLORS['mean_identity']"])=='#484848'
    trajectories=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and node.func.attr=='plot']
    assert len(trajectories)==1
    assert len(trajectories[0].args)>=2
    keywords={item.arg:item.value for item in trajectories[0].keywords}
    assert not {'marker','markersize','markevery'} & set(keywords)
    assert ast.literal_eval(keywords['linestyle'])=='-'
    for actual,expected in zip(trajectories[0].args[:2],('points[:,0]','points[:,1]')):
        assert ast.dump(actual)==ast.dump(ast.parse(expected,mode='eval').body)
    band_condition = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'fill_between' for call in ast.walk(node)))
    assert ast.dump(band_condition.test) == ast.dump(
        ast.parse("method in ('mean', 'mean_identity')", mode='eval').body)
    band = next(node for node in ast.walk(band_condition) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'fill_between')
    for actual, expected in zip(band.args, ('points[:,0]', 'points[:,2]', 'points[:,3]')):
        assert ast.dump(actual) == ast.dump(ast.parse(expected, mode='eval').body)
    band_keywords = {item.arg: item.value for item in band.keywords}
    assert ast.literal_eval(band_keywords['alpha']) == .14
    assert ast.literal_eval(band_keywords['zorder']) == 1
    assert 'label' not in band_keywords
    assert any(isinstance(node,ast.Subscript) and isinstance(node.slice,ast.Constant)
               and node.slice.value=='sphere_curves' for node in ast.walk(tree))
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                   and node.func.attr in ('interp','interpolate') for node in ast.walk(tree))
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                   and node.func.attr=='axvline' for node in ast.walk(tree)), 'Sphere history has no event line'
    record=data['conditions']['sphere']
    assert {curve['method'] for curve in record['curves']}=={'mean','mode','iso'}
    identity=[curve for curve in record['all_curves'] if curve['method']=='mean_identity']
    assert len(identity)==1
    np.testing.assert_array_equal(identity[0]['points'][:,0],[4,16,32,64,128,192])
    dense=transport['sphere_curves']
    assert {curve['method'] for curve in dense}=={'mean','mode','iso','mean_identity'}
    for curve in dense:
        points=np.asarray(curve['points']);assert points.shape==(48,4)
        np.testing.assert_array_equal(points[:,0],np.arange(4,193,4))
        original=next(row for row in record['all_curves'] if row['method']==curve['method'])
        selected=np.isin(points[:,0],np.asarray(original['points'])[:,0])
        np.testing.assert_allclose(points[selected],original['points'],rtol=2e-10,atol=2e-9)
    pairs=data['paired_episode_average_summary']
    saved=pairs[pairs.condition.eq('sphere')&pairs.reference.eq('mean_identity')]
    assert len(saved)==1
    row=saved.iloc[0]
    assert row.endpoint=='episode_average' and int(row.t)==192 and int(row.simulations)==256
    np.testing.assert_allclose([row['gain'],row['lower'],row['upper']],
        [2.6548246409743737,2.095611015675944,3.2140382662728033],rtol=0,atol=1e-10)
    results=_source(by_id['contact-prediction-results'])
    for value in (row['gain'],row['lower'],row['upper']):
        assert f'{value:.4f}' in results
    assert '48' in results
    return dict(status='passed',methods=['mean','mode','iso','mean_identity'],curves=4,
        reporting_measurements=list(range(4,193,4)),original_six_checkpoints_unchanged=True,
        identity_label=r'$\overline P_t^+,\ G_t=I_3$',identity_color='#484848',
        identity_linestyle='-',identity_marker=None,original_mean_interval_retained=True,
        identity_minus_measured_transport=dict(gain=float(row['gain']),lower=float(row['lower']),
            upper=float(row['upper']),units='(mm/s)^2',simulations=256,endpoint='episode_average',
            internal_action_times=48,not_average_of_sparse_checkpoints=True))


def _transport_numeric_check(root,data):
    """Recompute dense contact intervals and diagnostics from sealed trial rows."""
    import numpy as np
    import pandas as pd
    from scipy.stats import t as student_t,skew
    folder=root/TRANSPORT_OUTPUT;manifest_path=folder/'data/manifest.json'
    manifest=json.loads(manifest_path.read_text());cfg=manifest['config']
    assert manifest['status']=='completed' and cfg['action_times']==list(range(4,193,4))
    assert cfg['n']==3 and cfg['k']==2 and cfg['T']==192 and cfg['beta']==.9
    assert cfg['test_simulations']==256 and cfg['original_validation_simulations']==64
    assert cfg['moment_tolerance']==5e-8 and cfg['algebra_tolerance']==1e-10
    for key in ('source_hashes','input_hashes','artifacts'):
        _verify_hashes(root,manifest[key],'dense contact '+key)
    declaration=json.loads((folder/'data/preregistered.json').read_text())
    assert declaration==cfg
    assert hashlib.sha256(json.dumps(cfg,sort_keys=True,allow_nan=False).encode()).hexdigest()==manifest['declaration_sha256']
    preflight=json.loads((folder/'data/preflight.json').read_text())
    assert preflight['status']=='passed' and preflight['declaration_sha256']==manifest['declaration_sha256']
    methods={'mean','mode','isotropic','regularized','great_window','great_ew','mean_identity','mode_identity'}
    frame=data['dense_metrics'];physical=data['physical_diagnostics'];diagnostics=data['trial_diagnostics']
    assert len(frame)==256*8*49 and set(frame.method)==methods and frame.condition.eq('sphere').all()
    assert not frame.duplicated(['trial','method','t','endpoint']).any()
    assert frame.numerical_ok.astype(bool).all() and np.isfinite(frame.response_mse).all()
    old=pd.read_csv(root/'results/contact/tables/paper/performance/metrics.csv')
    old=old[old.condition.eq('sphere')]
    keys=['condition','trial','method','t','endpoint']
    joined=old.merge(frame,on=keys,suffixes=('_old','_new'),validate='one_to_one')
    assert len(joined)==len(old)==256*56
    numeric=['time_s','beta','sigma2','moment_resolution_error','response_mse','response_map_frobenius',
             'command_cost','posterior_response_excess','posterior_command_excess','command_excess_identity_error']
    for column in numeric:
        np.testing.assert_allclose(joined[column+'_new'],joined[column+'_old'],rtol=1e-10,atol=1e-10)
    for column in ('numerical_ok','mode_unique'):
        np.testing.assert_array_equal(joined[column+'_new'],joined[column+'_old'])
    def interval(values):
        values=np.asarray(values,float);assert len(values)==256 and np.isfinite(values).all()
        mean=float(values.mean());half=float(student_t.ppf(.975,255)*values.std(ddof=1)/16.)
        return dict(simulations=256,mean=mean,lower=mean-half,upper=mean+half)
    def compare(actual,expected,columns):
        actual=actual.sort_values(columns).reset_index(drop=True)
        expected=expected.sort_values(columns).reset_index(drop=True)
        assert set(actual.columns)==set(expected.columns)
        pd.testing.assert_frame_equal(actual,expected[actual.columns],check_dtype=False,
            check_exact=False,rtol=3e-10,atol=2e-9)
    checkpoint=frame[frame.endpoint.eq('checkpoint')];episodes=frame[frame.endpoint.eq('episode_average')]
    assert len(checkpoint)==256*8*48 and len(episodes)==256*8 and episodes.t.eq(192).all()
    for (_,method),rows in checkpoint.groupby(['trial','method']):
        np.testing.assert_array_equal(np.sort(rows.t),np.arange(4,193,4))
    average=checkpoint.groupby(['trial','method']).response_mse.mean().rename('whole_run')
    merged=episodes.set_index(['trial','method']).join(average)
    np.testing.assert_allclose(merged.response_mse,merged.whole_run,rtol=1e-10,atol=1e-10)
    summaries=[]
    for (endpoint,t,method),rows in frame.groupby(['endpoint','t','method']):
        np.testing.assert_array_equal(np.sort(rows.trial),np.arange(256))
        summaries.append(dict(condition='sphere',endpoint=endpoint,t=int(t),
            method='iso' if method=='isotropic' else method,**interval(rows.response_mse.to_numpy()*1e6)))
    summaries=pd.DataFrame(summaries)
    for endpoint in ('checkpoint','episode_average'):
        compare(data[endpoint+'_summary'],summaries[summaries.endpoint.eq(endpoint)],['endpoint','t','method'])
    def distribution(values,**attributes):
        values=np.asarray(values,float);absolute=np.sort(abs(values))[::-1]
        positive=np.sort(np.maximum(values,0))[::-1];stats=interval(values)
        return dict(attributes,**stats,fraction_positive=float(np.mean(values>0)),median=float(np.median(values)),
            **{'q'+str(q).zfill(2):float(np.quantile(values,q/100)) for q in (1,5,25,75,95,99)},
            minimum=float(values.min()),maximum=float(values.max()),sd=float(values.std(ddof=1)),
            skewness=float(skew(values,bias=False)),top_1pct_absolute_contribution=float(absolute[:3].sum()/absolute.sum()),
            top_5pct_absolute_contribution=float(absolute[:13].sum()/absolute.sum()),
            top_5pct_positive_contribution=float(positive[:13].sum()/positive.sum()) if positive.sum() else 0.,
            tail_definition='Top 1% and 5% use respectively the largest 3 and 13 absolute changes among all 256; primary statistics always retain every trial.')
    rises=[];rise_trials=[];plane=[]
    original_plane=pd.read_csv(root/'results/contact/tables/paper/performance/metrics.csv')
    original_plane=original_plane[original_plane.condition.eq('plane')&original_plane.endpoint.eq('checkpoint')]
    for condition,rows,target in [('sphere',checkpoint,rises),('plane',original_plane,plane)]:
        for method,group in rows.groupby('method'):
            wide=group.pivot(index='trial',columns='t',values='response_mse')*1e6
            np.testing.assert_array_equal(wide.index,np.arange(256))
            delta=wide[192]-wide[128];alias='iso' if method=='isotropic' else method
            target.append(distribution(delta,condition=condition,method=alias))
            if condition=='sphere':
                rise_trials.extend(dict(trial=int(trial),method=alias,error128_mm2_s2=float(wide.loc[trial,128]),
                    error192_mm2_s2=float(wide.loc[trial,192]),difference_mm2_s2=float(value)) for trial,value in delta.items())
    compare(data['paired_rise'],pd.DataFrame(rises),['condition','method'])
    compare(data['plane_control_late_rise'],pd.DataFrame(plane),['condition','method'])
    rise_trials=pd.DataFrame(rise_trials)
    compare(data['late_rise_trials'],rise_trials,['trial','method'])
    wide=rise_trials.pivot(index='trial',columns='method',values='difference_mm2_s2')
    difference=wide['mean']-wide['mean_identity']
    compare(data['paired_rise_control'],pd.DataFrame([distribution(difference,
        comparison='(E192-E128)_mean-(E192-E128)_mean_identity')]),['comparison'])
    compare(data['paired_rise_control_trials'],pd.DataFrame(dict(trial=wide.index,
        difference_mm2_s2=difference.to_numpy())),['trial'])
    assert len(physical)==256*192 and len(diagnostics)==256*48
    assert not physical.duplicated(['trial','t']).any() and not diagnostics.duplicated(['trial','t']).any()
    for _,rows in physical.groupby('trial'):
        np.testing.assert_array_equal(np.sort(rows.t),np.arange(1,193))
    np.testing.assert_allclose(physical.time_s,(physical.t-1)*.05,rtol=1e-12,atol=1e-12)
    for column in ('sensor_increment_fro','transported_increment_fro','world_increment_fro'):
        assert physical.loc[physical.t.eq(1),column].isna().all()
        assert np.isfinite(physical.loc[physical.t.ge(2),column]).all()
    np.testing.assert_allclose(physical.transported_increment_fro,physical.world_increment_fro,
                               rtol=1e-10,atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(diagnostics.effective_input_mass,(1-.9**diagnostics.t)/.1,rtol=1e-12,atol=1e-12)
    for prefix in ('input_gram','input_gram_identity'):
        values=diagnostics[[prefix+'_lambda_'+str(i) for i in (1,2,3)]].to_numpy()
        assert np.isfinite(values).all() and values.min()>-1e-10 and np.diff(values,axis=1).min()>-1e-10
    for prefix in ('curvature','curvature_identity'):
        assert (diagnostics[prefix+'_min']>=-1e-10).all()
        assert (diagnostics[prefix+'_max']>=diagnostics[prefix+'_min']-1e-10).all()
    for prefix in ('transported','identity'):
        for metric in ('mean_resolution_error','log_normalizer_resolution_error'):
            assert diagnostics[prefix+'_'+metric].ge(0).all()
            assert diagnostics[prefix+'_'+metric].lt(5e-8).all()
    groups={
        'projector_changes':(physical,['sensor_increment_fro','transported_increment_fro','world_increment_fro']),
        'frame_rotation':(physical,['frame_angular_speed_rad_s']),
        'excitation':(diagnostics,['input_gram_lambda_'+str(i) for i in (1,2,3)]),
        'curvature':(diagnostics,['curvature_min','curvature_max','curvature_identity_min','curvature_identity_max']),
        'numerical_accuracy':(diagnostics,[prefix+'_'+metric for prefix in ('transported','identity')
            for metric in ('mean_resolution_error','log_normalizer_resolution_error','angular_order')])}
    diagnostic_intervals=0
    for name,(raw,metrics) in groups.items():
        expected=[]
        for t,rows in raw.groupby('t'):
            np.testing.assert_array_equal(np.sort(rows.trial),np.arange(256))
            for metric in metrics:
                values=rows[metric].to_numpy(float)
                if not np.isfinite(values).all():continue
                q25,q75=np.quantile(values,[.25,.75])
                expected.append(dict(t=int(t),metric=metric,**interval(values),q25=float(q25),q75=float(q75),
                                     deterministic=bool(np.ptp(values)==0.)))
        compare(data[name],pd.DataFrame(expected),['t','metric']);diagnostic_intervals+=len(expected)
    story_checks=_transport_story_check(root,data,cfg)
    checks=json.loads((folder/'data/trial_checks.json').read_text())
    assert len(checks)==256 and [row['trial'] for row in checks]==list(range(256))
    assert sum(row['parity']['matched_rows'] for row in checks)==256*56
    assert all(row['same_observations_variance_beta_action_times'] and row['only_row_retention_changed']
               and row['all_native_moments_resolved'] for row in checks)
    assert all(row['native_checked_moment_calls']==96 for row in checks)
    assert len({row['exact_seed_generated_arrays']['U'] for row in checks})==256
    assert len({row['exact_seed_generated_arrays']['G'] for row in checks})==1
    return dict(status='passed',manifest=str(manifest_path.relative_to(root)),manifest_sha256=_sha(manifest_path),
        simulations=256,methods=8,action_times=list(range(4,193,4)),dense_rows=len(frame),
        original_rows_matched=len(joined),whole_run_averages_verified=256*8,
        summary_intervals=len(summaries),paired_late_change_intervals=17,diagnostic_intervals=diagnostic_intervals,
        saved_story=story_checks,rise_means={row['method']:row['mean'] for row in rises},
        all_trials_retained=True,original_accuracy_tolerances_preserved=True)


def _transport_story_check(root,data,cfg):
    """Reconstruct geometry, input Gramians and Hessians from saved story arrays."""
    import numpy as np
    with np.load(root/'results/contact/data/paper/performance/story.npz',allow_pickle=False) as saved:
        U,V,G,R,P,nworld=[saved['sphere_'+name] for name in ('U','V','G','R','P','world_normal')]
        saved_A=saved['sphere_A']
    world=np.eye(3)-np.einsum('ti,tj->tij',nworld,nworld)
    normal=np.einsum('tji,tj->ti',R,nworld)
    transported=np.einsum('tij,tjk,tlk->til',G[1:],P[:-1],G[1:])
    sensor=np.r_[np.nan,np.linalg.norm(np.diff(P,axis=0),axis=(1,2))]
    residual=np.r_[np.nan,np.linalg.norm(P[1:]-transported,axis=(1,2))]
    world_step=np.r_[np.nan,np.linalg.norm(np.diff(world,axis=0),axis=(1,2))]
    speed=np.arccos(np.clip((np.trace(G,axis1=1,axis2=2)-1)/2,-1,1))/.05
    physical=data['physical_diagnostics'];physical=physical[physical.trial.eq(0)].sort_values('t')
    for name,values in [('sensor_increment_fro',sensor),('transported_increment_fro',residual),
            ('world_increment_fro',world_step),('frame_angular_speed_rad_s',speed)]:
        np.testing.assert_allclose(physical[name],values,rtol=1e-10,atol=1e-10,equal_nan=True)
    rows=data['trial_diagnostics'];rows=rows[rows.trial.eq(0)].set_index('t')
    S=np.zeros((3,3));Si=S.copy();A=S.copy();Ai=S.copy();mass=0.;maximum_A_error=0.
    for j,(u,v,g) in enumerate(zip(U,V,G)):
        t=j+1;S=.9*g@S@g.T+np.outer(v,v);Si=.9*Si+np.outer(v,v);mass=.9*mass+1
        B=.5/cfg['sigma2']*(np.outer(u,v)+np.outer(v,u)-np.outer(v,v))
        A=.9*g@A@g.T+B;A=(A+A.T)/2;A-=np.trace(A)*np.eye(3)/3
        Ai=.9*Ai+B;Ai=(Ai+Ai.T)/2;Ai-=np.trace(Ai)*np.eye(3)/3
        maximum_A_error=max(maximum_A_error,float(np.max(abs(A-saved_A[j]))))
        np.testing.assert_allclose(A,saved_A[j],rtol=1e-10,atol=1e-10)
        expected=float(normal[j]@S@normal[j]/np.trace(S))
        np.testing.assert_allclose(physical.normal_input_fraction.iloc[j],expected,rtol=1e-10,atol=1e-10)
        if t not in rows.index:continue
        for prefix,matrix in [('input_gram',S),('input_gram_identity',Si)]:
            eig=np.linalg.eigvalsh((matrix+matrix.T)/(2*cfg['sigma2']*mass))
            np.testing.assert_allclose(rows.loc[t,[prefix+'_lambda_'+str(i) for i in (1,2,3)]].to_numpy(float),eig,
                                       rtol=1e-10,atol=1e-10)
        for prefix,matrix in [('curvature',A),('curvature_identity',Ai)]:
            eig=np.linalg.eigvalsh(matrix);curvature=2*(eig[1:]-eig[0])
            np.testing.assert_allclose(rows.loc[t,[prefix+'_min',prefix+'_max']].to_numpy(float),curvature,
                                       rtol=1e-10,atol=1e-10)
    return dict(simulation=0,scope='Existing saved trajectory only; no new data or posterior moment integration.',
        geometry=True,input_Gramian=True,normal_density_curvature=True,maximum_parameter_absolute_error=maximum_A_error)


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
    path = root/'results/contact/data/paper/performance/anisotropy.npz'
    with np.load(path,allow_pickle=False) as saved:
        a = {name:saved[name] for name in saved.files}
    # The visible risk block overrides the generic loader with family 0. The
    # underlying loader and all simulation arrays intentionally remain intact.
    notebook=json.loads((root/'notebooks/03_contact.ipynb').read_text())
    cell=next(cell for cell in notebook['cells'] if cell.get('id')=='45ec7869')
    source=_source(cell)
    risk_start=source.rfind('# Edit these controls',0,source.index("'posterior_anisotropy_risk.pdf'"))
    risk_source=source[risk_start:]
    tree=ast.parse(risk_source)
    assignments={ast.unparse(target):node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
                 for target in node.targets}
    required={'query_family':"np.asarray(saved['query_family']) == 0",
        'theta':"np.asarray(saved['query_angles'])[query_family].copy()",
        'unit_queries':"np.asarray(saved['unit_queries'])[query_family].copy()",
        'risk_full':"np.asarray(saved['risks_full'])[query_family].copy()",
        'risk_iso':"np.asarray(saved['risks_isotropic'])[query_family].copy()"}
    for key,expected in required.items():
        assert ast.dump(assignments[key],include_attributes=False)==ast.dump(
            ast.parse(expected,mode='eval').body,include_attributes=False),key
    selected = a['query_family']==0
    theta=a['query_angles'][selected]
    unit=a['unit_queries'][selected]
    np.testing.assert_allclose(unit,np.column_stack([np.cos(theta),np.sin(theta),np.zeros_like(theta)]),
                               rtol=0,atol=1e-14)
    plotted=dict(risk_full=a['risks_full'][selected],risk_iso=a['risks_isotropic'][selected])
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
    endpoints=[]
    for angle,expected in ((0.,(.019462209657903684,.08842183901123923)),
                           (np.pi/2,(.14346703590668408,.09824824499508511))):
        index=int(np.argmin(abs(theta-angle)))
        np.testing.assert_allclose([plotted['risk_full'][index],plotted['risk_iso'][index]],
                                   expected,rtol=0,atol=1e-14)
        endpoints.append(dict(theta=float(theta[index]),risk_full=float(plotted['risk_full'][index]),
                              risk_isotropic=float(plotted['risk_iso'][index])))
    risk_exports=_exports_check(root,{'results/main/contact/posterior_anisotropy_risk.pdf':dict(
        local_pdf='results/notebook_exports/03_contact/posterior_anisotropy_risk.pdf')})
    return dict(status='passed',original_arrays_sha256=_sha(path),
        each_curve_uses_its_own_law_mean=True,normalization='r_t(v(theta))/v0^2',
        query_family=0,query_definition='v(theta)=v0*(cos(theta)*e1+sin(theta)*e2)',
        saved_unit_risks_used_directly=True,endpoints=endpoints,exports=risk_exports,
        maximum_absolute_errors=errors,
        interpretation='Posterior response variance under each displayed law; not an empirical loss under one shared law.')


def _preservation_check(root):
    from lib.presentation_update_provenance import verify_presentation_preservation
    proof_path = root/OUTPUT/'preserved_files.json'
    proof = json.loads(proof_path.read_text())
    mapping = proof['tracked_files']
    tracked_check=verify_presentation_preservation(root,mapping)
    baseline_path = Path('/tmp/contact-prediction-preservation-baseline.json')
    local = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        checked=verify_presentation_preservation(root,baseline)
        local = dict(files=len(baseline),sha256=_sha(baseline_path),
                     all_files_unchanged=not checked['authorized_presentation_updates'],
                     authorized_presentation_updates=checked['authorized_presentation_updates'])
    return dict(status='passed',tracked_files=len(mapping),tracked_proof_sha256=_sha(proof_path),
                all_original_tracked_files_unchanged=not tracked_check['authorized_presentation_updates'],
                authorized_presentation_updates=tracked_check['authorized_presentation_updates'],
                scientific_data_and_settings_unchanged=True,local_snapshot=local)


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
    assert meta['snapshot_time']==96
    assert set(meta['figures'])=={route['filename'] for route in routes.values()}
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
    if 'methods' in meta['figures']['prediction_error_sphere.pdf']:
        assert meta['figures']['prediction_error_sphere.pdf']['methods']==['mean','mode','iso','mean_identity']
    assert str(OUTPUT/'data/manifest.json') in meta['files']
    _verify_hashes(root,meta['files'],'contact diagnostic manifest file')
    transport_path=root/TRANSPORT_OUTPUT/'figure_manifest.json'
    transport=json.loads(transport_path.read_text())
    assert transport['status']=='completed' and transport['notebook']==NOTEBOOK
    assert transport['notebook_sha256']==_sha(root/NOTEBOOK)
    assert transport['producer']==meta['producer'] and transport['producer_sha256']==_sha(root/meta['producer'])
    assert transport['data_manifest']==str(TRANSPORT_OUTPUT/'data/manifest.json')
    assert transport['data_manifest_sha256']==_sha(root/transport['data_manifest'])
    assert transport['figures']=={name:meta['figures'][name] for name in TRANSPORT_FILENAMES}
    for key in ('input_hashes','source_hashes','files'):
        _verify_hashes(root,transport[key],'separate transport figure '+key)
    return dict(manifest=str(path.relative_to(root)),sha256=_sha(path),
                figure_copies=len(expected_outputs),all_bound_files=len(meta['files']),
                transport_manifest=str(transport_path.relative_to(root)),transport_manifest_sha256=_sha(transport_path))


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
        if route['filename']!='moving_sphere_posterior_spheres.pdf':
            tree=ast.parse(_source(notebook['cells'][route['cell']]))
            controls={target.id:node.value for node in tree.body if isinstance(node,ast.Assign)
                      for target in node.targets if isinstance(target,ast.Name)}
            border_width=next((float(ast.literal_eval(controls[name])) for name in ('BORDER_WIDTH','SPINE_WIDTH')
                               if name in controls),.9)
            if route['filename'] in ORIGINAL_FILENAMES:
                assert border_width==.9
            with fitz.open(root/destination) as document:
                page=document[0]
                horizontal=[];vertical=[]
                for drawing in page.get_drawings():
                    width=drawing.get('width')
                    if width is None or abs(float(width)-border_width)>1e-4:
                        continue
                    for item in drawing['items']:
                        if item[0]!='l':
                            continue
                        first,second=item[1:]
                        if abs(first.y-second.y)<1e-5 and abs(first.x-second.x)>72:
                            horizontal.append([float(first.x),float(first.y),float(second.x),float(second.y)])
                        if abs(first.x-second.x)<1e-5 and abs(first.y-second.y)>72:
                            vertical.append([float(first.x),float(first.y),float(second.x),float(second.y)])
                assert len(horizontal)>=2 and len(vertical)>=2,'Missing full PDF borders'
                if route['filename']=='prediction_error_sphere.pdf':
                    assert 'Gt=I3' in ''.join(page.get_text().split())
            report['ordinary_plot_borders']=dict(width_points=border_width,horizontal=horizontal,vertical=vertical)
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


def refresh_presentation_metadata(root=ROOT):
    """Refresh only the authorized figure manifest and read-only PDF report.

    No notebook, image, snapshot, numerical record, data manifest, or historical
    independent-validation report is written by this maintenance operation.
    """
    root=Path(root).resolve()
    path=root/OUTPUT/'figure_manifest.json'
    meta=json.loads(path.read_text())
    old_meta=json.loads(path.read_text())
    routes=optional_contact_diagnostic_routes(root)
    notebook=json.loads((root/NOTEBOOK).read_text())
    pdfs=_pdf_audit(root)
    assert pdfs['status']=='passed'
    asset_names={record[key] for record in meta['figures'].values()
                 for key in ('local_pdf','local_png','draft_pdf','draft_png')}
    # Scientific-data hashes must stay bound to their original bytes. Only
    # presentation assets can receive refreshed hashes here.
    for name,digest in meta['files'].items():
        if name not in asset_names:
            assert _sha(root/name)==digest,'Scientific diagnostic evidence changed: '+name
    _verify_hashes(root,meta['input_hashes'],'unchanged contact diagnostic input')
    _verify_hashes(root,meta['source_hashes'],'unchanged contact numerical source')
    meta['notebook_sha256']=_sha(root/NOTEBOOK)
    meta['producer_sha256']=_sha(root/meta['producer'])
    methods={'prediction_error_stationary.pdf':['mean','mode','iso'],
        'prediction_error_sphere.pdf':['mean','mode','iso','mean_identity'],
        'prediction_error_references.pdf':['mean','mode','iso','regularized','great_window','great_ew']}
    local_reports={}
    for destination,route in routes.items():
        record=meta['figures'][route['filename']]
        record['hashes']={name:_sha(root/name) for name in record['hashes']}
        meta['files'].update(record['hashes'])
        if route['filename'] in methods:
            record['methods']=methods[route['filename']]
            record['border']=dict(width_points=.9,sides=4,inward_ticks=True,
                                  tick_labels='bottom and left')
        else:
            assert record==old_meta['figures'][route['filename']], 'The 3D figure metadata changed'
        local_reports[record['local_pdf']]=pdfs['figures'][destination]
    meta['presentation_updates']=dict(
        directional_risk='Saved query family0; unit-query risks are used directly without rescaling.',
        sphere_prediction='Original mean_identity control added to the three original curves.',
        borders='Three ordinary diagnostic plots have four .9-point spines and inward ticks.',
        applied_by=['scripts/update_contact_plotting_addendum.py',
                    'scripts/update_contact_diagnostics_borders.py'],
        no_new_simulations=True,no_new_parameter_selection=True,no_new_posterior_fits=True)
    layout=dict(status='passed',pdfs=local_reports,
        executed_code_cells=sum(cell['cell_type']=='code' and bool(_source(cell).strip())
                                for cell in notebook['cells']),
        figure_cells=len(routes),one_png_per_figure=True,
        audit_source_sha256=_sha(root/'scripts/check_manuscript_plot_exports.py'))
    path.write_text(json.dumps(meta,indent=2)+'\n')
    layout_path=root/OUTPUT/'pdf_layout_checks.json'
    layout_path.write_text(json.dumps(layout,indent=2)+'\n')
    return dict(status='completed',refreshed_files={str(path.relative_to(root)):_sha(path),
        str(layout_path.relative_to(root)):_sha(layout_path)},
        notebook_sha256=meta['notebook_sha256'],producer_sha256=meta['producer_sha256'],
        unchanged_scientific_data=True,unchanged_3d_figure_metadata=True)


def fullcheck(root=ROOT):
    """Verify the complete optional addendum without changing its files."""
    root = Path(root).resolve()
    routes = optional_contact_diagnostic_routes(root)
    if not routes:
        return dict(status='skipped', reason='Contact prediction diagnostics notebook is absent', routes={})
    from lib.contact_prediction_diagnostics import load_contact_prediction_diagnostics
    from lib.contact_moving_snapshot import load_moving_sphere_snapshot
    from lib.contact_transport_diagnostics import load_contact_transport_diagnostics
    from lib.final_validation import require_current_gate
    gate=require_current_gate(root)
    data=load_contact_prediction_diagnostics(root)
    transport=load_contact_transport_diagnostics(root)
    snapshot=load_moving_sphere_snapshot(root,t=96)
    inputs=dict(data['input_hashes'])
    for name,expected in snapshot['input_hashes'].items():
        assert name not in inputs or inputs[name]==expected
        inputs[name]=expected
    data_manifest=root/OUTPUT/'data/manifest.json'
    inputs[str(data_manifest.relative_to(root))]=_sha(data_manifest)
    for name,expected in transport['input_hashes'].items():
        assert name not in inputs or inputs[name]==expected
        inputs[name]=expected
    transport_manifest=TRANSPORT_OUTPUT/'data/manifest.json'
    inputs[str(transport_manifest)]=_sha(root/transport_manifest)
    notebook=json.loads((root/NOTEBOOK).read_text())
    transport_meta=notebook['metadata']['contact_transport_diagnostics']
    assert transport_meta['manifest']==str(transport_manifest)
    assert transport_meta['manifest_sha256']==_sha(root/transport_manifest)
    assert transport_meta['action_times']==list(range(4,193,4)) and transport_meta['simulations']==256
    assert transport_meta['identity_color']=='#484848'
    assert set(transport_meta['figures'])=={Path(name).stem for name in TRANSPORT_FILENAMES}
    assert transport_meta['checks']==transport['checks']
    assert transport_meta['input_hashes']==transport['input_hashes']
    assert transport_meta['source_hashes']==transport['source_hashes']
    notebook_checks=_notebook_check(root,notebook,routes,inputs)
    files=_exports_check(root,routes)
    numerical=_numeric_check(root,data)
    identity_curve=_identity_curve_check(root,notebook,data,transport)
    dense_transport=_transport_numeric_check(root,transport)
    snapshot_checks=_snapshot_check(root,snapshot)
    static_variance=_static_response_variance_check(root)
    manifest=_figure_manifest_check(root,notebook,routes,inputs)
    preservation=_preservation_check(root)
    pdf_audit=_pdf_audit(root)
    return dict(status='passed',notebook=NOTEBOOK,notebook_sha256=_sha(root/NOTEBOOK),routes=routes,
        notebook_checks=notebook_checks,files=files,numerical=numerical,
        identity_curve=identity_curve,dense_transport=dense_transport,snapshot=snapshot_checks,
        retained_static_response_variance=static_variance,figure_manifest=manifest,
        preservation=preservation,pdf_audit=pdf_audit,scientific_sha256=gate['final_science_hash'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--routes-only', action='store_true')
    parser.add_argument('--pdf-only',action='store_true',help='Run only the independent PyMuPDF audit.')
    parser.add_argument('--refresh-presentation-metadata',action='store_true',
        help='Refresh only figure_manifest.json and pdf_layout_checks.json from current saved figures.')
    parser.add_argument('--report', type=Path, help='Optional JSON report; result files are read-only.')
    args = parser.parse_args()
    report = (refresh_presentation_metadata(args.root) if args.refresh_presentation_metadata else
              optional_contact_diagnostic_routes(args.root) if args.routes_only else
              audit_new_pdfs(args.root) if args.pdf_only else fullcheck(args.root))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
