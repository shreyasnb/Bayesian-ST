"""Read-only verification of the current final-study scientific artifacts."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.final_validation import (
    BASE_COMMIT, EXAMPLES, STAGES, COMPONENTS, require_gate, verify_component,
    verify_history, verify_mapping, sha, gate_path)

INDIVIDUAL = (
    'setup', 'task_performance', 'posterior_anisotropy', 'noise_adaptation',
    'information', 'posterior_curvature', 'tracking_bound', 'block_updates',
    'additional_comparisons', 'variance_estimation', 'predictive_calibration',
    'predictive_scores', 'model_mismatch', 'computation_time', 'numerical_accuracy',
    'paper_summary')
BUNDLES = ('supp_information_tracking', 'supp_comparisons',
           'supp_variance_prediction_cost')


def verify_paper_budget(example, component, meta):
    cfg, checks = meta['config'], meta.get('checks', {})
    if component in ('performance', 'prediction'):
        assert cfg['validation_episodes'] == 64 and cfg['test_episodes'] == 256
    if component == 'performance':
        if example == 'beamforming':
            assert cfg['dimensions'] == [32, 64, 128]
            assert checks['stationary_episodes'] == 768 and checks['moving_episodes'] == 1280
            assert checks['native_mean_calls'] == 44032 and checks['native_mean_unresolved'] == 0
            assert checks['n128_task_completed'] is True
        else:
            assert cfg['n'] == 3 and cfg['k'] == 2
            assert checks['episodes'] == 1024 and checks['validation_episodes'] == 256
            assert checks['numerical_failures'] == 0
    elif component == 'prediction':
        assert cfg['calibration_validation_episodes'] == 32
        assert cfg['calibration_test_episodes'] == 128
        assert cfg['mismatch_episodes'] == 64
        assert checks['requested_jobs'] == (1568 if example == 'beamforming' else 1440)
        assert checks['adaptation_rows'] == (55296 if example == 'beamforming' else 27648)
        assert checks['calibration_rows'] == 57600 and checks['failed_jobs'] == 0
        assert checks['maximum_state_change_from_fit'] == 0
    elif component == 'computation':
        assert cfg['repetitions'] == 7 and cfg['warmup_repetitions'] >= 1
        assert checks['timed_cycles'] == 7 * checks['pipeline_targets']
        assert meta['timing_setting']['isolated_asserted'] is True
        assert all(value == '1' for value in meta['timing_setting']['blas_environment'].values())


def verify_figures(example):
    folder = ROOT / 'results' / example
    meta = json.loads((folder / 'figure_manifest.json').read_text())
    assert meta['status'] == 'completed' and meta['stage'] == 'paper'
    assert meta['example'] == example and not meta['unresolved']
    assert meta['text_usetex'] is True and meta['titles'] is False
    assert meta['preview_dpi'] == meta['rasterized_layer_dpi'] == 200
    assert meta['paper_width_in'] == 7
    assert meta['shared_camera'] == dict(elev=25., azim=-50.)
    assert meta['validation_sha256'] == sha(gate_path(ROOT))
    verify_mapping(ROOT, meta['sources'])
    verify_mapping(ROOT, meta['input_manifests'])
    verify_mapping(folder, meta['files'])
    expected = {name + '.pdf' for name in INDIVIDUAL + BUNDLES}
    assert set(meta['figures']) == expected, (example, set(meta['figures']) ^ expected)
    assert {p.name for p in folder.glob('*.pdf')} == expected
    required = ('experiment', 'formula_or_claim', 'producer', 'configurations',
                'source_hashes', 'data_manifest_hashes', 'seed_sets',
                'repetition_unit', 'repetition_counts', 'numerical_tolerances',
                'numerical_status', 'proposed_latex_caption', 'placement')
    page_count = 0
    for name, record in meta['figures'].items():
        assert record['status'] == 'completed'
        assert all(key in record for key in required), (example, name)
        assert record['proposed_latex_caption'].strip()
        verify_mapping(ROOT, record['source_hashes'])
        verify_mapping(ROOT, record['data_manifest_hashes'])
        verify_mapping(folder, record['output_hashes'])
        assert record['pages']
        for page in record['pages']:
            assert page['titles'] is False
            assert abs(page['figure_size_in'][0] - 7.) < 1e-10
            page_count += 1
    from PIL import Image
    previews = [folder / name for name in meta['files'] if name.endswith('.png')]
    for path in previews:
        with Image.open(path) as im:
            assert all(abs(d - 200.) < .1 for d in im.info['dpi']), path
    return dict(pdf_files=len(expected), pdf_pages_including_bundles=page_count,
                previews=len(previews), status='verified')


def verify_notebook(name):
    import nbformat
    path = ROOT / 'notebooks' / name
    nb = nbformat.read(path, as_version=4)
    example = 'beamforming' if name == '01_array.ipynb' else 'contact'
    record = nb.metadata['final_study']
    assert record['example'] == example and record['stage'] == 'paper'
    assert record['producer_sha256'] == sha(ROOT / 'scripts/make_final_notebooks.py')
    assert record['figure_manifest_sha256'] == sha(
        ROOT / 'results' / example / 'figure_manifest.json')
    code = [cell for cell in nb.cells if cell.cell_type == 'code' and cell.source.strip()]
    assert code and all(cell.execution_count is not None for cell in code), name
    assert not any(output.output_type == 'error'
                   for cell in code for output in cell.get('outputs', [])), name
    embedded = [hashlib.sha256(base64.b64decode(output['data']['image/png'])).hexdigest()
                for cell in code for output in cell.get('outputs', [])
                if 'image/png' in output.get('data', {})]
    expected = record['displayed_previews']
    verify_mapping(ROOT, expected)
    assert sorted(embedded) == sorted(expected.values()), name
    images = len(embedded)
    assert 5 <= images <= 10, (name, images)
    return dict(code_cells=len(code), embedded_images=images, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--science-only', action='store_true')
    args = parser.parse_args()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT,
                                     text=True).strip()
    assert branch == 'test', branch
    gate = require_gate(ROOT)
    history = verify_history(ROOT)
    result = dict(status='verified', branch=branch,
        final_science_hash=gate['final_science_hash'],
        archived_files=len(history['files']),
        unchanged_historical_results=len(history['historical_results']), components={})
    for example in EXAMPLES:
        result['components'][example] = {}
        for stage in STAGES:
            result['components'][example][stage] = {}
            for component in COMPONENTS:
                meta = verify_component(ROOT, example, stage, component)
                if stage == 'paper':
                    verify_paper_budget(example, component, meta)
                result['components'][example][stage][component] = dict(
                    status=meta['status'], artifacts=len(meta['artifacts']))
    if not args.science_only:
        result['figures'] = {example: verify_figures(example) for example in EXAMPLES}
        result['notebooks'] = {name: verify_notebook(name)
                               for name in ('01_array.ipynb', '03_contact.ipynb')}
        report = (ROOT / 'completion_report.md').read_text()
        assert 'runs are in progress' not in report and 'will be replaced' not in report
        inventory = json.loads((ROOT / 'final_artifact_inventory.json').read_text())
        verify_mapping(ROOT, inventory['files'])
        assert inventory['final_science_hash'] == gate['final_science_hash']
        changed = set(subprocess.check_output(
            ['git', 'diff', '--name-only', BASE_COMMIT, '--'], cwd=ROOT,
            text=True).splitlines())
        changed.update(subprocess.check_output(
            ['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT,
            text=True).splitlines())
        changed.discard('final_artifact_inventory.json')
        assert set(inventory['files']) == changed, 'Rebuild the complete artifact inventory'
        assert inventory['retained_files'] == len(changed)
        assert inventory['retained_bytes'] == sum((ROOT / p).stat().st_size for p in changed)
        result['retained_artifacts'] = inventory['retained_files']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
