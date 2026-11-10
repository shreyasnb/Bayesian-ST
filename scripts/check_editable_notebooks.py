"""Read-only checks for visible, editable Matplotlib cells in root notebooks.

Plot controls are deliberately not content-addressed: changing DPI, FIGSIZE,
labels, styles or PDF names is the purpose of these notebooks. Frozen inputs
and data-loader sources remain verified independently of those controls.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    '00_theory.ipynb', '01_array.ipynb', '02_field.ipynb',
    '03_contact.ipynb', '04_implementation_audit.ipynb')
DEFAULT_FIGURES = dict(zip(NOTEBOOKS, (4, 8, 17, 8, 12)))
FORBIDDEN_CALLS = {'show', 'show_saved', 'save', 'exec', 'eval'}
HIDDEN_RENDER_CALLS = {
    'setup', 'information', 'task_performance', 'posterior_anisotropy',
    'noise_adaptation', 'predictive_calibration', 'computation_time',
    'paper_summary', 'density_sphere', 'prepare_paper_axes',
    'prediction_fans', 'projected_clouds', 'drift_certificate',
    'monte_carlo_uncertainty', 'field_overview', 'paired_field', 'field_maps',
    'calibration_overview', 'gain_maps', 'decision_calibration',
    'coverage_regions', 'noise_overview', 'predictive_scores',
    'cost_tradeoff', 'sampler_overview', 'fisher', 'count_errors', 'curvature',
    'bound_envelopes', 'bound_phase', 'heldout', 'tracking_decisions',
    'calibration', 'block_evidence', 'adaptation', 'mismatch'}
HIDDEN_RENDER_MODULES = {
    'lib.final_paper_visuals', 'lib.final_information_visuals', 'lib.visuals',
    'lib.story_visuals', 'lib.decision_visuals',
    'scripts.plot_final_simulations', 'scripts.plot_implementation_audit'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(cell):
    value = cell.get('source', '')
    return ''.join(value) if isinstance(value, list) else value


def _tree(source, location):
    try:
        return ast.parse(source)
    except SyntaxError:
        # Notebook setup can contain ordinary IPython magics. Transforming
        # their syntax does not execute them, nor hide Matplotlib source.
        try:
            from IPython.core.inputtransformer2 import TransformerManager
            return ast.parse(TransformerManager().transform_cell(source))
        except (ImportError, SyntaxError) as error:
            raise AssertionError(f'{location}: code cannot be parsed') from error


def _qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _qualified(node.value) + '.' + node.attr
    return ''


def _verify_mapping(root, mapping, category):
    assert isinstance(mapping, dict), category + ' must map paths to SHA-256 values'
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected.get('sha256')
        assert isinstance(expected, str), f'{category}: missing hash for {name}'
        path = root / name
        assert path.is_file(), f'{category}: missing {name}'
        assert sha(path) == expected, f'{category}: changed {name}'


def _metadata_mappings(meta):
    inputs = {}
    for name in ('input_hashes', 'data_hashes'):
        inputs.update(meta.get(name, {}))
    loaders = dict(meta.get('loader_sources', {}))
    if meta.get('loader'):
        assert meta.get('loader_sha256'), 'The data loader needs its source hash'
        loaders[meta['loader']] = meta['loader_sha256']
    return inputs, loaders


def _expected_count(meta, name):
    expected = meta.get('figure_count', meta.get('figure_cells'))
    if isinstance(expected, (list, dict)):
        expected = len(expected)
    if expected is None and meta.get('figure_ids'):
        expected = len(meta['figure_ids'])
    return int(expected if expected is not None else DEFAULT_FIGURES[name])


def _check_numerical_cells(nb, meta, name):
    expected = meta.get('numerical_cell_hashes', {})
    if isinstance(expected, list):
        expected = {row.get('id', row.get('cell_id')): row['sha256'] for row in expected}
    by_id = {cell.get('id'): cell for cell in nb['cells']}
    for cell_id, digest in expected.items():
        assert cell_id in by_id, f'{name}: missing numerical cell {cell_id}'
        if isinstance(digest, dict):
            digest = digest['sha256']
        actual = hashlib.sha256(_source(by_id[cell_id]).encode()).hexdigest()
        assert actual == digest, f'{name}: changed preserved numerical cell {cell_id}'
    return len(expected)


def verify_notebook(root, name, require_executed=True, check_pdfs=False):
    """Verify editable plot structure without constraining user plot choices."""
    root = Path(root).resolve()
    name = Path(name).name
    assert name in NOTEBOOKS, 'Unknown root notebook: ' + name
    path = root / 'notebooks' / name
    nb = json.loads(path.read_text())
    meta = nb.get('metadata', {}).get('editable_plotting')
    assert isinstance(meta, dict), name + ': missing editable_plotting metadata'
    if 'mode' in meta:
        assert meta['mode'] == 'inline_matplotlib', name + ': unexpected plotting mode'
    if 'kind' in meta:
        assert meta['kind'] in ('explicit_matplotlib', 'inline_matplotlib'), name
    inputs, loaders = _metadata_mappings(meta)
    _verify_mapping(root, inputs, name + ' numerical inputs')
    _verify_mapping(root, loaders, name + ' data-loader sources')
    numerical_cells = _check_numerical_cells(nb, meta, name)

    code_cells = [cell for cell in nb['cells']
                  if cell.get('cell_type') == 'code' and _source(cell).strip()]
    plots, rendered, embedded, explicit_pdf_names = [], 0, 0, set()
    for index, cell in enumerate(code_cells, 1):
        source = _source(cell)
        location = f'{name}, code cell {index}'
        tree = _tree(source, location)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for call in calls:
            function = _qualified(call.func)
            assert function not in FORBIDDEN_CALLS, f'{location}: hidden call {function}()'
        figures = [call for call in calls
                   if _qualified(call.func) in ('plt.figure', 'plt.subplots')]
        if figures:
            plots.append(cell)
            rendered += len(figures)
            # Require editable controls to live in this figure's own cell.
            assigned = {node.id for node in ast.walk(tree)
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
            assert {'DPI', 'FIGSIZE'} <= assigned, f'{location}: expose DPI and FIGSIZE assignments'
            for figure in figures:
                kwargs = {item.arg: item.value for item in figure.keywords}
                assert isinstance(kwargs.get('dpi'), ast.Name) and kwargs['dpi'].id == 'DPI', (
                    location + ': figure creation must use dpi=DPI')
                assert isinstance(kwargs.get('figsize'), ast.Name) and kwargs['figsize'].id == 'FIGSIZE', (
                    location + ': figure creation must use figsize=FIGSIZE')
            shows = sum(_qualified(call.func) == 'plt.show' for call in calls)
            saves = sum(isinstance(call.func, ast.Attribute) and call.func.attr == 'savefig'
                        for call in calls)
            assert shows >= len(figures), (
                location + ': each figure needs its own plt.show()')
            assert saves >= len(figures), (
                location + ': each figure needs a direct figure.savefig()')
            for call in calls:
                function = _qualified(call.func)
                assert function.rsplit('.', 1)[-1] not in HIDDEN_RENDER_CALLS, (
                    location + ': hidden rendering function ' + function)
                assert not (isinstance(call.func, ast.Name)
                            and function.startswith(('_draw_', '_plot_'))), (
                    location + ': hidden plotting helper ' + function)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in HIDDEN_RENDER_MODULES, (
                        location + ': imports a hidden renderer ' + str(node.module))
                if isinstance(node, ast.Import):
                    assert not any(alias.name in HIDDEN_RENDER_MODULES for alias in node.names), (
                        location + ': imports a hidden renderer')
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.endswith('.pdf') and Path(node.value).name != '.pdf':
                        explicit_pdf_names.add(Path(node.value).name)
        if require_executed:
            assert cell.get('execution_count') is not None, location + ': not executed'
        for output in cell.get('outputs', []):
            assert output.get('output_type') != 'error', location + ': saved execution error'
            embedded += int('image/png' in output.get('data', {}))

    expected = _expected_count(meta, name)
    expected_rendered = int(meta.get('rendered_figure_count', expected))
    assert len(plots) == expected, f'{name}: expected {expected} visible plot cells, found {len(plots)}'
    assert rendered == expected_rendered, (
        f'{name}: expected {expected_rendered} figure creations, found {rendered}')
    if require_executed:
        assert embedded == expected_rendered, (
            f'{name}: expected {expected_rendered} embedded figures, found {embedded}')
    pdfs_checked = 0
    if check_pdfs:
        directory = root / meta['output_directory']
        names = explicit_pdf_names or {str(key) + '.pdf' for key in meta.get('figure_ids', [])}
        assert len(names) == expected_rendered, name + ': cannot identify every explicit PDF output'
        for filename in names:
            output = directory / filename
            assert output.is_file(), f'{name}: missing editable PDF {output}'
            with output.open('rb') as handle:
                assert handle.read(5) == b'%PDF-', f'{name}: output is not a PDF: {filename}'
            pdfs_checked += 1
    return dict(status='verified', notebook=name, code_cells=len(code_cells),
                plotting_cells=len(plots), rendered_figures=rendered, embedded_images=embedded,
                verified_input_files=len(inputs), verified_loader_sources=len(loaders),
                preserved_numerical_cells=numerical_cells, pdf_headers_checked=pdfs_checked,
                plotting_controls='User editable; DPI, FIGSIZE and label values are not restricted',
                editable_metadata_sha256=hashlib.sha256(
                    json.dumps(meta, sort_keys=True).encode()).hexdigest(),
                sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', choices=NOTEBOOKS, action='append')
    parser.add_argument('--allow-unexecuted', action='store_true')
    parser.add_argument('--check-pdfs', action='store_true',
                        help='Also require the optional local PDF exports; not needed for a fresh checkout.')
    args = parser.parse_args()
    result = {name: verify_notebook(ROOT, name, not args.allow_unexecuted, args.check_pdfs)
              for name in args.notebook or NOTEBOOKS}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
