"""Append editable saved-study figures without replacing existing notebook cells."""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NOTEBOOKS = {'beamforming': '01_array.ipynb', 'contact': '03_contact.ipynb'}
FAMILIES = ('tracking', 'theory', 'comparisons', 'adaptation', 'calibration', 'accuracy')
ORDER = ('tracking_error', 'tracking_bound', 'posterior_curvature', 'block_updates',
         'additional_comparisons', 'task_performance', 'posterior_anisotropy',
         'variance_estimation', 'noise_adaptation', 'predictive_calibration',
         'predictive_scores', 'model_mismatch', 'computation_time',
         'numerical_accuracy', 'paper_summary')

SETUP = r'''from pathlib import Path
import os
import sys
ROOT = Path.cwd() if (Path.cwd() / 'lib').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
LOCAL_TEX = Path('/workspace/.local/tex')
if (LOCAL_TEX / 'usr/bin/dvipng').exists():
    os.environ['PATH'] = str(LOCAL_TEX / 'usr/bin') + os.pathsep + os.environ['PATH']
    os.environ.setdefault('TEXMFHOME', str(LOCAL_TEX / 'usr/share/texmf'))

# These readers verify saved numerical evidence; no experiment is rerun.
from lib.notebook_supplementary_data import (
    load_supplementary_evidence, load_constructed_geometry,
    _normalize, _primary, _column, _finite_column, _mean_interval,
    _episode_losses, _saved, _optional, _reference_frame,
    bingham_sphere_mesh, real_bingham_log_density, display_sphere_geometry,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython

EXAMPLE = __EXAMPLE__
SUPPLEMENTARY_EVIDENCE = load_supplementary_evidence(ROOT, EXAMPLE)
EDITABLE_OUTPUT = ROOT / 'results/notebook_exports' / __STEM__
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
    get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': None}")
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
    'font.family': 'serif', 'font.size': 25,
    'axes.labelsize': 25, 'axes.titlesize': 25,
    'legend.fontsize': 22, 'xtick.labelsize': 23, 'ytick.labelsize': 23,
    'lines.linewidth': 1.6, 'lines.markersize': 4,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': None,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.max_open_warning': 0,
})
print('Supplementary figures use verified saved data:', EXAMPLE)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_hash(cell):
    return hashlib.sha256(json.dumps(cell, sort_keys=True).encode()).hexdigest()


def figure_count(cells):
    return sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'plt'
        and node.func.attr in ('figure', 'subplots')
        for cell in cells if cell['cell_type'] == 'code'
        for node in ast.walk(ast.parse(''.join(cell['source']))))


def make_cell(kind, source, identifier, metadata=None):
    cell = dict(cell_type=kind, id=identifier, metadata=metadata or {},
                source=source.splitlines(keepends=True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    return cell


def catalogue(root, example):
    manifest = json.loads((root/'results'/example/'figure_manifest.json').read_text())
    return {(Path(name).stem, int(page['page']))
        for name, entry in manifest['figures'].items() if not name.startswith('supp_')
        for page in entry['pages']}


def append(example, root=ROOT, refresh_added=False):
    root = Path(root).resolve()
    path = root/'notebooks'/NOTEBOOKS[example]
    notebook = json.loads(path.read_text())
    previous = notebook.get('metadata', {}).get('supplementary_appendix')
    if previous and not refresh_added:
        print('Kept already appended cells:', path.name)
        return
    prefix = (notebook['cells'][:previous['original_cell_count']]
              if previous else notebook['cells'][:])
    original_hashes = [cell_hash(cell) for cell in prefix]
    existing_pages = {(cell.get('metadata', {}).get('figure_id'),
                       int(cell.get('metadata', {}).get('source_page', 1)))
                      for cell in prefix
                      if cell.get('metadata', {}).get('editable_plotting_role') == 'plot'
                      and cell.get('metadata', {}).get('source_catalogue_page', True)}
    records = [record for family in FAMILIES
        for record in import_module('scripts.notebook_supp_'+family).build_cells(example)]
    missing = catalogue(root, example) - existing_pages
    supplied = [(record['stem'], record['page']) for record in records
                if record['stem'] != 'tracking_error']
    assert len(supplied) == len(set(supplied)), 'Duplicated supplementary source page'
    assert set(supplied) == missing, (example, sorted(missing-set(supplied)),
                                      sorted(set(supplied)-missing))
    assert sum(record['stem'] == 'tracking_error' for record in records) == 1
    records.sort(key=lambda record: (ORDER.index(record['stem']), record['page']))
    additions = [make_cell('markdown',
        '## Supplementary figures\n\n'
        'The following cells expose the remaining saved figure pages and application '
        'tracking errors. Each figure uses an editable `(15, 5)` canvas, 300 dpi and '
        'LaTeX. Saving writes the local PDF and its supplementary manuscript copy. '
        'These cells read the saved Monte Carlo simulations and numerical checks; they do not '
        'rerun inference, simulation or benchmark selection.\n',
        'supplementary-heading'),
        make_cell('code', SETUP.replace('__EXAMPLE__', repr(example)).replace(
            '__STEM__', repr(Path(NOTEBOOKS[example]).stem)),
            'supplementary-setup', {'supplementary_setup': True})]
    entries = []
    for record in records:
        stem, page = record['stem'], record['page']
        identifier = f'supp-{stem.replace("_", "-")}-{page}'
        additions.append(make_cell('markdown', '### '+record['title']+'\n\n'
            +record['note']+'\n', identifier+'-note'))
        metadata = dict(editable_plotting_role='plot', figure_id=stem,
            source_page=page, supplementary_figure=True,
            source_catalogue_page=stem != 'tracking_error')
        additions.append(make_cell('code', record['source'], identifier, metadata))
        entries.append(dict(figure_id=stem, source_page=page,
            canonical_page=stem != 'tracking_error', cell_id=identifier))
    notebook['cells'] = prefix + additions
    assert notebook['cells'][:len(prefix)] == prefix
    meta = notebook['metadata']['editable_plotting']
    meta['figure_count'] = sum(cell.get('metadata', {}).get('editable_plotting_role') == 'plot'
                               for cell in notebook['cells'])
    meta['rendered_figure_count'] = figure_count(notebook['cells'])
    meta.setdefault('loader_sources', {})['lib/notebook_supplementary_data.py'] = sha(
        root/'lib/notebook_supplementary_data.py')
    notebook['metadata']['supplementary_appendix'] = dict(schema=1,
        original_cell_count=len(prefix), original_cell_hashes=original_hashes,
        appended_cells=len(additions), appended_figures=len(records),
        catalogue_pages_added=len(supplied), catalogue_pages_total=len(catalogue(root, example)),
        original_catalogue_pages=len(existing_pages), figures=entries,
        source_figure_manifest_sha256=sha(root/'results'/example/'figure_manifest.json'),
        generation_sources={f'scripts/notebook_supp_{family}.py': sha(
            root/f'scripts/notebook_supp_{family}.py') for family in FAMILIES},
        prefix_scope='Original cells preserved including source, outputs, execution counts and metadata. The stored hashes document this append operation; they do not lock future user styling edits.')
    notebook['metadata']['supplementary_appendix']['generation_sources']['scripts/notebook_publication_style.py'] = sha(root/'scripts/notebook_publication_style.py')
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')
    saved = json.loads(path.read_text())
    assert saved['cells'][:len(prefix)] == prefix
    print(json.dumps(dict(notebook=path.name, original_cells=len(prefix),
        appended_figures=len(records), canonical_pages_added=len(supplied),
        rendered_figures=meta['rendered_figure_count'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=NOTEBOOKS, action='append')
    parser.add_argument('--refresh-added', action='store_true',
        help='Explicitly reset only this script\'s appended cells; preserve the original prefix.')
    args = parser.parse_args()
    for example in args.example or NOTEBOOKS:
        append(example, refresh_added=args.refresh_added)


if __name__ == '__main__':
    main()
