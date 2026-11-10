"""Exercise notebook-local dpi, figure-size and axis-label edits in fresh kernels.

The probes execute temporary mini-notebooks, never modify root notebooks or
published PDFs, and save their figures under /tmp by default. Each probe uses
one existing two-dimensional plotting cell. The theory probe reuses its saved
tracking CSV instead of rerunning validation, simulations or posterior fits.
PyMuPDF is needed only for the separate actual-PDF check; --pdf-python may
select an interpreter containing the optional presentation dependencies.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR', '/tmp/bayesian-st-ipython')

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('00_theory', '01_array', '02_field', '03_contact', '04_implementation_audit')
CUSTOM_DPI = 120
CUSTOM_SIZE = (8., 4.)
CUSTOM_LABEL = 'Custom axis label'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plot_cell(cell):
    if cell.cell_type != 'code':
        return False
    source = cell.source
    return 'plt.figure(' in source or 'plt.subplots(' in source


def _snapshot(baseline):
    if baseline:
        mapping = json.loads(Path(baseline).read_text())
        if 'files' in mapping and isinstance(mapping['files'], dict):
            mapping = mapping['files']
    else:
        names = subprocess.check_output(['git', 'ls-files', 'results'], cwd=ROOT,
                                        text=True).splitlines()
        mapping = {name: sha(ROOT/name) for name in names}
    for name, expected in mapping.items():
        digest = expected.get('sha256') if isinstance(expected, dict) else expected
        if sha(ROOT/name) != digest:
            raise AssertionError('Frozen result changed: '+name)
    return mapping


def _unchanged(mapping):
    for name, expected in mapping.items():
        digest = expected.get('sha256') if isinstance(expected, dict) else expected
        if sha(ROOT/name) != digest:
            raise AssertionError('Frozen result changed during customization probes: '+name)


def _selected_cells(name, nb):
    plots = [(index, cell) for index, cell in enumerate(nb.cells) if _plot_cell(cell)]
    if not plots:
        raise ValueError('Notebook has no visible Matplotlib cells: '+name)
    if name == '00_theory':
        # The original theory setup removes an old certificate and its numerical
        # cells intentionally run tests/studies. None belongs in a render probe.
        selected = next((pair for pair in plots if 'track.copy()' in pair[1].source), None)
        if selected is None:
            raise ValueError('The theory probe requires the saved tracking plot')
        table = ROOT/'results/notebook_runs/00_theory/theory_tracking.csv'
        if not table.is_file():
            raise FileNotFoundError('Execute the theory notebook once before probing: '+str(table))
        setup = '''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
ROOT = Path(%r)
sys.path.insert(0, str(ROOT))
from lib.plotting import plt
plt.rcParams.update({'text.usetex': True, 'savefig.bbox': None})
track = pd.read_csv(Path(%r))
''' % (str(ROOT), str(table))
        return selected[0], [nbformat.v4.new_code_cell(setup)], selected[1], [str(table.relative_to(ROOT))]
    if name in ('01_array', '03_contact'):
        selected = next((pair for pair in plots if "'information'" in pair[1].source), None)
        if selected is None:
            raise ValueError('The application probe requires its 2D information figure')
        # These notebooks' first setup cell loads/validates frozen inputs only.
        first = next(cell for cell in nb.cells if cell.cell_type == 'code')
        return selected[0], [nbformat.v4.new_code_cell(first.source)], selected[1], []
    selected = plots[0]
    prefix = [nbformat.v4.new_code_cell(cell.source) for cell in nb.cells[:selected[0]]
              if cell.cell_type == 'code' and cell.source.strip() and not _plot_cell(cell)]
    return selected[0], prefix, selected[1], []


class _LocalControls(ast.NodeTransformer):
    def __init__(self):
        self.changed = set()

    def visit_Assign(self, node):
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if 'DPI' in names:
            node.value = ast.Constant(CUSTOM_DPI)
            self.changed.add('DPI')
        if 'FIGSIZE' in names:
            node.value = ast.Tuple([ast.Constant(value) for value in CUSTOM_SIZE], ast.Load())
            self.changed.add('FIGSIZE')
        return self.generic_visit(node)


def _customized(source, probe_pdf, custom_label):
    controls = _LocalControls()
    tree = controls.visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    if controls.changed != {'DPI', 'FIGSIZE'}:
        raise AssertionError('The plot must expose both local controls')
    source = ast.unparse(tree)
    if source.count('plt.show()') != 1:
        raise AssertionError('Expected exactly one visible plt.show()')
    inserted = '''assert abs(float(fig.dpi)-%r)<1e-12, 'The figure ignored local DPI'
assert np.allclose(fig.get_size_inches(), %r, rtol=0, atol=1e-12), 'The figure ignored local FIGSIZE'
fig.axes[0].set_xlabel(%r)
fig.savefig(Path(%r), dpi=DPI, bbox_inches=None)
print('Customization probe saved:', %r)
plt.show()''' % (CUSTOM_DPI, list(CUSTOM_SIZE), custom_label, str(probe_pdf), str(probe_pdf))
    return source.replace('plt.show()', inserted)


def _check_pdf(path, pdf_python, custom_label):
    program = r'''import json,sys
try:
 import pymupdf as fitz
except ImportError:
 import fitz
with fitz.open(sys.argv[1]) as doc:
 assert len(doc)==1, 'Expected one probe page'
 page=doc[0]
 width,height=page.mediabox.width,page.mediabox.height
 assert abs(width-576.)<.01 and abs(height-288.)<.01, (width,height)
 text=' '.join(page.get_text().split())
 assert sys.argv[2] in text, text
 print(json.dumps({'page_width_pt':width,'page_height_pt':height,'custom_label_present':True,'tool':'PyMuPDF '+fitz.VersionBind}))
'''
    result = subprocess.run([pdf_python, '-c', program, str(path), custom_label], cwd=ROOT,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Actual PDF customization check failed. Select a Python with '
                           'PyMuPDF using --pdf-python.\n'+result.stderr+result.stdout)
    return json.loads(result.stdout)


def probe(name, folder, pdf_python):
    path = ROOT/'notebooks'/(name+'.ipynb')
    original = sha(path)
    nb = nbformat.read(path, as_version=4)
    source_fingerprint = hashlib.sha256(json.dumps(
        [(cell.cell_type, cell.source) for cell in nb.cells],
        ensure_ascii=False).encode()).hexdigest()
    index, prefix, plotting, extra_inputs = _selected_cells(name, nb)
    output = folder/name
    output.mkdir(parents=True, exist_ok=True)
    pdf = output/'probe.pdf'
    # Unique text forces a fresh TeX label even when the previous probe's label
    # is cached. This catches missing font setup that cached figures can hide.
    custom_label = CUSTOM_LABEL+' '+name.split('_', 1)[0]
    reset = nbformat.v4.new_code_cell('from pathlib import Path\nimport numpy as np\n'
        +'EDITABLE_OUTPUT=Path('+repr(str(output))+')\n'
        +'EDITABLE_OUTPUT.mkdir(parents=True,exist_ok=True)\n')
    cells = [*prefix, reset, nbformat.v4.new_code_cell(_customized(plotting.source, pdf, custom_label))]
    mini = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python'}})
    NotebookClient(mini, timeout=240, kernel_name='python3',
                   resources={'metadata': {'path':str(ROOT)}}).execute()
    executed = output/'probe.ipynb'
    nbformat.write(mini, executed)
    after = nbformat.read(path, as_version=4)
    after_fingerprint = hashlib.sha256(json.dumps(
        [(cell.cell_type, cell.source) for cell in after.cells],
        ensure_ascii=False).encode()).hexdigest()
    if after_fingerprint != source_fingerprint:
        raise AssertionError('Original notebook cell sources changed during the probe: '+name)
    record = _check_pdf(pdf, pdf_python, custom_label)
    return dict(status='passed', notebook=str(path.relative_to(ROOT)),
                original_notebook_sha256=original,
                original_notebook_sources_sha256=source_fingerprint,
                original_notebook_sources_unchanged=True,
                original_notebook_bytes_unchanged=sha(path)==original,
                original_plot_cell=index,
                customized_dpi=CUSTOM_DPI, customized_figsize_in=list(CUSTOM_SIZE),
                customized_xlabel=custom_label, pdf=str(pdf), pdf_sha256=sha(pdf),
                temporary_executed_notebook=str(executed),
                actual_pdf=record, extra_saved_inputs=extra_inputs,
                computation='Existing saved-data setup, original lightweight pre-plot checks '
                    'and one plotting cell; the theory probe skips original setup, '
                    'pytest and numerical studies.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/tmp/editable-plot-probes'))
    parser.add_argument('--baseline', type=Path,
                        help='Optional pre-refactor frozen-result hash map to verify before and after.')
    parser.add_argument('--pdf-python', default=shutil.which('python') or sys.executable)
    parser.add_argument('--output', type=Path, help='Optional JSON report path (normally under /tmp).')
    args = parser.parse_args()
    mapping = _snapshot(args.baseline)
    records = []
    try:
        for name in NAMES:
            records.append(probe(name, args.directory.resolve(), args.pdf_python))
            _unchanged(mapping)
            print(json.dumps(dict(notebook=name, status='passed', dpi=CUSTOM_DPI,
                                  figsize_in=list(CUSTOM_SIZE))), file=sys.stderr, flush=True)
    finally:
        _unchanged(mapping)
    report = dict(status='passed', producer='scripts/probe_editable_plots.py',
        producer_sha256=sha(Path(__file__)), probes=records,
        frozen_results_checked=len(mapping), frozen_results_unchanged=True,
        scope='Actual rendering customization in fresh kernels; root notebook sources '
              'and canonical numerical/PDF results remain untouched.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
