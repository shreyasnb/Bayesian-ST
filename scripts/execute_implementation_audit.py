"""Execute both application narratives and the guide supplement from scratch."""
from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import re
import base64
import hashlib
import subprocess
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR', '/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.implementation_validation import require_gate, sha, verify_preservation
from scripts.summarize_implementation_audit import verify_report, verify_figures

BASE = ROOT / 'results/implementation_audit'


def notebook_source(nb):
    value = [(cell.cell_type, cell.source) for cell in nb.cells]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def inspect(path, figure_folder):
    nb = nbformat.read(path, as_version=4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    names = [name for c in cells for name in re.findall(r"show\('([^']+)'\)", c.source)]
    png = [out['data']['image/png'] for c in cells for out in c.outputs
           if out.output_type in ('display_data', 'execute_result')
           and 'image/png' in out.get('data', {})]
    errors = [out for c in cells for out in c.outputs if out.output_type == 'error']
    if errors or any(c.execution_count is None for c in cells) or len(png) != len(names):
        raise RuntimeError('Incomplete notebook execution: ' + str(path))
    figures = json.loads((figure_folder / 'manifest.json').read_text())
    for name, encoded in zip(names, png):
        if hashlib.sha256(base64.b64decode(encoded)).hexdigest() != figures['files'][name + '.png']:
            raise RuntimeError('Embedded figure does not match its verified export: ' + name)
    return dict(sha256=sha(path), source_hash=notebook_source(nb), code_cells=len(cells),
                error_outputs=len(errors), image_outputs=len(png), figure_names=names)


def main():
    gate = require_gate(ROOT)
    verify_preservation(ROOT)
    verify_report(ROOT)
    verify_figures(ROOT, 'paper')
    inputs = [BASE / 'validation/validation.json', BASE / 'summary_manifest.json',
              BASE / 'paper/figures/manifest.json',
              ROOT / 'results/simulation_redesign/summary_manifest.json',
              ROOT / 'results/simulation_redesign/paper/figures/manifest.json']
    before = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    subprocess.run([sys.executable, str(ROOT / 'scripts/execute_geometry_notebooks.py')],
                   cwd=ROOT, check=True)
    path = ROOT / 'notebooks/04_implementation_audit.ipynb'
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    try:
        NotebookClient(nb, timeout=3600, kernel_name='python3',
            resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        nbformat.write(nb, path)
    reports = {str((ROOT / 'notebooks' / (name + '.ipynb')).relative_to(ROOT)):
        inspect(ROOT / 'notebooks' / (name + '.ipynb'),
                ROOT / 'results/simulation_redesign/paper/figures')
        for name in ('01_array', '03_contact')}
    reports[str(path.relative_to(ROOT))] = inspect(path, BASE / 'paper/figures')
    for name, expected in before.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Execution inputs changed: ' + name)
    require_gate(ROOT)
    sources = ['scripts/execute_implementation_audit.py',
        'scripts/execute_geometry_notebooks.py', 'scripts/make_geometry_notebooks.py',
        'scripts/make_implementation_audit_notebook.py', 'scripts/summarize_implementation_audit.py']
    meta = dict(status='passed', tests=gate['tests'],
        executed_utc=datetime.now(timezone.utc).isoformat(), notebooks=reports,
        input_hashes=before, sources={name: sha(ROOT / name) for name in sources},
        scope='Clean-kernel execution of the two applications and computational supplement. '
              'Every embedded PNG is byte-checked against the final source-verified figure export. '
              'No new inference or performance tuning occurs during notebook execution.')
    (BASE / 'execution.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('Passed clean-kernel execution and embedded-image verification:', ', '.join(reports))


if __name__ == '__main__':
    main()
