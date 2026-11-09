"""Execute the two saved-evidence walkthroughs in fresh Python kernels."""
from pathlib import Path
import argparse
import json
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR', '/tmp/bayesian-st-ipython')

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.final_validation import require_gate, verify_history, science_hash, sha
from scripts.make_final_notebooks import verify_figures


def execute(example):
    name = '01_array.ipynb' if example == 'beamforming' else '03_contact.ipynb'
    path = ROOT / 'notebooks' / name
    verify_figures(ROOT, example)
    manifest = ROOT / 'results' / example / 'figure_manifest.json'
    manifest_hash = sha(manifest)
    fingerprint = science_hash(ROOT)
    original = path.read_bytes()
    nb = nbformat.read(path, as_version=4)
    assert nb.metadata.final_study.figure_manifest_sha256 == manifest_hash
    assert nb.metadata.final_study.producer_sha256 == sha(ROOT / 'scripts/make_final_notebooks.py')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    NotebookClient(nb, timeout=900, kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}}).execute()
    if path.read_bytes() != original:
        raise RuntimeError('Notebook changed during execution; preserving the new file: ' + name)
    if sha(manifest) != manifest_hash or science_hash(ROOT) != fingerprint:
        raise RuntimeError('Scientific or figure evidence changed during execution')
    verify_figures(ROOT, example)
    target = path.with_suffix('.executed.tmp')
    nbformat.write(nb, target)
    target.replace(path)
    print(json.dumps(dict(status='executed', notebook=name,
        code_cells=sum(cell.cell_type == 'code' for cell in nb.cells),
        embedded_images=sum('image/png' in output.get('data', {})
            for cell in nb.cells if cell.cell_type == 'code'
            for output in cell.get('outputs', [])), sha256=sha(path))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['beamforming', 'contact', 'both'], default='both')
    args = parser.parse_args()
    require_gate(ROOT)
    verify_history(ROOT)
    for example in (('beamforming', 'contact') if args.example == 'both' else (args.example,)):
        execute(example)


if __name__ == '__main__':
    main()
