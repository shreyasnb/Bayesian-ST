"""Execute appended figure cells while preserving every original notebook cell."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
from lib.final_validation import science_hash, require_gate
from scripts.check_editable_notebooks import verify_notebook


def execute(example, figures=None):
    filename = '01_array.ipynb' if example == 'beamforming' else '03_contact.ipynb'
    path = ROOT/'notebooks'/filename
    original = path.read_bytes()
    notebook = json.loads(original)
    start = notebook['metadata']['supplementary_appendix']['original_cell_count']
    prefix = notebook['cells'][:start]
    fingerprint = science_hash(ROOT)
    selected = [index for index in range(start, len(notebook['cells']))
        if not figures or notebook['cells'][index].get('metadata', {}).get('supplementary_setup')
        or notebook['cells'][index].get('metadata', {}).get('figure_id') in figures]
    added = nbformat.reads(json.dumps(dict(nbformat=4, nbformat_minor=5,
        cells=[notebook['cells'][index] for index in selected],
        metadata={'kernelspec': notebook['metadata']['kernelspec']})), as_version=4)
    NotebookClient(added, timeout=900, kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}}).execute()
    if path.read_bytes() != original:
        raise RuntimeError('Notebook changed during execution; preserving the newer file')
    if science_hash(ROOT) != fingerprint:
        raise RuntimeError('Scientific implementation changed during figure execution')
    executed = json.loads(nbformat.writes(added))['cells']
    for index, cell in zip(selected, executed):
        notebook['cells'][index] = cell
    temporary = path.with_suffix('.executed.tmp')
    temporary.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')
    assert json.loads(temporary.read_text())['cells'][:start] == prefix
    temporary.replace(path)
    result = verify_notebook(ROOT, filename, check_pdfs=True)
    result['preserved_original_cells'] = start
    result['executed_added_code_cells'] = sum(cell['cell_type']=='code' for cell in executed)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['beamforming', 'contact'], action='append')
    parser.add_argument('--figure', action='append',
        help='Execute only this appended figure family, plus its setup cell.')
    args = parser.parse_args()
    require_gate(ROOT)
    for example in args.example or ['beamforming', 'contact']:
        execute(example, figures=args.figure)


if __name__ == '__main__':
    main()
