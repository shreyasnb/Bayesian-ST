"""Execute selected editable notebooks in clean kernels without rebuilding them."""
import argparse
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_editable_notebooks import NOTEBOOKS, verify_notebook
from lib.final_validation import science_hash, require_gate


def execute(name):
    path = ROOT/'notebooks'/name
    original = path.read_bytes()
    fingerprint = science_hash(ROOT)
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=3600, kernel_name='python3',
        resources={'metadata':{'path':str(ROOT)}}).execute()
    if path.read_bytes()!=original:
        raise RuntimeError('Notebook was edited during execution; preserving the newer source: '+name)
    if science_hash(ROOT)!=fingerprint:
        raise RuntimeError('Numerical implementation changed during execution')
    temporary = path.with_suffix('.executed.tmp')
    nbformat.write(notebook, temporary)
    temporary.replace(path)
    result = verify_notebook(ROOT,name,check_pdfs=True)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', choices=NOTEBOOKS, action='append')
    args = parser.parse_args()
    require_gate(ROOT)
    for name in args.notebook or NOTEBOOKS:
        execute(name)


if __name__=='__main__':
    main()
