"""Execute the existing general notebook with isolated extension artifacts.

The original notebook and results stay byte-for-byte unchanged. Only its OUT
assignment is redirected; every regression, stochastic check, and final source
hash assertion executes as written. A failure removes the new certificate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/dev_prediction')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'notebooks/00_theory.ipynb'
    original=source.read_bytes();notebook=nbformat.read(source,as_version=4)
    target="OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)"
    matches=[c for c in notebook.cells if c.cell_type=='code' and target in c.source]
    if len(matches)!=1:raise RuntimeError('General notebook OUT assignment changed; review the runner')
    matches[0].source=matches[0].source.replace(target,
        f'OUT=Path({str(out)!r}); OUT.mkdir(parents=True,exist_ok=True)')
    for cell in notebook.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    destination=out/'validation.ipynb'
    (out/'validation.json').unlink(missing_ok=True)
    try:
        NotebookClient(notebook,timeout=3600,kernel_name='python3',
                       resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook,destination)
    assert source.read_bytes()==original,'Original notebook changed during validation'
    gate=json.loads((out/'validation.json').read_text())
    gate.update(original_notebook_sha256=hashlib.sha256(original).hexdigest(),
                validation_notebook=str(destination.relative_to(ROOT)) if destination.is_relative_to(ROOT) else str(destination),
                execution='Unchanged general-notebook checks with isolated OUT directory')
    (out/'validation.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps(gate,indent=2))


if __name__=='__main__':
    main()
