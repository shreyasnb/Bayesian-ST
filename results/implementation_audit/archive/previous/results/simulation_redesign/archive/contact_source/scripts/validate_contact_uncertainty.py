"""Run all theory/regression checks with isolated uncertainty-study artifacts."""
from pathlib import Path
import hashlib
import json
import os
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.contact_uncertainty_validation import uncertainty_hash


def main():
    out=ROOT/'results/contact_uncertainty/validation';out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'notebooks/00_theory.ipynb';original=path.read_bytes()
    source_hash=uncertainty_hash(ROOT);notebook=nbformat.read(path,as_version=4)
    assignment="OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)"
    setup=[c for c in notebook.cells if c.cell_type=='code' and assignment in c.source]
    if len(setup)!=1:raise RuntimeError('General notebook setup changed; review isolated-output routing')
    setup[0].source=setup[0].source.replace(assignment,
        f'OUT=Path({str(out)!r}); OUT.mkdir(parents=True,exist_ok=True)')
    for c in notebook.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    gate_path=out/'validation.json';gate_path.unlink(missing_ok=True)
    try:
        NotebookClient(notebook,timeout=3600,kernel_name='python3',
                       resources={'metadata':{'path':str(ROOT)}}).execute()
        if uncertainty_hash(ROOT)!=source_hash or path.read_bytes()!=original:
            raise RuntimeError('Validation sources changed while running; rerun')
        gate=json.loads(gate_path.read_text())
        gate.update(uncertainty_hash=source_hash,original_notebook_sha256=hashlib.sha256(original).hexdigest(),
            execution='Complete general notebook and regression suite, isolated uncertainty outputs',
            scope='Finite software/numerical checks; reference and empirical outcome flags have separate scopes')
        gate_path.write_text(json.dumps(gate,indent=2));print(json.dumps(gate,indent=2))
    except BaseException:
        gate_path.unlink(missing_ok=True);raise
    finally:nbformat.write(notebook,out/'validation.ipynb')


if __name__=='__main__':main()
