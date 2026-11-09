"""Execute the full general notebook and contact tests with isolated outputs."""
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
from lib.contact_validation import contact_hash


def main():
    out=ROOT/'results/contact_pilot/validation';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'notebooks/00_theory.ipynb';original=source.read_bytes()
    signature=contact_hash(ROOT);nb=nbformat.read(source,as_version=4)
    assignment="OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)"
    cells=[c for c in nb.cells if c.cell_type=='code' and assignment in c.source]
    if len(cells)!=1:raise RuntimeError('General notebook setup changed; review the output redirection')
    cells[0].source=cells[0].source.replace(assignment,
        f'OUT=Path({str(out)!r}); OUT.mkdir(parents=True,exist_ok=True)')
    for c in nb.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    gate_path=out/'validation.json';gate_path.unlink(missing_ok=True)
    try:
        NotebookClient(nb,timeout=3600,kernel_name='python3',
                       resources={'metadata':{'path':str(ROOT)}}).execute()
        if source.read_bytes()!=original or contact_hash(ROOT)!=signature:
            raise RuntimeError('Validation sources changed during execution; rerun')
        gate=json.loads(gate_path.read_text())
        gate.update(contact_hash=signature,original_notebook_sha256=hashlib.sha256(original).hexdigest(),
                    execution='Full general notebook and complete regression suite, isolated contact outputs',
                    scope='Finite model, mechanics, deterministic-posterior and numerical checks; empirical superiority and coverage remain experimental outcomes')
        gate_path.write_text(json.dumps(gate,indent=2));print(json.dumps(gate,indent=2))
    except BaseException:
        gate_path.unlink(missing_ok=True)
        raise
    finally:
        nbformat.write(nb,out/'validation.ipynb')


if __name__=='__main__':main()
