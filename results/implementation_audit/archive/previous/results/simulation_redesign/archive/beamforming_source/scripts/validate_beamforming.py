"""Run the general theory notebook and all regression checks in isolation."""
from pathlib import Path
import hashlib,json,os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.beamforming_validation import beamforming_hash,science_sources,sha


def main():
    out=ROOT/'results/beamforming_unified/validation';out.mkdir(parents=True,exist_ok=True)
    original=ROOT/'notebooks/00_theory.ipynb';saved=original.read_bytes()
    fingerprint=beamforming_hash(ROOT);nb=nbformat.read(original,as_version=4)
    assignment="OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)"
    cells=[c for c in nb.cells if c.cell_type=='code' and assignment in c.source]
    if len(cells)!=1:raise RuntimeError('Review the general-notebook output redirection')
    cells[0].source=cells[0].source.replace(assignment,
        f'OUT=Path({str(out)!r}); OUT.mkdir(parents=True,exist_ok=True)')
    for c in nb.cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    certificate=out/'validation.json';certificate.unlink(missing_ok=True)
    try:
        NotebookClient(nb,timeout=3600,kernel_name='python3',
            resources={'metadata':{'path':str(ROOT)}}).execute()
        if original.read_bytes()!=saved or beamforming_hash(ROOT)!=fingerprint:
            raise RuntimeError('Numerical sources changed during validation')
        gate=json.loads(certificate.read_text())
        gate.update(beamforming_hash=fingerprint,
            sources={name:sha(ROOT/name) for name in science_sources(ROOT)},
            original_notebook_sha256=hashlib.sha256(saved).hexdigest(),
            scope='Complete general notebook and regression suite; empirical and numerical study flags have separate scopes')
        certificate.write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate,indent=2))
    except BaseException:
        certificate.unlink(missing_ok=True);raise
    finally:nbformat.write(nb,out/'validation.ipynb')


if __name__=='__main__':main()
