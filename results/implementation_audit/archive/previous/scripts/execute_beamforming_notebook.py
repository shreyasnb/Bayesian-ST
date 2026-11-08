"""Execute the verified paper narrative without changing scientific outputs."""
from pathlib import Path
import os,hashlib,json
from datetime import datetime,timezone
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/01_array.ipynb'
notebook=nbformat.read(path,as_version=4)
for cell in notebook.cells:
    if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
try:
    NotebookClient(notebook,timeout=3600,kernel_name='python3',
        resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    nbformat.write(notebook,path)
errors=[out for cell in notebook.cells if cell.cell_type=='code'
        for out in cell.outputs if out.output_type=='error']
if errors:raise RuntimeError('Notebook has error outputs')
base=ROOT/'results/beamforming_unified'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
metadata=dict(status='passed',executed_utc=datetime.now(timezone.utc).isoformat(),
    notebook='notebooks/01_array.ipynb',notebook_sha256=sha(path),
    code_cells=sum(c.cell_type=='code' for c in notebook.cells),error_outputs=len(errors),
    executed_cells=sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells),
    sources={name:sha(ROOT/name) for name in ['scripts/make_beamforming_notebook.py',
        'scripts/execute_beamforming_notebook.py','lib/beamforming_visuals.py',
        'scripts/plot_beamforming_unified.py']},
    figure_manifest_sha256=sha(base/'paper/figures/manifest.json'),
    validation_sha256=sha(base/'validation/validation.json'),
    findings_sha256=sha(base/'findings.md'),
    main_summary_sha256=sha(base/'main_summary.md'),
    summary_manifest_sha256=sha(base/'summary_manifest.json'),
    scope='The notebook checks source/table/figure hashes, displays the frozen paper figures and verifies selected identities. Scientific simulation is run by the component commands, separately from rendering.')
(base/'execution.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Executed array notebook:',sum(c.cell_type=='code' for c in notebook.cells),'code cells; no errors.')
