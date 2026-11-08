"""Execute the two source-verified draft-5 narratives without rerunning science."""
from pathlib import Path
import os,sys,json,hashlib
from datetime import datetime,timezone
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.simulation_validation import require_gate,verify_component,verify_legacy,sha,gate_path
BASE=ROOT/'results/simulation_redesign'


def main():
    require_gate(ROOT)
    for name in ('geometry','beamforming','contact','scaling'):verify_component(ROOT,'paper',name)
    verify_legacy(ROOT)
    figure_path=BASE/'paper/figures/manifest.json';before=sha(figure_path)
    summary=BASE/'summary_manifest.json'
    if not summary.exists():raise RuntimeError('Generate the verified paper findings first')
    summ=json.loads(summary.read_text())
    if summ.get('profile')!='paper' or not summ.get('paper_evidence'):
        raise RuntimeError('Findings do not describe the paper profile')
    reports={}
    for name in ['01_array','03_contact']:
        path=ROOT/'notebooks'/f'{name}.ipynb';nb=nbformat.read(path,as_version=4)
        for c in nb.cells:
            if c.cell_type=='code':c.outputs=[];c.execution_count=None
        try:
            NotebookClient(nb,timeout=3600,kernel_name='python3',
                resources={'metadata':{'path':str(ROOT)}}).execute()
        finally:nbformat.write(nb,path)
        code=[c for c in nb.cells if c.cell_type=='code']
        errors=[o for c in code for o in c.outputs if o.output_type=='error']
        if errors or any(c.execution_count is None for c in code):raise RuntimeError('Incomplete notebook '+name)
        reports[str(path.relative_to(ROOT))]=dict(sha256=sha(path),code_cells=len(code),error_outputs=len(errors),
            image_outputs=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs))
        print(name,':',len(code),'code cells, no errors',flush=True)
    if sha(figure_path)!=before:raise RuntimeError('Figure inputs changed during notebook execution')
    names=['scripts/make_geometry_notebooks.py','scripts/execute_geometry_notebooks.py',
        'scripts/plot_geometry_studies.py','lib/simulation_story_visuals.py',
        'lib/distribution_geometry_visuals.py','lib/plotting.py']
    meta=dict(status='passed',executed_utc=datetime.now(timezone.utc).isoformat(),notebooks=reports,
        sources={name:sha(ROOT/name) for name in names},figure_manifest_sha256=before,
        validation_sha256=sha(gate_path(ROOT)),
        validation_path=str(gate_path(ROOT).relative_to(ROOT)),summary_manifest_sha256=sha(summary),
        scope='Executed narratives verify and display frozen paper results. Simulation and figure layout are separate commands; retained scientific limitations are not relabeled as successes.')
    (BASE/'execution.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
