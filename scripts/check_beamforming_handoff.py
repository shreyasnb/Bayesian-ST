"""Read-only final verification of the isolated beamforming handoff."""
from pathlib import Path
import hashlib,json,subprocess,sys
import nbformat
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.beamforming_validation import require_beamforming_gate,verify_component
from lib.beamforming_history import verify_historical
BASE=ROOT/'results/beamforming_unified'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def verify_map(mapping,relative=ROOT):
    for name,expected in mapping.items():
        if isinstance(expected,dict):expected=expected['sha256']
        if sha(relative/name)!=expected:raise RuntimeError('Changed content: '+name)

def main():
    if git('branch','--show-current')!='test':raise RuntimeError('Handoff is not on test')
    baseline=json.loads((BASE/'branch_baseline.json').read_text())
    for ref in ['refs/heads/main','refs/heads/dev']:
        if git('rev-parse',ref)!=baseline[ref]:raise RuntimeError('Protected local branch moved: '+ref)
    original=['lib/models.py','lib/filter.py','lib/geometry.py','lib/theory.py','lib/em.py',
        'lib/moments.py','lib/spectral_series.py','lib/process_prediction.py',
        'notebooks/00_theory.ipynb','notebooks/02_field.ipynb','notebooks/03_contact.ipynb']
    if git('diff','--name-only',baseline['refs/heads/dev'],'--',*original):
        raise RuntimeError('Unintended changes to existing machinery/notebooks')
    gate=require_beamforming_gate(ROOT)
    components={};figures={}
    for profile in ['quick','paper']:
        for component in ['information','numerics','decisions','tracking','process','inference']:
            verify_component(ROOT,profile,component)
            components[profile+'/'+component]='source and saved tables verified'
        folder=BASE/profile/'figures';m=json.loads((folder/'manifest.json').read_text())
        if m['figure_dpi']!=200 or m['savefig_dpi']!=200 or m['text_usetex'] is not True:
            raise RuntimeError('Wrong figure protocol')
        verify_map(m['sources']);verify_map(m['input_manifests']);verify_map(m['files'],folder)
        pngs=0
        for name in m['files']:
            if name.endswith('.png'):
                with Image.open(folder/name) as img:
                    if tuple(round(d) for d in img.info['dpi'])!=(200,200):raise RuntimeError('Wrong PNG dpi')
                pngs+=1
            elif name.endswith('.pdf'):
                if not (folder/name).read_bytes().startswith(b'%PDF'):raise RuntimeError('Invalid PDF')
        if pngs!=11 or len(m['files'])!=22:raise RuntimeError('Expected eight main and three supplementary figures')
        figures[profile]=pngs
    history=verify_historical(ROOT)
    audit_folder=BASE/'development'
    audit=json.loads((audit_folder/'independent_audit_manifest.json').read_text())
    verify_map(audit['source_hashes'])
    if sha(audit_folder/audit['raw_output'])!=audit['raw_output_sha256']:
        raise RuntimeError('Independent audit output changed')
    if sha(audit_folder/'independent_audit_README.md')!=audit['readme_sha256']:
        raise RuntimeError('Independent audit interpretation changed')
    summary=json.loads((BASE/'summary_manifest.json').read_text())
    if summary['profile']!='paper' or not summary['paper_evidence'] or summary['status']!='completed':
        raise RuntimeError('Findings are not verified paper evidence')
    for name in ['verified_input_hashes','source_hashes','outputs']:verify_map(summary[name])
    generator=summary['generator']
    if sha(ROOT/generator['path'])!=generator['sha256']:raise RuntimeError('Stale findings generator')
    execution=json.loads((BASE/'execution.json').read_text())
    if execution['status']!='passed' or execution['error_outputs']!=0:raise RuntimeError('Notebook failed')
    if sha(ROOT/execution['notebook'])!=execution['notebook_sha256']:raise RuntimeError('Executed notebook changed')
    verify_map(execution['sources'])
    for key,path in [('figure_manifest_sha256',BASE/'paper/figures/manifest.json'),
        ('validation_sha256',BASE/'validation/validation.json'),('findings_sha256',BASE/'findings.md'),
        ('main_summary_sha256',BASE/'main_summary.md'),('summary_manifest_sha256',BASE/'summary_manifest.json')]:
        if sha(path)!=execution[key]:raise RuntimeError('Notebook input changed: '+key)
    n=nbformat.read(ROOT/execution['notebook'],as_version=4)
    codes=[c for c in n.cells if c.cell_type=='code']
    if any(c.execution_count is None or any(o.output_type=='error' for o in c.outputs) for c in codes):
        raise RuntimeError('Unexecuted or failed notebook cell')
    inventory=json.loads((BASE/'artifact_manifest.json').read_text())
    verify_map(inventory['retained']);verify_map(inventory['sources'])
    # Excluded hashes document regeneration; a clean clone need not contain checkpoints.
    for name,entry in inventory['excluded'].items():
        path=ROOT/name
        if path.exists() and sha(path)!=entry['sha256']:raise RuntimeError('Changed checkpoint: '+name)
    print(json.dumps(dict(status='passed',branch='test',general_tests=gate['tests'],
        components=len(components),figures=figures,executed_code_cells=len(codes),
        historical_artifacts=len(history['artifacts']),retained_artifacts=len(inventory['retained']),
        independent_audit_checks=audit['summary']['checks'],
        protected_local_branches='main/dev unchanged',
        scope='Content/source/protocol/execution verification; retained empirical and numerical limitations are not relabeled successes'),indent=2))

if __name__=='__main__':main()
