"""Inventory committed evidence and regenerable checkpoints for the redesign."""
from pathlib import Path
import hashlib,json,subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/simulation_redesign'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='test':raise RuntimeError('The redesign handoff belongs on test')
    retained={};excluded={}
    for p in sorted(BASE.rglob('*')):
        if not p.is_file() or p.name=='artifact_manifest.json' or '__pycache__' in p.parts:continue
        target=excluded if 'checkpoints' in p.parts or p.name=='pytest.xml' else retained
        target[str(p.relative_to(ROOT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    names=['lib/simulation_validation.py','lib/array_geometry_study.py','lib/contact_geometry_study.py',
        'lib/anisotropy_study.py','lib/simulation_scaling.py','lib/simulation_story_visuals.py',
        'lib/distribution_geometry_visuals.py','notebooks/01_array.ipynb','notebooks/03_contact.ipynb',
        '.gitignore','README.md','pyproject.toml']
    names+=['scripts/'+p.name for p in (ROOT/'scripts').glob('*geometry*.py')]
    names+=['scripts/'+p.name for p in (ROOT/'scripts').glob('*simulation_redesign*.py')]
    names+=['scripts/run_anisotropy_study.py','scripts/run_simulation_scaling.py']
    names+=['docs/'+p.name for p in (ROOT/'docs').glob('simulation_redesign*.md')]
    names+=['tests/'+p.name for p in (ROOT/'tests').glob('test_*geometry*.py')]
    names+=['tests/test_anisotropy_study.py','tests/test_simulation_scaling.py']
    meta=dict(branch=branch,base_commit='2a8697c7e7fa27b3779f3abeaed32ca8b323a9ef',
        created_utc=datetime.now(timezone.utc).isoformat(),retained=retained,excluded=excluded,
        sources={name:sha(ROOT/name) for name in sorted(set(names))},
        regeneration='See results/simulation_redesign/README.md for all profiles. Use fresh --output directories to regenerate scientific worker records without replacing the verified published evidence. Draws are seeded; timing and timestamp fields naturally vary.',
        scope='All paired scientific metrics, configurations, numerical/local validity flags and figure outputs are retained. Only reconstructible worker checkpoints and raw pytest XML are omitted. This inventory excludes itself.')
    (BASE/'artifact_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(len(retained),'retained files;',len(excluded),'regenerable worker files')

if __name__=='__main__':main()
