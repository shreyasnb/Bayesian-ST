"""Inventory retained results and regenerable worker checkpoints without hiding flags."""
from pathlib import Path
import hashlib,json,subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/beamforming_unified'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def excluded(path):return 'checkpoints' in path.parts or path.name=='pytest.xml'

def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='test':raise RuntimeError('The unified handoff must stay on test')
    retained={};omitted={}
    for path in sorted(BASE.rglob('*')):
        if not path.is_file() or path.name=='artifact_manifest.json':continue
        entry=dict(sha256=sha(path),bytes=path.stat().st_size)
        target=omitted if excluded(path.relative_to(BASE)) else retained
        target[str(path.relative_to(ROOT))]=entry
    sources=sorted(set(list((ROOT/'lib').glob('beamforming_*.py'))+
        list((ROOT/'scripts').glob('*beamforming*.py'))+
        list((ROOT/'tests').glob('test_beamforming*.py'))+
        list((ROOT/'docs').glob('beamforming*.md'))+
        [ROOT/'README.md',ROOT/'.gitignore',ROOT/'pyproject.toml',ROOT/'notebooks/01_array.ipynb']))
    meta=dict(branch=branch,base_commit='3b47f8d7765d125fff6a1bbe8c8e9d43a8627517',
        created_utc=datetime.now(timezone.utc).isoformat(),
        retained=retained,excluded=omitted,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        regeneration=dict(checkpoints='Re-run the same profile/component command in results/beamforming_unified/README.md; seed/task/source hashes reconstruct every worker result.',
            full_gibbs_states='Full state chains are intentionally not serialized; regenerate each fit using the declared data/chain seed and inference configuration.',
            general_validation='python scripts/validate_beamforming.py',
            figures='python scripts/plot_beamforming_unified.py --profile paper',
            notebook='python scripts/make_beamforming_notebook.py && python scripts/execute_beamforming_notebook.py'),
        interpretation='Excluded checkpoints retain no unique favorable/unfavorable scientific summary: all trajectory metrics, numerical/mixing flags and configurations are committed. All retained files are content-addressed; this inventory excludes itself.')
    (BASE/'artifact_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Manifest:',len(retained),'retained files;',len(omitted),'regenerable worker files;',
        sum(x['bytes'] for x in retained.values()),'retained bytes')

if __name__=='__main__':main()
