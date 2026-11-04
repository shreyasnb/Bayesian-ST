"""Audit the contact notebook and preserve every pre-pilot artifact and library."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.contact_validation import require_contact_gate
from lib.contact_reporting import load_contact
BASE='bef07fe301d03ecf4bd714e0e15999514c24a2a2'
OUT=ROOT/'results/contact_pilot'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate=require_contact_gate(ROOT);load_contact(ROOT)
    preserved={}
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--',
        'lib','benchmarks','tests','notebooks','results','docs','scripts'],cwd=ROOT).decode().splitlines()
    for name in names:
        original=subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=ROOT)
        digest=hashlib.sha256(original).hexdigest()
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Pre-pilot source/artifact changed: {name}')
        preserved[name]=digest
    numerical=json.loads((OUT/'numerics/checks.json').read_text())
    for name,digest in numerical['sources'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Stale numerical reference source: {name}')
    for name,digest in numerical['tables'].items():assert sha(OUT/'numerics'/name)==digest,name
    mechanics=json.loads((OUT/'numerics/mechanics_checks.json').read_text())
    for name,digest in mechanics['sources'].items():assert sha(ROOT/name)==digest,name
    assert sha(OUT/'numerics/mechanics_checks.csv')==mechanics['table_sha256']
    assert sha(OUT/'config.json')==mechanics['config_sha256']
    figures=json.loads((OUT/'figures/manifest.json').read_text())
    for name,digest in figures['sources'].items():assert sha(ROOT/name)==digest,name
    for name,digest in figures['figures'].items():assert sha(OUT/'figures'/name)==digest,name
    assert figures['dpi']==200 and figures['text_usetex']
    assert figures['metadata_sha256']==sha(OUT/'config.json')
    assert figures['core_hash']==gate['core_hash'] and figures['contact_hash']==gate['contact_hash']
    notebooks={}
    for path in [ROOT/'notebooks/03_contact.ipynb',OUT/'validation/validation.ipynb']:
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        code=[c for c in notebook.cells if c.cell_type=='code']
        assert code and all(c.execution_count is not None for c in code),path
        assert not any(o.output_type=='error' for c in code for o in c.outputs),path
        notebooks[str(path.relative_to(ROOT))]=dict(executed_cells=len(code),errors=0)
    files={}
    for pattern in ['lib/contact*.py','scripts/*contact*.py','tests/test_contact*.py',
                    'docs/contact*.md','notebooks/03_contact.ipynb']:
        for path in ROOT.glob(pattern):files[str(path.relative_to(ROOT))]=sha(path)
    for path in OUT.rglob('*'):
        if (not path.is_file() or 'checkpoints' in path.parts or path==OUT/'manifest.json'
                or path.name=='pytest.xml'):continue
        if path.suffix in ['.json','.csv','.md','.png','.pdf','.ipynb','.txt']:
            files[str(path.relative_to(ROOT))]=sha(path)
    for name in ['README.md','.gitignore']:files[name]=sha(ROOT/name)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),'unknown')
    record=dict(baseline_commit=BASE,core_hash=gate['core_hash'],contact_hash=gate['contact_hash'],
        guide_sha256=sha(ROOT/'docs/contact_pilot_spec.md'),preserved_pre_pilot=preserved,
        existing_numerical_core_modified=False,files=files,notebooks=notebooks,
        hardware=dict(platform=platform.platform(),cpu_model=cpu),
        timing_scope='Cloud implementation measurements at stated reporting frequencies; not a real-time control guarantee')
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2))
    print('Preserved',len(preserved),'pre-pilot files; verified',len(files),'current files;',notebooks)


if __name__=='__main__':main()
