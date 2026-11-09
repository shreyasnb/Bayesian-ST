"""Verify the extension gate and preserve historical artifacts before handoff."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.validation import require_gate

BASE='57f520cbe860b73e13d71bae5c31391e2a009482'
OUT=ROOT/'results/dev_prediction'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate=require_gate(ROOT,gate_path=OUT/'validation.json')
    historical=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,
                                        '--','results','notebooks'],cwd=ROOT).decode().splitlines()
    preserved={}
    for name in historical:
        old=subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=ROOT)
        digest=hashlib.sha256(old).hexdigest()
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Historical artifact changed: {name}')
        preserved[name]=digest
    files={}
    for directory in ['lib','tests','benchmarks','scripts','docs','results/dev_prediction']:
        for path in sorted((ROOT/directory).rglob('*')):
            if (not path.is_file() or any(part in path.parts for part in
                    ['__pycache__','checkpoints','.ipynb_checkpoints'])
                    or path==OUT/'manifest.json' or path.name=='pytest.xml'):continue
            if path.suffix in ['.py','.sh','.md','.json','.csv','.pdf','.png','.ipynb','.txt','.log']:
                files[str(path.relative_to(ROOT))]=sha(path)
    for name in ['README.md','pyproject.toml','requirements-lock.txt','.gitignore']:
        files[name]=sha(ROOT/name)
    reviewed_sources=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--',
        'lib','tests','benchmarks','scripts','docs','README.md','.gitignore'],cwd=ROOT).decode().splitlines()
    changes={}
    for name in reviewed_sources:
        if not (ROOT/name).is_file():raise RuntimeError(f'Reviewed source removed: {name}')
        before=hashlib.sha256(subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=ROOT)).hexdigest()
        after=sha(ROOT/name)
        if before!=after:changes[name]=dict(reviewed_sha256=before,current_sha256=after)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
              if line.startswith('model name')),'unknown')
    quota=Path('/sys/fs/cgroup/cpu.max')
    document=dict(schema=1,baseline_commit=BASE,core_hash=gate['core_hash'],
                  historical_artifacts_preserved=preserved,changed_reviewed_sources=changes,files=files,
                  hardware=dict(platform=platform.platform(),cpu_model=cpu,
                                cgroup_cpu_max=quota.read_text().strip() if quota.exists() else None),
                  timing_scope='Wall-clock measurements on this cloud worker; studies may overlap and timings are not dedicated-machine benchmarks')
    (OUT/'manifest.json').write_text(json.dumps(document,indent=2))
    print(f'Verified {len(preserved)} preserved historical artifacts and {len(files)} current files')


if __name__=='__main__':
    main()
