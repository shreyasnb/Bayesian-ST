"""Archive the preceding source/notebook version and bind retained old results."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a91faabeb10a63b1d1c613d75cc32503cfc0cac6'
OUT = ROOT / 'archive/a91faab'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = OUT / 'preservation.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        for name, expected in meta['files'].items():
            if sha((OUT / name).read_bytes()) != expected:
                raise RuntimeError('Changed archive: ' + name)
        for name, expected in meta['historical_results'].items():
            if sha((ROOT / name).read_bytes()) != expected:
                raise RuntimeError('Historical result changed: ' + name)
        print('Verified', len(meta['files']), 'archived files and',
              len(meta['historical_results']), 'unchanged historical outputs')
        return
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE],
                                    cwd=ROOT, text=True).splitlines()
    files, results = {}, {}
    for name in names:
        if name.startswith('results/'):
            results[name] = sha((ROOT / name).read_bytes())
        elif (name.startswith(('lib/', 'scripts/', 'tests/', 'benchmarks/', 'notebooks/', 'docs/'))
              or name in ('README.md', 'pyproject.toml', '.gitignore')):
            data = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT)
            path = OUT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files[name] = sha(data)
    meta = dict(source_commit=BASE, files=files, historical_results=results,
        branch_tips={name: subprocess.check_output(['git', 'rev-parse', name],
            cwd=ROOT, text=True).strip() for name in ('main', 'dev', 'test')},
        scope='Exact preceding code and all executed notebook versions. Original result '
              'files remain byte-identical in place. Historical notebooks retain their '
              'original formulas and source gates; use the recorded commit to execute '
              'them. Final figures/data use separate beamforming/contact directories.')
    manifest.write_text(json.dumps(meta, indent=2) + '\n')
    print('Archived', len(files), 'files; bound', len(results), 'historical outputs')


if __name__ == '__main__':
    main()
