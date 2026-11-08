"""Preserve the preceding executed redesign and its producer sources exactly."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '6e78cc9911e7e78452633f13dd947afa669c3470'
OUT = ROOT / 'results/implementation_audit/archive'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = OUT / 'preservation.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        for name, expected in meta['files'].items():
            if sha((OUT / name).read_bytes()) != expected:
                raise RuntimeError('Changed archived evidence: ' + name)
        print('Verified', len(meta['files']), 'archived files')
        return
    names = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', BASE_COMMIT], cwd=ROOT, text=True).splitlines()
    prefixes = ('lib/', 'scripts/', 'tests/', 'benchmarks/', 'docs/',
                'results/simulation_redesign/')
    explicit = {'.gitignore', 'README.md', 'pyproject.toml',
                'notebooks/00_theory.ipynb', 'notebooks/01_array.ipynb',
                'notebooks/02_field.ipynb', 'notebooks/03_contact.ipynb'}
    files = {}
    for name in names:
        if name not in explicit and not name.startswith(prefixes):
            continue
        data = subprocess.check_output(['git', 'show', BASE_COMMIT + ':' + name], cwd=ROOT)
        target = OUT / 'previous' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[str(target.relative_to(OUT))] = sha(data)
    meta = dict(source_commit=BASE_COMMIT, files=files,
                branch_tips={name: subprocess.check_output(['git', 'rev-parse', name],
                    cwd=ROOT, text=True).strip() for name in ('main', 'dev', 'test')},
                scope='Byte-exact Git blobs for the preceding executed notebooks, all '
                      'redesign evidence and its full producer/test sources. Historical '
                      'certificates keep their original assumptions and fingerprints.')
    manifest.write_text(json.dumps(meta, indent=2) + '\n')
    print('Preserved', len(files), 'files from', BASE_COMMIT)


if __name__ == '__main__':
    main()
