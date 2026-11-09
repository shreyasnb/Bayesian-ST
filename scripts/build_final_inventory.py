"""Inventory retained final-study changes without including checkpoint caches."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.final_validation import BASE_COMMIT, require_gate, sha


def git_lines(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--content-commit')
    args = parser.parse_args()
    gate = require_gate(ROOT)
    destination = ROOT / 'final_artifact_inventory.json'
    names = set(git_lines('diff', '--name-only', BASE_COMMIT, '--'))
    names.update(git_lines('ls-files', '--others', '--exclude-standard'))
    names.discard(str(destination.relative_to(ROOT)))
    files = {}
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError('A retained change is missing: ' + name)
        if path.stat().st_size >= 100 * 1024**2:
            raise RuntimeError('Compress or exclude the oversized artifact: ' + name)
        files[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    ignored = git_lines('ls-files', '--others', '--ignored', '--exclude-standard',
                        'results/beamforming', 'results/contact')
    excluded = [ROOT / name for name in ignored if (ROOT / name).is_file()]
    result = dict(schema=1, status='inventoried', base_commit=BASE_COMMIT,
        scientific_content_commit=args.content_commit,
        final_science_hash=gate['final_science_hash'], files=files,
        retained_files=len(files), retained_bytes=sum(v['bytes'] for v in files.values()),
        excluded_regenerable_files=len(excluded),
        excluded_regenerable_bytes=sum(p.stat().st_size for p in excluded),
        excluded_scope='Git-ignored checkpoint caches, pytest XML and original general-notebook diagnostic PDF/PNG exports. Component manifests exclude these caches; required raw arrays/tables and executed general notebook are retained.',
        self_reference_scope='Only this root inventory excludes itself. Nested historical manifests remain inventoried. The scientific content commit precedes a metadata-only publication commit that records its hash.')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
