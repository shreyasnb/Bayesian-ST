"""Record publishable sources and artifacts, excluding ignored worker caches."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.contact_tracking_evidence import check_evidence


def selected_files(*paths):
    output=subprocess.check_output(
        ['git','ls-files','--cached','--others','--exclude-standard','-z','--',*paths],cwd=ROOT)
    return sorted(set(name.decode() for name in output.split(b'\0') if name))


def record(name):
    path=ROOT/name;digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return dict(sha256=digest.hexdigest(),bytes=path.stat().st_size)


def main():
    gate=check_evidence(ROOT)
    target='results/contact_tracking/manifest.json'
    artifacts=[name for name in selected_files('results/contact_tracking','results/contact_uncertainty')
               if name!=target]
    sources=selected_files('lib','scripts','tests','docs')
    files={name:record(name) for name in sorted(set(artifacts+sources+[
        'notebooks/03_contact.ipynb','README.md','pyproject.toml','requirements-lock.txt']))}
    oversized=[name for name,value in files.items() if value['bytes']>=100*1024**2]
    if oversized:raise RuntimeError(f'Artifacts exceed the GitHub file limit: {oversized}')
    manifest=dict(starting_commit='f09d3fecd3af11fff5af708c8d1d31a35db75ca4',
        validation_tests=gate['tests'],core_hash=gate['core_hash'],
        uncertainty_hash=gate['uncertainty_hash'],files=files,
        excluded='Git-ignored resumable worker checkpoints, bytecode and pytest XML',
        total_bytes=sum(v['bytes'] for v in files.values()))
    (ROOT/target).write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Recorded {len(files)} source/artifact hashes; all files below 100 MiB.')


if __name__=='__main__':main()
