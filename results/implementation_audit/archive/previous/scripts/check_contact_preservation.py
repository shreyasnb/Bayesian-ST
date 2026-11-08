"""Verify the frozen repository artifacts were preserved during this study."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f09d3fecd3af11fff5af708c8d1d31a35db75ca4'


def git_blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    entries = subprocess.check_output(['git','ls-tree','-rz',BASE],cwd=ROOT).split(b'\0')
    allowed = {'.gitignore','README.md','notebooks/03_contact.ipynb'}
    checked = {}
    for entry in entries:
        if not entry:
            continue
        info, name = entry.split(b'\t',1)
        mode, kind, expected = info.decode().split()
        name = name.decode()
        if kind != 'blob' or name in allowed:
            continue
        data = (ROOT/name).read_bytes()
        if git_blob(data) != expected:
            raise RuntimeError(f'Historical artifact changed: {name}')
        checked[name] = hashlib.sha256(data).hexdigest()
    old = subprocess.check_output(['git','show',f'{BASE}:notebooks/03_contact.ipynb'],cwd=ROOT)
    archived = ROOT/'results/contact_tracking/archive/03_contact_f09d3fe.ipynb'
    assert archived.read_bytes() == old
    manifest = dict(status='passed',starting_commit=BASE,unchanged_files=len(checked),
        unchanged_contact_pilot_files=sum(p.startswith('results/contact_pilot/') for p in checked),
        explicitly_revised=sorted(allowed),archived_notebook_sha256=hashlib.sha256(old).hexdigest(),
        sha256=checked)
    (ROOT/'results/contact_tracking/preservation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Preserved {len(checked)} historical files; original contact notebook archived exactly.')


if __name__ == '__main__':
    main()
