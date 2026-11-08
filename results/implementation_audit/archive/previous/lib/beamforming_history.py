"""Read-only validation of the independently archived earlier array studies."""
from pathlib import Path
import hashlib,json


def verify_historical(root):
    root=Path(root).resolve();out=root/'results/beamforming_unified/archive'
    meta=json.loads((out/'historical_manifest.json').read_text())
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for name,expected in meta['artifacts'].items():
        if digest(root/name)!=expected:raise RuntimeError(f'Historical artifact changed: {name}')
    for name,expected in meta['snapshot_files'].items():
        if digest(out/name)!=expected:raise RuntimeError(f'Historical source changed: {name}')
    h=hashlib.sha256()
    for name in meta['certificate_core_paths']:
        h.update(name.encode());h.update((out/'certificate_source'/name).read_bytes())
    if h.hexdigest()!=meta['certificate']['core_hash']:
        raise RuntimeError('The archived numerical certificate does not match its source tree')
    return meta
