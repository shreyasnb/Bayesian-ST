"""Source-verified gate for the draft-5 simulation redesign."""
from pathlib import Path
import hashlib
import json
from .validation import core_hash

BASE_COMMIT='2a8697c7e7fa27b3779f3abeaed32ca8b323a9ef'
NEW_SOURCES=[
    'lib/array_geometry_study.py','lib/contact_geometry_study.py',
    'lib/anisotropy_study.py','lib/simulation_scaling.py','lib/simulation_validation.py',
    'scripts/run_array_geometry.py','scripts/run_contact_geometry.py',
    'scripts/run_anisotropy_study.py','scripts/run_simulation_scaling.py',
    'scripts/validate_simulation_redesign.py']
SHARED_SOURCES=[
    'lib/models.py','lib/filter.py','lib/geometry.py','lib/theory.py','lib/em.py',
    'lib/moments.py','lib/spectral_series.py','lib/reference.py',
    'lib/contact_mechanics.py','lib/contact_posterior.py','lib/contact_gaussian.py',
    'lib/beamforming_numerics.py','lib/beamforming_decisions.py',
    'benchmarks/case1/baselines.py','benchmarks/case2/baselines.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def science_sources(root):
    return sorted(set(NEW_SOURCES+SHARED_SOURCES))


def science_hash(root):
    root=Path(root);digest=hashlib.sha256()
    for name in science_sources(root):
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def require_gate(root):
    root=Path(root).resolve();path=root/'results/simulation_redesign/validation/validation.json'
    if not path.exists():raise RuntimeError('Run scripts/validate_simulation_redesign.py first')
    gate=json.loads(path.read_text())
    if (gate.get('status')!='passed' or gate.get('core_hash')!=core_hash(root)
            or gate.get('science_hash')!=science_hash(root)):
        # Later guide additions run the entire notebook and regression suite
        # into a separate directory. Keep the earlier certificate byte-exact;
        # accept only a current, fully source-verified superset validation.
        from .implementation_validation import require_gate as require_implementation_gate
        return require_implementation_gate(root)
    return gate


def gate_path(root):
    """Location of the current validated gate, including a later superset."""
    root=Path(root).resolve()
    gate=require_gate(root)
    namespace='implementation_audit' if gate.get('implementation_hash') else 'simulation_redesign'
    return root/'results'/namespace/'validation/validation.json'


def verify_component(root,profile,component):
    root=Path(root).resolve()
    if profile not in ('quick','paper'):raise ValueError('Use quick or paper')
    folder=root/'results/simulation_redesign'/profile/component
    choices=[folder/'manifest.json',folder/'config.json']
    path=next((p for p in choices if p.exists()),None)
    if path is None:raise RuntimeError(f'Complete {profile}/{component} first')
    meta=json.loads(path.read_text())
    status=meta.get('status')
    status_path=folder/'status.json'
    if status_path.exists():status=json.loads(status_path.read_text()).get('status')
    if status not in ('completed','passed','verified'):
        raise RuntimeError(f'Incomplete {profile}/{component}: {status}')
    if not meta.get('sources') or not meta.get('tables'):
        raise RuntimeError(f'Missing content hashes: {path}')
    for name,digest in meta['sources'].items():
        if isinstance(digest,dict):digest=digest['sha256']
        if sha(root/name)!=digest:raise RuntimeError('Changed producer: '+name)
    for name,digest in meta['tables'].items():
        if isinstance(digest,dict):digest=digest['sha256']
        if sha(folder/name)!=digest:raise RuntimeError('Changed artifact: '+str(folder/name))
    return meta


def verify_legacy(root):
    root=Path(root).resolve();folder=root/'results/simulation_redesign/archive'
    meta=json.loads((folder/'preservation.json').read_text())
    for name,digest in meta['unchanged_results'].items():
        if sha(root/name)!=digest:raise RuntimeError('Historical evidence changed: '+name)
    for name,digest in meta['archive_files'].items():
        if sha(folder/name)!=digest:raise RuntimeError('Historical archive changed: '+name)
    for record in meta['certificates']:
        digest=hashlib.sha256();source=folder/record['source_directory']
        for name in record['core_paths']:
            digest.update(name.encode());digest.update((source/name).read_bytes())
        if digest.hexdigest()!=record['core_hash']:
            raise RuntimeError('Historical certificate source mismatch')
    return meta
