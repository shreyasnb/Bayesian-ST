"""Content-addressed validation for the implementation-guide additions."""
from pathlib import Path
import hashlib
import json
from .validation import core_hash
from .simulation_validation import science_sources as redesign_sources
from .simulation_validation import science_hash as redesign_hash

BASE_COMMIT = '6e78cc9911e7e78452633f13dd947afa669c3470'
COMPONENTS = ('information', 'tracking', 'em', 'reliability')
NEW_SOURCES = (
    'lib/block_em.py', 'lib/guide_information.py', 'lib/guide_tracking.py',
    'lib/guide_reliability.py', 'lib/implementation_validation.py',
    'lib/beamforming_theory.py', 'lib/beamforming_protocol.py',
    'lib/contact_spherical_reference.py',
    'scripts/run_block_em_study.py', 'scripts/run_guide_information.py',
    'scripts/run_guide_tracking.py', 'scripts/run_guide_reliability.py',
    'scripts/validate_implementation_audit.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def science_sources(root):
    return sorted(set(redesign_sources(root)) | set(NEW_SOURCES))


def science_hash(root):
    root = Path(root)
    digest = hashlib.sha256()
    for name in science_sources(root):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def require_gate(root):
    root = Path(root).resolve()
    path = root / 'results/implementation_audit/validation/validation.json'
    if not path.exists():
        raise RuntimeError('Run scripts/validate_implementation_audit.py first')
    gate = json.loads(path.read_text())
    if (gate.get('status') != 'passed' or gate.get('core_hash') != core_hash(root)
            or gate.get('science_hash') != redesign_hash(root)
            or gate.get('implementation_hash') != science_hash(root)):
        raise RuntimeError('Implementation-guide validation is missing or stale')
    return gate


def verify_mapping(root, mapping):
    root = Path(root)
    if not mapping:
        raise RuntimeError('An evidence manifest needs nonempty content hashes')
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        if sha(root / name) != expected:
            raise RuntimeError('Changed evidence: ' + str(root / name))


def verify_component(root, profile, component):
    root = Path(root).resolve()
    if profile not in ('smoke', 'paper') or component not in COMPONENTS:
        raise ValueError('Unknown implementation-guide profile/component')
    folder = root / 'results/implementation_audit' / profile / component
    path = folder / 'manifest.json'
    meta = json.loads(path.read_text())
    if meta.get('status') not in ('completed', 'passed', 'verified'):
        raise RuntimeError('Incomplete evidence: ' + str(path))
    verify_mapping(root, meta.get('sources', {}))
    verify_mapping(folder, meta.get('tables', {}))
    return meta


def verify_preservation(root):
    root = Path(root).resolve()
    folder = root / 'results/implementation_audit/archive'
    meta = json.loads((folder / 'preservation.json').read_text())
    verify_mapping(folder, meta['files'])
    return meta
