"""Scientific and artifact gates for the final two-application study."""
from pathlib import Path
import hashlib
import json
from .validation import core_hash
from .implementation_validation import science_sources as previous_sources

BASE_COMMIT = 'a91faabeb10a63b1d1c613d75cc32503cfc0cac6'
EXAMPLES = ('beamforming', 'contact')
STAGES = ('verify', 'pilot', 'paper')
COMPONENTS = ('theory', 'performance', 'prediction', 'computation')
NEW_SOURCES = (
    'lib/final_validation.py', 'lib/final_theory.py',
    'lib/final_beamforming.py', 'lib/final_contact.py',
    'lib/causal_variance.py', 'lib/final_prediction.py',
    'lib/final_computation.py', 'lib/final_accuracy.py',
    'lib/final_stationary_evidence.py',
    'lib/simulation_scaling.py', 'benchmarks/common/__init__.py',
    'benchmarks/common/great.py', 'scripts/validate_final_simulations.py',
    'scripts/run_final_theory.py', 'scripts/run_final_beamforming.py',
    'scripts/run_final_contact.py', 'scripts/run_final_prediction.py',
    'scripts/run_final_computation.py', 'pyproject.toml',
    'results/beamforming/data/specification.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def science_sources(root):
    return sorted(set(previous_sources(root)) | set(NEW_SOURCES))


def science_hash(root):
    root = Path(root)
    digest = hashlib.sha256()
    for name in science_sources(root):
        path = root / name
        if not path.is_file():
            raise RuntimeError('Final scientific source is not complete: ' + name)
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


HISTORICAL_SOURCE_PROOF = 'archive/dd399ba/stationary_source_proof.json'
HISTORICAL_SOURCE_PROOF_SHA256 = 'c693f5b5f874f08a245ef6187c072e1e93c30c2242fde8d7119e8015a8e30222'


def gate_path(root):
    """Original certificate location, retained for historical figure provenance."""
    return Path(root) / 'results/beamforming/data/validation/validation.json'


def current_gate_path(root):
    """Certificate for the revised stationary experiment and current sources."""
    return gate_path(root).parent / 'stationary_T200/validation.json'


def _historical_proof(root):
    path = Path(root) / HISTORICAL_SOURCE_PROOF
    if not path.is_file() or sha(path) != HISTORICAL_SOURCE_PROOF_SHA256:
        raise RuntimeError('Historical stationary source proof is missing or changed')
    proof = json.loads(path.read_text())
    if proof.get('revision') != 'stationary_T200' or proof.get('schema') != 1:
        raise RuntimeError('Unknown historical source proof')
    return proof


def verify_recorded_sources(root, mapping, *, manifest_path, certificate=False):
    """Verify current sources, or the exact proof-bound historical producer.

    Only the immutable pre-revision manifest/certificate digests in the source
    proof can use archived producer bytes. Artifact checks never use this path.
    """
    root = Path(root).resolve()
    if not mapping:
        raise RuntimeError('Evidence requires nonempty source hashes')
    proof = None
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        path = root / name
        if path.is_file() and sha(path) == expected:
            continue
        if proof is None:
            proof = _historical_proof(root)
            record = Path(manifest_path).resolve()
            try:
                record_name = str(record.relative_to(root))
            except ValueError as error:
                raise RuntimeError('Historical evidence must be inside the repository') from error
            allowed = proof['general_certificates' if certificate else 'component_manifests']
            if record_name not in allowed or sha(record) != allowed[record_name]:
                raise RuntimeError('Unregistered historical evidence: ' + record_name)
        archived = proof['source_files'].get(name)
        if archived is None or archived['sha256'] != expected:
            raise RuntimeError('Changed source has no historical proof: ' + name)
        archive_path = root / archived['archive_path']
        if not archive_path.is_file() or sha(archive_path) != expected:
            raise RuntimeError('Changed historical source archive: ' + name)


def _current_gate(root, path):
    if not path.is_file():
        return None
    gate = json.loads(path.read_text())
    if (gate.get('status') == 'passed' and gate.get('core_hash') == core_hash(root)
            and gate.get('final_science_hash') == science_hash(root)):
        verify_mapping(root, gate.get('sources', {}))
        return gate
    return None


def require_current_gate(root):
    """Require a fresh complete validation; historical evidence cannot authorize a run."""
    root = Path(root).resolve()
    gate = _current_gate(root, current_gate_path(root))
    if gate is None:
        raise RuntimeError('Run scripts/validate_final_simulations.py --revision stationary_T200 first')
    return gate


def require_gate(root):
    root = Path(root).resolve()
    current = _current_gate(root, current_gate_path(root))
    if current is not None:
        return current
    path = gate_path(root)
    if not path.exists():
        raise RuntimeError('Run scripts/validate_final_simulations.py first')
    gate = json.loads(path.read_text())
    if gate.get('status') != 'passed':
        raise RuntimeError('The final general validation is missing or stale')
    if (gate.get('core_hash') == core_hash(root)
            and gate.get('final_science_hash') == science_hash(root)):
        verify_mapping(root, gate.get('sources', {}))
        return gate
    verify_recorded_sources(root, gate.get('sources', {}),
                            manifest_path=path, certificate=True)
    proof = _historical_proof(root)
    verify_recorded_sources(root, proof['core_sources'],
                            manifest_path=path, certificate=True)
    core_digest = hashlib.sha256()
    for name in proof['core_source_order']:
        source = root / name
        if not source.is_file() or sha(source) != proof['core_sources'][name]:
            source = root / proof['source_files'][name]['archive_path']
        core_digest.update(name.encode())
        core_digest.update(source.read_bytes())
    if core_digest.hexdigest() != gate.get('core_hash'):
        raise RuntimeError('Historical core certificate does not match its archived tests')
    digest = hashlib.sha256()
    for name, expected in sorted(gate['sources'].items()):
        path = root / name
        if not path.is_file() or sha(path) != expected:
            path = root / _historical_proof(root)['source_files'][name]['archive_path']
        digest.update(name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != gate.get('final_science_hash'):
        raise RuntimeError('Historical certificate does not match its archived sources')
    return gate


def certificate_path(root, *, current=False):
    """Return the actual verified certificate, with a strict current-only option."""
    root = Path(root).resolve()
    if current:
        require_current_gate(root)
        return current_gate_path(root)
    require_gate(root)
    if _current_gate(root, current_gate_path(root)) is not None:
        return current_gate_path(root)
    return gate_path(root)


def verify_mapping(root, mapping, *, nonempty=True):
    root = Path(root)
    if nonempty and not mapping:
        raise RuntimeError('Evidence requires nonempty source/artifact hashes')
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Changed or missing evidence: ' + str(path))


def component_path(root, example, stage, component):
    if example not in EXAMPLES or stage not in STAGES or component not in COMPONENTS:
        raise ValueError('Unknown final example, stage or component')
    return Path(root) / 'results' / example / 'data' / stage / component / 'manifest.json'


def verify_component(root, example, stage, component):
    root = Path(root).resolve()
    path = component_path(root, example, stage, component)
    meta = json.loads(path.read_text())
    if (meta.get('status') not in ('completed', 'passed', 'verified')
            or meta.get('example') != example or meta.get('stage') != stage
            or meta.get('component') != component):
        raise RuntimeError('Incomplete or incorrectly identified evidence: ' + str(path))
    verify_recorded_sources(root, meta.get('sources', {}), manifest_path=path)
    verify_mapping(root, meta.get('artifacts', {}))
    return meta


def verify_history(root):
    root = Path(root).resolve()
    folder = root / 'archive/a91faab'
    meta = json.loads((folder / 'preservation.json').read_text())
    verify_mapping(folder, meta['files'])
    verify_mapping(root, meta['historical_results'])
    return meta
