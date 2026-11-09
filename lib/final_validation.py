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


def gate_path(root):
    return Path(root) / 'results/beamforming/data/validation/validation.json'


def require_gate(root):
    root = Path(root).resolve()
    path = gate_path(root)
    if not path.exists():
        raise RuntimeError('Run scripts/validate_final_simulations.py first')
    gate = json.loads(path.read_text())
    if (gate.get('status') != 'passed' or gate.get('core_hash') != core_hash(root)
            or gate.get('final_science_hash') != science_hash(root)):
        raise RuntimeError('The final general validation is missing or stale')
    return gate


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
    verify_mapping(root, meta.get('sources', {}))
    verify_mapping(root, meta.get('artifacts', {}))
    return meta


def verify_history(root):
    root = Path(root).resolve()
    folder = root / 'archive/a91faab'
    meta = json.loads((folder / 'preservation.json').read_text())
    verify_mapping(folder, meta['files'])
    verify_mapping(root, meta['historical_results'])
    return meta
