"""Read-only provenance checks for the independent fixed-horizon cohort."""
from pathlib import Path
import json

from .final_validation import (
    STAGES, current_gate_path, require_current_gate, sha, verify_mapping,
)


def stationary_component_path(root, stage='paper'):
    if stage not in STAGES:
        raise ValueError('Use verify, pilot or paper')
    return (Path(root) / 'results/beamforming/data' / stage / 'performance'
            / 'stationary_T200/manifest.json')


def verify_stationary_component(root, stage='paper'):
    """Verify fresh sources and artifacts; never substitute historical sources."""
    root = Path(root).resolve()
    path = stationary_component_path(root, stage)
    meta = json.loads(path.read_text())
    if (meta.get('status') not in ('completed', 'passed', 'verified')
            or meta.get('example') != 'beamforming'
            or meta.get('stage') != stage
            or meta.get('component') != 'stationary_T200'):
        raise RuntimeError('Incomplete fixed-horizon stationary evidence: ' + str(path))
    cfg = meta.get('config', {})
    if (cfg.get('stationary_T') != 200
            or cfg.get('stationary_primary_t') != 200
            or cfg.get('stationary_times') != [25, 50, 100, 200]
            or cfg.get('dimensions') != [32, 64, 128]):
        raise RuntimeError('Unexpected stationary dimensions or measurement horizon')
    if stage == 'paper':
        if cfg.get('validation_episodes') != 64 or cfg.get('test_episodes') != 256:
            raise RuntimeError('Paper stationary evidence requires 64 validation and 256 test simulations')
        gate = require_current_gate(root)
        certificate = current_gate_path(root)
        if (meta.get('validation_path') != str(certificate.relative_to(root))
                or meta.get('validation_sha256') != sha(certificate)
                or meta.get('gate_science_hash') != gate.get('final_science_hash')):
            raise RuntimeError('Stationary evidence is not bound to its current validation certificate')
    verify_mapping(root, meta.get('sources', {}))
    verify_mapping(root, meta.get('artifacts', {}))
    expected = str(path.with_name('stationary.csv').relative_to(root))
    if expected not in meta['artifacts']:
        raise RuntimeError('Stationary raw records are missing from the manifest')
    return meta
