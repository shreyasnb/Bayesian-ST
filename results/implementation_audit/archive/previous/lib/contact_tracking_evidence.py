"""Check saved study sources and artifacts before displaying their results."""
import hashlib
import json
from pathlib import Path
from .contact_uncertainty_validation import require_uncertainty_gate


def check_evidence(root):
    root = Path(root).resolve()
    gate = require_uncertainty_gate(root)
    out = root / 'results/contact_tracking'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    mean_manifest = json.loads((out/'mean/manifest.json').read_text())
    for name, expected in mean_manifest['files'].items():
        if digest(out/'mean'/name) != expected:
            raise RuntimeError(f'Saved mean-tracking evidence changed: {name}')
    uncertainty = json.loads((out/'uncertainty/config.json').read_text())
    for name, expected in uncertainty['sources'].items():
        if digest(root/name) != expected:
            raise RuntimeError(f'Uncertainty-study source changed: {name}')
    for name, expected in uncertainty['tables'].items():
        if digest(out/'uncertainty'/name) != expected:
            raise RuntimeError(f'Saved uncertainty evidence changed: {name}')
    selected = json.loads((out/'mean/selected.json').read_text())
    for name, expected in selected['sources'].items():
        if digest(root/name) != expected:
            raise RuntimeError(f'Mean-tracking source changed: {name}')
    return gate
