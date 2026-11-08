"""Content-addressed gate for the separate contact-uncertainty comparison."""
from pathlib import Path
import hashlib
import json
from .validation import core_hash

UNCERTAINTY_SOURCES=[
    'lib/contact_mechanics.py','lib/contact_study.py','lib/contact_posterior.py',
    'lib/contact_process_prediction.py','lib/contact_gaussian.py',
    'lib/contact_spherical_reference.py','lib/contact_uncertainty_study.py',
    'lib/contact_uncertainty_validation.py','scripts/run_contact_uncertainty.py',
    'scripts/run_contact_matched.py','scripts/check_contact_process.py',
    'scripts/check_contact_gaussian.py','scripts/validate_contact_uncertainty.py',
    'lib/contact_tracking.py','lib/contact_tracking_uncertainty.py',
    'scripts/run_contact_tracking.py','scripts/run_contact_tracking_uncertainty.py']


def uncertainty_hash(root):
    root=Path(root);digest=hashlib.sha256()
    for name in UNCERTAINTY_SOURCES:
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def require_uncertainty_gate(root):
    root=Path(root).resolve();path=root/'results/contact_uncertainty/validation/validation.json'
    if not path.exists():raise RuntimeError('Run scripts/validate_contact_uncertainty.py successfully first')
    gate=json.loads(path.read_text())
    if (gate.get('status')!='passed' or gate.get('core_hash')!=core_hash(root)
            or gate.get('uncertainty_hash')!=uncertainty_hash(root)):
        raise RuntimeError('Contact uncertainty validation is missing or stale; rerun its validation script')
    return gate
