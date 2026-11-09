"""Separate validation for the contact pilot; historical certificates stay intact."""
import hashlib
import json
from pathlib import Path
from .validation import core_hash

CONTACT_SOURCES=['lib/contact_mechanics.py','lib/contact_posterior.py',
                 'lib/contact_study.py','lib/contact_validation.py',
                 'scripts/run_contact_pilot.py','scripts/check_contact_posterior.py',
                 'scripts/validate_contact.py']


def contact_hash(root):
    root=Path(root);digest=hashlib.sha256()
    for name in CONTACT_SOURCES:
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def require_contact_gate(root):
    root=Path(root).resolve();path=root/'results/contact_pilot/validation/validation.json'
    if not path.exists():raise RuntimeError('Run scripts/validate_contact.py successfully first')
    gate=json.loads(path.read_text())
    if (gate.get('status')!='passed' or gate.get('core_hash')!=core_hash(root)
            or gate.get('contact_hash')!=contact_hash(root)):
        raise RuntimeError('Contact validation is missing or stale; rerun scripts/validate_contact.py')
    return gate
