"""Read the separately gated contact pilot without changing historical evidence."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from .contact_validation import require_contact_gate


def load_contact(root,directory=None):
    root=Path(root).resolve();out=Path(directory or root/'results/contact_pilot').resolve()
    gate=require_contact_gate(root);metadata=json.loads((out/'config.json').read_text())
    if metadata.get('development_only'):raise RuntimeError('Development smoke data are not the contact pilot')
    if any(metadata['gate'][key]!=gate[key] for key in ['core_hash','contact_hash']):
        raise RuntimeError('Contact evidence and validation certificate differ')
    for name,digest in metadata['sources'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Stale contact source: {name}')
    frames=dict(config=metadata['config'],metadata=metadata)
    for name,digest in metadata['tables'].items():
        path=out/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Changed contact table: {name}')
        if path.suffix=='.csv':frames[path.stem]=pd.read_csv(path)
    return frames
