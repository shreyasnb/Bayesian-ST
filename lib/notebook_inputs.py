"""Read-only numerical inputs for user-editable notebook figures.

This module has no plotting imports. Input checks protect the saved experiment;
they deliberately do not constrain a notebook's labels, canvas size or dpi.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_input_hashes(root, example):
    root = Path(root).resolve()
    meta = json.loads((root/'results'/example/'figure_manifest.json').read_text())
    hashes = dict(meta['input_manifests'])
    for record in meta['figures'].values():
        hashes.update(record.get('data_hashes', {}))
    return {name: value['sha256'] if isinstance(value, dict) else value
            for name, value in hashes.items()}


def verify_saved_inputs(root, example):
    """Verify numerical data only; notebook formatting remains unrestricted."""
    root = Path(root).resolve()
    from .final_validation import require_gate
    require_gate(root)
    hashes = saved_input_hashes(root, example)
    for name, digest in hashes.items():
        if sha(root/name) != digest:
            raise RuntimeError('Saved numerical input changed: '+name)
    return hashes


def load_final_config(root, example, component):
    path = Path(root)/'results'/example/'data/paper'/component/'manifest.json'
    return json.loads(path.read_text())['config']


def load_final_array(root, example, component, stem):
    """Load the same saved NPZ used by a deterministic scientific check."""
    root = Path(root).resolve()
    folder = root/'results'/example/'data/paper'/component
    meta = json.loads((folder/'manifest.json').read_text())
    matches = [name for name in meta['artifacts'] if Path(name).name == stem+'.npz']
    if len(matches) != 1:
        raise KeyError('Expected one saved array: '+stem)
    name = matches[0]
    expected = meta['artifacts'][name]
    expected = expected['sha256'] if isinstance(expected, dict) else expected
    if sha(root/name) != expected:
        raise RuntimeError('Saved numerical array changed: '+name)
    with np.load(root/name, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}
