"""Read-only, hash-checked historical arrays for the editable audit notebook.

This module deliberately imports no plotting library and performs no inference,
fitting or simulation. Historical component manifests identify the saved data;
their old whole-code validation gates are not represented as current gates.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = ('information', 'information_counts', 'tracking', 'em', 'reliability')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_input_hashes(root):
    """Return the historical input-file closure used by the notebook."""
    root = Path(root).resolve()
    mapping = {}
    for component in COMPONENTS:
        folder = root/'results/implementation_audit/paper'/component
        manifest = folder/'manifest.json'
        meta = json.loads(manifest.read_text())
        if meta.get('status') not in ('completed', 'passed', 'verified'):
            raise RuntimeError('Incomplete historical component: '+component)
        mapping[str(manifest.relative_to(root))] = _sha(manifest)
        for name, expected in meta['tables'].items():
            mapping[str((folder/name).relative_to(root))] = (
                expected['sha256'] if isinstance(expected, dict) else expected)
    folder = root/'results/simulation_redesign/paper/beamforming'
    manifest = folder/'manifest.json'
    meta = json.loads(manifest.read_text())
    mapping[str(manifest.relative_to(root))] = _sha(manifest)
    expected = meta['tables']['metrics.csv']
    mapping[str((folder/'metrics.csv').relative_to(root))] = (
        expected['sha256'] if isinstance(expected, dict) else expected)
    return mapping


def verify_audit_inputs(root):
    """Validate saved arrays/tables without validating a different source tree."""
    root = Path(root).resolve()
    mapping = audit_input_hashes(root)
    for name, expected in mapping.items():
        if _sha(root/name) != expected:
            raise RuntimeError('Changed historical notebook input: '+name)
    return mapping


@lru_cache(maxsize=6)
def _load_component(root_string, component):
    root = Path(root_string)
    if component == 'heldout':
        folder = root/'results/simulation_redesign/paper/beamforming'
        meta = json.loads((folder/'manifest.json').read_text())
        path = folder/'metrics.csv'
        expected = meta['tables'][path.name]
        if _sha(path) != (expected['sha256'] if isinstance(expected, dict) else expected):
            raise RuntimeError('Changed historical held-out data: '+str(path))
        return {'beamforming': {'metrics': pd.read_csv(path)}}
    if component not in COMPONENTS:
        raise ValueError('Unknown historical audit component: '+str(component))
    folder = root/'results/implementation_audit/paper'/component
    meta = json.loads((folder/'manifest.json').read_text())
    result = {}
    for name, expected in meta['tables'].items():
        path = folder/name
        if _sha(path) != (expected['sha256'] if isinstance(expected, dict) else expected):
            raise RuntimeError('Changed historical notebook input: '+str(path))
        if path.suffix == '.csv':
            result[path.stem] = pd.read_csv(path)
        elif path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as saved:
                result[path.stem] = {key: saved[key].copy() for key in saved.files}
    return result


def load_audit_data(root, component):
    """One-line data loading for an editable plot; returns independent copies."""
    return deepcopy(_load_component(str(Path(root).resolve()), str(component)))
