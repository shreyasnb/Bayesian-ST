"""Read-only inputs for the editable historical field notebook.

This module loads saved arrays/tables only. It never simulates observations,
fits a posterior, selects a policy, creates figures, or writes study outputs.
The historical preservation ledger and the original table manifests provide
independent content checks even when old rendering-source gates are obsolete.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


LEDGER = 'archive/a91faab/preservation.json'
LEDGER_SHA256 = '3dbe1854921661384db890c82e31953f5944cf331fcd243f0f4229cd439c8dda'
DATASETS = {'application': 'case2', 'drift': 'drift_study', 'decisions': 'decision_study'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field_input_hashes(root, dataset=None):
    """Verify the historical data and return the exact files used by a load."""
    root = Path(root).resolve()
    ledger_path = root / LEDGER
    if _sha(ledger_path) != LEDGER_SHA256:
        raise RuntimeError('The original historical preservation ledger changed')
    preserved = json.loads(ledger_path.read_text())['historical_results']
    groups = list(DATASETS) if dataset is None else [dataset]
    verified = {LEDGER: LEDGER_SHA256}
    for group in groups:
        if group not in DATASETS:
            raise ValueError(f'Unknown field notebook dataset: {group}')
        prefix = DATASETS[group]
        config_name = f'results/{prefix}_config.json'
        expected = preserved[config_name]
        if _sha(root / config_name) != expected:
            raise RuntimeError('Historical field input changed: ' + config_name)
        verified[config_name] = expected
        config = json.loads((root / config_name).read_text())
        for table, digest in config['tables'].items():
            name = f'results/{prefix}_{table}.csv'
            if preserved[name] != digest or _sha(root / name) != digest:
                raise RuntimeError('Historical field input changed: ' + name)
            verified[name] = digest
    return verified


def load_field_data(root, dataset='application'):
    """Load one saved table group, without rerunning any technical procedure.

    ``application`` contains the passive field tables, ``drift`` contains the
    recorded tracking/prediction tables, and ``decisions`` contains the shared-
    data stationary decision experiment. Returned frames are ordinary editable
    in-memory DataFrames; modifying them does not modify their source files.
    """
    root = Path(root).resolve()
    field_input_hashes(root, dataset)
    prefix = DATASETS[dataset]
    metadata = json.loads((root / 'results' / f'{prefix}_config.json').read_text())
    frames = {name: pd.read_csv(root / 'results' / f'{prefix}_{name}.csv')
              for name in metadata['tables']}
    return dict(**frames, config=metadata['config'],
                checks=metadata.get('checks', {}), metadata=metadata)
