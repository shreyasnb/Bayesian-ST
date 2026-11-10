"""Read verified saved evidence for the appended supplementary figure cells.

This module loads tables and arrays only. It does not call a renderer, run a
simulation, fit a parameter, or draw posterior samples. Imports finish in the
supplementary setup cell before its visible Matplotlib defaults are applied.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the original selection and reporting calculations without changing the
# scientific implementation. None of these helpers creates a Matplotlib artist.
from .final_paper_visuals import (
    load_evidence, _normalize, _primary, _column, _finite_column,
    _mean_interval, _episode_losses, _saved, _optional, _reference_frame,
    display_sphere_geometry,
)
from .distribution_geometry_visuals import (
    bingham_sphere_mesh, real_bingham_log_density,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def _load_evidence(root, example):
    return load_evidence(Path(root), example, 'paper')


def load_supplementary_evidence(root, example):
    """Load the unchanged four-component paper evidence once per application."""
    if example not in ('beamforming', 'contact'):
        raise ValueError('Use beamforming or contact saved evidence')
    return _load_evidence(str(Path(root).resolve()), example)


@lru_cache(maxsize=2)
def _constructed_geometry(root):
    from .simulation_validation import verify_component
    root = Path(root)
    meta = verify_component(root, 'paper', 'geometry')
    folder = root/'results/simulation_redesign/paper/geometry'
    names = ('manifest.json', 'matrices.npz', 'directional.csv', 'distance_cdf.csv')
    inputs = {str((folder/name).relative_to(root)): _sha(folder/name)
              for name in names}
    with np.load(folder/'matrices.npz', allow_pickle=False) as saved:
        parameters = saved['A'][:2].copy()
        normalizers = saved['log_normalizer'][:2].copy()
        modes = saved['mode'][:2].copy()
    return dict(parameters=parameters, log_normalizers=normalizers, modes=modes,
        directional=pd.read_csv(folder/'directional.csv'),
        distance_cdf=pd.read_csv(folder/'distance_cdf.csv'),
        input_hashes=inputs, manifest=meta)


def load_constructed_geometry(root):
    """Read the original two-case deterministic geometry counterexample."""
    record = _constructed_geometry(str(Path(root).resolve()))
    return dict(parameters=record['parameters'].copy(),
        log_normalizers=record['log_normalizers'].copy(),
        modes=record['modes'].copy(),
        directional=record['directional'].copy(),
        distance_cdf=record['distance_cdf'].copy(),
        input_hashes=dict(record['input_hashes']), manifest=record['manifest'])
