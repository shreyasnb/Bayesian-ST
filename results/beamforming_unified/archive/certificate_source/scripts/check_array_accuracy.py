"""Recheck failed array posterior diagnostics at a larger, separate budget.

Run after the case1 notebook has saved its aggregate tables::

    python scripts/check_array_accuracy.py --workers 4

Original tables and their failed diagnostic flags are preserved. This follow-up
uses new reproducible random streams; it does not replay original sampler RNGs
or select a budget based on which beamformer performs better.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import sys

# Bound BLAS threads before importing numerical libraries in spawned workers.
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy

from benchmarks.case1 import mvdr, output_power
from lib.experiments import _array_data, _credible, _posterior
from lib.filter import BinghamFilter
from lib.reporting import config_digest, simulation_hash
from lib.validation import require_gate

SCENARIOS = (
    ('Static', 0., .25, 'Matched'),
    ('Near interference', .003, .85, 'Matched'),
    ('Far from interference', .003, .05, 'Matched'),
    ('Drift', .012, .25, 'Matched'),
    ('Unequal powers', .003, .25, 'Unequal powers'),
    ('Correlation', .003, .25, 'Correlation'),
    ('Contamination', .003, .25, 'Contamination'),
)
SEED_OFFSET = 7331


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    paths = [*ROOT.joinpath('lib').glob('*.py'),
             *ROOT.joinpath('benchmarks').rglob('*.py'), Path(__file__)]
    files = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    return dict(files=files, sha256=hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest())


def _bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError(f'Unrecognized sampling status: {value!r}')


def _check_point(task):
    row, config, budgets = task
    matches = [(i, item) for i, item in enumerate(SCENARIOS)
               if item[0] == row['scenario']]
    if len(matches) != 1:
        raise ValueError(f"Unknown array scenario {row['scenario']!r}")
    case, (name, drift, overlap, mismatch) = matches[0]
    seed, rep, t = int(config['seed']), int(row['repeat']), int(row['t'])
    if row['mismatch'] != mismatch:
        raise ValueError('Saved scenario and mismatch disagree')
    U, X, a, covariance, G = _array_data(seed + 100*case + rep,
                                       int(config['T']), drift, overlap, mismatch)
    f = BinghamFilter(16, 3, .3, 2., beta=float(row['beta']), complex_data=True)
    for u in U[:t]:
        f.update(u, G=G)
    # This deliberately differs from the original stream _rng(seed,case,rep,17),
    # whose exact state also depends on earlier sampling/evaluation checkpoints.
    rng = np.random.default_rng(np.random.SeedSequence([seed, case, rep, t, SEED_OFFSET]))
    M, bases, diagnostics = _posterior(f.A, 3, rng, budgets['draws'],
        budgets['burnin'], True, budgets['max_draws'], budgets['max_burnin'])
    credible = _credible(f.A, f.basis, X[t-1], bases, mass=float(config['nominal_mass']))
    w = mvdr(.3*np.eye(16) + 2*M, a[t-1])
    new_power = float(output_power(w, covariance[t-1]))
    old_power = float(row['output_power'])
    result = dict(scenario=name, repeat=rep, t=t, beta=float(row['beta']),
        old_sampling_ok=_bool(row['sampling_ok']),
        new_sampling_ok=bool(diagnostics['sampling_ok']),
        old_output_power=old_power, new_output_power=new_power,
        output_power_relative_change=(new_power-old_power)/old_power,
        output_power_absolute_change=new_power-old_power,
        target_A_sha256=hashlib.sha256(np.ascontiguousarray(f.A).tobytes()).hexdigest(),
        seed_offset=SEED_OFFSET, old_sinr_db=float(row['sinr_db']),
        new_sinr_db=float(-10*np.log10(new_power)))
    for key in ('min_ess', 'max_rhat', 'max_mcse', 'max_standardized_difference',
                'actual_draws_per_chain', 'actual_burnin', 'sampling_attempts',
                'moment_seconds'):
        result['old_'+key] = float(row[key])
        result['new_'+key] = diagnostics[key]
    for key in ('radius', 'hpd_threshold', 'radius_squared_mcse', 'hpd_threshold_mcse',
                'ball_mass', 'hpd_mass', 'ball_covered', 'hpd_covered'):
        result['old_'+key] = row[key]
        result['new_'+key] = credible[key]
    result['radius_change'] = credible['radius'] - float(row['radius'])
    result['hpd_threshold_change'] = credible['hpd_threshold'] - float(row['hpd_threshold'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--draws', type=int, default=48000)
    parser.add_argument('--burnin', type=int, default=12000)
    parser.add_argument('--max-draws', type=int, default=96000)
    parser.add_argument('--max-burnin', type=int, default=24000)
    args = parser.parse_args(argv)
    if (args.workers < 1 or args.draws < 2 or args.burnin < 0
            or args.max_draws < args.draws or args.max_burnin < args.burnin):
        parser.error('Require positive workers and valid increasing sampling budgets')
    output = ROOT/'results'
    metrics_path, config_path = output/'case1_metrics.csv', output/'case1_config.json'
    if not metrics_path.exists() or not config_path.exists():
        raise RuntimeError('Finish and save the array notebook before this follow-up')
    gate = require_gate(ROOT)
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in
              (metrics_path, config_path, output/'validation.json')}
    provenance = source_manifest()
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    metadata = json.loads(config_path.read_text())
    config = metadata['config']
    if (metadata.get('source_hash') != simulation_hash(ROOT, 'case1', metadata['requested_config'])
            or metadata.get('config_digest') != config_digest(config)
            or metadata['tables'].get('metrics') != digest(metrics_path)):
        raise RuntimeError('Original array provenance does not match current source/config/table')
    if any(config.get(key) != value for key, value in
           dict(n=16, k=3, s=.3, tau2=2.).items()):
        raise RuntimeError('Follow-up reconstruction supports the documented n=16,k=3 protocol')
    metrics = pd.read_csv(metrics_path)
    posterior = metrics.loc[metrics.method == 'Posterior mean'].copy()
    failed = posterior.loc[~posterior.sampling_ok.map(_bool)].copy()
    if failed.duplicated(['scenario', 'repeat', 't']).any():
        raise ValueError('Duplicate failed posterior points in the original table')
    budgets = {name: getattr(args, name) for name in
               ('draws', 'burnin', 'max_draws', 'max_burnin')}
    tasks = [(row, config, budgets) for row in failed.to_dict('records')]
    # Cache only this follow-up, after independently verifying the original
    # experiment above. A changed script is itself a changed source manifest.
    destination = output/'case1_accuracy.csv'
    followup_metadata = output/'case1_accuracy.json'
    if followup_metadata.exists() and destination.exists():
        try:
            saved = json.loads(followup_metadata.read_text())
            reusable = (saved.get('schema') == 1
                and saved.get('input_sha256') == inputs
                and saved.get('source') == provenance
                and saved.get('config') == budgets
                and saved.get('versions') == versions
                and saved.get('core_hash') == gate['core_hash']
                and saved.get('original_source_hash') == metadata['source_hash']
                and saved.get('original_failed_points') == len(tasks)
                and saved.get('result_sha256') == digest(destination))
            if reusable:
                print('Array accuracy: loading verified extended-budget follow-up', flush=True)
                return 0
        except (OSError, ValueError, TypeError):
            # Missing/corrupt metadata cannot certify a cached result.
            pass
    results = []
    print(f'Follow-up points: {len(tasks)}; new random streams; budgets {budgets}', flush=True)
    if args.workers == 1:
        for task in tasks:
            result = _check_point(task)
            results.append(result)
            print(f"Accuracy {len(results)}/{len(tasks)}: {result['scenario']} "
                  f"repeat={result['repeat']} t={result['t']} "
                  f"ESS={result['new_min_ess']:.1f} "
                  f"passed={result['new_sampling_ok']}", flush=True)
    elif tasks:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(_check_point, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Accuracy {len(results)}/{len(tasks)}: {result['scenario']} "
                      f"repeat={result['repeat']} t={result['t']} "
                      f"ESS={result['new_min_ess']:.1f} "
                      f"passed={result['new_sampling_ok']}", flush=True)
    if provenance != source_manifest():
        raise RuntimeError('Source changed during follow-up; discard and rerun this check')
    for relative, original_digest in inputs.items():
        if digest(ROOT/relative) != original_digest:
            raise RuntimeError('Original validation/config/metrics changed during follow-up')
    frame = pd.DataFrame(results)
    if len(frame):
        frame = frame.sort_values(['scenario', 'repeat', 't']).reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=['scenario', 'repeat', 't', 'old_sampling_ok', 'new_sampling_ok'])
    frame.to_csv(destination, index=False)
    record = dict(schema=1, scope='Separate extended-budget follow-up of failed array posterior diagnostics',
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        original_failed_points=len(tasks), followup_passed=int(frame.new_sampling_ok.sum()),
        original_tables_preserved=True, core_hash=gate['core_hash'], source=provenance,
        input_sha256=inputs, result_sha256=digest(destination),
        original_source_hash=metadata['source_hash'], config=budgets, workers=args.workers,
        random_stream=dict(seed_sequence=['original seed', 'scenario index', 'repeat',
                                         'observation time', SEED_OFFSET],
                           same_as_original=False),
        acceptance_rule='min ESS >= 50, max split Rhat < 1.10, standardized chain difference < 5',
        versions=versions,
        interpretation='Diagnostic agreement and budget sensitivity, not proof of convergence or a ranking claim')
    (output/'case1_accuracy.json').write_text(json.dumps(record, indent=2))
    print(f"Saved {destination}; {record['followup_passed']}/{len(tasks)} follow-up points pass", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
