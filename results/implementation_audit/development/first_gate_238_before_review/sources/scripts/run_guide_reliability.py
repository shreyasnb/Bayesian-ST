"""Execute the frozen calibration, block-adaptation and mismatch protocol."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from lib.guide_reliability import (configuration, summarize_episode, adaptation_episode,
    mismatch_episode, sensor_diagnostics, interval, binomial_interval)
from lib.implementation_validation import require_gate

SOURCES = ['lib/guide_reliability.py', 'scripts/run_guide_reliability.py',
    'tests/test_guide_reliability.py', 'lib/block_em.py', 'lib/models.py',
    'lib/geometry.py', 'lib/contact_posterior.py', 'lib/contact_mechanics.py',
    'lib/beamforming_numerics.py', 'lib/spectral_series.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(v) for v in value]
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + '\n')


def task(arguments):
    kind, model, regime, stage, episode, cfg = arguments
    try:
        if kind == 'calibration':
            return kind, summarize_episode(model, regime, stage, episode, cfg)
        if kind == 'adaptation':
            return kind, adaptation_episode(model, regime, episode, cfg)
        return kind, mismatch_episode(model, episode, cfg)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
        return 'failure', dict(kind=kind, model=model, regime=regime, stage=stage,
                              episode=episode, error=type(error).__name__ + ': ' + str(error))


def summaries(calibration, scores):
    output = []
    keys = ['model', 'regime', 'stage', 'method', 'kind', 'nominal']
    for values, frame in calibration.groupby(keys, sort=False):
        count, successes = len(frame), int(frame.covered.sum())
        lo, hi = binomial_interval(successes, count)
        width, width_lo, width_hi = interval(frame.width)
        output.append(dict(zip(keys, values), episodes=count, successes=successes,
            coverage=successes / count, lower=lo, upper=hi, width=width,
            width_lower=width_lo, width_upper=width_hi,
            unresolved=int((~frame.numerical_ok).sum()), numerical_ok=bool(frame.numerical_ok.all())))
    score_rows = []
    for values, frame in scores.groupby(['model', 'regime', 'stage'], sort=False):
        mixture = interval(frame.mixture_log_score)
        gaussian = interval(frame.gaussian_log_score)
        difference = interval(frame.mixture_log_score - frame.gaussian_log_score)
        risk_difference = interval(frame.risk_realized - frame.risk_predicted)
        score_rows.append(dict(zip(['model', 'regime', 'stage'], values),
            episodes=len(frame), mixture_score=mixture[0], gaussian_score=gaussian[0],
            score_gain=difference[0], score_lower=difference[1], score_upper=difference[2],
            predicted_risk=float(frame.risk_predicted.mean()), realized_risk=float(frame.risk_realized.mean()),
            risk_difference=risk_difference[0], risk_lower=risk_difference[1], risk_upper=risk_difference[2],
            unresolved=int((~frame.numerical_ok).sum())))
    return pd.DataFrame(output), pd.DataFrame(score_rows)


def matched_widths(calibration, cfg):
    rows = []
    selected = calibration[calibration.nominal.eq(cfg['target_coverage'])
        & calibration.kind.isin(['clean_response', 'noisy_observation'])]
    for values, frame in selected.groupby(['model', 'regime', 'method', 'kind'], sort=False):
        validation = frame[frame.stage.eq('validation')]
        test = frame[frame.stage.eq('test')]
        factor = cfg['width_multipliers'][-1]
        achieved = 0.
        for candidate in cfg['width_multipliers']:
            lower = validation.midpoint + candidate * (validation.lower - validation.midpoint)
            upper = validation.midpoint + candidate * (validation.upper - validation.midpoint)
            coverage = float(((validation.truth >= lower) & (validation.truth <= upper)).mean())
            if coverage >= cfg['target_coverage']:
                factor, achieved = candidate, coverage
                break
            achieved = coverage
        lower = test.midpoint + factor * (test.lower - test.midpoint)
        upper = test.midpoint + factor * (test.upper - test.midpoint)
        included = (test.truth >= lower) & (test.truth <= upper)
        lo, hi = binomial_interval(int(included.sum()), len(test))
        widths = interval(upper - lower)
        rows.append(dict(zip(['model', 'regime', 'method', 'kind'], values),
            target=cfg['target_coverage'], selected_multiplier=factor,
            validation_episodes=len(validation), validation_coverage=achieved,
            validation_target_reached=bool(achieved >= cfg['target_coverage']),
            test_episodes=len(test), test_coverage=float(included.mean()),
            test_lower=lo, test_upper=hi, matched_width=widths[0],
            width_lower=widths[1], width_upper=widths[2],
            numerical_ok=bool(validation.numerical_ok.all() and test.numerical_ok.all()),
            scope='Validation-calibrated interval, not a posterior credible interval; all test episodes retained.'))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'paper'], default='smoke')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('Positive worker count required')
    gate = require_gate(ROOT) if args.profile == 'paper' else None
    cfg = configuration(args.profile)
    output = args.output or ROOT / 'results/implementation_audit' / args.profile / 'reliability'
    output.mkdir(parents=True, exist_ok=True)
    sources = {name: sha(ROOT / name) for name in SOURCES}
    if (output / 'manifest.json').exists():
        old = json.loads((output / 'manifest.json').read_text())
        if old.get('config') == cfg and old.get('sources') == sources:
            if all(sha(output / name) == expected for name, expected in old['tables'].items()):
                print('Verified completed reliability cache', flush=True)
                return
        raise RuntimeError('Preserve stale evidence and use a fresh output directory')
    save(output / 'preregistered.json', dict(config=cfg, sources=sources,
        created_utc=datetime.now(timezone.utc).isoformat(), gate=gate))
    jobs = []
    for model in cfg['models']:
        for regime in cfg['regimes']:
            for stage, count in [('validation', cfg['validation_episodes']), ('test', cfg['test_episodes'])]:
                jobs += [('calibration', model, regime, stage, i, cfg) for i in range(count)]
        for change in ['noise_only', 'motion_only']:
            jobs += [('adaptation', model, change, 'test', i, cfg) for i in range(cfg['adaptation_episodes'])]
        jobs += [('mismatch', model, '', 'test', i, cfg) for i in range(cfg['mismatch_episodes'])]
    start = perf_counter()
    calibration, scores, adaptation, mismatches, failures = [], [], [], [], []
    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'))
        stream = executor.map(task, jobs, chunksize=1)
    else:
        stream = map(task, jobs)
    try:
        for index, (kind, value) in enumerate(stream, 1):
            if kind == 'calibration':
                rows, score = value
                calibration.extend(rows)
                scores.append(score)
            elif kind == 'adaptation':
                adaptation.extend(value)
            elif kind == 'mismatch':
                mismatches.extend(value)
            else:
                failures.append(value)
            if index % 32 == 0 or index == len(jobs):
                print('reliability completed', index, '/', len(jobs), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    frames = dict(calibration=pd.DataFrame(calibration), scores=pd.DataFrame(scores),
                  adaptation=pd.DataFrame(adaptation), mismatch=pd.DataFrame(mismatches))
    frames['calibration_summary'], frames['score_summary'] = summaries(frames['calibration'], frames['scores'])
    frames['matched_widths'] = matched_widths(frames['calibration'], cfg)
    # JSON histories preserve all EM starts, rather than selecting a start by
    # a held-out score or pretending unconverged fits are a global optimum.
    starts = frames['adaptation'].pop('starts').tolist()
    save(output / 'em_start_histories.json', starts)
    for name, frame in frames.items():
        frame.to_csv(output / (name + '.csv'), index=False)
    save(output / 'failures.json', failures)
    sensor = sensor_diagnostics(cfg)
    save(output / 'sensor_diagnostics.json', sensor)
    if any(sha(ROOT / name) != expected for name, expected in sources.items()):
        raise RuntimeError('Scientific producer changed during execution')
    checks = dict(seconds=perf_counter() - start, requested_jobs=len(jobs),
        completed_jobs=len(jobs) - len(failures), failed_jobs=len(failures),
        unresolved_calibration_rows=int((~frames['calibration'].numerical_ok).sum()),
        unresolved_score_rows=int((~frames['scores'].numerical_ok).sum()),
        unresolved_adaptation_rows=int((~frames['adaptation'].numerical_ok).sum()),
        unresolved_mismatch_rows=int((~frames['mismatch'].numerical_ok).sum()),
        scaled_projection_residual=sensor['scaled_projection_residual'],
        error_scope='Mean accuracy, finite posterior integration precision, repeated-data inclusion and model correctness have separate flags.')
    save(output / 'checks.json', checks)
    files = sorted(p for p in output.iterdir() if p.is_file() and p.name != 'manifest.json')
    save(output / 'manifest.json', dict(status='completed', config=cfg, sources=sources,
        tables={p.name: sha(p) for p in files}, checks=checks,
        figures=['B6', 'C5', 'B7b', 'C6b'],
        assumptions='Exact static prior-predictive experiments are model matched; fixed-state and deterministic drift experiments are distinct. Fits use one boundary prior per disjoint block. No fitted-scale interval includes hyperparameter uncertainty. Mismatch rows do not invoke the theorem.'))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
