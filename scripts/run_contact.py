from pathlib import Path
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lib.final_contact import test_episode, anisotropy_fixture
from lib.contact_reporting import save_tables


def run_trial(arguments):
    config, selection, trial = arguments
    return test_episode(config, 'sphere', trial, selection)['metrics']


def main():
    parser = argparse.ArgumentParser(description='Reproduce the frozen contact experiment and fixed-plane illustration.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/rerun/contact/data')
    parser.add_argument('--trials', type=int, default=256)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.trials < 2 or args.workers < 1:
        parser.error('At least two trials and one worker are required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a fresh directory')
    frozen = json.loads((ROOT / 'configs/contact.json').read_text())
    config, selection = frozen['config'], frozen['selection']
    tasks = [(config, selection, trial) for trial in range(args.trials)]
    if args.workers == 1:
        results = list(map(run_trial, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            results = list(pool.map(run_trial, tasks))
    metrics = pd.DataFrame([row for rows in results for row in rows])
    _, paired = save_tables(metrics, args.output)
    arrays, checks = anisotropy_fixture(config)
    if not checks['numerical_ok']:
        raise ArithmeticError('Fixed-plane illustration failed its moment checks')
    np.savez_compressed(args.output / 'anisotropy.npz', **arrays)
    (args.output / 'checks.json').write_text(json.dumps(dict(
        source_commit=frozen['source_commit'], trials=args.trials, methods=4,
        action_times=list(range(4,193,4)), full_paper_cohort=args.trials==256,
        maximum_mean_resolution_error=float(metrics.moment_resolution_error.max()),
        anisotropy=checks), indent=2)+'\n')
    print(paired.to_string(index=False))


if __name__ == '__main__':
    main()
