import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Rerun the frozen paper simulations, exports and checks.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'rerun')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    commands = [
        ['-m', 'pytest', '-q'],
        ['scripts/run_contact.py', '--output', str(output / 'contact' / 'data'),
         '--workers', str(args.workers)],
        ['scripts/run_beamforming.py', '--output', str(output / 'beamforming' / 'data'),
         '--workers', str(args.workers)],
        ['scripts/export_paper.py', '--results', str(output)],
        ['scripts/check_paper.py', '--results', str(output)],
    ]
    for command in commands:
        print('+', sys.executable, *command, flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
