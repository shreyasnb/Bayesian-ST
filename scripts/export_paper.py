from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.paper_plots import export_figures


def main():
    parser = argparse.ArgumentParser(description='Export the paper figures from saved simulation records.')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--example', choices=['all', 'contact', 'beamforming'], default='all')
    args = parser.parse_args()
    results = args.results.resolve()
    for path in export_figures(ROOT, results=results, example=args.example):
        print(path)


if __name__ == '__main__':
    main()
