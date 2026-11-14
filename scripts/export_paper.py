from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.paper_plots import export_figures
from lib.beamforming_tables import export_tables
from lib.contact_reporting import save_tables
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Export all five paper figures and numerical tables from saved records.')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    results = parser.parse_args().results.resolve()
    contact = results / 'contact/data'
    save_tables(pd.read_csv(contact / 'metrics.csv.gz', float_precision='round_trip'),
                contact, write_metrics=False)
    export_tables(ROOT, results=results)
    for path in export_figures(ROOT, results=results):
        print(path)


if __name__ == '__main__':
    main()
