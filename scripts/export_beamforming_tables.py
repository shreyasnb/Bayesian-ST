import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.beamforming_tables import export_tables,validate_tables


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--results',type=Path)
    args=parser.parse_args()
    print(json.dumps(dict(outputs=export_tables(args.root,args.results),checks=validate_tables(args.root,args.results)),indent=2))


if __name__=='__main__':main()
