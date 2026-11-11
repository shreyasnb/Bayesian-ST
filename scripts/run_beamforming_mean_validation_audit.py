"""Run the isolated independent posterior-mean memory-selection audit."""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from lib.beamforming_mean_validation_audit import (
    prepare_audit,verify_preserved_replay,validate_mean_memory,
    evaluate_selected_memory,summarize_mean_validation)


def progress(scope,phase,case,trial):
    print(json.dumps(dict(scope=scope,phase=phase,case=case,completed_trial=trial)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--scope',choices=('moving','stationary','both'),default='both')
    parser.add_argument('--phase',choices=('prepare','replay','validation','evaluation','summary','all'),default='all')
    args=parser.parse_args()
    scopes=('moving','stationary') if args.scope=='both' else (args.scope,)
    # Freeze every requested scope before looking at either new test outcome.
    for scope in scopes:
        cfg,destination,binding=prepare_audit(args.root,scope)
        print(json.dumps(dict(scope=scope,phase='prepared',destination=str(destination),
            binding=binding,criterion=cfg['validation_criterion'])),flush=True)
    for scope in scopes:
        if args.phase in ('all','replay'):
            print(json.dumps(verify_preserved_replay(args.root,scope)),flush=True)
        if args.phase in ('all','validation'):
            print(json.dumps(validate_mean_memory(args.root,scope,progress)),flush=True)
        if args.phase in ('all','evaluation'):
            frame=evaluate_selected_memory(args.root,scope,progress)
            print(json.dumps(dict(scope=scope,phase='evaluated',checkpoint_method_rows=len(frame))),flush=True)
        if args.phase in ('all','summary'):
            result=summarize_mean_validation(args.root,scope)
            print(json.dumps(dict(scope=scope,status=result['status'])),flush=True)


if __name__=='__main__':
    main()
