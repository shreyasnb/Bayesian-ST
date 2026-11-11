"""Replay original contact simulations with dense output retention only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import sys
from time import perf_counter

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from lib.contact_transport_diagnostics import OUTPUT,load_context,declaration,replay_trial,summarize,sha


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def trial_task(trial,decl_hash):
    context=load_context(ROOT);folder=ROOT/OUTPUT/'data/checkpoints'/str(trial).zfill(3)
    record=folder/'record.json'
    if record.exists():
        saved=json.loads(record.read_text())
        if saved['declaration_sha256']!=decl_hash:raise RuntimeError('Preserve obsolete dense checkpoint '+str(folder))
        for name,digest in saved['artifacts'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Changed dense checkpoint '+name)
        return saved
    metrics,physical,diagnostics,checks=replay_trial(context,trial)
    folder.mkdir(parents=True,exist_ok=True)
    for name,frame in [('metrics',metrics),('physical',physical),('diagnostics',diagnostics)]:
        frame.to_csv(folder/(name+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
    result=dict(trial=trial,declaration_sha256=decl_hash,checks=checks,
        artifacts={path.relative_to(ROOT).as_posix():sha(path) for path in folder.glob('*.csv.gz')})
    write(record,result);return result


def run(preflight=False,workers=1):
    context=load_context(ROOT);decl=declaration(context);decl_hash=signature(decl)
    base=ROOT/OUTPUT;data=base/'data';tables=base/'tables'
    data.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    if preflight:
        start=perf_counter();checks=[]
        for trial in (0,1):
            _,_,_,check=replay_trial(context,trial);checks.append(check)
        seconds=perf_counter()-start
        report=dict(status='passed',declaration_sha256=decl_hash,config=decl,first_two_trial_checks=checks,
            seconds=seconds,estimated_serial_256_seconds=seconds*128,
            scope='Preflight execution metadata only, not a computational benchmark.')
        write(data/'preflight.json',report);print(json.dumps(report,indent=2));return report
    preflight_path=data/'preflight.json'
    if not preflight_path.exists():raise RuntimeError('Run --preflight for the first two exact-parity trials first')
    preflight_record=json.loads(preflight_path.read_text())
    if preflight_record['status']!='passed' or preflight_record['declaration_sha256']!=decl_hash:
        raise RuntimeError('Dense preflight is stale; preserve it and rerun for the current sources')
    registered=data/'preregistered.json'
    if registered.exists() and json.loads(registered.read_text())!=decl:raise RuntimeError('Preserve the prior dense replay declaration')
    write(registered,decl)
    manifest=data/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['declaration_sha256']!=decl_hash:raise RuntimeError('Completed dense replay uses another declaration')
        for name,digest in old['artifacts'].items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Changed completed dense artifact '+name)
        print('Verified completed dense contact diagnostics');return old
    completed={}
    if workers==1:
        for trial in range(256):
            completed[trial]=trial_task(trial,decl_hash)
            if (trial+1)%16==0:print('Dense contact',trial+1,'/256',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs={pool.submit(trial_task,trial,decl_hash):trial for trial in range(256)}
            for job in as_completed(jobs):
                trial=jobs[job];completed[trial]=job.result()
                if len(completed)%16==0:print('Dense contact',len(completed),'/256',flush=True)
    def combined(stem):
        return pd.concat([pd.read_csv(data/'checkpoints'/str(trial).zfill(3)/(stem+'.csv.gz'))
                          for trial in range(256)],ignore_index=True)
    metrics=combined('metrics');physical=combined('physical');diagnostics=combined('diagnostics')
    result=summarize(context,metrics,physical,diagnostics,[completed[trial]['checks'] for trial in range(256)])
    for name,frame in [('dense_metrics',metrics),('physical_diagnostics',physical),('trial_diagnostics',diagnostics)]:
        frame.to_csv(data/(name+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
    for name,frame in result.items():
        if name!='checks':frame.to_csv(tables/(name+'.csv'),index=False)
    write(data/'trial_checks.json',[completed[trial]['checks'] for trial in range(256)])
    write(data/'checks.json',result['checks'])
    for mapping in (decl['source_hashes'],decl['input_hashes']):
        for name,digest in mapping.items():
            if sha(ROOT/name)!=digest:raise RuntimeError('Dense replay source/input changed: '+name)
    artifacts={path.relative_to(ROOT).as_posix():sha(path) for folder in (data,tables)
        for path in sorted(folder.glob('*')) if path.is_file() and path.name!='manifest.json'}
    meta=dict(schema=1,status='completed',example='contact_transport_diagnostics',
        config=decl,declaration_sha256=decl_hash,source_hashes=decl['source_hashes'],
        input_hashes=decl['input_hashes'],artifacts=artifacts,checks=result['checks'],
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Original exact-seed observations, settings and all eight methods. Only output retention expands. Single saved-story checks are not Monte Carlo summaries.')
    write(manifest,meta);print('Completed dense contact diagnostics',flush=True);return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    run(args.preflight,args.workers)


if __name__=='__main__':main()
