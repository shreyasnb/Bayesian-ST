"""Replay final-time selection into a new, separately declared audit directory.

All random keys and measurement models come from the frozen T=200 cohort.
Original observations, results, settings and validation certificates are read
only. Concurrent elapsed times below are execution metadata, not benchmarks.
"""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.beamforming_stationary_selection_audit import (
    OUTPUT,SOURCES,configuration,inputs,reference_frame,candidate_grid,validation_trial,
    select_dimension,test_trial,summaries,sha,fingerprint,replay_stream,array_sha,
)
from lib.final_validation import require_current_gate,current_gate_path


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def load_validation(folder,n,trial,signature):
    path=folder/'checkpoints/validation'/f'n{n}_{trial:03d}.json'
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value['signature']!=signature:raise RuntimeError('Preserve the obsolete new audit and use another output directory')
    if value['result_hash']!=fingerprint(value['rows']):raise RuntimeError('Changed audit validation worker')
    target=folder/value['statistics_file']
    if sha(target)!=value['statistics_sha256']:raise RuntimeError('Changed loading validation statistics')
    with np.load(target,allow_pickle=False) as saved:
        statistics={name:saved[name] for name in saved.files}
    return value['rows'],statistics,value['replay'],value['seconds']


def validation_phase(folder,cfg,source_signature):
    allrows=[];allsummaries=[];selected={};replay=[];timings=[]
    for n in cfg['n']:
        rows=[];statistics=[]
        for trial in range(64):
            signature=fingerprint(dict(source=source_signature,n=n,trial=trial,split='validation'))
            output=load_validation(folder,n,trial,signature)
            if output is None:
                started=perf_counter();records,stats,proof=validation_trial(cfg,n,trial)
                seconds=perf_counter()-started
                target=folder/'checkpoints/validation'/f'n{n}_{trial:03d}.npz'
                target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**stats)
                value=dict(signature=signature,rows=records,result_hash=fingerprint(records),
                    statistics_file=str(target.relative_to(folder)),statistics_sha256=sha(target),replay=proof,seconds=seconds)
                write(target.with_suffix('.json'),value)
                output=records,stats,proof,seconds
            records,stats,proof,seconds=output
            rows.extend(records);statistics.append(stats);replay.append(proof)
            timings.append(dict(phase='validation',n=n,trial=trial,seconds=seconds))
            if (trial+1)%8==0:print(f'validation n={n}: {trial+1}/64',flush=True)
        selection,records,summary=select_dimension(cfg,n,rows,statistics)
        selected[str(n)]=selection;allrows.extend(records);allsummaries.extend(summary)
        print('selected n='+str(n),json.dumps(selection['settings']),flush=True)
    pd.DataFrame(allrows).to_csv(folder/'validation_trials.csv',index=False)
    pd.DataFrame(allsummaries).to_csv(folder/'validation_candidates.csv',index=False)
    write(folder/'selected_settings.json',selected)
    write(folder/'selection_freeze.json',dict(status='selected_before_test_scoring',
        selected_settings_sha256=sha(folder/'selected_settings.json'),
        validation_trials_sha256=sha(folder/'validation_trials.csv'),
        validation_candidates_sha256=sha(folder/'validation_candidates.csv'),
        source_signature=source_signature,frozen_utc=datetime.now(timezone.utc).isoformat()))
    return selected,replay,timings


def test_phase(folder,cfg,source_signature,selected,reference):
    freeze=json.loads((folder/'selection_freeze.json').read_text())
    assert freeze['source_signature']==source_signature
    for key,name in [('selected_settings_sha256','selected_settings.json'),
                     ('validation_trials_sha256','validation_trials.csv'),
                     ('validation_candidates_sha256','validation_candidates.csv')]:
        assert freeze[key]==sha(folder/name)
    rows=[];proofs=[];timings=[]
    for n in cfg['n']:
        for trial in range(256):
            old=reference[reference.n.eq(n)&reference.trial.eq(trial)]
            assert len(old)==10
            signature=fingerprint(dict(source=source_signature,n=n,trial=trial,split='test',selection=selected[str(n)]))
            path=folder/'checkpoints/test'/f'n{n}_{trial:03d}.json'
            if path.exists():
                saved=json.loads(path.read_text())
                assert saved['signature']==signature and saved['result_hash']==fingerprint(saved['rows'])
            else:
                start=perf_counter();records,proof=test_trial(cfg,n,trial,selected[str(n)],old)
                saved=dict(signature=signature,rows=records,result_hash=fingerprint(records),replay=proof,seconds=perf_counter()-start)
                write(path,saved)
            rows.extend(saved['rows']);proofs.append(saved['replay'])
            timings.append(dict(phase='test',n=n,trial=trial,seconds=saved['seconds']))
            if (trial+1)%32==0:print(f'test n={n}: {trial+1}/256',flush=True)
    original=reference[['n','trial','t','T','method','true_power','oracle_power','true_ratio_db','numerical_ok','unit_gain_error']].copy()
    original['source']='saved_original'
    frame=pd.concat([original,pd.DataFrame(rows)],ignore_index=True)
    assert len(frame)==3*256*14 and not frame.duplicated(['n','trial','method']).any()
    assert frame.groupby(['n','method']).trial.nunique().eq(256).all()
    assert frame.numerical_ok.astype(bool).all()
    np.testing.assert_allclose(frame.true_ratio_db,10*np.log10(frame.true_power/frame.oracle_power),rtol=1e-11,atol=1e-11)
    summary,paired=summaries(frame)
    frame.to_csv(folder/'test_metrics.csv',index=False)
    summary.to_csv(folder/'test_summary.csv',index=False)
    paired.to_csv(folder/'paired_vs_mean.csv',index=False)
    return frame,proofs,timings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();folder=args.output or ROOT/OUTPUT
    folder=folder.resolve()
    try:folder.relative_to(ROOT/OUTPUT)
    except ValueError as error:raise ValueError('Audit writes must remain inside the new stationary selection directory') from error
    folder.mkdir(parents=True,exist_ok=True)
    gate=require_current_gate(ROOT);cfg=configuration(ROOT)
    sources={name:sha(ROOT/name) for name in SOURCES};raw_inputs=inputs(ROOT)
    signature=fingerprint(dict(config=cfg,sources=sources,inputs=raw_inputs))
    declaration=dict(config=cfg,sources=sources,inputs=raw_inputs,source_signature=signature,
        validation_path=str(current_gate_path(ROOT).relative_to(ROOT)),
        validation_sha256=sha(current_gate_path(ROOT)),scientific_sha256=gate['final_science_hash'])
    target=folder/'preregistered.json'
    if target.exists():
        old=json.loads(target.read_text());assert all(old[key]==value for key,value in declaration.items())
    else:write(target,dict(declaration,registered_utc=datetime.now(timezone.utc).isoformat()))
    if (folder/'manifest.json').exists():
        meta=json.loads((folder/'manifest.json').read_text())
        assert meta['source_signature']==signature
        for name,digest in meta['artifacts'].items():assert sha(ROOT/name)==digest
        print('Verified existing stationary final-time selection audit');return
    began=perf_counter();reference=reference_frame(ROOT)
    # An exact saved trajectory verifies the replay generator independently of
    # all candidate values. The original initialization is its unchanged seed child.
    stream=replay_stream(cfg,64,0,'test')
    original_manifest=json.loads((ROOT/'results/beamforming/data/paper/performance/stationary_T200/manifest.json').read_text())
    original_story=next(name for name in original_manifest['artifacts'] if Path(name).name=='stories.npz')
    with np.load(ROOT/original_story,allow_pickle=False) as saved:
        np.testing.assert_array_equal(stream['U'],saved['static_n64_U'])
        np.testing.assert_array_equal(stream['h'],saved['static_n64_h'])
        np.testing.assert_array_equal(stream['a'],saved['static_n64_a'])
    selected,val_replay,clocks=validation_phase(folder,cfg,signature)
    frame,test_replay,test_clocks=test_phase(folder,cfg,signature,selected,reference)
    clocks.extend(test_clocks)
    pd.DataFrame(val_replay+test_replay).to_csv(folder/'replay_fingerprints.csv',index=False)
    pd.DataFrame(clocks).to_csv(folder/'execution_metadata.csv',index=False)
    assert not set(row['observation_sha256'] for row in val_replay)&set(row['observation_sha256'] for row in test_replay)
    checks=dict(status='passed',dimensions=[32,64,128],T=200,t=200,
        validation_simulations_per_dimension=64,test_simulations_per_dimension=256,
        great_candidates_per_dimension={str(n):len(candidate_grid(n,cfg)) for n in cfg['n']},
        full_history_window=200,full_history_control_separate=True,
        candidate_selection_uses_final_measured_db_only=True,selection_frozen_before_test=True,
        old_mean_mode_isotropic_values_preserved=True,all_original_methods_retained=10,
        additional_methods=4,new_test_records=int(frame.source.eq('final_time_selection_audit').sum()),
        maximum_mode_replay_error_db=max(row['mode_reference_db_error'] for row in test_replay),
        maximum_old_great_window_replay_error_db=max(row['great_window_reference_db_error'] for row in test_replay),
        maximum_old_great_ew_replay_error_db=max(row['great_ew_reference_db_error'] for row in test_replay),
        maximum_old_covariance_loading_replay_error_db=max(row['covariance_loading_reference_db_error'] for row in test_replay),
        exact_saved_n64_story_measurements_replayed=True,disjoint_validation_test_replay_hashes=True,
        initial_and_observation_hashes_recorded=True,all_test_methods_resolved=bool(frame.numerical_ok.all()),
        maximum_unit_gain_error=float(frame.unit_gain_error.max()),
        moment_and_normalizer_recomputations=0,old_files_written=0,
        timing_scope='Concurrent execution metadata; no computational benchmark claim',
        total_seconds=perf_counter()-began)
    write(folder/'replay_checks.json',checks)
    assert all(sha(ROOT/name)==digest for name,digest in sources.items()),'Audit producer changed during execution'
    assert all(sha(ROOT/name)==digest for name,digest in raw_inputs.items()),'Original evidence changed during execution'
    artifacts={str(path.relative_to(ROOT)):sha(path) for path in sorted(folder.iterdir())
               if path.is_file() and path.name!='manifest.json'}
    write(folder/'manifest.json',dict(schema=1,status='completed',example='beamforming',
        component='stationary_final_time_selection_audit',config=cfg,sources=sources,inputs=raw_inputs,
        source_signature=signature,artifacts=artifacts,checks=checks,
        validation_path=declaration['validation_path'],validation_sha256=declaration['validation_sha256'],
        scientific_sha256=gate['final_science_hash'],
        scope='Additional references on exact original streams and initials; original selections/results remain separate and immutable',
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
