"""Repair held-out query pairing without changing any inference or tuning.

The inherited posterior summary uses its method-specific simulation seed for a
fresh noisy response. This script preserves those raw rows and evaluates the
same intervals on one new method-shared physical response per trajectory/law.
No model setting, state draw, training observation, or posterior is changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from lib.reporting import wilson

QUERY_SEED=8291747


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/dev_prediction/motion')
    args=parser.parse_args();out=args.output_dir.resolve()
    meta_path=out/'config.json';meta=json.loads(meta_path.read_text())
    raw_path=out/'posterior.csv'
    assert sha(raw_path)==meta['tables']['posterior.csv'],'Raw posterior table changed'
    for name,digest in meta['sources'].items():
        assert sha(ROOT/name)==digest,name
    script_digest=sha(__file__)
    evaluation_path=out/'paired_noise_evaluation.json'
    if evaluation_path.exists():
        old=json.loads(evaluation_path.read_text())
        if old['script_sha256']==script_digest and old['input_sha256']==sha(raw_path) and all(
            sha(out/p)==h for p,h in old['tables'].items()):
            print('Method-shared noisy-query evaluation: verified cache');return
        raise RuntimeError('Existing paired-query evaluation differs; preserve and use a new output directory')
    # Preserve the exact producer metadata before recording additional derived tables.
    original_meta=out/'config_before_paired_noise.json'
    original_meta.write_bytes(meta_path.read_bytes())
    frame=pd.read_csv(raw_path)
    if not set(frame.law).issubset({'slow','fast'}):
        raise ValueError('This paired-query pilot is specified for the fixed-noise slow/fast laws only')
    frame['original_future_observation']=frame.future_observation
    frame['original_predictive_covered']=frame.predictive_covered
    law_ids={law:i for i,law in enumerate(meta['config']['posterior_laws'])}
    shared={}
    for (trial,law,input_frame),g in frame.groupby(['trial','law','input_frame'],sort=False):
        assert np.ptp(g.true_response.to_numpy())<1e-12,'Methods do not share the same clean query target'
        frame_id=['sensor','co_rotating'].index(input_frame)
        rng=np.random.default_rng(np.random.SeedSequence([QUERY_SEED,int(trial),law_ids[law],frame_id]))
        observed=float(g.true_response.iloc[0]+np.sqrt(meta['config']['s'])*rng.normal())
        shared[(trial,law,input_frame)]=observed
    frame['future_observation']=[shared[(r.trial,r.law,r.input_frame)] for r in frame.itertuples()]
    frame['predictive_covered']=(frame.future_observation>=frame.predictive_q025)&(frame.future_observation<=frame.predictive_q975)
    rows=[]
    for (law,input_frame,method),g in frame.groupby(['law','input_frame','method'],sort=False):
        for region,width in [('ball_covered','posterior_radius'),('hpd_covered','hpd_threshold'),
                             ('clean_response_covered','clean_width'),('predictive_covered','predictive_width')]:
            count=int(g[region].sum());low,high=wilson(count,len(g))
            rows.append(dict(law=law,input_frame=input_frame,method=method,region=region,
                successes=count,trials=len(g),coverage=count/len(g),low=low,high=high,
                mean_width=float(g[width].mean()),width_quantity=width,
                sampling_passes=int(g.sampling_ok.sum()),prior_sampling_passes=int(g.prior_sampling_ok.sum())))
    tables={}
    for name,table in [('posterior_paired_noise',frame),('coverage_paired_noise',pd.DataFrame(rows))]:
        path=out/f'{name}.csv';table.to_csv(path,index=False);tables[path.name]=sha(path)
    record=dict(status='checked',query_seed=QUERY_SEED,script_sha256=script_digest,
        input_sha256=sha(raw_path),original_metadata_sha256=sha(original_meta),
        original_producer_source_hash=meta['source_hash'],tables=tables,
        scope='Evaluation pairing correction only; identical inference, tuning, hidden states, training observations and posterior draws',
        raw_evidence='posterior.csv and original future-response columns are preserved',
        query_law='One independent N(0,s) held-out noise draw per trial/law/input frame, shared across methods',
        paired_groups=len(shared),posterior_rows=len(frame),
        interval_scope=f'Terminal pilot, {meta["config"]["posterior_trials"]} independent trajectories per regime/method, wide Wilson intervals')
    evaluation_path.write_text(json.dumps(record,indent=2))
    meta['tables'].update(tables)
    meta['postprocessing_sources']={'scripts/check_motion_paired_noise.py':script_digest}
    meta['postprocessing']=[dict(name='method_shared_query_noise',metadata='paired_noise_evaluation.json',
                                scope=record['scope'],original_metadata='config_before_paired_noise.json')]
    meta_path.write_text(json.dumps(meta,indent=2))
    findings=out/'findings.md'
    findings.write_text(findings.read_text()+'\nMethod-shared held-out query evaluation: use posterior_paired_noise.csv and\ncoverage_paired_noise.csv for the final noisy-response coverage comparison.\nThe inherited raw summary used method-specific held-out noise; its original\nrows and producer metadata remain preserved. This paired-evaluation repair\nchanges no inference, tuning, training data or posterior draws. See\npaired_noise_evaluation.json for the independent query seed and checksums.\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
