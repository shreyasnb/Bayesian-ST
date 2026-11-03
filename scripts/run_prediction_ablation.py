"""Run the frozen conditional motion/memory controls in an isolated directory.

Validation selects one policy over a declared mixture. Test streams are fixed
before validation, and failures are retained. Runtime measurements are hardware
dependent; source, software, settings, table digests and checkpoints are saved.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import subprocess
import sys

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.prediction_study import default_config,tune_task,test_task,posterior_task,choose,summaries
from lib.validation import require_gate


def canonical(value):
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if isinstance(value,np.generic):return canonical(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths=['lib/prediction_study.py','lib/prediction.py','scripts/run_prediction_ablation.py',
           'lib/geometry.py','lib/models.py','lib/filter.py','lib/em.py','lib/moments.py',
           'lib/drift_study.py','lib/theory.py','lib/reporting.py']
    return {p:sha(ROOT/p) for p in paths}


def save_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(canonical(value),indent=2,allow_nan=False));temporary.replace(path)


def run_jobs(function,tasks,stage,key,out,workers):
    directory=out/'checkpoints'/stage;directory.mkdir(parents=True,exist_ok=True)
    completed={};pending=[]
    for idx,task in enumerate(tasks):
        path=directory/f'{idx:04d}_{key[:16]}.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old.get('source_hash')==key and old.get('result_hash')==digest(old['result']):
                completed[idx]=old['result'];continue
        pending.append((idx,task))
    print(f'{stage}: restored {len(completed)}/{len(tasks)}',flush=True)
    def retain(idx,result):
        save_json(directory/f'{idx:04d}_{key[:16]}.json',
                  dict(source_hash=key,result_hash=digest(result),result=result))
        completed[idx]=result
        if len(completed)%8==0 or len(completed)==len(tasks):
            print(f'{stage}: completed {len(completed)}/{len(tasks)}',flush=True)
    if workers==1:
        for idx,task in pending:retain(idx,function(task))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):idx for idx,task in pending}
            for future in as_completed(futures):retain(futures[future],future.result())
    return [completed[idx] for idx in range(len(tasks))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/dev_prediction/motion')
    parser.add_argument('--validation-trials',type=int,default=32)
    parser.add_argument('--test-trials',type=int,default=64)
    parser.add_argument('--posterior-trials',type=int,default=8)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--gate-path',type=Path,default=ROOT/'results/dev_prediction/validation.json')
    args=parser.parse_args()
    if min(args.validation_trials,args.test_trials,args.workers)<1 or not 0<=args.posterior_trials<=args.test_trials:
        parser.error('Require positive validation/test/worker counts and 0 <= posterior_trials <= test_trials')
    config=default_config();config.update(validation_trials=args.validation_trials,
        test_trials=args.test_trials,posterior_trials=args.posterior_trials)
    gate=require_gate(ROOT,gate_path=args.gate_path)
    source=sources();versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=config,sources=source,versions=versions))
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    meta_path=out/'config.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old.get('source_hash')==key and old.get('core_hash')==gate['core_hash'] and all(
            (out/p).exists() and sha(out/p)==h for p,h in old['tables'].items()):
            print('Prediction ablation: verified cached tables');return
        # Preserve completed evidence; scientific revisions require a new output directory.
        raise RuntimeError('Existing completed run differs. Preserve it and choose a new --output-dir (and new test seed after test-driven revisions).')
    register=dict(config=config,configuration_hash=digest(config),sources=source,versions=versions,
        source_hash=key,core_hash=gate['core_hash'],
        reviewed_commit='57f520cbe860b73e13d71bae5c31391e2a009482',
        checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        selection='One common policy; equal weighting across slow/fast/acceleration and both input frames',
        cost_scope='Prediction and update cumulative seconds, separately; posterior moments only at terminal pilot',
        primary='Fast drift, sensor-fixed inputs: terminal causal velocity versus tuned forgetting; other intervals exploratory')
    frozen=out/'preregistered.json'
    if frozen.exists() and json.loads(frozen.read_text()).get('source_hash')!=key:
        raise RuntimeError('Incomplete run has different frozen sources/configuration. Use a new output directory.')
    save_json(frozen,register)
    save_json(out/'status.json',dict(status='running',stage='memory_validation',source_hash=key))
    tasks=[(config,j,'memory',None) for j in range(config['validation_trials'])]
    memory=run_jobs(tune_task,tasks,'memory_validation',key,out,args.workers)
    tuning=pd.DataFrame([row for group in memory for row in group]);selected=choose(tuning)
    tasks=[(config,j,'velocity',selected['beta']) for j in range(config['validation_trials'])]
    velocity=run_jobs(tune_task,tasks,'velocity_validation',key,out,args.workers)
    tuning=pd.concat([tuning,pd.DataFrame([row for group in velocity for row in group])],ignore_index=True)
    selected=choose(tuning)
    selected['beta_boundary']=selected['beta'] in (min(config['betas']),max(config['betas']))
    selected['window_boundary']=selected['window'] in (min(config['windows']),max(config['windows']))
    selected['lag_boundary']=selected['lag'] in (min(config['lags']),max(config['lags']))
    selected['damping_boundary']=selected['damping'] in (min(config['damping']),max(config['damping']))
    save_json(out/'selection.json',dict(selected=selected,source_hash=key,
        test_observations_used=False,
        grid_expansion='Initial validation selected lag 16 at the upper boundary; added lag 32 using validation only before any final test. Other grids unchanged; all remaining boundary selections explicit.',
        policy_scope='One setting shared across all test rates, laws and input frames'))
    tuning.to_csv(out/'tuning.csv',index=False)
    tuning_summary=[]
    for keys,g in tuning.groupby(['stage','family','candidate'],sort=False):
        # Law/frame cells in a trial share the Haar state and noise stream.
        # Average within each trial before computing a standard error.
        trial_means=g.groupby('trial').terminal_error.mean()
        tuning_summary.append(dict(stage=keys[0],family=keys[1],candidate=keys[2],
            mean_error=float(trial_means.mean()),se=float(trial_means.std(ddof=1)/np.sqrt(len(trial_means))),
            evaluations=len(g),independent_trials=len(trial_means),total_seconds=float(g.seconds.sum())))
    tuning_summary=pd.DataFrame(tuning_summary)
    tuning_summary.to_csv(out/'tuning_summary.csv',index=False)
    save_json(out/'status.json',dict(status='running',stage='final_holdout',source_hash=key))
    results=run_jobs(test_task,[(config,j,selected) for j in range(config['test_trials'])],
                     'test',key,out,args.workers)
    frames={name:pd.DataFrame([r for group in results for r in group[name]]) for name in ['metrics','paths','bounds','cost']}
    if config['posterior_trials']:
        pilots=run_jobs(posterior_task,[(config,j,selected) for j in range(config['posterior_trials'])],
                        'posterior',key,out,args.workers)
        frames['posterior']=pd.DataFrame([r for group in pilots for r in group])
    else:frames['posterior']=pd.DataFrame()
    frames.update(summaries(frames,config))
    if sources()!=source:raise RuntimeError('A numerical source changed during execution. Preserve and rerun with the final sources.')
    tables={name:sha(out/name) for name in ['tuning.csv','tuning_summary.csv']}
    for name,frame in frames.items():
        path=out/f'{name}.csv';frame.to_csv(path,index=False);tables[path.name]=sha(path)
    b=frames['bounds'];applicable=b[b.theorem_applicable.astype(bool)];valid=applicable[applicable.valid.astype(bool)]
    posterior=frames['posterior']
    checks=dict(terminal_rows=len(frames['cost']),numerical_finite=bool(np.isfinite(frames['cost'].terminal_error).all()),
        theorem_applicable=len(applicable),theorem_valid=len(valid),
        valid_certificate_violations=int(valid.bound_violated.sum()),
        posterior_evaluations=len(posterior),
        posterior_sampling_failures=int((~posterior.sampling_ok.astype(bool)).sum()) if len(posterior) else 0,
        prior_sampling_failures=int((~posterior.prior_sampling_ok.astype(bool)).sum()) if len(posterior) else 0)
    record=dict(**register,selected=selected,checks=checks,tables=tables,workers=args.workers,
        inference_scope='Unknown-motion Bingham laws are approximate; no credible-coverage theorem or learned-motion tracking certificate',
        innovation_scope='Terminal prior trace identity across independent trajectories; temporal correlation not assessed in this bounded pilot',
        noise_learning='Fixed true variance only; separate noise-fit ablation required',
        coverage_scope='Eight independent trajectories per law/method, with Wilson intervals; no calibration claim')
    save_json(meta_path,record)
    save_json(out/'status.json',dict(status='completed',source_hash=key,checks=checks,
        checkpoints_2_3='implemented and checked',empirical_claim='See paired.csv; software correctness does not imply superiority'))
    lines=['# Conditional motion and memory findings','',
        'All test filters receive identical observations within each trajectory. Unknown-motion',
        'settings are selected once using independent validation over a declared mixture.',
        f'Selected beta={selected["beta"]}, W={selected["window"]}, lag={selected["lag"]}, damping={selected["damping"]}.',
        f'Boundary selections: beta={selected["beta_boundary"]}, W={selected["window_boundary"]}, lag={selected["lag_boundary"]}, damping={selected["damping_boundary"]}.','',
        'The primary condition is fast drift with sensor-fixed inputs and terminal chordal',
        'error; other paired intervals are exploratory pointwise intervals. No method is',
        'selected using a hidden test rate. Partial motion is a declared synthetic control.','']
    primary=frames['paired'][frames['paired'].primary.astype(bool)]
    for r in primary.itertuples():
        outcome='supports lower error' if r.low>0 else 'supports higher error' if r.high<0 else 'is inconclusive'
        lines.extend([f'The primary paired gain (tuned forgetting minus causal velocity) is {r.mean_gain:.5f}',
                      f'with 95% interval [{r.low:.5f}, {r.high:.5f}]; it {outcome}.',''])
    lines.extend(['See summary.csv and paths.csv for all regimes and initial/forecast errors;',
        'cost.csv separates prediction from correction, and tuning.csv preserves every',
        'validation candidate. bounds.csv retains failed sufficient conditions and excludes',
        'learned rotations, finite windows, and corrupted observations from theorem claims.',
        f'Valid certificate violations: {checks["valid_certificate_violations"]}/{checks["theorem_valid"]}.','',
        'posterior.csv and coverage.csv retain ball, HPD, clean-response and noisy-observation',
        'coverage and widths separately, including every capped sampler failure. The terminal',
        'innovation check uses independently sampled priors before correction. It is a trace',
        'identity; it is not a chi-squared rule. Temporal innovation correlation and noise',
        'learning are not evaluated by this bounded terminal pilot.',''])
    (out/'findings.md').write_text('\n'.join(lines))
    print(json.dumps(dict(selected=selected,checks=checks),indent=2))
    print(frames['summary'][frames['summary'].input_frame=='sensor'].to_string(index=False))


if __name__=='__main__':main()
