"""Existing-data comparisons, including both original GREAT references.

This reader does not generate data, replay estimators, retune parameters or
change a scientific producer. All four moving-data aggregation choices are
retained; their orderings never determine which one is reported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .beamforming_moving_evidence import _verified_original
from .final_stationary_evidence import stationary_component_path, verify_stationary_component
from .final_validation import HISTORICAL_SOURCE_PROOF, HISTORICAL_SOURCE_PROOF_SHA256


MOVING_METHODS=('mean','mode','isotropic','great_window','great_ew')
STATIONARY_METHODS=('mean','mode','isotropic','modal_loading','covariance_loading',
                    'great_window','great_ew','grouse','ledoit_wolf','oracle')
AGGREGATIONS=(
    dict(aggregation='mean_db',window='all',start_t=8,reporting_points=32,primary=True),
    dict(aggregation='mean_db',window='t_ge_32',start_t=32,reporting_points=29,primary=False),
    dict(aggregation='log_mean_ratio',window='all',start_t=8,reporting_points=32,primary=False),
    dict(aggregation='log_mean_ratio',window='t_ge_32',start_t=32,reporting_points=29,primary=False),
)
OUTPUT='results/beamforming_comparison_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(test,message):
    if not test:raise RuntimeError(message)


def _digest(value):
    return value['sha256'] if isinstance(value,dict) else value


def _artifact(meta,name):
    matches=[path for path in meta['artifacts'] if Path(path).name==name]
    _require(len(matches)==1,'Expected one verified artifact: '+name)
    return matches[0]


def _interval(sample):
    sample=np.asarray(sample,float)
    _require(len(sample)==256 and np.isfinite(sample).all(),
             'Comparison intervals require all 256 finite independent simulations')
    mean=float(sample.mean());half=float(student_t.ppf(.975,255)*sample.std(ddof=1)/16.)
    return dict(simulations=256,mean=mean,lower=mean-half,upper=mean+half)


def _summary(frame,keys):
    rows=[]
    for values,group in frame.groupby(keys,sort=True,dropna=False):
        values=values if isinstance(values,tuple) else (values,)
        group=group.sort_values('trial')
        np.testing.assert_array_equal(group.trial,np.arange(256))
        rows.append(dict(zip(keys,values),**_interval(group.value_db)))
    return pd.DataFrame(rows)


def _paired(frame,keys,methods):
    rows=[];differences=[]
    for values,group in frame.groupby(keys,sort=True,dropna=False):
        values=values if isinstance(values,tuple) else (values,)
        attributes=dict(zip(keys,values))
        wide=group.pivot(index='trial',columns='method',values='value_db')
        np.testing.assert_array_equal(wide.index,np.arange(256))
        for reference in methods:
            if reference=='mean':continue
            delta=wide[reference]-wide['mean'];stats=_interval(delta)
            rows.append(dict(attributes,reference=reference,gain=stats['mean'],**stats))
            differences.extend(dict(attributes,reference=reference,trial=int(trial),
                difference_db=float(value)) for trial,value in delta.items())
    return pd.DataFrame(rows),pd.DataFrame(differences)


def _check_raw(frame,methods,dimensions,times):
    _require(not frame.duplicated(['n','condition','t','method','trial']).any(),
             'Duplicate comparison simulation/checkpoint')
    _require(set(frame.method)==set(methods) and set(frame.n)==set(dimensions),
             'Missing declared comparison method/dimension')
    np.testing.assert_array_equal(np.sort(frame.t.unique()),times)
    _require(frame.sigma2.eq(.2).all() and frame.tau2.eq(1.).all(),
             'The generating variances must remain fixed')
    _require(frame.numerical_ok.astype(bool).all() and frame.mean_ok.astype(bool).all(),
             'Unresolved original comparison records must not be dropped')
    _require(np.isfinite(frame.true_ratio_db).all(),'Missing exact-covariance comparison')
    ratio=frame.true_power/frame.oracle_power
    np.testing.assert_allclose(frame.true_ratio,ratio,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(frame.true_ratio_db,10*np.log10(ratio),rtol=1e-12,atol=1e-12)


def _moving(root):
    meta,proof,sources=_verified_original(root)
    manifest='results/beamforming/data/paper/performance/manifest.json'
    raw_name=_artifact(meta,'moving_checkpoints.csv.gz')
    selected_name=_artifact(meta,'selected.json')
    inputs={manifest:sha(root/manifest),raw_name:_digest(meta['artifacts'][raw_name]),
        selected_name:_digest(meta['artifacts'][selected_name]),
        HISTORICAL_SOURCE_PROOF:HISTORICAL_SOURCE_PROOF_SHA256}
    inputs.update({record['path']:record['sha256'] for record in sources.values()})
    raw=pd.read_csv(root/raw_name)
    raw=raw[raw.condition.isin(['smooth','abrupt'])&raw.n.eq(32)&raw.method.isin(MOVING_METHODS)].copy()
    _check_raw(raw,MOVING_METHODS,[32],np.arange(8,257,8))
    _require(len(raw)==2*5*32*256,'Incomplete original moving comparison')
    shared=raw[raw.method.isin(('mean','mode','isotropic','great_ew'))]
    _require(shared.beta.eq(.97).all(),'Original common Bingham/EW forgetting factor changed')
    _require(raw[raw.method.isin(('mean','mode','isotropic'))].same_parameter.astype(bool).all(),
             'Mean, mode and isotropic mean must use the same posterior')
    selected=json.loads((root/selected_name).read_text());settings={}
    for condition in ('smooth','abrupt'):
        matches=[policy for policy in selected if policy['condition']==condition
                 and policy['n']==32 and policy['stationary'] is False]
        _require(len(matches)==1,'Missing original moving settings')
        policy=matches[0];settings[condition]=policy
        _require(policy['shared_beta']==.97 and policy['great_ew']['beta']==.97,
                 'Original matched EW settings changed')
        for method in ('great_window','great_ew'):
            rows=raw[raw.condition.eq(condition)&raw.method.eq(method)]
            params=policy[method]
            _require(rows.gradient_steps.eq(params['inner_steps']).all()
                and np.allclose(rows.gradient_stepsize,params['stepsize'],rtol=1e-13,atol=0.),
                'Saved GREAT records disagree with their selected settings')
            if method=='great_window':
                _require(rows.optimization_window.eq(params['window']).all(),
                         'Saved GREAT window changed')
    checkpoint=raw[['condition','n','trial','t','method','true_ratio_db']].rename(columns={'true_ratio_db':'value_db'})
    checkpoint_summary=_summary(checkpoint,['condition','n','t','method'])
    checkpoint_pairs,checkpoint_differences=_paired(checkpoint,['condition','n','t'],MOVING_METHODS)
    aggregates=[]
    for specification in AGGREGATIONS:
        subset=raw[raw.t.ge(specification['start_t'])]
        for values,group in subset.groupby(['condition','trial','method'],sort=True):
            np.testing.assert_array_equal(np.sort(group.t),np.arange(specification['start_t'],257,8))
            value=(float(group.true_ratio_db.mean()) if specification['aggregation']=='mean_db'
                   else float(10*np.log10(group.true_ratio.mean())))
            aggregates.append(dict(condition=values[0],trial=int(values[1]),method=values[2],
                value_db=value,**specification))
    aggregates=pd.DataFrame(aggregates)
    keys=['condition','aggregation','window','start_t','reporting_points','primary']
    summary=_summary(aggregates,[*keys,'method'])
    paired,differences=_paired(aggregates,keys,MOVING_METHODS)
    curves={condition:[dict(method=method,points=group[['t','mean','lower','upper']].to_numpy())
             for method in MOVING_METHODS
             for group in [checkpoint_summary[checkpoint_summary.condition.eq(condition)
                 &checkpoint_summary.method.eq(method)].sort_values('t')]]
            for condition in ('smooth','abrupt')}
    config=dict(n=32,T=256,sigma2=.2,tau2=1.,beta=.97,methods=list(MOVING_METHODS),
        conditions=['smooth','abrupt'],reporting_t=list(range(8,257,8)),event_abrupt=129,
        validation_simulations=64,test_simulations=256,aggregations=list(AGGREGATIONS),
        primary='mean_db/all: mean individual true_ratio_db over all 32 reporting measurements within each simulation',
        mean_db='Arithmetic mean of 10*log10(true_power/oracle_power) within each simulation.',
        log_mean_ratio='10*log10(arithmetic mean of true_power/oracle_power within each simulation).',
        all='t=8,16,...,256 (32 reporting measurements)',
        t_ge_32='t=32,40,...,256 (29 reporting measurements); omits only t=8,16,24',
        selection='Unchanged original measured held-out-power selection, on 64 separate validation simulations.',
        great_ew_weights=meta['config']['great_ew_weights'])
    return dict(config=config,curves=curves,time=np.arange(8,257,8),raw_stats=raw,
        selected_settings=settings,checkpoint_summary=checkpoint_summary,
        paired_checkpoint_summary=checkpoint_pairs,paired_checkpoint_differences=checkpoint_differences,
        simulation_aggregates=aggregates,aggregate_summary=summary,paired_aggregate_summary=paired,
        paired_aggregate_differences=differences,primary_summary=summary[summary.primary].reset_index(drop=True),
        paired_primary_summary=paired[paired.primary].reset_index(drop=True),
        input_hashes=inputs,source_hashes=meta['sources'],
        provenance=dict(original_manifest=manifest,original_manifest_sha256=inputs[manifest],
            original_source_commit=proof['source_commit'],verified_source_files=sources,
            original_config=meta['config'],raw_input=raw_name,
            scope='Existing smooth/abrupt measurements and original settings; no replays, tuning or simulation.',
            aggregation_policy='All four user-specified aggregations retained regardless of method ranking.'),
        checks=dict(status='passed',raw_rows=len(raw),checkpoint_summaries=len(checkpoint_summary),
            aggregate_summaries=len(summary),all_simulation_ids_verified=True,
            original_sources_and_artifacts_verified=True,exact_covariance_power_identity_verified=True,
            new_simulations=0,new_replays=0,new_parameter_selections=0))


def _stationary(root):
    meta=verify_stationary_component(root,stage='paper')
    manifest=stationary_component_path(root,stage='paper').relative_to(root).as_posix()
    raw_name=_artifact(meta,'stationary.csv');selected_name=_artifact(meta,'selected.json')
    raw=pd.read_csv(root/raw_name)
    _require(raw.condition.eq('static').all() and raw['T'].eq(200).all(),
             'The stationary comparison requires the exact fixed-T200 component')
    _check_raw(raw,STATIONARY_METHODS,[32,64,128],np.array([25,50,100,200]))
    _require(len(raw)==3*10*4*256,'Incomplete fixed-T200 stationary comparison')
    _require(raw[raw.method.isin(('mean','mode','isotropic','great_ew'))].beta.eq(1.).all(),
             'Stationary Bingham/EW comparisons require beta=1')
    checkpoint=raw[['n','trial','t','method','true_ratio_db']].rename(columns={'true_ratio_db':'value_db'})
    summary=_summary(checkpoint,['n','t','method'])
    pairs,differences=_paired(checkpoint,['n','t'],STATIONARY_METHODS)
    inputs={manifest:sha(root/manifest)}
    inputs.update({path:_digest(value) for path,value in meta['artifacts'].items()})
    inputs.update({name:_digest(value) for name,value in meta['sources'].items()})
    return dict(config=dict(T=200,primary_t=200,dimensions=[32,64,128],sigma2=.2,tau2=1.,
        beta=1.,methods=list(STATIONARY_METHODS),main_methods=list(MOVING_METHODS),
        reporting_t=[25,50,100,200],validation_simulations=64,test_simulations=256,
        endpoint='true_ratio_db',primary='Corrected estimate after exactly t=T=200 training measurements'),
        raw_stats=raw,selected_settings=json.loads((root/selected_name).read_text()),
        checkpoint_summary=summary,paired_checkpoint_summary=pairs,
        paired_checkpoint_differences=differences,
        primary_summary=summary[summary.t.eq(200)].reset_index(drop=True),
        paired_primary_summary=pairs[pairs.t.eq(200)].reset_index(drop=True),
        input_hashes=inputs,source_hashes=meta['sources'],
        provenance=dict(original_manifest=manifest,original_manifest_sha256=inputs[manifest],
            raw_input=raw_name,original_config=meta['config'],
            scope='Strict verified fixed-T200 sidecar, all ten original methods, no replays or retuning.'),
        checks=dict(status='passed',raw_rows=len(raw),checkpoint_summaries=len(summary),
            primary_summaries=int(summary.t.eq(200).sum()),all_simulation_ids_verified=True,
            strict_T200_sources_and_artifacts_verified=True,exact_covariance_power_identity_verified=True,
            new_simulations=0,new_replays=0,new_parameter_selections=0))


def load_existing_beamforming_comparisons(root):
    """Return moving aggregation checks and fixed-T200 comparisons as data."""
    root=Path(root).resolve()
    return dict(moving=_moving(root),stationary=_stationary(root))
