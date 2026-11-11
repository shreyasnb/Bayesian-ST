"""Separate final-time selection audit on the frozen stationary observations.

This module changes no tracker or original result. Selection receives training
measurements, known scales, the desired signature, the original initialization,
and independent measured holdouts only. Latent signatures are scoring-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .final_beamforming import (
    beam_episode, heldout_snapshots, modal_weight, covariance_power,
    covariance_validation_statistics, projector_mode,
)
from .array_geometry_study import (
    covariance_shrinkage, distortionless_weight, continuous_minimum,
)
from .final_stationary_evidence import stationary_component_path, verify_stationary_component
from benchmarks.common.great import GREAT
from benchmarks.case1.baselines import sample_output_power

OUTPUT = 'results/beamforming_comparison_audit/stationary/selection'
NEW_METHODS = ('great_window_final_db', 'great_ew_final_db',
               'great_full_history_final_db', 'covariance_loading_final_db')
SOURCES = ('lib/beamforming_stationary_selection_audit.py',
    'scripts/run_beamforming_stationary_selection_audit.py',
    'lib/final_beamforming.py','benchmarks/common/great.py',
    'lib/array_geometry_study.py','lib/beamforming_protocol.py',
    'lib/final_stationary_evidence.py','lib/final_validation.py',
    'benchmarks/case1/baselines.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def configuration(root):
    root = Path(root).resolve()
    original = verify_stationary_component(root)
    cfg = original['config']
    selected_path=next(name for name in original['artifacts'] if Path(name).name=='selected.json')
    original_settings={}
    for policy in json.loads((root/selected_path).read_text()):
        if policy['stationary']:
            loading=next(row for row in policy['policies'] if 200 in row['times'])
            original_settings[str(policy['n'])]=dict(great_window=policy['great_window'],
                great_ew=policy['great_ew'],covariance_loading_alpha=loading['alpha'])
    return dict(schema=1,n=[32,64,128],T=200,t=200,validation_simulations=64,
        test_simulations=256,sigma2=.2,tau2=1.,beta=1.,
        original_configuration=cfg,original_configuration_sha256=fingerprint(cfg),
        original_settings=original_settings,
        windows='Original n/2,n,2n settings plus explicit full-history window 200',
        step_multipliers=[.02,.08,.24],inner_steps=[1,3],
        ew_beta=1.,ew_step_memory='2*n',covariance_loading_grid_points=33,
        covariance_loading_bounds=[0.,1.],
        validation_criterion='Arithmetic mean across 64 validation simulations of 10*log10(independent measured heldout output power) at t=200 only; no oracle normalization',
        test_endpoint='true_ratio_db at t=200',
        numerical_tolerances={key:cfg[key] for key in ('moment_tolerance','moment_max_order','moment_fallback_max_order')},
        original_results_unchanged=True)


def inputs(root):
    root=Path(root).resolve();meta=verify_stationary_component(root)
    manifest=stationary_component_path(root)
    names=[str(manifest.relative_to(root))]
    names.extend(name for name in meta['artifacts'] if Path(name).name in
                 ('stationary.csv','selected.json','stories.npz','validation_curves.csv'))
    return {name:sha(root/name) for name in names}


def reference_frame(root):
    root=Path(root).resolve();meta=verify_stationary_component(root)
    path=next(name for name in meta['artifacts'] if Path(name).name=='stationary.csv')
    frame=pd.read_csv(root/path)
    frame=frame[frame.t.eq(200)].copy()
    assert len(frame)==3*256*10 and frame.groupby(['n','method']).trial.nunique().eq(256).all()
    return frame


def candidate_grid(n,cfg):
    out=[]
    for family in ('great_window','great_ew'):
        windows=sorted(set([n//2,n,2*n,200])) if family=='great_window' else [None]
        for window in windows:
            memory=window if family=='great_window' else 2*n
            for multiplier in cfg['step_multipliers']:
                for inner in cfg['inner_steps']:
                    out.append(dict(candidate_id=f'{family}_{len(out):02d}',family=family,
                        window=window,beta=None if family=='great_window' else 1.,
                        stepsize=float(multiplier/(memory*(cfg['sigma2']+cfg['tau2']))),
                        step_multiplier=multiplier,inner_steps=inner))
    return out


def replay_stream(cfg,n,trial,split):
    stream=beam_episode(cfg['original_configuration'],n,'static',trial,split,
                        stationary=True,length=200)
    assert stream['U'].shape==(200,n)
    assert np.array_equal(stream['sigma2'],np.full(200,.2))
    assert np.array_equal(stream['tau2'],np.ones(200))
    return stream


def observed_validation(stream):
    """Discard latent fields before any candidate score or loading fit."""
    Z=heldout_snapshots(stream,200,256)
    R=Z.T@Z.conj()/len(Z)
    return dict(U=stream['U'],a=stream['a'],initial=stream['initial'],
                sigma2=.2,tau2=1.,holdout=Z,measured_covariance=R)


def great_final(U,initial,candidate):
    tracker=GREAT(U.shape[1],1,complex_data=True,initial_basis=initial[:,None],
        **{key:candidate[key] for key in ('window','beta','stepsize','inner_steps')})
    for u in U:
        tracker.update(u)
    return tracker.basis[:,0].copy()


def validation_trial(cfg,n,trial):
    stream=replay_stream(cfg,n,trial,'validation')
    data=observed_validation(stream)
    rows=[]
    for candidate in candidate_grid(n,cfg):
        h=great_final(data['U'],data['initial'],candidate)
        w=modal_weight(h,data['a'],data['sigma2'],data['tau2'])
        measured=sample_output_power(w,data['holdout'])
        assert np.isfinite(measured) and measured>0
        rows.append(dict(n=n,trial=trial,t=200,T=200,**candidate,
            measured_power=float(measured),measured_log_power_db=float(10*np.log10(measured)),
            numerical_ok=True))
    S=data['U'].T@data['U'].conj()/200
    statistics=covariance_validation_statistics(S,data['a'],data['measured_covariance'])
    hashes=dict(n=n,trial=trial,split='validation',T=200,
        observation_sha256=array_sha(data['U']),initial_sha256=array_sha(data['initial']),
        holdout_sha256=array_sha(data['holdout']),desired_sha256=array_sha(data['a']))
    return rows,statistics,hashes


def measured_loading_powers(alpha,statistics):
    """One measured power per validation simulation, with no latent input."""
    values=np.asarray([row['values'] for row in statistics])
    p=np.asarray([row['p'] for row in statistics])
    quadratic=np.asarray([row['quadratic'] for row in statistics])
    d=(1-alpha)*values+alpha*values.mean(axis=1)[:,None]
    if np.any(d<=1e-13*np.max(values,axis=1)[:,None]) or not np.isfinite(d).all():
        return np.full(len(statistics),np.inf)
    inverse=1/d
    numerator=np.einsum('ei,eij,ej->e',inverse,quadratic,inverse,optimize=True)
    powers=numerator/np.sum(p*inverse,axis=1)**2
    if np.any(powers<=0) or not np.isfinite(powers).all():
        return np.full(len(statistics),np.inf)
    return powers


def select_dimension(cfg,n,rows,statistics):
    frame=pd.DataFrame(rows)
    grouped=frame.groupby('candidate_id',sort=False)
    summaries=[]
    for candidate_id,group in grouped:
        assert len(group)==64 and set(group.trial)==set(range(64))
        record=group.iloc[0]
        values=group.measured_log_power_db.to_numpy()
        summary=dict(n=n,candidate_id=candidate_id,
            **{key:None if pd.isna(record[key]) else record[key].item() if isinstance(record[key],np.generic) else record[key]
               for key in ('family','window','beta','stepsize','step_multiplier','inner_steps')},
            **sample_interval(values,64))
        summaries.append(summary)
    by_id={item['candidate_id']:item for item in candidate_grid(n,cfg)}
    selected={}
    for method,family,window in [('great_window_final_db','great_window',None),
            ('great_ew_final_db','great_ew',None),('great_full_history_final_db','great_window',200)]:
        candidates=[row for row in summaries if row['family']==family and (window is None or row['window']==window)]
        best=min(candidates,key=lambda row:(row['mean'],row['candidate_id']))
        selected[method]=dict(by_id[best['candidate_id']],validation_mean_db=best['mean'],
                             validation_simulations=64,selection_t=200)
    def objective(alpha):
        powers=measured_loading_powers(alpha,statistics)
        return float(np.mean(10*np.log10(powers))) if np.isfinite(powers).all() else np.inf
    fit=continuous_minimum(objective,33)
    selected['covariance_loading_final_db']=dict(alpha=fit['value'],beta=1.,
        validation_mean_db=fit['loss'],validation_simulations=64,selection_t=200,
        grid_points=33,bounds=[0.,1.],optimizer_evaluations=fit['evaluations'])
    for index,alpha in enumerate(fit['grid']+[fit['value']]):
        identifier='covariance_loading_selected' if index==33 else f'covariance_loading_grid_{index:02d}'
        powers=measured_loading_powers(alpha,statistics)
        assert np.isfinite(powers).all()
        values=10*np.log10(powers)
        for trial,value in enumerate(values):
            rows.append(dict(n=n,trial=trial,t=200,T=200,candidate_id=identifier,
                family='covariance_loading',alpha=float(alpha),beta=1.,window=None,
                stepsize=None,step_multiplier=None,inner_steps=None,
                measured_power=float(powers[trial]),measured_log_power_db=float(value),numerical_ok=True))
        summaries.append(dict(n=n,candidate_id=identifier,family='covariance_loading',alpha=float(alpha),
            beta=1.,window=None,stepsize=None,step_multiplier=None,inner_steps=None,
            **sample_interval(values,64)))
    np.testing.assert_allclose(selected['covariance_loading_final_db']['validation_mean_db'],
                               summaries[-1]['mean'],rtol=1e-13,atol=1e-13)
    return dict(n=n,settings=selected,loading_search=fit),rows,summaries


def test_trial(cfg,n,trial,selection,old_rows):
    stream=replay_stream(cfg,n,trial,'test')
    U,a,initial=stream['U'],stream['a'],stream['initial']
    S=U.T@U.conj()/200
    _,mode,_=projector_mode(S)
    mode_w=modal_weight(mode,a,.2,1.)
    h=stream['h'][-1]
    oracle=covariance_power(modal_weight(h,a,.2,1.),h,.2,1.)
    mode_power=covariance_power(mode_w,h,.2,1.)
    old_mode=old_rows[old_rows.method.eq('mode')].iloc[0]
    np.testing.assert_allclose(10*np.log10(mode_power/oracle),old_mode.true_ratio_db,rtol=1e-11,atol=1e-11)
    rows=[];cached={};reference_errors={}
    original_settings=cfg['original_settings'][str(n)]
    for family in ('great_window','great_ew'):
        chosen=original_settings[family]
        key=fingerprint({key:chosen[key] for key in ('window','beta','stepsize','inner_steps')})
        if key not in cached:
            line=great_final(U,initial,chosen)
            cached[key]=modal_weight(line,a,.2,1.)
        replay_db=10*np.log10(covariance_power(cached[key],h,.2,1.)/oracle)
        recorded=float(old_rows[old_rows.method.eq(family)].true_ratio_db.iloc[0])
        np.testing.assert_allclose(replay_db,recorded,rtol=1e-11,atol=1e-11)
        reference_errors[family+'_reference_db_error']=float(abs(replay_db-recorded))
    old_weight=distortionless_weight(covariance_shrinkage(S,original_settings['covariance_loading_alpha']),a)
    replay_db=10*np.log10(covariance_power(old_weight,h,.2,1.)/oracle)
    recorded=float(old_rows[old_rows.method.eq('covariance_loading')].true_ratio_db.iloc[0])
    np.testing.assert_allclose(replay_db,recorded,rtol=1e-11,atol=1e-11)
    reference_errors['covariance_loading_reference_db_error']=float(abs(replay_db-recorded))
    for method in NEW_METHODS:
        chosen=selection['settings'][method]
        if method.startswith('great'):
            key=fingerprint({key:chosen[key] for key in ('window','beta','stepsize','inner_steps')})
            if key not in cached:
                line=great_final(U,initial,chosen)
                cached[key]=modal_weight(line,a,.2,1.)
            w=cached[key]
        else:
            w=distortionless_weight(covariance_shrinkage(S,chosen['alpha']),a)
        power=covariance_power(w,h,.2,1.)
        rows.append(dict(n=n,trial=trial,t=200,T=200,method=method,source='final_time_selection_audit',
            true_power=float(power),oracle_power=float(oracle),true_ratio_db=float(10*np.log10(power/oracle)),
            numerical_ok=bool(np.isfinite(w).all()),unit_gain_error=float(abs(np.vdot(w,a)-1))))
    hashes=dict(n=n,trial=trial,split='test',T=200,observation_sha256=array_sha(U),
        initial_sha256=array_sha(initial),desired_sha256=array_sha(a),**reference_errors,mode_reference_db_error=float(
            abs(10*np.log10(mode_power/oracle)-old_mode.true_ratio_db)))
    return rows,hashes


def sample_interval(values,count=256):
    values=np.asarray(values,float)
    assert len(values)==count and np.isfinite(values).all()
    mean=float(values.mean());half=float(student_t.ppf(.975,count-1)*values.std(ddof=1)/np.sqrt(count))
    return dict(simulations=count,mean=mean,lower=mean-half,upper=mean+half)


def summaries(frame):
    rows=[];pairs=[]
    for n,group in frame.groupby('n'):
        own=group[group.method.eq('mean')].set_index('trial').true_ratio_db
        for method,values in group.groupby('method'):
            values=values.set_index('trial').true_ratio_db
            rows.append(dict(n=int(n),t=200,method=method,**sample_interval(values)))
            if method!='mean':
                pairs.append(dict(n=int(n),t=200,reference=method,
                    **sample_interval(values.loc[own.index]-own),
                    sign='Reference minus original posterior mean; positive favors posterior mean'))
    return pd.DataFrame(rows),pd.DataFrame(pairs)


def load_stationary_selection_audit(root):
    """Read completed new-only audit exports for notebook use."""
    root=Path(root).resolve();folder=root/OUTPUT
    meta=json.loads((folder/'manifest.json').read_text())
    assert meta['status']=='completed'
    for mapping in (meta['sources'],meta['inputs'],meta['artifacts']):
        for name,expected in mapping.items():
            assert sha(root/name)==expected, 'Changed selection-audit evidence: '+name
    result=dict(config=meta['config'],manifest=meta,source_hashes=meta['sources'],original_inputs=meta['inputs'],input_hashes={str((folder/'manifest.json').relative_to(root)):sha(folder/'manifest.json')})
    result['input_hashes'].update(meta['artifacts'])
    for key in ('test_metrics','test_summary','paired_vs_mean','validation_trials','validation_candidates'):
        result[key]=pd.read_csv(folder/(key+'.csv'))
    result['selected_settings']=json.loads((folder/'selected_settings.json').read_text())
    return result
