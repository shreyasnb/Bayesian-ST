"""Exact-seed dense replay and diagnostics for the original contact study.

The original test routine is reused without changing its estimator, plant,
selection, action times or tolerances. Only checkpoint retention expands from
six rows to all 48 original action times. Read-only observers retain its actual
generated arrays and native moment certificates for additional diagnostics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, skew

from .final_validation import require_gate, verify_component


OUTPUT='results/contact_transport_diagnostics'
TIMES=tuple(range(4,193,4))
METHODS=('mean','mode','isotropic','regularized','great_window','great_ew','mean_identity','mode_identity')
MAIN_METHODS=('mean','mode','iso','mean_identity')
SCALE=1e6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value,message):
    if not value:raise RuntimeError(message)


def _array_hash(value):
    a=np.ascontiguousarray(value)
    digest=hashlib.sha256(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def _interval(values):
    a=np.asarray(values,float)
    _require(len(a)==256 and np.isfinite(a).all(),'All 256 finite simulations are required')
    mean=float(a.mean());half=float(student_t.ppf(.975,255)*a.std(ddof=1)/16.)
    return dict(simulations=256,mean=mean,lower=mean-half,upper=mean+half)


def load_context(root):
    root=Path(root).resolve();require_gate(root)
    meta=verify_component(root,'contact','paper','performance')
    cfg=meta['config']
    for key,value in dict(n=3,k=2,T=192,dt=.05,v0=.02,resistance=10.,radius=.05,
        sigma2=.004**2,action_every=4,validation_episodes=64,test_episodes=256,
        moment_tolerance=5e-8,algebra_tolerance=1e-10).items():
        _require(cfg.get(key)==value,'Original contact setting changed: '+key)
    for name,digest in meta['sources'].items():
        _require(sha(root/name)==digest,'Original contact source changed: '+name)
    def artifact(basename):
        names=[name for name in meta['artifacts'] if Path(name).name==basename]
        _require(len(names)==1,'Expected one original '+basename)
        return names[0]
    names={name:artifact(name) for name in ('metrics.csv','selected.json','story.npz')}
    selection=json.loads((root/names['selected.json']).read_text())
    policy=next(row for row in selection if row['condition']=='sphere')
    _require(policy['beta']==.9,'The original selected sphere forgetting factor changed')
    raw=pd.read_csv(root/names['metrics.csv'])
    with np.load(root/names['story.npz'],allow_pickle=False) as archive:
        story={name:archive[name] for name in archive.files}
    manifest='results/contact/data/paper/performance/manifest.json'
    inputs={manifest:sha(root/manifest),**{name:sha(root/name) for name in names.values()}}
    sources={name:sha(root/name) for name in (
        'lib/contact_transport_diagnostics.py','scripts/run_contact_transport_diagnostics.py',
        *meta['sources'])}
    return dict(root=root,meta=meta,config=cfg,selection=selection,policy=policy,
        original=raw,story=story,input_hashes=inputs,source_hashes=sources)


def declaration(context):
    cfg=context['config']
    return dict(schema=1,condition='sphere',n=3,k=2,T=192,dt=.05,v0=.02,
        resistance=10.,radius=.05,sigma2=cfg['sigma2'],beta=.9,
        action_times=list(TIMES),original_saved_checkpoints=[4,16,32,64,128,192],
        methods=list(METHODS),test_simulations=256,original_validation_simulations=64,
        validation_seed=cfg['validation_seed'],test_seed=cfg['test_seed'],
        stage_seed_offset=cfg['stage_seed_offset'],moment_tolerance=cfg['moment_tolerance'],
        algebra_tolerance=cfg['algebra_tolerance'],selected_settings=context['policy'],
        source_hashes=context['source_hashes'],input_hashes=context['input_hashes'],
        change_scope='Only checkpoint_times used by test_episode to retain output rows changes; all 48 action times were already evaluated by the original producer.',
        excitation_definition='S_t=beta G_t S_(t-1) G_t^T+v_t v_t^T; s_t=beta s_(t-1)+1; H_t=S_t/(sigma2*s_t). Eigenvalues are ascending; this dimensionless observable input Gram matrix is not called Fisher information.',
        curvature_definition='For ascending eigenvalues a1<=a2<=a3 of A_t, normal-density tangent curvature eigenvalues are 2(a2-a1),2(a3-a1). These are dimensionless in unit-sphere tangent coordinates (inverse squared radians for angular coordinates).',
        projector_definition='P_t is the true sensor-frame tangent-plane projector; world P=R_t P_t R_t^T, G_t=R_t^T R_(t-1). Increments start at t=2; no state at t=0 is invented.',
        frame_speed_definition='acos(clip((trace(G_t)-1)/2,-1,1))/dt. R and G are common deterministic measured-frame schedules, not independent Monte Carlo draws.',
        averaging='episode_average at t=192 averages all 48 action times within each simulation; checkpoint rows are never substituted.',
        late_change='E_192-E_128 within each simulation, in (mm/s)^2; positive means a late increase.',
        native_accuracy='Original fast_moments(checked=True), doubled angular orders, unchanged 5e-8 mean and log-normalizer resolution criteria.')


def _parity(context,trial,frame):
    old=context['original'];old=old[old.condition.eq('sphere')&old.trial.eq(trial)]
    keys=['condition','trial','method','t','endpoint']
    joined=old.merge(frame,on=keys,suffixes=('_original','_replay'),validate='one_to_one')
    _require(len(joined)==len(old)==56,'All six checkpoints and whole-run endpoints for eight methods are required')
    absolute={};relative={}
    fields=['time_s','beta','sigma2','moment_resolution_error','response_mse','response_map_frobenius',
            'command_cost','posterior_response_excess','posterior_command_excess','command_excess_identity_error']
    tolerance=context['config']['algebra_tolerance']
    for field in fields:
        a=joined[field+'_original'].to_numpy(float);b=joined[field+'_replay'].to_numpy(float)
        np.testing.assert_allclose(b,a,rtol=tolerance,atol=tolerance)
        difference=abs(b-a);absolute[field]=float(difference.max())
        nonzero=abs(a)>0
        relative[field]=float(np.max(difference[nonzero]/abs(a[nonzero]))) if nonzero.any() else 0.
    for field in ['numerical_ok','mode_unique']:
        np.testing.assert_array_equal(joined[field+'_original'],joined[field+'_replay'])
    _require(frame.numerical_ok.astype(bool).all(),'An unresolved replay must not be dropped')
    return dict(original_rows=len(old),matched_rows=len(joined),rtol=tolerance,atol=tolerance,
        maximum_absolute_error_by_field=absolute,maximum_relative_error_on_nonzero_references_by_field=relative,
        note='Relative errors near algebraic zero are reported without a fitted denominator floor; absolute tolerance is the unchanged original algebra tolerance.')


def replay_trial(context,trial):
    import lib.final_contact as original
    capture={};generate=original.generate_contact_trial;bingham=original.bingham_records
    def observe_data(*args,**kwargs):
        result=generate(*args,**kwargs);capture['data']=result;return result
    def observe_records(*args,**kwargs):
        result=bingham(*args,**kwargs)
        capture['transported' if kwargs.get('transport',True) else 'identity']=result
        return result
    # This configuration field controls retention only; action_times already
    # uses every fourth measurement, independently of checkpoint_times.
    local=dict(context['config'],checkpoint_times=list(TIMES))
    with patch.object(original,'generate_contact_trial',observe_data),patch.object(original,'bingham_records',observe_records):
        result=original.test_episode(local,'sphere',int(trial),context['selection'])
    metrics=pd.DataFrame(result['metrics'])
    _require(len(metrics)==392,'The replay must retain 48 checkpoints plus one average for every original method')
    parity=_parity(context,trial,metrics)
    data=capture['data'];records,Apath=capture['transported'];identity_records,identity_A=capture['identity']
    np.testing.assert_array_equal(sorted(records),TIMES)
    np.testing.assert_array_equal(sorted(identity_records),TIMES)
    if trial==0:
        for key in ('U','V','G','R','P','world_normal','world_V','times'):
            np.testing.assert_array_equal(data[key],context['story']['sphere_'+key])
        np.testing.assert_array_equal(Apath,context['story']['sphere_A'])
    cfg=context['config'];R=data['R'];G=data['G'];P=data['P'];nworld=data['world_normal']
    np.testing.assert_allclose(G[1:],np.einsum('tji,tjk->tik',R[1:],R[:-1]),rtol=0.,atol=cfg['algebra_tolerance'])
    np.testing.assert_allclose(data['normal'],np.einsum('tji,tj->ti',R,nworld),rtol=0.,atol=cfg['algebra_tolerance'])
    worldP=np.eye(3)-np.einsum('ti,tj->tij',nworld,nworld)
    np.testing.assert_allclose(P,np.einsum('tji,tjk,tkl->til',R,worldP,R),rtol=0.,atol=cfg['algebra_tolerance'])
    transported=np.einsum('tij,tjk,tlk->til',G[1:],P[:-1],G[1:])
    sensor_step=np.r_[np.nan,np.linalg.norm(np.diff(P,axis=0),axis=(1,2))]
    residual=np.r_[np.nan,np.linalg.norm(P[1:]-transported,axis=(1,2))]
    world_step=np.r_[np.nan,np.linalg.norm(np.diff(worldP,axis=0),axis=(1,2))]
    np.testing.assert_allclose(residual[1:],world_step[1:],rtol=1e-10,atol=cfg['algebra_tolerance'])
    frame_speed=np.arccos(np.clip((np.trace(G,axis1=1,axis2=2)-1)/2,-1.,1.))/cfg['dt']
    physical=[];diagnostics=[];S=np.zeros((3,3));S_identity=np.zeros((3,3));mass=0.
    max_mean=max_log=0.;moment_calls=0
    for j in range(192):
        t=j+1;v=data['V'][j];S=.9*G[j]@S@G[j].T+np.outer(v,v)
        S_identity=.9*S_identity+np.outer(v,v);mass=.9*mass+1.
        physical.append(dict(trial=int(trial),t=t,time_s=float(data['times'][j]),
            frame_angular_speed_rad_s=float(frame_speed[j]),sensor_increment_fro=float(sensor_step[j]),
            transported_increment_fro=float(residual[j]),world_increment_fro=float(world_step[j]),
            normal_input_fraction=float((data['normal'][j]@S@data['normal'][j])/np.trace(S))))
        if t not in records:continue
        eigenvalues=np.linalg.eigvalsh((S+S.T)/(2*cfg['sigma2']*mass))
        identity_eigenvalues=np.linalg.eigvalsh((S_identity+S_identity.T)/(2*cfg['sigma2']*mass))
        row=dict(physical[-1],effective_input_mass=float(mass),
            input_gram_lambda_1=float(eigenvalues[0]),input_gram_lambda_2=float(eigenvalues[1]),input_gram_lambda_3=float(eigenvalues[2]),
            input_gram_identity_lambda_1=float(identity_eigenvalues[0]),input_gram_identity_lambda_2=float(identity_eigenvalues[1]),input_gram_identity_lambda_3=float(identity_eigenvalues[2]))
        for key,record in [('transported',records[t]),('identity',identity_records[t])]:
            values=np.linalg.eigvalsh(record['A']);curvature=2*(values[1:]-values[0])
            prefix='' if key=='transported' else 'identity_'
            row['curvature_'+prefix+'min']=float(curvature[0]);row['curvature_'+prefix+'max']=float(curvature[1])
            diag=record['diagnostics'];moment_calls+=1
            _require(diag['numerical_ok'] and diag['mean_resolution_error']<cfg['moment_tolerance']
                     and diag['log_normalizer_resolution_error']<cfg['moment_tolerance'],
                     'Original native moment-resolution check failed')
            for name in ['mean_resolution_error','log_normalizer_resolution_error','angular_order']:
                row[key+'_'+name]=diag[name]
            max_mean=max(max_mean,float(diag['mean_resolution_error']));max_log=max(max_log,float(diag['log_normalizer_resolution_error']))
            M=record['M'];mode=record['Pmode'];iso=record['isotropic']
            np.testing.assert_allclose(M,M.T,rtol=0.,atol=cfg['algebra_tolerance'])
            np.testing.assert_allclose(np.trace(M),2.,rtol=0.,atol=cfg['algebra_tolerance'])
            overlap=np.trace(M@mode)/2;expected=overlap*mode+(2-2*overlap)*(np.eye(3)-mode)
            np.testing.assert_allclose(iso,expected,rtol=cfg['algebra_tolerance'],atol=cfg['algebra_tolerance'])
        diagnostics.append(row)
    checks=dict(trial=int(trial),status='passed',parity=parity,native_checked_moment_calls=moment_calls,
        maximum_mean_resolution_error=max_mean,maximum_log_normalizer_resolution_error=max_log,
        original_tolerance=cfg['moment_tolerance'],all_native_moments_resolved=True,
        exact_seed_generated_arrays={key:_array_hash(data[key]) for key in ('U','V','G','R','P','world_normal','times')},
        saved_story_exact_match=bool(trial==0),saved_story_scope='Single simulation zero only, not a Monte Carlo result.',
        maximum_transport_geometry_identity_error=float(np.max(abs(residual[1:]-world_step[1:]))),
        same_observations_variance_beta_action_times=True,only_row_retention_changed=True)
    return metrics,pd.DataFrame(physical),pd.DataFrame(diagnostics),checks


def _long_summary(frame,metrics):
    rows=[]
    for t,group in frame.groupby('t',sort=True):
        for metric in metrics:
            values=group[metric].to_numpy(float)
            if not np.isfinite(values).all():continue
            stats=_interval(values);q25,q75=np.quantile(values,[.25,.75])
            rows.append(dict(t=int(t),metric=metric,**stats,q25=float(q25),q75=float(q75),
                deterministic=bool(np.ptp(values)==0.)))
    return pd.DataFrame(rows)


def _rise_distribution(values,**attributes):
    a=np.asarray(values,float);stats=_interval(a);absolute=np.sort(abs(a))[::-1]
    positive=np.sort(np.maximum(a,0))[::-1];positive_sum=float(positive.sum())
    return dict(attributes,**stats,fraction_positive=float(np.mean(a>0)),median=float(np.median(a)),
        q01=float(np.quantile(a,.01)),q05=float(np.quantile(a,.05)),q25=float(np.quantile(a,.25)),
        q75=float(np.quantile(a,.75)),q95=float(np.quantile(a,.95)),q99=float(np.quantile(a,.99)),
        minimum=float(a.min()),maximum=float(a.max()),sd=float(a.std(ddof=1)),skewness=float(skew(a,bias=False)),
        top_1pct_absolute_contribution=float(absolute[:3].sum()/absolute.sum()),
        top_5pct_absolute_contribution=float(absolute[:13].sum()/absolute.sum()),
        top_5pct_positive_contribution=float(positive[:13].sum()/positive_sum) if positive_sum else 0.,
        tail_definition='Top 1% and 5% use respectively the largest 3 and 13 absolute changes among all 256; primary statistics always retain every trial.')


def summarize(context,metrics,physical,diagnostics,trial_checks):
    summaries=[]
    for (endpoint,t,method),group in metrics.groupby(['endpoint','t','method'],sort=True):
        np.testing.assert_array_equal(np.sort(group.trial),np.arange(256))
        summaries.append(dict(condition='sphere',endpoint=endpoint,t=int(t),method='iso' if method=='isotropic' else method,
            **_interval(group.response_mse.to_numpy()*SCALE)))
    summary=pd.DataFrame(summaries);rise_rows=[];rise_trials=[]
    control=context['original'];control=control[control.condition.eq('plane')&control.endpoint.eq('checkpoint')]
    plane_rows=[]
    for condition,frame,target in [('sphere',metrics[metrics.endpoint.eq('checkpoint')],rise_rows),('plane',control,plane_rows)]:
        for method in METHODS:
            wide=frame[frame.method.eq(method)].pivot(index='trial',columns='t',values='response_mse')*SCALE
            np.testing.assert_array_equal(wide.index,np.arange(256));delta=wide[192]-wide[128]
            target.append(_rise_distribution(delta,condition=condition,method='iso' if method=='isotropic' else method))
            if condition=='sphere':
                rise_trials.extend(dict(trial=int(trial),method='iso' if method=='isotropic' else method,
                    error128_mm2_s2=float(wide.loc[trial,128]),error192_mm2_s2=float(wide.loc[trial,192]),
                    difference_mm2_s2=float(value)) for trial,value in delta.items())
    rise_trials=pd.DataFrame(rise_trials)
    wide=rise_trials.pivot(index='trial',columns='method',values='difference_mm2_s2')
    difference=wide['mean']-wide['mean_identity']
    rise_control=pd.DataFrame([_rise_distribution(difference,
        comparison='(E192-E128)_mean-(E192-E128)_mean_identity')])
    paired_control_trials=pd.DataFrame(dict(trial=wide.index,difference_mm2_s2=difference.to_numpy()))
    projector=_long_summary(physical,['sensor_increment_fro','transported_increment_fro','world_increment_fro'])
    frame_rotation=_long_summary(physical,['frame_angular_speed_rad_s'])
    excitation=_long_summary(diagnostics,['input_gram_lambda_1','input_gram_lambda_2','input_gram_lambda_3'])
    curvature=_long_summary(diagnostics,['curvature_min','curvature_max','curvature_identity_min','curvature_identity_max'])
    quality=_long_summary(diagnostics,[prefix+'_'+field for prefix in ['transported','identity']
        for field in ['mean_resolution_error','log_normalizer_resolution_error','angular_order']])
    abs_errors={field:max(check['parity']['maximum_absolute_error_by_field'][field] for check in trial_checks)
                for field in trial_checks[0]['parity']['maximum_absolute_error_by_field']}
    relative={field:max(check['parity']['maximum_relative_error_on_nonzero_references_by_field'][field] for check in trial_checks)
              for field in trial_checks[0]['parity']['maximum_relative_error_on_nonzero_references_by_field']}
    checks=dict(status='passed',trials=256,all_trials_present=True,methods=list(METHODS),
        dense_action_times=list(TIMES),native_checked_moment_calls=sum(check['native_checked_moment_calls'] for check in trial_checks),
        all_native_moments_resolved=True,original_moment_tolerance=context['config']['moment_tolerance'],
        maximum_mean_resolution_error=max(check['maximum_mean_resolution_error'] for check in trial_checks),
        maximum_log_normalizer_resolution_error=max(check['maximum_log_normalizer_resolution_error'] for check in trial_checks),
        original_checkpoint_and_average_parity=True,original_matched_rows=sum(check['parity']['matched_rows'] for check in trial_checks),
        maximum_absolute_error_by_field=abs_errors,maximum_relative_error_on_nonzero_references_by_field=relative,
        parity_atol=context['config']['algebra_tolerance'],parity_rtol=context['config']['algebra_tolerance'],
        moment_tolerance_not_relaxed=True,maximum_transport_geometry_identity_error=max(check['maximum_transport_geometry_identity_error'] for check in trial_checks),
        same_observations_variance_beta_action_times=True,only_row_retention_changed=True,
        original_files_not_modified=True,new_parameter_selections=0,
        plane_control_scope='Existing 256-trial checkpoint records, unchanged; its measured frame schedule matches sphere.')
    return dict(checkpoint_summary=summary[summary.endpoint.eq('checkpoint')].reset_index(drop=True),
        episode_average_summary=summary[summary.endpoint.eq('episode_average')].reset_index(drop=True),
        paired_rise=pd.DataFrame(rise_rows),late_rise_trials=rise_trials,
        paired_rise_control=rise_control,paired_rise_control_trials=paired_control_trials,
        plane_control_late_rise=pd.DataFrame(plane_rows),projector_changes=projector,
        frame_rotation=frame_rotation,excitation=excitation,curvature=curvature,
        numerical_accuracy=quality,checks=checks)


def load_contact_transport_diagnostics(root):
    """Load sealed dense replay records and summaries; never replay here."""
    root=Path(root).resolve();base=root/OUTPUT
    meta=json.loads((base/'data/manifest.json').read_text())
    _require(meta['status']=='completed','Dense contact diagnostics are incomplete')
    for mapping in (meta['source_hashes'],meta['input_hashes'],meta['artifacts']):
        for name,digest in mapping.items():_require(sha(root/name)==digest,'Dense diagnostic evidence changed: '+name)
    names=['checkpoint_summary','episode_average_summary','paired_rise','late_rise_trials','paired_rise_control',
           'paired_rise_control_trials','plane_control_late_rise','projector_changes','frame_rotation',
           'excitation','curvature','numerical_accuracy']
    tables={name:pd.read_csv(base/'tables'/(name+'.csv')) for name in names}
    checkpoint=tables['checkpoint_summary'];curves=[]
    for method in MAIN_METHODS:
        group=checkpoint[checkpoint.method.eq(method)].sort_values('t')
        np.testing.assert_array_equal(group.t,TIMES)
        curves.append(dict(method=method,points=group[['t','mean','lower','upper']].to_numpy()))
    return dict(config=meta['config'],checks=meta['checks'],provenance=meta,
        input_hashes=meta['input_hashes'],source_hashes=meta['source_hashes'],sphere_curves=curves,
        dense_metrics=pd.read_csv(base/'data/dense_metrics.csv.gz'),
        physical_diagnostics=pd.read_csv(base/'data/physical_diagnostics.csv.gz'),
        trial_diagnostics=pd.read_csv(base/'data/trial_diagnostics.csv.gz'),**tables)
