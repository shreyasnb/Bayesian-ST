"""Read-only contact-prediction summaries from the original saved trials.

Only algebra, sample summaries and provenance checks are performed. In
particular, sparse saved checkpoints never replace the saved full-run average.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .final_validation import require_gate, verify_component


CONDITIONS = ('static_concentrated','plane','sphere')
MAIN_METHODS = ('mean','mode','iso')
OUTPUT = 'results/contact_prediction_diagnostics'
ORIGINAL = 'results/contact/data/paper/performance'
SCALE = 1e6
AVERAGE_DEFINITION = (
    'The sphere and rotating-plane full-run value is the original '
    'episode_average record at t=192. The producer averages the 48 internal '
    'action times t=4,8,...,192 within each simulation. The six saved checkpoint '
    'rows are not used to reconstruct this average. Stationary full-run values '
    'average its four original action times t=1,4,16,32.'
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value,message):
    if not value:
        raise RuntimeError(message)


def _interval(values):
    a=np.asarray(values,float)
    _require(len(a)==256 and np.isfinite(a).all(),
             'Contact intervals require all 256 finite independent simulations')
    mean=float(a.mean());half=float(student_t.ppf(.975,255)*a.std(ddof=1)/16.)
    return dict(simulations=256,mean=mean,lower=mean-half,upper=mean+half)


def _artifact(meta,basename):
    matches=[name for name in meta['artifacts'] if Path(name).name==basename]
    _require(len(matches)==1,'Expected one original contact artifact: '+basename)
    return matches[0]


def _physical_checks(story,cfg,selections,metrics):
    checks={};identity=np.eye(3)
    for condition in CONDITIONS:
        prefix=condition+'_';total=32 if condition.startswith('static_') else 192
        beta=float(selections[condition]['beta'])
        sigma2=float(cfg['static_sigma2'] if condition.startswith('static_') else cfg['sigma2'])
        R,G=story[prefix+'R'],story[prefix+'G']
        normal=story[prefix+'world_normal'];body_normal=story[prefix+'normal']
        P=story[prefix+'P'];U,V=story[prefix+'U'],story[prefix+'V']
        np.testing.assert_equal(R.shape,(total,3,3))
        np.testing.assert_allclose(np.einsum('tji,tjk->tik',R,R),
            np.broadcast_to(identity,(total,3,3)),rtol=0.,atol=2e-12)
        np.testing.assert_allclose(G[0],R[0].T,rtol=0.,atol=2e-12)
        np.testing.assert_allclose(G[1:],np.einsum('tji,tjk->tik',R[1:],R[:-1]),rtol=0.,atol=2e-12)
        np.testing.assert_allclose(body_normal,np.einsum('tji,tj->ti',R,normal),rtol=0.,atol=2e-12)
        world_P=identity-np.einsum('ti,tj->tij',normal,normal)
        np.testing.assert_allclose(P,np.einsum('tji,tjk,tkl->til',R,world_P,R),rtol=0.,atol=2e-12)
        for body,world in [('U','world_U'),('V','world_V'),('clean','world_clean')]:
            np.testing.assert_allclose(story[prefix+body],
                np.einsum('tji,tj->ti',R,story[prefix+world]),rtol=0.,atol=2e-12)
        np.testing.assert_allclose(story[prefix+'forces'],cfg['resistance']*story[prefix+'world_V'],rtol=0.,atol=2e-12)
        clean=story[prefix+'world_clean']
        np.testing.assert_allclose(clean,np.einsum('tij,tj->ti',world_P,story[prefix+'world_V']),rtol=0.,atol=2e-12)
        np.testing.assert_allclose(np.sum(normal*clean,axis=1),0.,rtol=0.,atol=2e-12)
        transported=np.einsum('tij,tjk,tlk->til',G[1:],P[:-1],G[1:])
        residual=np.linalg.norm(P[1:]-transported,axis=(1,2))
        physical=np.linalg.norm(np.diff(world_P,axis=0),axis=(1,2))
        np.testing.assert_allclose(residual,physical,rtol=1e-10,atol=2e-12)
        if condition=='sphere':
            np.testing.assert_allclose(np.linalg.norm(story[prefix+'positions'],axis=1),cfg['radius'],rtol=0.,atol=2e-12)
            np.testing.assert_allclose(normal,story[prefix+'positions']/cfg['radius'],rtol=0.,atol=2e-12)
            _require(float(residual.max())>1e-5,'Sphere physical tangent drift is missing')
        else:
            np.testing.assert_allclose(normal,np.broadcast_to(normal[0],normal.shape),rtol=0.,atol=2e-12)
            np.testing.assert_allclose(residual,0.,rtol=0.,atol=2e-12)
        rotation_change=float(np.max(np.linalg.norm(R-identity,axis=(1,2))))
        if condition=='plane':
            _require(rotation_change>.1,'The saved plane must include measured frame rotation')
        if condition.startswith('static_'):
            np.testing.assert_array_equal(G,np.broadcast_to(identity,G.shape))

        # Verify the transported conditional statistic from the saved inputs;
        # no moments, normalizers, posterior fits or trajectories are recomputed.
        A=np.zeros((3,3));maximum_A_error=0.
        for j,(u,v,g) in enumerate(zip(U,V,G)):
            A=beta*g@A@g.T+(np.outer(u,v)+np.outer(v,u)-np.outer(v,v))/(2*sigma2)
            A=(A+A.T)/2;A-=np.trace(A)/3*identity
            np.testing.assert_allclose(A,story[prefix+'A'][j],rtol=2e-11,atol=2e-10)
            maximum_A_error=max(maximum_A_error,float(np.max(abs(A-story[prefix+'A'][j]))))
        reports=story[prefix+'report_times'].astype(int)
        expected=[1,4,16,32] if condition.startswith('static_') else list(range(4,193,4))
        np.testing.assert_array_equal(reports,expected)
        trial_zero=metrics[metrics.condition.eq(condition)&metrics.trial.eq(0)]
        whole_run={}
        for method,key in [('mean','M'),('mode','Pmode'),('isotropic','isotropic')]:
            matrices=story[prefix+key]
            losses=cfg['v0']**2/3*np.sum((matrices-P[reports-1])**2,axis=(1,2))
            average=trial_zero[trial_zero.method.eq(method)&trial_zero.endpoint.eq('episode_average')]
            _require(len(average)==1,'Missing original full-run story endpoint')
            np.testing.assert_allclose(losses.mean(),average.response_mse.iloc[0],rtol=1e-11,atol=1e-15)
            whole_run[method]=float(losses.mean()*SCALE)
            for row in trial_zero[trial_zero.method.eq(method)&trial_zero.endpoint.eq('checkpoint')].itertuples():
                index=np.flatnonzero(reports==row.t)
                _require(len(index)==1,'Checkpoint is not a saved action time')
                np.testing.assert_allclose(losses[index[0]],row.response_mse,rtol=1e-11,atol=1e-15)
        checks[condition]=dict(saved_simulation=0,T=total,beta=beta,sigma2=sigma2,
            measured_rotation_only=True,coordinate_transport_identity_verified=True,
            world_normal_fixed=condition!='sphere',
            maximum_sensor_rotation_distance=rotation_change,
            maximum_physical_projector_increment=float(physical.max()),
            maximum_residual_after_measured_transport=float(residual.max()),
            maximum_conditional_parameter_error=maximum_A_error,
            actual_internal_action_times=expected,
            whole_run_saved_story_mean_errors_mm2_s2=whole_run)
    return checks


def load_contact_prediction_diagnostics(root,config=None):
    """Return original checkpoint errors, full-run endpoints and paired data.

    Main curves use mean, mode and isotropic mean. Supplementary summaries
    retain every original method, including the separately named identity-
    transport comparisons. All displayed summaries have units (mm/s)^2.
    """
    root=Path(root).resolve();require_gate(root)
    meta=verify_component(root,'contact','paper','performance')
    cfg=meta['config']
    required=dict(n=3,k=2,T=192,static_T=32,dt=.05,v0=.02,resistance=10.,radius=.05,
        sigma2=.004**2,static_sigma2=.01**2,validation_episodes=64,test_episodes=256,action_every=4)
    for key,value in required.items():
        _require(cfg.get(key)==value,'Unexpected original contact setting: '+key)
    _require(cfg['validation_seed']!=cfg['test_seed'],'Validation and test seeds must be disjoint')
    # Contact producers are unchanged. Do not reinterpret any source digest as
    # a hash of revised bytes or add this reader to the scientific source set.
    for name,digest in meta['sources'].items():
        _require(sha(root/name)==digest,'Original contact producer bytes changed: '+name)
    names={key:_artifact(meta,filename) for key,filename in (
        ('metrics','metrics.csv'),('paired','paired.csv'),('selected','selected.json'),
        ('story','story.npz'),('validation','validation.csv'))}
    input_hashes={ORIGINAL+'/manifest.json':sha(root/(ORIGINAL+'/manifest.json'))}
    input_hashes.update({name:(value['sha256'] if isinstance(value,dict) else value)
                        for name,value in meta['artifacts'].items() if name in names.values()})
    input_hashes.update(meta['sources'])
    selections={record['condition']:record for record in json.loads((root/names['selected']).read_text())}
    for condition in CONDITIONS:
        expected=1. if condition!='sphere' else .9
        _require(selections[condition]['beta']==expected,'Original selected contact beta changed')
    validation=pd.read_csv(root/names['validation'])
    for condition in CONDITIONS:
        rows=validation[validation.condition.eq(condition)]
        _require(rows.trial.nunique()==64 and rows.numerical_ok.astype(bool).all(),
                 'Incomplete or unresolved original validation records')
        means=rows[rows.family.eq('bingham')].groupby('beta').loss.mean()
        best=min(means.index,key=lambda beta:(means.loc[beta],-beta))
        _require(best==selections[condition]['beta'],'Stored beta disagrees with original validation')

    metrics=pd.read_csv(root/names['metrics'])
    _require(not metrics.duplicated(['condition','endpoint','t','method','trial']).any(),
             'Duplicate original contact test record')
    _require(metrics.numerical_ok.astype(bool).all(),'Unresolved contact test records must not be dropped')
    np.testing.assert_allclose(metrics.response_mse,cfg['v0']**2/3*metrics.response_map_frobenius,
                               rtol=1e-11,atol=1e-15)
    # Recheck the original paired table using its unchanged computational
    # procedure, for every method, condition and both stored task metrics.
    from .final_contact import paired_table
    old_pairs=pd.read_csv(root/names['paired'])
    recomputed=pd.DataFrame(paired_table(metrics))
    keys=['condition','endpoint','t','reference','metric']
    old_pairs=old_pairs.sort_values(keys).reset_index(drop=True)
    recomputed=recomputed.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(old_pairs[keys],recomputed[keys],check_dtype=False)
    for column in ['gain','lower','upper']:
        np.testing.assert_allclose(old_pairs[column],recomputed[column],rtol=1e-11,atol=1e-15)
    for column in ['episodes','finite_episodes','numerical_ok']:
        np.testing.assert_array_equal(old_pairs[column],recomputed[column])
    with np.load(root/names['story'],allow_pickle=False) as archive:
        story={name:archive[name] for name in archive.files}
    physical=_physical_checks(story,cfg,selections,metrics)
    declared=dict(n=3,k=2,T=192,static_T=32,dt=.05,v0=.02,resistance=10.,radius=.05,
        sigma2=.004**2,static_sigma2=.01**2,static_beta=1.,plane_beta=1.,sphere_beta=.9,
        conditions=list(CONDITIONS),methods=list(MAIN_METHODS),
        validation_simulations=64,test_simulations=256,
        stationary_checkpoints=[1,4,16,32],moving_checkpoints=[4,16,32,64,128,192],
        internal_moving_action_times=list(range(4,193,4)),unit_factor=SCALE,
        error_definition='E_t(D)=(1/3) sum_j ||D v_t^(j)-P_Xt v_t^(j)||^2',
        full_run_average_definition=AVERAGE_DEFINITION)
    for key,value in (config or {}).items():
        _require(key in declared and value==declared[key],
                 'Requested contact configuration differs from the original data: '+key)
    selected=metrics[metrics.condition.isin(CONDITIONS)].copy()
    selected['method']=selected.method.replace({'isotropic':'iso'})
    selected['response_mse_mm2_s2']=selected.response_mse*SCALE
    checkpoint=[];averages=[];pairs=[];differences=[];conditions={}
    for condition in CONDITIONS:
        subset=selected[selected.condition.eq(condition)]
        methods=list(MAIN_METHODS)+[method for method in ('regularized','great_window','great_ew',
            'mean_identity','mode_identity') if subset.method.eq(method).any()]
        times=[1,4,16,32] if condition.startswith('static_') else [4,16,32,64,128,192]
        total=32 if condition.startswith('static_') else 192
        curves=[]
        for method in methods:
            points=[]
            for t in times:
                group=subset[subset.method.eq(method)&subset.endpoint.eq('checkpoint')&subset.t.eq(t)].sort_values('trial')
                np.testing.assert_array_equal(group.trial,np.arange(256))
                stats=_interval(group.response_mse_mm2_s2)
                checkpoint.append(dict(condition=condition,endpoint='checkpoint',t=t,method=method,**stats))
                points.append((t,stats['mean'],stats['lower'],stats['upper']))
            curves.append(dict(method=method,points=np.asarray(points)))
            group=subset[subset.method.eq(method)&subset.endpoint.eq('episode_average')&subset.t.eq(total)].sort_values('trial')
            np.testing.assert_array_equal(group.trial,np.arange(256))
            averages.append(dict(condition=condition,endpoint='episode_average',t=total,method=method,
                                 **_interval(group.response_mse_mm2_s2)))
        for (endpoint,t),group in subset.groupby(['endpoint','t'],sort=True):
            wide=group.pivot(index='trial',columns='method',values='response_mse_mm2_s2')
            np.testing.assert_array_equal(wide.index,np.arange(256))
            for reference in methods:
                if reference=='mean':continue
                delta=wide[reference]-wide['mean'];stats=_interval(delta)
                pairs.append(dict(condition=condition,endpoint=endpoint,t=int(t),reference=reference,
                    gain=stats['mean'],**stats))
                old_reference='isotropic' if reference=='iso' else reference
                recorded=old_pairs[old_pairs.condition.eq(condition)&old_pairs.endpoint.eq(endpoint)
                    &old_pairs.t.eq(t)&old_pairs.reference.eq(old_reference)&old_pairs.metric.eq('response_mse')]
                _require(len(recorded)==1,'Missing original paired comparison')
                np.testing.assert_allclose([stats['mean'],stats['lower'],stats['upper']],
                    recorded[['gain','lower','upper']].iloc[0].to_numpy()*SCALE,rtol=1e-11,atol=1e-10)
                differences.extend(dict(condition=condition,endpoint=endpoint,t=int(t),trial=int(trial),
                    reference=reference,difference_mm2_s2=float(value)) for trial,value in delta.items())
        conditions[condition]=dict(curves=[curve for curve in curves if curve['method'] in MAIN_METHODS],
            all_curves=curves,time=np.asarray(times,dtype=int),T=total,
            beta=float(selections[condition]['beta']),
            sigma2=float(cfg['static_sigma2'] if condition.startswith('static_') else cfg['sigma2']),
            selected_settings=selections[condition])
    pairs=pd.DataFrame(pairs);differences=pd.DataFrame(differences)
    histograms={}
    for key,condition,endpoint,t in [('static_concentrated_t16','static_concentrated','checkpoint',16),
                                   ('sphere_episode_average','sphere','episode_average',192)]:
        subset=selected[selected.condition.eq(condition)&selected.endpoint.eq(endpoint)&selected.t.eq(t)]
        histograms[key]={method:group.sort_values('trial').response_mse_mm2_s2.to_numpy()
                         for method,group in subset.groupby('method',sort=True)}
    sparse=selected[selected.condition.eq('sphere')&selected.method.eq('mean')&selected.endpoint.eq('checkpoint')].groupby('trial').response_mse_mm2_s2.mean()
    full=selected[selected.condition.eq('sphere')&selected.method.eq('mean')&selected.endpoint.eq('episode_average')].set_index('trial').response_mse_mm2_s2
    checks=dict(status='passed',original_sources_and_all_artifacts_verified=True,
        original_contact_sources_match_current_bytes=True,original_paired_rows_verified=len(old_pairs),
        response_error_frobenius_identity_verified=True,unit_factor=SCALE,no_extra_factor_of_three=True,
        test_simulations_per_condition=256,validation_simulations_per_condition=64,
        stationary_saved_checkpoints=[1,4,16,32],moving_saved_checkpoints=[4,16,32,64,128,192],
        sphere_average_uses_original_endpoint=True,sphere_internal_action_count=48,
        maximum_sparse_vs_whole_run_mean_difference_mm2_s2=float(np.max(abs(sparse-full))),
        saved_story_physical_checks=physical,
        new_simulations=0,new_parameter_selections=0,new_posterior_fits=0)
    provenance=dict(original_manifest=ORIGINAL+'/manifest.json',
        original_manifest_sha256=input_hashes[ORIGINAL+'/manifest.json'],
        original_sources=meta['sources'],original_config=cfg,
        primary_error_definition=declared['error_definition'],units='(mm/s)^2',
        full_run_average_definition=AVERAGE_DEFINITION,
        paired_definition='Reference minus posterior mean on the same simulation; positive favors posterior mean.',
        confidence_interval='Pointwise Student 95% interval across 256 independent simulations.',
        selection=cfg['selection'],information_by_method=meta['information_by_method'],
        limitations=['Dense test metrics are not retained for every simulation. Saved episode_average records and the unchanged verified producer establish the 48-time average; the saved trial-zero trajectory independently verifies it for mean, mode and isotropic mean.',
                    'GREAT receives measured velocities and rotations; conditional estimators additionally receive the known applied inputs.'])
    return dict(config=declared,conditions=conditions,physical_checks=physical,checks=checks,
        provenance=provenance,input_hashes=input_hashes,source_hashes=meta['sources'],
        selected_settings={condition:selections[condition] for condition in CONDITIONS},
        raw_stats=selected.reset_index(drop=True),histogram_samples=histograms,histograms=histograms,
        checkpoint_summary=pd.DataFrame(checkpoint),episode_average_summary=pd.DataFrame(averages),
        paired_checkpoint_summary=pairs[pairs.endpoint.eq('checkpoint')].reset_index(drop=True),
        paired_episode_average_summary=pairs[pairs.endpoint.eq('episode_average')].reset_index(drop=True),
        paired_checkpoint_differences=differences[differences.endpoint.eq('checkpoint')].reset_index(drop=True),
        paired_episode_average_differences=differences[differences.endpoint.eq('episode_average')].reset_index(drop=True))
