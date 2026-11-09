"""Separate contact prediction, posterior spread and distribution shape.

Estimation functions accept force/velocity records and causal rotations only.
Geometry is passed to separate scoring/reporting functions. The raw likelihood
recursion is never changed by report-only temperature calibration.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .geometry import center
from .models import increment,constant
from .contact_study import query_vectors,query_truth
from .contact_mechanics import generate_contact
from .contact_tracking import make_tracker,mobility_maps
from .contact_process_prediction import fast_moments,predict_contact,kernel_parameters
from .contact_gaussian import RecursiveTangentGaussian,_MappedGaussian,gaussian_summary,laplace_summary
from .contact_posterior import contact_summary
from .contact_uncertainty_study import interval_score,_directional,_direction_rows,_summary_flags

METHOD_LABELS={'raw':'Posterior prediction','temperature':'Calibrated posterior',
    'process_identity':'Process posterior, identity','process_motion':'Process posterior, motion',
    'gaussian_identity':'Gaussian filter, identity','gaussian_motion':'Gaussian filter, motion'}


def default_config():
    return dict(schema=1,validation_seed=6245201,test_seed=6245202,
        validation_trials=16,test_trials=64,T=600,dt=.05,d=10.,v0=.02,sigma=.001,
        main_conditions=['plane','sphere_large','sphere_small','weak'],
        curved_conditions=['sphere_large','sphere_small','weak'],stress_conditions=['anisotropic'],
        report_times=[60,180,600],story_times=list(range(30,601,30)),story_query=1,story_component=0,
        temperatures=[.5,1.,2.,4.,8.],kappas=[20.,100.,500.,2000.,10000.,None],
        temperature_expansion=[.25,16.,32.],kappa_expansion=[5.,2.,50000.,200000.],
        warmup=6,alpha=.05,primary_condition='sphere_small',primary_time=600,
        primary_endpoint='Terminal and time-averaged clean vector-response MSE, nine fixed queries averaged within trajectory',
        secondary_endpoint='Terminal 95% clean scalar-response interval score, nine queries and three components averaged within trajectory',
        point_policy='Stage1 validation-selected practical policy; no oracle enters any uncertainty control',
        temperature_selection='Terminal clean interval score on validation; reporting q_(A/T) preserves raw recursion and mode; true geometry required for validation score only',
        process_selection='Mean pre-correction negative log predictive observation density on validation; true sensor variance and full Gaussian constants retained',
        scopes='Stationary plane separately; equally weighted large/small/one-axis force conditions share one curved policy; resistance stress never tunes',
        nominal_motion='Identity or exact same G_t from the Stage1 causal velocity predictor, available through t-1; no process method learns its own favourable rotation',
        kernel='Exact conditional first moment; inverse-moment fit projects predictive density approximately into Bingham family',
        gaussian='Two normal-line rotation coordinates, Joseph update and coordinate transport; kernel-matched E(theta^2)/2; local approximation, not exact transition density',
        shrinkage='Continuous rho in [0,1] minimizes pooled validation clean MSE for rho*mode separately for each selected method',
        statistics='Trajectory-level paired Student intervals; queries/components/times are not replicates; all contrasts exploratory except predeclared small-sphere terminal process versus Gaussian MSE',
        timings='Prediction includes inverse fit; reporting includes means/regions/intervals. Shared causal-forecast tracker cost is counted once per process/Gaussian method; never add nested timers twice.',
        sensor_variance_fixed=True,no_truth_in_estimation=True,starting_commit='f09d3fecd3af11fff5af708c8d1d31a35db75ca4')


def scope(condition):return 'plane' if condition=='plane' else 'curved'
def as_kappa(value):return np.inf if value is None else float(value)


def policies_from_stage1(document):
    """Only practical, frozen validation winners; oracle is rejected explicitly."""
    result={}
    for s in ['plane','curved']:
        best=document['best_per_scope'][s]
        velocity=next(p for p in document['selected'] if p['scope']==s and p['family']=='velocity')
        result[s]=dict(raw=best['specification'],motion=velocity['specification'])
        for p in result[s].values():
            if p['prediction']=='oracle':raise ValueError('Truth-dependent oracle may not enter posterior comparison')
    return result


def shared_inputs(config,U,V,policies):
    """Materialize a common causal forecast once, before using current u_t."""
    paths={}
    for label,spec in policies.items():
        if spec['prediction']=='oracle':raise ValueError('Oracle is not a practical forecast')
        f=make_tracker(config,spec);As=[];priors=[];Gs=[];updates=[];predictions=[]
        for u,v in zip(U,V):
            priors.append(f.predict());Gs.append(f.last_motion.copy())
            f.update(u,v=v);As.append(f.A.copy());updates.append(f.update_seconds);predictions.append(f.prediction_seconds)
        paths[label]=dict(A=np.asarray(As),prior=np.asarray(priors),G=np.asarray(Gs),
            update_seconds=f.update_seconds,prediction_seconds=f.prediction_seconds,update_path=np.asarray(updates),prediction_path=np.asarray(predictions))
    return paths


def _mode(A):
    values,E=np.linalg.eigh(A);return np.eye(3)-np.outer(E[:,0],E[:,0]),float(values[1]-values[0])


def _check(moment):
    if not moment.numerical_ok or not np.isfinite(moment.log_normalizer):
        raise ArithmeticError('Unresolved fixed-budget moment/normalizer')
    return moment


def estimate_path(config,U,V,spec,shared):
    """One filter run, without normals, positions, resistance or scoring data.

    ``shared['raw']`` is the unmodified likelihood path. ``shared['motion']``
    supplies only the causal nominal G for matched process controls.
    """
    path_begin=perf_counter();T=len(U);s=config['sigma']**2;kind=spec['kind'];temperature=float(spec.get('temperature',1.))
    if temperature<=0:raise ValueError('Positive temperature required')
    nominal=spec.get('nominal','identity');G=(shared['raw']['G'] if kind=='raw' else shared['motion']['G'] if nominal=='motion' else np.broadcast_to(np.eye(3),(T,3,3)))
    corrected=np.full((T,3,3),np.nan);prior_mean=corrected.copy();modes=corrected.copy();forecast_modes=corrected.copy()
    Aall=corrected.copy();priors=corrected.copy();gap=np.full(T,np.nan);forecast_gap=np.full(T,np.nan);nll=np.full(T,np.nan)
    normals=np.full((T,3),np.nan);covs=np.full((T,2,2),np.nan);bases=np.full((T,3,2),np.nan)
    clocks=dict(update_seconds=0.,prediction_seconds=0.,inverse_seconds=0.,score_seconds=0.,mean_seconds=0.,shared_forecast_seconds=0.,filter_wall_seconds=0.,end_to_end_seconds=0.,warmup_seconds=0.)
    diagnostics=dict(completed_steps=0,inverse_failures=0,maximum_inverse_residual=0.,maximum_moment_uncertainty=0.,failure_reason='',nonlocal_updates=0)
    clock_path={k:np.zeros(T) for k in clocks};A=np.zeros((3,3));M=2*np.eye(3)/3
    if kind=='gaussian':
        f=RecursiveTangentGaussian(s,as_kappa(spec['kappa']),warmup=config['warmup'])
        clocks['inverse_seconds']=np.nan  # Warmup fits are nested in prediction; no separate timer exposed.
    if nominal=='motion':
        # Includes the point tracker needed to construct the common G sequence.
        clocks['shared_forecast_seconds']=shared['motion']['prediction_seconds']+shared['motion']['update_seconds']
    try:
        for j,(u,v) in enumerate(zip(U,V)):
            if kind=='raw':
                prior=shared['raw']['prior'][j]/temperature;A=shared['raw']['A'][j]/temperature
                begin=perf_counter();before=_check(fast_moments(prior));after=_check(fast_moments(A))
                prior_mean[j]=before.mean;corrected[j]=after.mean;clocks['mean_seconds']+=perf_counter()-begin
                clocks['prediction_seconds']=shared['raw']['prediction_path'][j]
                clocks['update_seconds']=shared['raw']['update_path'][j]
                begin=perf_counter();augmented=_check(fast_moments(prior+increment(u,s,v=v)))
                nll[j]=-(constant(u,2,s,v=v)+augmented.log_normalizer-before.log_normalizer)
                clocks['score_seconds']+=perf_counter()-begin
            elif kind=='process':
                begin=perf_counter();prediction=predict_contact(A,as_kappa(spec['kappa']),G=G[j],previous_mean=M)
                prior=prediction.prior;clocks['prediction_seconds']+=perf_counter()-begin
                clocks['inverse_seconds']+=prediction.fit.seconds if np.isfinite(as_kappa(spec['kappa'])) else 0.
                diagnostics['maximum_inverse_residual']=max(diagnostics['maximum_inverse_residual'],float(prediction.fit.residual))
                diagnostics['maximum_moment_uncertainty']=max(diagnostics['maximum_moment_uncertainty'],float(prediction.fit.moment_uncertainty))
                if not prediction.fit.converged:raise ArithmeticError('Inverse moment fit did not converge')
                prior_mean[j]=prediction.target_mean
                begin=perf_counter();B=increment(u,s,v=v);A=center(prior+B)
                clocks['update_seconds']+=perf_counter()-begin
                begin=perf_counter();after=_check(fast_moments(A));M=after.mean;corrected[j]=M
                clocks['mean_seconds']+=perf_counter()-begin
                begin=perf_counter();before=_check(fast_moments(prior))
                # A is trace-centred, so restore the correction's removed trace.
                augmented_logz=after.log_normalizer+2*np.trace(prior+B)/3
                nll[j]=-(constant(u,2,s,v=v)+augmented_logz-before.log_normalizer)
                clocks['score_seconds']+=perf_counter()-begin
            elif kind=='gaussian':
                state=f.predict(G[j]);begin=perf_counter()
                if f.initialized:
                    prior_mean[j]=_MappedGaussian(f.normal,f.covariance,f.basis,32).mean()
                    forecast_modes[j]=np.eye(3)-np.outer(f.normal,f.normal);forecast_gap[j]=np.inf
                    prior=np.full((3,3),np.nan)
                else:
                    prior=f._warmup_prior.copy();prior_mean[j]=_check(fast_moments(prior)).mean
                    forecast_modes[j],forecast_gap[j]=_mode(prior)
                clocks['mean_seconds']+=perf_counter()-begin
                nll[j]=-float(f.predictive_log_score(u,v));f.update(u,v,G=G[j])
                begin=perf_counter()
                if f.initialized:
                    normals[j]=f.normal;covs[j]=f.covariance;bases[j]=f.basis
                    corrected[j]=_MappedGaussian(f.normal,f.covariance,f.basis,32).mean()
                    modes[j]=np.eye(3)-np.outer(f.normal,f.normal)
                    A=np.full((3,3),np.nan)
                else:
                    A=f.A.copy();corrected[j]=_check(fast_moments(A)).mean;modes[j],gap[j]=_mode(A)
                clocks['mean_seconds']+=perf_counter()-begin
                clocks.update({k:float(f.timings[k]) for k in ['update_seconds','prediction_seconds','score_seconds','warmup_seconds']})
                diagnostics['nonlocal_updates']=int(f.nonlocal_updates)
            else:raise ValueError('Unknown posterior method')
            Aall[j]=A;priors[j]=prior
            if nominal=='motion':clocks['shared_forecast_seconds']=shared['motion']['prediction_path'][j]+shared['motion']['update_path'][j]
            if kind!='gaussian':modes[j],gap[j]=_mode(A);forecast_modes[j],forecast_gap[j]=_mode(prior)
            diagnostics['completed_steps']=j+1
            clocks['filter_wall_seconds']=perf_counter()-path_begin
            clocks['end_to_end_seconds']=clocks['filter_wall_seconds']+(clocks['update_seconds']+clocks['prediction_seconds'] if kind=='raw' else clocks['shared_forecast_seconds'])
            for key in clocks:clock_path[key][j]=clocks[key]
    except (ArithmeticError,np.linalg.LinAlgError,ValueError) as error:
        diagnostics['failure_reason']=str(error);diagnostics['inverse_failures']=int(kind=='process')
    return dict(A=Aall,prior=priors,mean=corrected,prior_mean=prior_mean,mode=modes,
        forecast_mode=forecast_modes,gap=gap,forecast_identified=forecast_gap>1e-10,nll=nll,normal=normals,covariance=covs,basis=bases,
        clocks=clocks,clock_path=clock_path,diagnostics=diagnostics,specification=dict(spec),G=G.copy())


def path_scores(config,data,path,rho=1.):
    """Tight-frame query MSE, with actual mobility under resistance mismatch."""
    truth=mobility_maps(data,config['d']);scale=config['v0']**2/3
    regularized=rho*path['mode']
    losses={name:scale*np.sum((D-truth)**2,axis=(1,2)) for name,D in
        [('mean_mse',path['mean']),('mode_mse',path['mode']),('regularized_mse',regularized),('forecast_mean_mse',path['prior_mean'])]}
    P=data['tangent_projectors']
    losses['forecast_chordal']=np.sqrt(np.maximum(0,2-np.einsum('tij,tji->t',path['forecast_mode'],P)))
    losses['corrected_chordal']=np.sqrt(np.maximum(0,2-np.einsum('tij,tji->t',path['mode'],P)))
    losses['posterior_risk']=config['v0']**2/3*np.trace(path['mean']-path['mean']@path['mean'],axis1=1,axis2=2)
    return losses


def shrinkage_coefficients(config,data,path):
    Q=query_vectors(config['v0']);truth,_=query_truth(data,config['T'],Q,config['d']);P=path['mode'][-1]
    contrast=Q@P.T;residual=-truth
    return dict(shrink_a=float(np.mean(np.sum(contrast**2,axis=1))),
        shrink_b=float(np.mean(np.sum(residual*contrast,axis=1))),shrink_c=float(np.mean(np.sum(residual**2,axis=1))))


def report(config,data,path,t,laplace=False,queries=None):
    j=t-1;queries=query_vectors(config['v0']) if queries is None else np.asarray(queries)
    normal=data['normals'][j];kind=path['specification']['kind'];begin=perf_counter()
    if path['diagnostics']['completed_steps']<t:raise ArithmeticError(path['diagnostics']['failure_reason'])
    if kind=='gaussian' and np.isfinite(path['normal'][j]).all():
        summary=gaussian_summary(path['normal'][j],path['covariance'][j],queries,
            true_normal=normal,basis=path['basis'][j])
    elif laplace:summary=laplace_summary(path['A'][j],queries,true_normal=normal)
    else:summary=contact_summary(path['A'][j],queries,true_normal=normal)
    elapsed=perf_counter()-begin;truth,geometric=query_truth(data,t,queries,config['d'])
    M=np.asarray(summary['mean']);lower=np.asarray(summary['response_lower']);upper=np.asarray(summary['response_upper']);d=summary['numerics']
    metrics=dict(clean_mse=float(np.mean(np.sum((queries@M.T-truth)**2,axis=1))),
        interval_score=float(np.mean(interval_score(lower,upper,truth,config['alpha']))),
        clean_coverage=float(np.mean((truth>=lower)&(truth<=upper))),clean_width=float(np.mean(upper-lower)),
        ball_covered=bool(summary.get('ball_covered',False)),hpd_covered=bool(summary.get('hpd_covered',False)),
        ball_radius=float(summary.get('ball_radius',np.nan)),hpd_threshold=float(summary.get('hpd_threshold',np.nan)),
        hpd_kind=str(d.get('hpd_kind',summary.get('hpd_kind','Invariant-density Bingham HPD'))),
        posterior_risk=float(np.mean(summary['risks'])),report_seconds=elapsed,
        report_mean_seconds=float(d.get('mean_seconds',0.)),
        report_interval_seconds=float(d.get('quantile_seconds',d.get('interval_seconds',0.))),
        mean_mode_difference=float(np.linalg.norm(M-summary['mode_projector'])),
        integration_numerical_ok=bool(d.get('integration_numerical_ok',d.get('numerical_ok',False))),
        local_approximation_ok=d.get('local_approximation_ok',np.nan),
        cut_locus_mass=d.get('cut_locus_mass',np.nan),
        **_summary_flags(summary))
    metrics['report_region_seconds']=max(0,elapsed-metrics['report_mean_seconds']-metrics['report_interval_seconds'])
    return summary,metrics,truth,geometric


def candidate_specs(config,stage='initial'):
    temperatures=config['temperatures'] if stage=='initial' else config.get('_temperatures',[])
    kappas=config['kappas'] if stage=='initial' else config.get('_kappas',[])
    result=[dict(method='temperature',kind='raw',temperature=float(T)) for T in temperatures]
    for kind in ['process','gaussian']:
        for nominal in ['identity','motion']:
            result.extend(dict(method=f'{kind}_{nominal}',kind=kind,nominal=nominal,kappa=k) for k in kappas)
    return result


def validation_task(task):
    config,condition,trial,policies,stage=task
    data=generate_contact(config['validation_seed'],trial,condition,**{k:config[k] for k in ['T','dt','d','v0','sigma']})
    shared=shared_inputs(config,data['U'],data['V'],policies[scope(condition)]);rows=[]
    for spec in candidate_specs(config,stage):
        start=perf_counter();path=estimate_path(config,data['U'],data['V'],spec,shared)
        scores=path_scores(config,data,path);complete=path['diagnostics']['completed_steps']==config['T']
        row=dict(condition=condition,scope=scope(condition),trial=trial,stage=stage,**spec,
            predictive_nll=float(np.mean(path['nll'])) if complete else np.inf,
            terminal_mse=float(scores['mean_mse'][-1]),time_mse=float(np.mean(scores['mean_mse'])),
            mode_mse=float(scores['mode_mse'][-1]),interval_score=np.nan,clean_coverage=np.nan,clean_width=np.nan,
            mean_ok=bool(complete and np.isfinite(path['mean']).all()),interval_ok=False,state_ok=False,
            numerical_ok=False,**path['diagnostics'],**shrinkage_coefficients(config,data,path))
        if spec['kind']=='raw' and complete:
            try:
                _,metrics,_,_=report(config,data,path,config['T']);row.update({k:metrics[k] for k in
                    ['interval_score','clean_coverage','clean_width','mean_ok','interval_ok','state_ok','numerical_ok']})
            except (ArithmeticError,np.linalg.LinAlgError) as error:row['failure_reason']=str(error)
        row['candidate_seconds']=perf_counter()-start;rows.append(row)
    return rows


def choose_policies(rows,config):
    frame=pd.DataFrame(rows);chosen=[];curves=[]
    for (s,method),g in frame.groupby(['scope','method'],sort=False):
        key='temperature' if method=='temperature' else 'kappa'
        objective='interval_score' if method=='temperature' else 'predictive_nll'
        choices=[];expected=config['validation_trials']*(1 if s=='plane' else len(config['curved_conditions']))
        for setting,h in g.groupby(key,dropna=False,sort=False):
            resolved=bool(len(h)==expected and h.mean_ok.astype(bool).all() and np.isfinite(h[objective]).all()
                and h.completed_steps.eq(config['T']).all())
            if method=='temperature':resolved=resolved and bool(h.interval_ok.astype(bool).all())
            score=float(h.groupby('trial')[objective].mean().mean()) if resolved else np.inf
            row=dict(scope=s,method=method,**{key:None if pd.isna(setting) else float(setting)},
                objective=objective,selection_value=score,terminal_mse=float(h.terminal_mse.mean()),
                time_mse=float(h.time_mse.mean()),mode_mse=float(h.mode_mse.mean()),
                mean_interval_score=float(h.interval_score.mean()),selected=False,resolved=resolved,
                evaluations=len(h),independent_trials=h.trial.nunique(),candidate_seconds=float(h.candidate_seconds.sum()),
                failures=int((h.completed_steps<config['T']).sum()))
            curves.append(row);choices.append((score,len(curves)-1,h))
        score,index,h=min(choices,key=lambda x:x[0]);curves[index]['selected']=True;winner=curves[index]
        spec=dict(method=method,kind='raw' if method=='temperature' else method.split('_')[0],
            **{key:winner[key]},selection_value=score,objective=objective,selection_resolved=bool(np.isfinite(score)))
        if method!='temperature':spec['nominal']=method.split('_')[1]
        a=float(h.shrink_a.mean());b=float(h.shrink_b.mean())
        spec['rho']=float(np.clip(-b/a,0,1)) if a>0 else 1.
        chosen.append(dict(scope=s,**spec))
    return chosen,pd.DataFrame(curves)


def boundary_expansion(selected,config):
    temperatures=[];kappas=[]
    if any(p.get('temperature')==min(config['temperatures']) for p in selected):temperatures.append(.25)
    if any(p.get('temperature')==max(config['temperatures']) for p in selected):temperatures.extend([16.,32.])
    finite=[k for k in config['kappas'] if k is not None]
    if any(p.get('kappa')==min(finite) for p in selected):kappas.extend([5.,2.])
    if any(p.get('kappa')==max(finite) for p in selected):kappas.extend([50000.,200000.])
    return dict(temperatures=temperatures,kappas=kappas)


def selected_for_condition(selected,condition):return [dict(p) for p in selected if p['scope']==scope(condition)]


def selected_validation_task(task):
    config,condition,trial,policies,selected=task
    data=generate_contact(config['validation_seed'],trial,condition,**{k:config[k] for k in ['T','dt','d','v0','sigma']})
    shared=shared_inputs(config,data['U'],data['V'],policies[scope(condition)]);rows=[]
    for spec in selected_for_condition(selected,condition):
        path=estimate_path(config,data['U'],data['V'],spec,shared)
        try:
            _,metrics,_,_=report(config,data,path,config['T']);rows.append(dict(condition=condition,trial=trial,**spec,**metrics))
        except (ArithmeticError,np.linalg.LinAlgError) as error:rows.append(dict(condition=condition,trial=trial,**spec,failure_reason=str(error),numerical_ok=False))
    return rows


def test_task(task):
    config,condition,trial,policies,selected=task
    data=generate_contact(config['test_seed'],trial,condition,**{k:config[k] for k in ['T','dt','d','v0','sigma']})
    shared=shared_inputs(config,data['U'],data['V'],policies[scope(condition)]);metrics=[];components=[];directional=[];shapes=[];story=[]
    specs=selected_for_condition(selected,condition)
    calibrated=next(p for p in specs if p['method']=='temperature')
    specs=[dict(calibrated,method='raw',temperature=1.)]+specs
    arrays={};Q=query_vectors(config['v0'])
    for spec in specs:
        method=spec['method'];path=estimate_path(config,data['U'],data['V'],spec,shared);scores=path_scores(config,data,path,spec['rho'])
        if trial==0:
            for field in ['A','prior','mean','prior_mean','mode','forecast_mode','normal','covariance','basis','G','nll','forecast_identified']:
                arrays[f'{method}__{field}']=path[field]
            for field,value in scores.items():arrays[f'{method}__{field}']=value
        for t in config['report_times']:
            j=t-1;base=dict(condition=condition,trial=trial,t=t,method=method,label=METHOD_LABELS[method],
                rho=spec['rho'],temperature=spec.get('temperature',np.nan),kappa=spec.get('kappa',np.nan),
                nominal=spec.get('nominal','raw'),**{k:float(v[j]) for k,v in scores.items()},
                time_mean_mse=float(np.mean(scores['mean_mse'][:t])),time_mode_mse=float(np.mean(scores['mode_mse'][:t])),
                time_regularized_mse=float(np.mean(scores['regularized_mse'][:t])),
                predictive_nll=float(np.mean(path['nll'][:t])),**path['diagnostics'],
                **{k:float(v[j]) for k,v in path['clock_path'].items()})
            try:
                summary,loss,truth,geometric=report(config,data,path,t);row=dict(base,**loss);metrics.append(row)
                for q in range(len(Q)):
                    for c in range(3):components.append(dict(condition=condition,trial=trial,t=t,method=method,query=q,component=c,
                        truth=float(truth[q,c]),mean=float((Q@summary['mean'].T)[q,c]),mode=float((Q@summary['mode_projector'].T)[q,c]),
                        lower=float(summary['response_lower'][q,c]),upper=float(summary['response_upper'][q,c]),interval_ok=loss['interval_ok']))
                if spec['kind']!='gaussian':
                    directions=_directional(path['A'][j],summary,data['normals'][j],data['normals'][max(0,j-1)])
                    try:
                        local,local_loss,_,_=report(config,data,path,t,laplace=True)
                        # A report-only control has no measured per-step local-mean path.
                        metrics.append(dict(base,method=method+'_laplace',label=METHOD_LABELS[method]+' (local approximation)',
                            **local_loss,mean_mse=local_loss['clean_mse'],time_mean_mse=np.nan,forecast_mean_mse=np.nan,
                            mean_seconds=np.nan,filter_wall_seconds=np.nan,end_to_end_seconds=np.nan))
                        shapes.append(dict(condition=condition,trial=trial,t=t,method=method,
                            mean_map_difference=float(np.linalg.norm(summary['mean']-local['mean'])),
                            interval_endpoint_difference=float(max(np.max(abs(summary['response_lower']-local['response_lower'])),np.max(abs(summary['response_upper']-local['response_upper'])))),
                            full_interval_ok=loss['interval_ok'],local_interval_ok=local_loss['interval_ok'],local_approximation_ok=local_loss['local_approximation_ok']))
                    except (ArithmeticError,np.linalg.LinAlgError) as error:
                        metrics.append(dict(base,method=method+'_laplace',mean_mse=np.nan,time_mean_mse=np.nan,
                            clean_mse=np.nan,interval_score=np.nan,mean_ok=False,interval_ok=False,state_ok=False,
                            numerical_ok=False,report_failure=str(error)))
                else:
                    n=path['normal'][j];E=path['basis'][j];cov=E@path['covariance'][j]@E.T
                    directions=_direction_rows(n,E,cov,cov,np.full(2,np.nan),data['normals'][j],data['normals'][max(0,j-1)])
                directional.extend(dict(condition=condition,trial=trial,t=t,method=method,**r) for r in directions)
            except (ArithmeticError,np.linalg.LinAlgError) as error:
                metrics.append(dict(base,clean_mse=np.nan,interval_score=np.nan,clean_coverage=np.nan,clean_width=np.nan,
                    mean_ok=False,interval_ok=False,state_ok=False,numerical_ok=False,report_failure=str(error)))
        if trial==0:
            # A fixed trajectory/query and fixed report schedule; illustrative, not extra replicates.
            for t in config['story_times']:
                try:
                    q=config['story_query'];summary,loss,truth,_=report(config,data,path,t,queries=Q[[q]])
                    for c in range(3):story.append(dict(condition=condition,trial=trial,t=t,method=method,query=q,component=c,
                        truth=float(truth[0,c]),mean=float((Q[[q]]@summary['mean'].T)[0,c]),
                        lower=float(summary['response_lower'][0,c]),upper=float(summary['response_upper'][0,c]),
                        numerical_ok=loss['numerical_ok'],interval_ok=loss['interval_ok']))
                except (ArithmeticError,np.linalg.LinAlgError) as error:story.append(dict(condition=condition,trial=trial,t=t,method=method,failure_reason=str(error),numerical_ok=False))
    if trial==0:
        arrays.update({k:data[k] for k in ['U','V','normals','positions','tangent_projectors','velocities']})
        arrays['time']=data['times']
        for label,p in shared.items():
            arrays[f'shared_{label}_A']=p['A'];arrays[f'shared_{label}_prior']=p['prior'];arrays[f'shared_{label}_G']=p['G']
    return dict(metrics=metrics,components=components,directional=directional,shape_comparison=shapes,story=story,arrays=arrays)


def summarize(frames,config):
    m=frames['metrics'];terminal=m[m.t==config['T']];paired=[];summary=[]
    for (condition,method),g in terminal.groupby(['condition','method'],sort=False):
        row=dict(condition=condition,method=method,trials=len(g),unresolved=int((~g.numerical_ok.astype(bool)).sum()))
        for metric in ['clean_mse','mean_mse','mode_mse','regularized_mse','time_mean_mse','time_mode_mse','time_regularized_mse',
            'interval_score','clean_coverage','clean_width','ball_covered','hpd_covered','posterior_risk','forecast_chordal','corrected_chordal',
            'predictive_nll','update_seconds','prediction_seconds','inverse_seconds','score_seconds','mean_seconds','shared_forecast_seconds','filter_wall_seconds','end_to_end_seconds','warmup_seconds']:
            if metric in g:row[metric]=float(pd.to_numeric(g[metric]).mean())
        costs=m[(m.condition==condition)&(m.method==method)]
        row['three_report_seconds']=float(costs.groupby('trial').report_seconds.sum().mean()) if 'report_seconds' in costs else np.nan
        summary.append(row)
    for condition,g in terminal.groupby('condition',sort=False):
        for endpoint in ['clean_mse','time_mean_mse','interval_score']:
            pivot=g.pivot(index='trial',columns='method',values=endpoint)
            for method,reference in [('process_motion','gaussian_motion'),('process_identity','gaussian_identity'),
                ('process_motion','temperature'),('process_identity','temperature'),('temperature','raw'),
                ('process_motion','process_identity'),('gaussian_motion','gaussian_identity')]:
                if method not in pivot or reference not in pivot:continue
                difference=pivot[reference]-pivot[method];n=len(difference);resolved=bool(np.isfinite(difference).all())
                mean=float(difference.mean()) if resolved else np.nan
                half=float(student_t.ppf(.975,n-1)*difference.std(ddof=1)/np.sqrt(n)) if resolved and n>1 else np.nan
                paired.append(dict(condition=condition,endpoint=endpoint,method=method,reference=reference,
                    gain=mean,low=mean-half,high=mean+half,trials=n,resolved=resolved,
                    primary=condition=='sphere_small' and endpoint=='clean_mse' and method=='process_motion' and reference=='gaussian_motion'))
    return dict(summary=pd.DataFrame(summary),paired=pd.DataFrame(paired))
