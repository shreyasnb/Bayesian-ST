"""Frozen physical comparison of process models, global shape and local filters.

Every estimator observes only the prescribed force and noisy instantaneous
velocity. Geometry is scoring information. The mechanical state law is not
the isotropic transition assumed by the process filters.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.integrate import IntegrationWarning

from .geometry import center,projector,risk
from .models import increment,constant
from .contact_mechanics import generate_contact
from .contact_study import query_vectors,query_truth,weighted_projector_ls
from .contact_posterior import contact_summary,normal_quadrature


METHODS=['Point mode','Point mean','Scaled Bingham','Process Bingham',
         'Scaled Laplace','Process Laplace','Recursive Gaussian']
UNCERTAINTY_METHODS=['Scaled Bingham','Process Bingham','Recursive Gaussian']


def default_config():
    return dict(schema=1,validation_seed=6245201,test_seed=6245202,
        validation_trials=16,test_trials=64,T=600,dt=.05,d=10.,v0=.02,sigma=.001,
        main_conditions=['plane','sphere_large','sphere_small','weak'],
        curved_conditions=['sphere_large','sphere_small','weak'],stress_conditions=['anisotropic'],
        betas=[1.,.99,.98,.95,.9,.8,.6,.4,.2],etas=[.125,.25,.5,1.,2.],
        process_rms=[0.,.0025,.005,.01,.02,.04,.08],
        beta_expansion=[.1,.05],process_expansion=[.16,.32],
        expansion_rule='One validation-only expansion: beta .2 -> .1,.05; maximum process RMS .08 -> .16,.32 for either process method; both methods receive the same expanded dispersion family',
        report_times=[60,180,600],warmup=6,alpha=.05,
        primary_condition='sphere_small',primary_time=600,
        primary_method='Process Bingham',primary_references=['Scaled Bingham','Recursive Gaussian'],
        primary_endpoint='95% clean-response interval score, averaged over nine queries and three components within each trajectory',
        uncertainty_selection='Mean pre-correction observation negative log density over all observations, averaged within each trajectory then across the declared scope',
        point_selection='Terminal clean vector-response MSE over nine fixed queries',
        secondary_validation='Terminal clean interval score retained for every candidate; never used to change the primary NLL-selected policy',
        scopes='Plane separately; one equally weighted curved-mixture policy shared across large/small/weak and reused in the resistance stress',
        motion='Identity nominal frame for every method; no learned motion, positions or true normals supplied',
        process_interpretation='Exact kernel first moment conditional on the maintained previous law; Bingham density projection and recursive tangent Gaussian are approximations',
        predictive_score='Bingham logZ(Aprior+B)-logZ(Aprior)+Gaussian constant; B uses true noise, not generalized-posterior eta',
        gaussian_initialization='Six data-only kernel-specific Bingham updates, then mode/Laplace initialization; all startup information and costs retained',
        sensing='Known applied force and noisy instantaneous velocity only; positions are plant/scoring records',
        fixed_noise=True,noisy_intervals=False,
        statistics='Independent trajectory IDs; average queries and components before intervals. Two primary references; all other pointwise intervals exploratory.',
        timing='Per-step correction/prediction/log-score clocks cumulative. Report clocks per requested report; inversion is nested in prediction, report means/regions/intervals nested in total reporting.',
        reviewed_commit='f09d3fecd3af11fff5af708c8d1d31a35db75ca4')


def interval_score(lower,upper,truth,alpha=.05):
    lower,upper,truth=np.broadcast_arrays(lower,upper,truth)
    if not 0<alpha<1 or np.any(upper<lower):raise ValueError('Invalid interval or alpha')
    return upper-lower+(2/alpha)*(lower-truth)*(truth<lower)+(2/alpha)*(truth-upper)*(truth>upper)


def _process_api():
    from .contact_process_prediction import fast_moments,predict_contact,kappa_from_chordal_rms
    return fast_moments,predict_contact,kappa_from_chordal_rms


def candidate_specs(config,scope,extra=False):
    if extra:
        betas=config['beta_expansion'] if config.get('_expand_beta',True) else []
        rms=config['process_expansion'] if config.get('_expand_process',True) else []
    else:
        betas=config['betas'];rms=config['process_rms']
    result=[dict(method='Scaled Bingham',beta=float(beta),eta=float(eta))
            for beta in betas for eta in config['etas']]
    result.extend(dict(method=method,process_rms=float(value))
                  for method in ['Process Bingham','Recursive Gaussian'] for value in rms)
    return result


def _scope(condition):return 'plane' if condition=='plane' else 'curved'


def _kappa(spec):
    if 'process_rms' not in spec:return np.inf
    return _process_api()[2](float(spec['process_rms']))


def _backend_ok(result):
    return bool(result.numerical_ok) and np.isfinite(result.log_normalizer)


def _bingham_path(config,data,spec,report_times,keep_startup=False):
    """One correction per observation; predictive density is never tempered."""
    fast,predict,_=_process_api();s=config['sigma']**2
    A=np.zeros((3,3));M=2*np.eye(3)/3;logz=0.
    beta=float(spec.get('beta',1.));eta=float(spec.get('eta',1.));kappa=_kappa(spec)
    process=spec['method']=='Process Bingham'
    snapshots={};startup=[];nll=[]
    clocks=dict(update_seconds=0.,prediction_seconds=0.,inverse_seconds=0.,score_seconds=0.)
    diagnostics=dict(inverse_failures=0,backend_failures=0,maximum_inverse_residual=0.,
                     maximum_moment_uncertainty=0.,completed_steps=0,failure_reason='')
    try:
        for t,(u,v) in enumerate(zip(data['U'],data['V']),1):
            start=perf_counter()
            if process and np.isfinite(kappa):
                predicted=predict(A,kappa,previous_mean=M)
                prior=predicted.prior
                if not predicted.fit.converged:raise ArithmeticError('Inverse moment fit did not converge')
                diagnostics['maximum_inverse_residual']=max(diagnostics['maximum_inverse_residual'],float(predicted.fit.residual))
                diagnostics['maximum_moment_uncertainty']=max(diagnostics['maximum_moment_uncertainty'],float(predicted.fit.moment_uncertainty))
                clocks['inverse_seconds']+=float(predicted.fit.seconds)
                prior_logz=float(predicted.fit.objective+np.trace(prior@predicted.target_mean))
            else:
                prior=center(beta*A) if not process else A.copy()
                if process or beta==1.:prior_logz=logz
                else:
                    evaluated=fast(prior)
                    if not _backend_ok(evaluated):raise ArithmeticError('Unresolved prior normalizer')
                    prior_logz=float(evaluated.log_normalizer)
            clocks['prediction_seconds']+=perf_counter()-start
            begin=perf_counter();B=increment(u,s,v=v);B_seconds=perf_counter()-begin
            begin=perf_counter();augmented=fast(prior+B)
            if not _backend_ok(augmented):raise ArithmeticError('Unresolved predictive normalizer')
            value=-(float(augmented.log_normalizer)-prior_logz+constant(u,2,s,v=v))
            if not np.isfinite(value):raise ArithmeticError('Nonfinite predictive log density')
            nll.append(value);clocks['score_seconds']+=perf_counter()-begin
            begin=perf_counter();A=center(prior+eta*B)
            if eta==1.:
                M=augmented.mean
                logz=float(augmented.log_normalizer-2*np.trace(prior+B)/3)
            else:
                evaluated=fast(A)
                if not _backend_ok(evaluated):raise ArithmeticError('Unresolved corrected moment')
                M=evaluated.mean;logz=float(evaluated.log_normalizer)
            clocks['update_seconds']+=B_seconds+perf_counter()-begin
            diagnostics['completed_steps']=t
            if keep_startup and t<=max(12,config['warmup']):
                basis=np.linalg.eigh(A)[1][:,-2:]
                truth=data['tangent_projectors'][t-1]
                startup.append(dict(t=t,modal_error=float(np.sqrt(max(0.,2-np.trace(projector(basis)@truth)))),
                                    nll=value,initialized=True))
            if t in report_times:
                snapshots[t]=dict(A=A.copy(),mean=M.copy(),nll_mean=float(np.mean(nll)),
                                  nll_sum=float(np.sum(nll)),terminal_nll=value,**clocks,**diagnostics)
    except (ArithmeticError,np.linalg.LinAlgError) as error:
        diagnostics['inverse_failures']=int(process);diagnostics['backend_failures']=1
        diagnostics['failure_reason']=str(error)
    return dict(snapshots=snapshots,startup=startup,nll=nll,clocks=clocks,diagnostics=diagnostics)


def _gaussian_path(config,data,spec,report_times,keep_startup=False):
    from .contact_gaussian import RecursiveTangentGaussian
    tracker=RecursiveTangentGaussian(config['sigma']**2,_kappa(spec),warmup=config['warmup'])
    snapshots={};startup=[];nll=[];failed=''
    try:
        for t,(u,v) in enumerate(zip(data['U'],data['V']),1):
            value=-float(tracker.predictive_log_score(u,v));nll.append(value)
            tracker.update(u,v)
            if keep_startup and t<=max(12,config['warmup']):
                normal=(np.asarray(tracker.normal) if tracker.normal is not None
                        else np.linalg.eigh(-tracker.A)[1][:,-1])
                P=np.eye(3)-np.outer(normal,normal)
                startup.append(dict(t=t,modal_error=float(np.sqrt(max(0.,2-np.trace(P@data['tangent_projectors'][t-1])))),
                                    nll=value,initialized=bool(tracker.initialized)))
            if t in report_times:
                if not tracker.initialized:continue
                snapshots[t]=dict(normal=tracker.normal.copy(),covariance=tracker.covariance.copy(),
                    basis=tracker.basis.copy(),nll_mean=float(np.mean(nll)),nll_sum=float(np.sum(nll)),
                    terminal_nll=value,timings=dict(tracker.timings),last_diagnostics=dict(tracker.last_diagnostics))
    except (ArithmeticError,np.linalg.LinAlgError) as error:failed=str(error)
    return dict(snapshots=snapshots,startup=startup,nll=nll,diagnostics=dict(
        completed_steps=len(nll) if not failed else max(0,len(nll)-1),failure_reason=failed),
        clocks=dict(tracker.timings))


def _summary(A,queries,true_normal=None):return contact_summary(A,queries,true_normal=true_normal)


def _summary_flags(summary):
    d=summary['numerics'];overall=bool(d.get('numerical_ok',False))
    return dict(mean_ok=bool(d.get('mean_numerical_ok',overall)),
                state_ok=bool(d.get('regions_numerical_ok',d.get('state_regions_ok',overall))),
                interval_ok=bool(d.get('response_quantiles_ok',d.get('interval_numerical_ok',overall))),
                numerical_ok=overall)


def _report_loss(summary,truth,queries,alpha):
    M=np.asarray(summary['mean']);lower=np.asarray(summary['response_lower']);upper=np.asarray(summary['response_upper'])
    predictions=queries@M.T
    return dict(clean_mse=float(np.mean(np.sum((predictions-truth)**2,axis=1))),
                interval_score=float(np.mean(interval_score(lower,upper,truth,alpha))),
                clean_coverage=float(np.mean((truth>=lower)&(truth<=upper))),
                clean_width=float(np.mean(upper-lower)),
                posterior_risk=float(np.mean(summary.get('risks',[risk(M,w) for w in queries]))),
                mean_mode_difference=float(np.linalg.norm(M-summary['mode_projector'],'fro')),
                **_summary_flags(summary))


def validation_task(task):
    config,condition,trial,stage=task
    data=generate_contact(config['validation_seed'],trial,condition,**{k:config[k] for k in ['T','dt','d','v0','sigma']})
    queries=query_vectors(config['v0']);truth,_=query_truth(data,config['T'],queries,config['d'])
    rows=[];scope=_scope(condition);specs=candidate_specs(config,scope,extra=stage=='expansion')
    for spec in specs:
        start=perf_counter();method=spec['method']
        path=_gaussian_path(config,data,spec,[config['T']]) if method=='Recursive Gaussian' else _bingham_path(config,data,spec,[config['T']])
        snapshot=path['snapshots'].get(config['T']);base=dict(scope=scope,condition=condition,trial=trial,stage=stage,
            method=method,beta=spec.get('beta',np.nan),eta=spec.get('eta',np.nan),process_rms=spec.get('process_rms',np.nan),
            kappa=_kappa(spec) if 'process_rms' in spec else np.nan,
            predictive_nll=float(np.mean(path['nll'])) if snapshot is not None else np.inf,
            completed_steps=path['diagnostics']['completed_steps'],failure_reason=path['diagnostics']['failure_reason'])
        if snapshot is None:
            rows.append(dict(base,clean_mse=np.nan,interval_score=np.nan,numerical_ok=False,mean_ok=False,state_ok=False,interval_ok=False,
                             candidate_seconds=perf_counter()-start));continue
        try:
            if method=='Recursive Gaussian':
                from .contact_gaussian import gaussian_summary
                summary=gaussian_summary(snapshot['normal'],snapshot['covariance'],queries,
                                         true_normal=data['normals'][-1],basis=snapshot['basis'])
            else:summary=_summary(snapshot['A'],queries,data['normals'][-1])
            loss=_report_loss(summary,truth,queries,config['alpha'])
            rows.append(dict(base,**loss,candidate_seconds=perf_counter()-start))
            if method=='Scaled Bingham' and spec['eta']==1.:
                M=np.asarray(summary['mean']);P=np.asarray(summary['mode_projector'])
                for name,D in [('Point mean',M),('Point mode',P)]:
                    rows.append(dict(base,method=name,clean_mse=float(np.mean(np.sum((queries@D.T-truth)**2,axis=1))),
                        interval_score=np.nan,mean_ok=loss['mean_ok'] if name=='Point mean' else True,state_ok=True,interval_ok=True,
                        numerical_ok=loss['mean_ok'] if name=='Point mean' else True,candidate_seconds=0.,
                        reused_validation_path='Same beta, eta=1 scaled trajectory; no extra candidate execution'))
        except (ArithmeticError,np.linalg.LinAlgError,IntegrationWarning) as error:
            rows.append(dict(base,clean_mse=np.nan,interval_score=np.nan,numerical_ok=False,mean_ok=False,state_ok=False,interval_ok=False,
                             failure_reason=str(error),candidate_seconds=perf_counter()-start))
    return rows


def choose_policies(validation,config):
    """Select by the declared loss on independent validation, retaining all flags."""
    frame=pd.DataFrame(validation);selected=[];curves=[]
    for (scope,method),g in frame.groupby(['scope','method'],sort=False):
        keys=['beta'] if method in ['Point mode','Point mean'] else ['beta','eta'] if method=='Scaled Bingham' else ['process_rms']
        objective='clean_mse' if method in ['Point mode','Point mean'] else 'predictive_nll'
        options=[]
        for settings,h in g.groupby(keys,sort=False,dropna=False):
            settings=settings if isinstance(settings,tuple) else (settings,)
            p=dict(zip(keys,map(float,settings)))
            # Missing/unresolved candidates cannot become a favorable score by
            # dropping a trajectory. Every scope must contain all planned rows.
            expected=config['validation_trials']*(1 if scope=='plane' else len(config['curved_conditions']))
            objective_ok=bool(len(h)==expected and np.isfinite(h[objective]).all()
                              and h.completed_steps.eq(config['T']).all() and h.mean_ok.astype(bool).all())
            independent=h.groupby('trial')[objective].mean()
            score=float(independent.mean()) if objective_ok else np.inf
            interval_by_trial=h.groupby('trial').interval_score.mean()
            point_by_trial=h.groupby('trial').clean_mse.mean()
            row=dict(scope=scope,method=method,**p,selection_loss=objective,selection_value=score,
                mean_interval_score=float(interval_by_trial.mean()),mean_clean_mse=float(point_by_trial.mean()),
                independent_trials=h.trial.nunique(),evaluations=len(h),
                mean_failures=int((~h.mean_ok.astype(bool)).sum()),state_failures=int((~h.state_ok.astype(bool)).sum()),
                interval_failures=int((~h.interval_ok.astype(bool)).sum()),
                failed_trajectories=int((h.completed_steps<config['T']).sum()),
                candidate_seconds=float(h.candidate_seconds.sum()),selected=False,
                selected_by_interval_score=False,objective_resolved=objective_ok)
            curves.append(row);options.append((score,len(curves)-1))
        score,index=min(options,key=lambda item:item[0]);curves[index]['selected']=True
        chosen=curves[index]
        selected.append(dict(scope=scope,method=method,**{key:chosen[key] for key in keys},
                             selection_resolved=bool(np.isfinite(score)),mean_failures=chosen['mean_failures'],
                             state_failures=chosen['state_failures'],interval_failures=chosen['interval_failures']))
        finite=[i for _,i in options if np.isfinite(curves[i]['mean_interval_score'])]
        if finite:
            winner=min(finite,key=lambda i:curves[i]['mean_interval_score']);curves[winner]['selected_by_interval_score']=True
    return selected,pd.DataFrame(curves)


def needs_expansion(selected,config):
    beta=any(p.get('beta')==min(config['betas']) for p in selected)
    process=any(p.get('process_rms')==max(config['process_rms']) for p in selected)
    return dict(beta=beta,process=process)


def _policy(selected,condition,method):
    scope=_scope(condition)
    return next(p for p in selected if p['scope']==scope and p['method']==method)


def _normal_error(normal,true_normal):
    dot=float(normal@true_normal);truth=true_normal if dot>=0 else -true_normal
    dot=abs(dot);theta=float(np.arccos(np.clip(dot,0.,1.)))
    tangent=truth-dot*normal
    return tangent*(theta/max(np.linalg.norm(tangent),1e-15)) if theta>1e-12 else np.zeros(3)


def _directional(A,summary,true_normal,previous_normal):
    """Scoring-only motion/cross-motion axes avoid Haar sign cancellation."""
    values,U=np.linalg.eigh(-A);values=values[::-1];U=U[:,::-1]
    normal=U[:,0];E=U[:,1:];gaps=values[0]-values[1:]
    quadrature=normal_quadrature(-A,radial_order=32,angular_order=96)
    nodes=quadrature['normal_nodes'];weights=quadrature['weights'];dot=nodes@normal
    nodes=np.where((dot<0)[:,None],-nodes,nodes);dot=abs(dot)
    theta=np.arccos(np.clip(dot,0.,1.));sine=np.sqrt(np.maximum(0.,1-dot*dot))
    logs=(nodes-dot[:,None]*normal)*(theta/np.maximum(sine,1e-15))[:,None]
    cov=(logs.T*weights)@logs
    laplace=E@np.diag(np.divide(1.,2*gaps,out=np.full(2,np.nan),where=gaps>1e-12))@E.T
    return _direction_rows(normal,E,cov,laplace,gaps,true_normal,previous_normal)


def _direction_rows(normal,E,cov,laplace,gaps,true_normal,previous_normal):
    # Antipodal normals represent the same plane. Align only in scoring so
    # arbitrary eigensolver signs cannot cancel physical drift bias.
    if normal@true_normal<0:normal=-normal;E=-E
    motion=true_normal-previous_normal
    motion=motion-normal*(normal@motion)
    moving=np.linalg.norm(motion)>1e-10
    if moving:first=motion/np.linalg.norm(motion)
    else:
        axis=np.eye(3)[:,int(np.argmin(abs(normal)))];first=axis-normal*(normal@axis)
        first/=np.linalg.norm(first)
    second=np.cross(normal,first);second/=np.linalg.norm(second)
    error=_normal_error(normal,true_normal);rows=[]
    for label,direction in [('Motion direction' if moving else 'Static direction 1',first),
                            ('Cross-motion direction' if moving else 'Static direction 2',second)]:
        rows.append(dict(direction=label,tangent_error=float(direction@error),
            posterior_sd=float(np.sqrt(max(0.,direction@cov@direction))),
            laplace_sd=float(np.sqrt(max(0.,direction@laplace@direction))) if np.isfinite(laplace).all() else np.nan,
            minimum_gap=float(np.min(gaps)),maximum_gap=float(np.max(gaps)),
            axes='Actual mechanical motion direction projected at estimated normal; scoring only',
            width_kind='Standard deviation in shortest normal-line logarithm coordinates; not a claimed central interval'))
    return rows


def _failed_report(condition,trial,t,method,spec,reason):
    return dict(condition=condition,trial=trial,t=t,method=method,
                beta=spec.get('beta',np.nan),eta=spec.get('eta',np.nan),process_rms=spec.get('process_rms',np.nan),
                clean_mse=np.nan,interval_score=np.nan,clean_coverage=np.nan,clean_width=np.nan,posterior_risk=np.nan,
                ball_covered=np.nan,hpd_covered=np.nan,mean_ok=False,state_ok=False,interval_ok=False,numerical_ok=False,
                failure_reason=reason,primary=condition=='sphere_small' and t==600)


def test_task(task):
    config,condition,trial,selected=task
    data=generate_contact(config['test_seed'],trial,condition,**{key:config[key] for key in ['T','dt','d','v0','sigma']})
    queries=query_vectors(config['v0']);metrics=[];components=[];directions=[];shapes=[];startup=[]
    # Point-only recursions do not pay for predictive density calculations.
    point_specs={method:_policy(selected,condition,method) for method in ['Point mode','Point mean']}
    point_As={beta:np.zeros((3,3)) for beta in {p['beta'] for p in point_specs.values()}}
    point_times={beta:0. for beta in point_As};point_snapshots={}
    for t,(u,v) in enumerate(zip(data['U'],data['V']),1):
        for beta in point_As:
            begin=perf_counter();point_As[beta]=center(beta*point_As[beta]+increment(u,config['sigma']**2,v=v))
            # The modal point implementation returns its basis each update.
            basis=np.linalg.eigh(point_As[beta])[1][:,-2:]
            point_times[beta]+=perf_counter()-begin
            if t in config['report_times']:point_snapshots[beta,t]=(point_As[beta].copy(),projector(basis),point_times[beta])
    for method,spec in point_specs.items():
        for t in config['report_times']:
            A,P,seconds=point_snapshots[spec['beta'],t];truth,_=query_truth(data,t,queries,config['d'])
            begin=perf_counter()
            if method=='Point mean':
                evaluated=_process_api()[0](A,checked=True);D=evaluated.mean;mean_ok=_backend_ok(evaluated)
            else:D=P;mean_ok=True
            mean_seconds=perf_counter()-begin
            least_squares,S,_=weighted_projector_ls(data['U'][:t],data['V'][:t],spec['beta'])
            if np.linalg.norm(P-least_squares)>1e-8:raise ArithmeticError('Point mode differs from weighted projector LS')
            predictions=queries@D.T
            metrics.append(dict(condition=condition,trial=trial,t=t,method=method,beta=spec['beta'],eta=1.,
                process_rms=np.nan,clean_mse=float(np.mean(np.sum((predictions-truth)**2,axis=1))),
                interval_score=np.nan,clean_coverage=np.nan,clean_width=np.nan,posterior_risk=float(np.mean([risk(D,w) for w in queries])) if method=='Point mean' else np.nan,
                ball_covered=np.nan,hpd_covered=np.nan,mean_ok=mean_ok,state_ok=False,interval_ok=False,numerical_ok=mean_ok,
                update_seconds=seconds,prediction_seconds=0.,inverse_seconds=0.,score_seconds=0.,
                report_seconds=mean_seconds,report_mean_seconds=mean_seconds,report_region_seconds=0.,report_interval_seconds=0.,
                predictive_nll=np.nan,mean_mode_difference=float(np.linalg.norm(D-P)),failure_reason='',
                information='Point prediction memory selected by validation clean-response MSE'))
    for method in UNCERTAINTY_METHODS:
        spec=_policy(selected,condition,method)
        path=_gaussian_path(config,data,spec,config['report_times'],True) if method=='Recursive Gaussian' else _bingham_path(config,data,spec,config['report_times'],True)
        for row in path['startup']:startup.append(dict(condition=condition,trial=trial,method=method,**row))
        for t in config['report_times']:
            snapshot=path['snapshots'].get(t)
            if snapshot is None:
                for label in [method]+([method.replace('Bingham','Laplace')] if 'Bingham' in method else []):
                    metrics.append(_failed_report(condition,trial,t,label,spec,path['diagnostics']['failure_reason']))
                continue
            truth,geometric=query_truth(data,t,queries,config['d']);normal=data['normals'][t-1]
            before=data['normals'][max(0,t-2)];reports=[]
            if method=='Recursive Gaussian':
                from .contact_gaussian import gaussian_summary
                try:
                    begin=perf_counter();full=gaussian_summary(snapshot['normal'],snapshot['covariance'],queries,
                        true_normal=normal,basis=snapshot['basis']);elapsed=perf_counter()-begin
                except (ArithmeticError,np.linalg.LinAlgError,IntegrationWarning) as error:
                    metrics.append(_failed_report(condition,trial,t,method,spec,str(error)));continue
                reports.append((method,full,elapsed))
            else:
                from .contact_gaussian import laplace_summary
                try:
                    begin=perf_counter();full=_summary(snapshot['A'],queries,normal);elapsed=perf_counter()-begin
                except (ArithmeticError,np.linalg.LinAlgError,IntegrationWarning) as error:
                    for label in [method,method.replace('Bingham','Laplace')]:
                        metrics.append(_failed_report(condition,trial,t,label,spec,str(error)))
                    continue
                reports.append((method,full,elapsed))
                try:
                    begin=perf_counter();local=laplace_summary(snapshot['A'],queries,true_normal=normal);local_elapsed=perf_counter()-begin
                    reports.append((method.replace('Bingham','Laplace'),local,local_elapsed))
                except (ArithmeticError,np.linalg.LinAlgError,IntegrationWarning) as error:
                    metrics.append(_failed_report(condition,trial,t,method.replace('Bingham','Laplace'),spec,str(error)))
                    local=None
                rows=_directional(snapshot['A'],full,normal,before)
                directions.extend(dict(condition=condition,trial=trial,t=t,method=method,**r) for r in rows)
                if local is not None:shapes.append(dict(condition=condition,trial=trial,t=t,method=method,
                    mean_map_difference=float(np.linalg.norm(full['mean']-local['mean'])),
                    interval_endpoint_difference=float(max(np.max(abs(full['response_lower']-local['response_lower'])),np.max(abs(full['response_upper']-local['response_upper'])))),
                    interval_endpoint_difference_over_v0=float(max(np.max(abs(full['response_lower']-local['response_lower'])),np.max(abs(full['response_upper']-local['response_upper'])))/config['v0']),
                    full_mean_ok=_summary_flags(full)['mean_ok'],local_mean_ok=_summary_flags(local)['mean_ok'],
                    full_interval_ok=_summary_flags(full)['interval_ok'],local_interval_ok=_summary_flags(local)['interval_ok']))
            for label,summary,elapsed in reports:
                loss=_report_loss(summary,truth,queries,config['alpha']);d=summary['numerics'];M=np.asarray(summary['mean']);P=np.asarray(summary['mode_projector'])
                clocks=snapshot.get('timings',snapshot)
                update_seconds=float(clocks.get('update_seconds',clocks.get('correction_seconds',0.)))
                prediction_seconds=float(clocks.get('prediction_seconds',0.))
                inverse_seconds=float(clocks.get('inverse_seconds',clocks.get('inversion_seconds',0.)))
                score_seconds=float(clocks.get('score_seconds',clocks.get('log_score_seconds',0.)))
                mean_time=float(d.get('mean_seconds',0.));interval_time=float(d.get('quantile_seconds',d.get('interval_seconds',0.)))
                region_time=max(0.,elapsed-mean_time-interval_time)
                metrics.append(dict(condition=condition,trial=trial,t=t,method=label,**loss,
                    beta=spec.get('beta',np.nan),eta=spec.get('eta',np.nan),process_rms=spec.get('process_rms',np.nan),
                    kappa=_kappa(spec) if 'process_rms' in spec else np.nan,
                    normalized_interval_score=loss['interval_score']/config['v0'],normalized_width=loss['clean_width']/config['v0'],
                    ball_covered=bool(summary.get('ball_covered',False)),hpd_covered=bool(summary.get('hpd_covered',False)),
                    ball_radius=float(summary.get('ball_radius',np.nan)),hpd_threshold=float(summary.get('hpd_threshold',np.nan)),
                    hpd_kind=str(summary.get('hpd_kind',d.get('hpd_kind','Bingham density level'))),
                    modal_error=float(np.sqrt(max(0.,2-np.trace(P@data['tangent_projectors'][t-1])))) if np.isfinite(P).all() else np.nan,
                    predictive_nll=snapshot['nll_mean'],terminal_nll=snapshot['terminal_nll'],
                    update_seconds=update_seconds,prediction_seconds=prediction_seconds,inverse_seconds=inverse_seconds,score_seconds=score_seconds,
                    report_seconds=elapsed,report_mean_seconds=mean_time,report_region_seconds=region_time,report_interval_seconds=interval_time,
                    maximum_inverse_residual=float(snapshot.get('maximum_inverse_residual',np.nan)),
                    maximum_moment_uncertainty=float(snapshot.get('maximum_moment_uncertainty',np.nan)),
                    local_approximation_ok=d.get('local_approximation_ok',np.nan),
                    integration_numerical_ok=d.get('integration_numerical_ok',loss['mean_ok'] and loss['state_ok'] and loss['interval_ok']),
                    cut_locus_mass=d.get('cut_locus_mass',np.nan),
                    interval_angular_order=d.get('quantile_angular_order',d.get('angular_order',np.nan)),
                    interval_resolution_error=d.get('quantile_relative_error',np.nan),
                    failure_reason='',geometric_clean_mse=float(np.mean(np.sum((queries@M.T-geometric)**2,axis=1))),
                    likelihood='Known scalar resistance maintained; physical anisotropic stress scored against actual mobility' if condition=='anisotropic' else 'Matched isotropic mechanical response'))
                lower=np.asarray(summary['response_lower']);upper=np.asarray(summary['response_upper'])
                prediction=queries@M.T;mode_prediction=queries@P.T
                for q in range(len(queries)):
                    for component in range(3):
                        components.append(dict(condition=condition,trial=trial,t=t,method=label,query=q,component=component,
                            truth=float(truth[q,component]),geometric_truth=float(geometric[q,component]),
                            mean=float(prediction[q,component]),mode=float(mode_prediction[q,component]),
                            lower=float(lower[q,component]),upper=float(upper[q,component]),
                            covered=bool(lower[q,component]<=truth[q,component]<=upper[q,component]),
                            interval_score=float(interval_score(lower[q,component],upper[q,component],truth[q,component],config['alpha'])),
                            interval_ok=loss['interval_ok']))
            if method=='Recursive Gaussian':
                n=snapshot['normal'];E=snapshot['basis'];cov=E@snapshot['covariance']@E.T
                rows=_direction_rows(n,E,cov,cov,np.array([np.nan,np.nan]),normal,before)
                directions.extend(dict(condition=condition,trial=trial,t=t,method=method,**r) for r in rows)
    return dict(metrics=metrics,components=components,directional=directions,shape_comparison=shapes,startup=startup)


def summaries(frames,config):
    from .reporting import wilson
    m=frames['metrics'];paired=[];coverage=[];costs=[]
    terminal=m[m.t==config['T']]
    for condition,g in terminal.groupby('condition',sort=False):
        pivot=g.pivot(index='trial',columns='method',values='interval_score')
        for reference in ['Scaled Bingham','Recursive Gaussian','Process Laplace','Scaled Laplace','Point mean']:
            if reference not in pivot:continue
            gain=pivot[reference]-pivot['Process Bingham'];finite=np.isfinite(gain)
            n=len(gain);resolved=bool(finite.all())
            sem=float(gain.std(ddof=1)/np.sqrt(n)) if resolved and n>1 else np.nan
            half=float(student_t.ppf(.975,n-1)*sem) if resolved else np.nan
            selected=g[g.method.isin(['Process Bingham',reference])]
            failures=int((~selected.interval_ok.astype(bool)).sum())
            primary=condition==config['primary_condition'] and reference in config['primary_references']
            paired.append(dict(condition=condition,reference=reference,trials=n,gain=float(gain.mean()) if resolved else np.nan,
                low=float(gain.mean()-half) if resolved else np.nan,high=float(gain.mean()+half) if resolved else np.nan,
                process_score=float(pivot['Process Bingham'].mean()),reference_score=float(pivot[reference].mean()),
                primary=primary,interval_failures=failures,resolved=resolved and failures==0,
                scope='Primary preselected comparison' if primary else 'Exploratory pointwise comparison'))
    for (condition,t,method),g in m.groupby(['condition','t','method'],sort=False):
        if method in ['Point mode','Point mean']:continue
        for flag,width,kind in [('ball_covered','ball_radius','State ball'),('hpd_covered','hpd_threshold','State region')]:
            valid=g[flag].notna();success=int(g.loc[valid,flag].astype(bool).sum());n=len(g)
            low,high=wilson(success,n) if n else (np.nan,np.nan)
            coverage.append(dict(condition=condition,t=t,method=method,kind=kind,coverage=success/n,
                low=low,high=high,mean_width=float(g[width].mean()),trials=n,successes=success,
                unresolved=int((~g.state_ok.astype(bool)).sum()),interval='Wilson across trajectories; unresolved states included in denominator'))
        for column,kind in [('clean_coverage','Clean response'),('interval_score','Interval score')]:
            n=len(g);values=g[column];resolved=np.isfinite(values).all()
            sem=float(values.std(ddof=1)/np.sqrt(n)) if resolved and n>1 else np.nan
            half=float(student_t.ppf(.975,n-1)*sem) if resolved else np.nan
            coverage.append(dict(condition=condition,t=t,method=method,kind=kind,
                coverage=float(values.mean()) if resolved else np.nan,
                low=float(values.mean()-half) if resolved else np.nan,high=float(values.mean()+half) if resolved else np.nan,
                mean_width=float(g.clean_width.mean()),trials=n,successes=np.nan,
                unresolved=int((~g.interval_ok.astype(bool)).sum()),interval='Student across trajectory averages; queries/components are not replicates'))
    for (condition,method),g in m.groupby(['condition','method'],sort=False):
        independent=g.groupby('trial').agg(update_seconds=('update_seconds','max'),prediction_seconds=('prediction_seconds','max'),
            inverse_seconds=('inverse_seconds','max'),score_seconds=('score_seconds','max'),
            report_seconds=('report_seconds','sum'),report_mean_seconds=('report_mean_seconds','sum'),
            report_region_seconds=('report_region_seconds','sum'),report_interval_seconds=('report_interval_seconds','sum'))
        costs.append(dict(condition=condition,method=method,**independent.mean().to_dict(),
                          trials=len(independent),reports_per_trajectory=len(config['report_times']),
                          scope='Cumulative per trajectory; inverse nested in prediction and report components nested in report total'))
    return dict(paired=pd.DataFrame(paired),coverage=pd.DataFrame(coverage),costs=pd.DataFrame(costs))
