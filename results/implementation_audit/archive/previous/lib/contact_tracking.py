"""Causal mean-tracking experiment for mechanically generated contact data.

Truth enters plant diagnostics and scores only. The oracle policy explicitly
uses the current true normal and is never a feasible policy or tuning candidate.
Noiseless ablations retain the physical noise scale in the update (sigma**2).
"""
from __future__ import annotations

from collections import deque
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .contact_mechanics import generate_contact
from .contact_study import query_vectors, query_truth
from .geometry import center, projector
from .prediction import PredictiveBinghamFilter

CURVED = ['sphere_large', 'sphere_small', 'weak']
MAIN = ['plane', *CURVED]
INFORMATION_COLUMNS = ['normal_step','history_rotation','history_turn','history_rms_sine',
    'mean_observation_age','retained_weight','input_min','input_max',
    'fisher_min','fisher_max','current_fisher_min','current_fisher_max','normal_input_alignment']


def default_config():
    return dict(schema=1, validation_seed=6245201, test_seed=6245202,
        validation_trials=16, velocity_scan_trials=8, test_trials=64,
        T=600, dt=.05, d=10., v0=.02, sigma=.001,
        betas=[.2,.4,.6,.7,.8,.85,.9,.95,.98,1.],
        windows=[1,2,3,5,8,12,20], velocity_betas=[.6,.8,.9,.95],
        lags=[2,8,16], dampings=[.25,.5,1.], refine_candidates=3,
        speed_cap=None, report_times=[60,180,600],
        main_conditions=MAIN, curved_conditions=CURVED, stress_conditions=['anisotropic'],
        selection='Minimum terminal clean query MSE, equally weighted curved conditions; separate plane selection. Noiseless and oracle runs never select practical policies.',
        boundary='One validation-only expansion per finite search direction; structural beta=1 and damping=1 are not finite boundaries.',
        observation='Known force and noisy instantaneous velocity only; truth and positions used for scoring only.',
        primary_condition='sphere_small', primary_time=600,
        uncertainty='Stage 1 is mode-only; no coverage or posterior claim.',
        statistics='Independent trajectories; paired Student intervals are exploratory except declared primary contrasts.',
        fixed_update_variance='sigma**2 even when measurement noise is removed')


def policy_id(spec):
    kind = spec['prediction']
    if kind == 'rolling': return f"rolling_W{int(spec['likelihood_window'])}"
    if kind == 'velocity':
        return f"velocity_b{spec['beta']:g}_l{int(spec['lag'])}_d{spec['damping']:g}"
    if kind == 'oracle': return 'oracle_alignment'
    return f"identity_b{spec['beta']:g}"


def candidate_specs(config, kind):
    if kind == 'identity': return [dict(prediction=kind, beta=b) for b in config['betas']]
    if kind == 'rolling': return [dict(prediction=kind, beta=1., likelihood_window=w) for w in config['windows']]
    if kind == 'velocity':
        return [dict(prediction=kind, beta=b, lag=l, damping=d)
                for b in config['velocity_betas'] for l in config['lags'] for d in config['dampings']]
    raise ValueError('Unknown practical policy family')


def make_tracker(config, spec):
    kind = spec['prediction']
    if kind not in ('identity','rolling','velocity','oracle'): raise ValueError('Unknown forecast')
    return PredictiveBinghamFilter(3, 2, config['sigma']**2, tau2=None,
        beta=spec.get('beta',1.), prediction='identity' if kind == 'oracle' else kind,
        lag=spec.get('lag',8), damping=spec.get('damping',.5),
        likelihood_window=spec.get('likelihood_window',16),
        speed_cap=spec.get('speed_cap',config.get('speed_cap')))


def normal_alignment(previous, current):
    """Minimal plane rotation using truth: diagnostic only, never causal input."""
    a=np.asarray(previous,float); b=np.asarray(current,float)
    if a.shape != (3,) or b.shape != (3,): raise ValueError('Need unit three-vectors')
    if not np.isclose(a@a,1.) or not np.isclose(b@b,1.): raise ValueError('Need unit normals')
    if a@b < 0: b=-b  # n and -n specify the same plane
    cross=np.cross(a,b); K=np.array([[0.,-cross[2],cross[1]],[cross[2],0.,-cross[0]],[-cross[1],cross[0],0.]])
    return np.eye(3)+K+K@K/(1+float(a@b))


def _mode(A):
    values,basis=np.linalg.eigh(A)
    P=np.eye(3)-np.outer(basis[:,0],basis[:,0])
    return P, float(values[1]-values[0])


def _loss(P, normal, mobility, v0):
    # The nine query vectors form a tight frame Q.T Q=3*v0**2 I.
    return float(v0**2/3*np.sum((P-mobility)**2))


def mobility_maps(data, d):
    """Scoring map for hypothetical forces d*w; not available to estimators."""
    inv=np.linalg.inv(data['resistance']); a=data['normals']@inv.T
    denominator=np.sum(data['normals']*a,axis=1)
    return d*(inv[None,:,:]-np.einsum('ti,tj->tij',a,a)/denominator[:,None,None])


def replay(config, data, spec, noiseless=False, keep_arrays=True, audit=True):
    """Replay one policy, exposing raw A, pre-correction prior and causal G.

    Calling predict before reading the current observation makes the available
    information explicit. Practical G is derived from past modal estimates.
    The returned arrays can be reused by posterior methods without relearning G.
    Forecasts of the initial uniform parameter are marked nonidentified.
    """
    tracker=make_tracker(config,spec); T=len(data['U']); s=config['sigma']**2
    observations=data['velocities'] if noiseless else data['U']
    normals=np.asarray(data['normals']); truthP=np.asarray(data['tangent_projectors'])
    mobility=mobility_maps(data,config['d']); v0=config['v0']
    forecast=np.zeros(T); corrected=np.zeros(T); forecast_loss=np.zeros(T); losses=np.zeros(T)
    forecast_gap=np.zeros(T); corrected_gap=np.zeros(T)
    Aall=np.zeros((T,3,3)) if keep_arrays else None
    priorall=np.zeros((T,3,3)) if keep_arrays else None
    Gall=np.zeros((T,3,3)) if keep_arrays else None
    modeall=np.zeros((T,3,3)) if keep_arrays else None
    S=np.zeros((3,3)); C=np.zeros((3,3)); weight=age=0.; queue=deque()
    normal_sum=np.zeros(3); history_oldest=0
    # Untransported normal scatter describes physical rotation over retained
    # evidence; transported input C describes the actual likelihood information.
    N=np.zeros((3,3)); diagnostics=[]; reports=[]
    information_path=np.zeros((T,len(INFORMATION_COLUMNS))) if audit else None
    normal_steps=np.r_[0.,np.arccos(np.clip(np.abs(np.sum(normals[1:]*normals[:-1],axis=1)),0.,1.))]
    cumulative_turn=np.cumsum(normal_steps)
    reasons={}; caps=0; ls_parameter=ls_objective=ls_projector=0.
    for j in range(T):
        G=normal_alignment(normals[j-1],normals[j]) if spec['prediction']=='oracle' and j else None
        prior=tracker.predict(G=G)
        actualG=tracker.last_motion.copy(); diagnostic=tracker.motion_diagnostics.copy()
        reasons[diagnostic['reason']]=reasons.get(diagnostic['reason'],0)+1
        caps+=int(diagnostic['capped'])
        Pprior,forecast_gap[j]=_mode(prior)
        forecast[j]=max(0.,2-float(np.trace(Pprior@truthP[j])))
        forecast_loss[j]=_loss(Pprior,normals[j],mobility[j],v0)
        # No current u is used until this point.
        u=observations[j]; v=data['V'][j]
        basis=tracker.update(u,v=v); P=projector(basis)
        values=np.linalg.eigvalsh(tracker.A); corrected_gap[j]=values[1]-values[0]
        corrected[j]=max(0.,2-float(np.trace(P@truthP[j])))
        losses[j]=_loss(P,normals[j],mobility[j],v0)
        if keep_arrays:
            Aall[j]=tracker.A; priorall[j]=prior; Gall[j]=actualG; modeall[j]=P
        if audit:
            # Independent projector-LS sufficient statistic; no increment().
            increment=np.outer(u,v)+np.outer(v,u)-np.outer(v,v)
            ci=np.outer(v,v); ni=np.outer(normals[j],normals[j])
            if spec['prediction']=='rolling':
                queue.append((increment,ci,ni))
                if len(queue)>spec['likelihood_window']:queue.popleft()
                S=sum((x[0] for x in queue),np.zeros((3,3)))
                C=sum((x[1] for x in queue),np.zeros((3,3)))
                N=sum((x[2] for x in queue),np.zeros((3,3)))
                weight=float(len(queue)); age=(len(queue)-1)/2
                history_oldest=max(0,j-len(queue)+1)
            else:
                beta=spec.get('beta',1.)
                S=beta*actualG@S@actualG.T+increment
                C=beta*actualG@C@actualG.T+ci
                N=beta*N+ni
                age=beta*(age+weight);weight=beta*weight+1
                # First retained observation above 1% of the newest weight.
                horizon=j if beta==1 else int(np.floor(np.log(.01)/np.log(beta)))
                history_oldest=max(0,j-horizon)
            reference=center(S/(2*s)); difference=float(np.max(abs(reference-tracker.A)))
            ls_parameter=max(ls_parameter,difference)
            n=normals[j]; X=data['X'][j]
            info=(X.T@C@X+(n@C@n)*np.eye(2))/s
            # Local information is an aligned-state likelihood diagnostic,
            # not information about a misspecified drifting posterior.
            inst=((n@v)**2*np.eye(2)+np.outer(X.T@v,X.T@v))/s
            actual_step=float(normal_steps[j])
            rotation=float(np.arccos(np.clip(abs(n@normals[history_oldest]),0.,1.)))
            history_turn=float(cumulative_turn[j]-cumulative_turn[history_oldest])
            expected_scatter=N/max(weight,1.)
            physical_dispersion=float(np.sqrt(np.clip(1-n@expected_scatter@n,0.,1.)))
            input_eigen=np.linalg.eigvalsh(C); fisher=np.linalg.eigvalsh(info)
            instantaneous=np.linalg.eigvalsh(inst)
            information_path[j]=[actual_step,rotation,history_turn,physical_dispersion,
                age/weight if spec['prediction']!='rolling' else age,weight,
                input_eigen[0],input_eigen[-1],fisher[0],fisher[-1],
                instantaneous[0],instantaneous[-1],
                abs(n@v)/max(np.linalg.norm(v),np.finfo(float).tiny)]
            if j+1 in config['report_times'] or j==T-1:
                L,ls_gap=_mode(S); objective=abs(float(np.trace(S@(P-L))))
                pdifference=float(np.linalg.norm(P-L))
                ls_objective=max(ls_objective,objective)
                if ls_gap>1e-10: ls_projector=max(ls_projector,pdifference)
                if difference>1e-7 or objective>1e-10 or (ls_gap>1e-10 and pdifference>1e-8):
                    raise ArithmeticError('Mode and matched transported projector least squares disagree')
                diagnostics.append(dict(t=j+1,retained_oldest_index=history_oldest,
                    **dict(zip(INFORMATION_COLUMNS,information_path[j])),
                    ls_parameter_error=difference,ls_objective_error=objective,ls_projector_error=pdifference))
        if j+1 in config['report_times'] or j==T-1:
            reports.append(dict(t=j+1,forecast_chordal=float(np.sqrt(forecast[j])),
                corrected_chordal=float(np.sqrt(corrected[j])),forecast_mse=forecast_loss[j],
                corrected_mse=losses[j],time_average_mse=float(losses[:j+1].mean()),
                forecast_time_average_mse=float(forecast_loss[:j+1].mean()),
                forecast_identified=bool(forecast_gap[j]>1e-10),corrected_identified=bool(corrected_gap[j]>1e-10),
                prediction_seconds=tracker.prediction_seconds,correction_seconds=tracker.update_seconds))
    return dict(A=Aall,prior=priorall,G=Gall,mode_projectors=modeall,
        forecast_chordal=np.sqrt(forecast),corrected_chordal=np.sqrt(corrected),
        forecast_mse=forecast_loss,corrected_mse=losses,
        forecast_gap=forecast_gap,corrected_gap=corrected_gap,reports=reports,
        diagnostics=diagnostics,information_path=information_path,terminal_mse=float(losses[-1]),time_average_mse=float(losses.mean()),
        motion_reasons=reasons,capped_rotations=caps,ls_parameter_error=ls_parameter,
        ls_objective_error=ls_objective,ls_projector_error=ls_projector,
        fixed_update_variance=s,oracle=spec['prediction']=='oracle')


def plant_data(config, seed, trial, condition):
    return generate_contact(seed,trial,condition,**{k:config[k] for k in ['T','dt','d','v0','sigma']})


def validation_task(task):
    config,condition,trial,specs,stage=task
    data=plant_data(config,config['validation_seed'],trial,condition);rows=[]
    for spec in specs:
        path=replay(config,data,spec,keep_arrays=False,audit=False)
        rows.append(dict(scope='plane' if condition=='plane' else 'curved',condition=condition,
            trial=trial,stage=stage,policy=policy_id(spec),family=spec['prediction'],
            specification=spec,loss=path['terminal_mse'],time_average_mse=path['time_average_mse']))
    return rows


def choose_policies(validation, config):
    frame=pd.DataFrame(validation); rows=[];selected=[]
    for (scope,family,policy),portion in frame.groupby(['scope','family','policy'],sort=False):
        # A bounded velocity scan must be followed by complete 16-trial refinement.
        portion=portion.sort_values('stage').drop_duplicates(['condition','trial'],keep='last')
        trials=portion.groupby('trial').loss.mean()
        expected=config['validation_trials']*(1 if scope=='plane' else len(CURVED))
        complete=len(portion)==expected
        rows.append(dict(scope=scope,family=family,policy=policy,
            specification=portion.specification.iloc[0],loss=float(trials.mean()),
            se=float(trials.std(ddof=1)/np.sqrt(len(trials))),
            time_average_mse=float(portion.time_average_mse.mean()),
            independent_trials=len(trials),evaluations=len(portion),complete=complete,selected=False))
    for (scope,family),portion in pd.DataFrame(rows).groupby(['scope','family'],sort=False):
        eligible=portion[portion.complete]
        if eligible.empty:continue
        winner=eligible.sort_values(['loss','policy']).iloc[0]
        selected.append(dict(scope=scope,family=family,policy=winner.policy,specification=winner.specification,
            validation_loss=winner.loss))
        for row in rows:
            if row['scope']==scope and row['policy']==winner.policy:row['selected']=True
    return selected, rows


def selected_specs(condition, selected):
    if isinstance(selected,dict):selected=selected['selected']
    scope='plane' if condition=='plane' else 'curved'
    specs={row['family']:dict(row['specification']) for row in selected if row['scope']==scope}
    specs['original']=dict(prediction='identity',beta=1. if scope=='plane' else .8)
    specs['oracle']=dict(prediction='oracle',beta=1.)
    return specs


def test_task(task):
    config,condition,trial,selected=task
    data=plant_data(config,config['test_seed'],trial,condition)
    metrics=[];diagnostics=[];motions=[];errors={};arrays={};information={}
    for label,spec in selected_specs(condition,selected).items():
        for noiseless in (False,True):
            path=replay(config,data,spec,noiseless=noiseless,keep_arrays=trial==0,audit=True)
            common=dict(condition=condition,trial=trial,method=label,policy=policy_id(spec),
                noise='off' if noiseless else 'on',oracle=label=='oracle')
            metrics.extend([dict(common,**x) for x in path['reports']])
            diagnostics.extend([dict(common,**x) for x in path['diagnostics']])
            motions.append(dict(common,capped_rotations=path['capped_rotations'],
                ls_parameter_error=path['ls_parameter_error'],ls_objective_error=path['ls_objective_error'],
                ls_projector_error=path['ls_projector_error'],fixed_update_variance=path['fixed_update_variance'],
                **path['motion_reasons']))
            key=f"{condition}__{trial}__{label}__{common['noise']}"
            information[key]=path['information_path']
            errors[key]=np.column_stack([path['forecast_chordal'],path['corrected_chordal'],
                path['forecast_mse'],path['corrected_mse'],path['forecast_gap'],path['corrected_gap']])
            if trial==0:
                for field in ['A','prior','G','mode_projectors']:arrays[key+'__'+field]=path[field]
    if trial==0:
        for field in ['positions','normals','velocities','observations','free_velocities','times','tangent_projectors']:
            arrays[condition+'__'+field]=data[field]
    return dict(metrics=metrics,diagnostics=diagnostics,motions=motions,errors=errors,arrays=arrays,information=information)


def paired_summary(metrics, config, endpoint="corrected_mse"):
    frame=pd.DataFrame(metrics); rows=[]
    for (condition,noise,t),portion in frame.groupby(['condition','noise','t'],sort=False):
        wide=portion.pivot(index='trial',columns='method',values=endpoint)
        for method in wide:
            for reference in ['original','identity']:
                if method==reference or reference not in wide:continue
                differences=(wide[reference]-wide[method]).dropna();n=len(differences)
                half=float(student_t.ppf(.975,n-1)*differences.std(ddof=1)/np.sqrt(n))
                gain=float(differences.mean())
                rows.append(dict(condition=condition,noise=noise,t=t,endpoint=endpoint,method=method,reference=reference,
                    trials=n,method_mse=float(wide[method].mean()),reference_mse=float(wide[reference].mean()),
                    gain=gain,lower=gain-half,upper=gain+half,
                    relative_reduction=gain/float(wide[reference].mean()) if wide[reference].mean()>1e-20 else np.nan,
                    oracle=method=='oracle',primary=bool(condition==config['primary_condition'] and noise=='on' and t==config['T'] and reference=='original' and method!='oracle')))
    return pd.DataFrame(rows)
