"""Controlled low-data posterior decisions with identical passive observations.

This is a stationary real Gr(1,3) experiment with a Haar initial subspace and
known noise levels. It isolates the value of posterior decisions, rather than
claiming general tracking superiority in the larger drifting applications.
"""
from functools import lru_cache
from time import perf_counter
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.stats import t as student_t
from benchmarks.case2 import GROUSE, EWCovariancePCA
from benchmarks.case1 import LoadedCovariance
from .filter import BinghamFilter
from .geometry import projector


@lru_cache(maxsize=8)
def _nodes(order):
    q,w=leggauss(order)
    phi=2*np.pi*np.arange(2*order)/(2*order)
    q=np.repeat(q,len(phi));phi=np.tile(phi,order)
    radius=np.sqrt(1-q*q)
    squares=np.column_stack((q*q,radius*radius*np.cos(phi)**2,radius*radius*np.sin(phi)**2))
    weights=np.repeat(w/(4*order),2*order)
    return squares,weights


def mean_projector(A,order=48):
    """Normalized invariant-measure sphere quadrature, with cached nodes.

    Antipodal symmetry identifies the sphere calculation with Gr(1,3). The
    largest-eigenvalue axis is placed at the poles for concentrated targets.
    Accuracy is checked by order doubling in the study, not inferred from PSD.
    """
    lam,U=np.linalg.eigh((A+A.T)/2);lam=lam[::-1];U=U[:,::-1]
    squares,weights=_nodes(order)
    weight=weights*np.exp(squares@(lam-lam[0]));mass=weight.sum()
    m=weight@squares/mass
    return (U*m)@U.T


def _trajectory(seed,regime,trial,T,s,tau2,validation=False):
    rng=np.random.default_rng(np.random.SeedSequence([seed,regime,trial,100 if validation else 200]))
    x=rng.normal(size=3);x/=np.linalg.norm(x)
    clean=np.sqrt(tau2)*rng.normal(size=(T,1))*x
    u=clean+np.sqrt(s)*rng.normal(size=(T,3))
    queries=rng.normal(size=(8,3));queries/=np.linalg.norm(queries,axis=1,keepdims=True)
    init_seed=np.random.SeedSequence([seed,regime,trial,300 if validation else 400])
    return x,clean,u,queries,init_seed


def _projection_loss(action,P,queries):
    return float(np.mean(np.sum((queries@(action-P).T)**2,axis=1)))


def _beam_weights(covariance,queries):
    """Batched standard unit-gain MVDR solves, one desired query per row."""
    solution=np.linalg.solve(covariance,queries.T).T
    denominator=np.sum(queries.conj()*solution,axis=1).real
    if np.any(denominator<=0):raise ArithmeticError('Invalid MVDR normalization')
    return solution/denominator[:,None]


def _power(weights,covariance):
    return float(np.mean(np.einsum('bi,ij,bj->b',weights.conj(),covariance,weights).real))


def _tune(seed,regime,s,tau2,checkpoints,trials,steps,rhos,loadings):
    tasks=('Projection','Reconstruction','Beamforming')
    families={task:[('GROUSE',steps),('Shrunk PCA',rhos)] for task in tasks}
    families['Beamforming'].append(('Loaded covariance',loadings))
    values={(task,method,t,param):[] for task in tasks for method,candidates in families[task]
            for t in checkpoints for param in candidates}
    shrink=tau2/(s+tau2)
    for trial in range(trials):
        x,clean,U,V,initial=_trajectory(seed,regime,trial,max(checkpoints),s,tau2,True)
        P=np.outer(x,x);scatter=np.zeros((3,3));R=s*np.eye(3)+tau2*P
        trackers={v:GROUSE(3,1,step=v,rng=np.random.default_rng(initial)) for v in steps}
        loaded={v:LoadedCovariance(3,beta=1.,loading=v) for v in loadings}
        for t,u in enumerate(U,1):
            scatter+=np.outer(u,u)
            for tracker in trackers.values():tracker.update(u)
            for tracker in loaded.values():tracker.update(u)
            if t not in checkpoints:continue
            mode=np.linalg.eigh(scatter)[1][:,-1:];Pm=projector(mode)
            for task in tasks:
                for method,candidates in families[task]:
                    for param in candidates:
                        action=(projector(trackers[param].basis) if method=='GROUSE' else
                                param*Pm+(1-param)*np.eye(3)/3 if method=='Shrunk PCA' else None)
                        if task=='Projection':loss=_projection_loss(action,P,V)
                        elif task=='Reconstruction':loss=float(np.sum((shrink*action@u-clean[t-1])**2))
                        else:
                            covariance=loaded[param].covariance if method=='Loaded covariance' else s*np.eye(3)+tau2*action
                            loss=_power(_beam_weights(covariance,V),R)
                        values[task,method,t,param].append(loss)
    rows=[];selected={}
    for task in tasks:
        for method,candidates in families[task]:
            for t in checkpoints:
                losses=[np.mean(values[task,method,t,p]) for p in candidates]
                best=candidates[int(np.argmin(losses))]
                selected[task,method,t]=best
                for param,loss in zip(candidates,losses):
                    rows.append(dict(regime=regime,s=s,tau2=tau2,task=task,method=method,t=t,
                        parameter='step' if method=='GROUSE' else 'rho' if method=='Shrunk PCA' else 'loading',
                        value=param,validation_loss=float(loss),selected=param==best,
                        selected_boundary=bool(param==best and param in (candidates[0],candidates[-1]))))
    return selected,rows


def paired_summary(metrics):
    """Paired 95% Student intervals across independent latent/data trajectories.

    Eight query directions are averaged within each trajectory. They are not
    counted as eight independent replicates. Positive gains favor the mean.
    """
    out=[]
    for (regime,t),g in metrics.groupby(['regime','t']):
        for task,column in [('Projection','projection_loss'),('Reconstruction','reconstruction_loss'),('Beamforming','beamforming_loss')]:
            table=g.pivot(index='trial',columns='method',values=column)
            mean=table['Posterior mean']
            for method in table:
                if method=='Posterior mean' or table[method].isna().all():continue
                baseline=table[method];d=baseline-mean;n=len(d)
                sem=float(d.std(ddof=1)/np.sqrt(n));half=float(student_t.ppf(.975,n-1)*sem)
                denominator=float(baseline.mean())
                ratio=float(100*(denominator-mean.mean())/denominator) if denominator>0 else np.nan
                out.append(dict(regime=regime,t=t,task=task,reference=method,trials=n,
                    baseline_loss=float(baseline.mean()),posterior_loss=float(mean.mean()),
                    paired_gain=float(d.mean()),gain_low=float(d.mean()-half),gain_high=float(d.mean()+half),
                    reduction_percent=ratio,win_fraction=float(np.mean(d>0))))
    return pd.DataFrame(out)


def _coverage_summary(coverage):
    rows=[]
    for (regime,s,t),group in coverage.groupby(['regime','s','t']):
        for region in ('ball','hpd'):
            n=len(group);successes=int(group[region+'_covered'].sum());p=successes/n;z=1.96
            denominator=1+z*z/n;mid=(p+z*z/(2*n))/denominator
            half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
            rows.append(dict(regime=regime,s=s,t=t,region='Ball' if region=='ball' else 'HPD',
                successes=successes,trials=n,coverage=p,low=min(p,max(0.,mid-half)),high=max(p,min(1.,mid+half)),nominal_mass=.95))
    return pd.DataFrame(rows)


def run_decision_study(seed=86021,trials=400,validation_trials=160,
                       checkpoints=(1,2,4,8,16),noise_levels=(1.,.2),tau2=1.,
                       steps=(.02,.05,.08,.2,.35,.5,1.,2.,4.),rhos=(0.,.1,.25,.4,.55,.7,.85,.95,1.),
                       loadings=(.01,.03,.1,.3,1.,3.,10.,30.,1e6),order=48,reference_order=96,tolerance=1e-8,coverage_trials=100):
    """Three downstream tasks from the same low-dimensional passive data.

    Fresh projection/desired steering vectors contain no state information and
    are supplied to all methods only at decision time. Point and shrinkage
    methods have nine validation candidates per task/horizon/noise level.
    Loaded covariance is a nine-candidate beamforming-only control. Noise and
    rank are known and shared. No observed performance ranking is asserted.
    """
    checkpoints=tuple(sorted(set(checkpoints)))
    if trials<2 or validation_trials<2 or min(checkpoints)<1 or not len(steps)==len(rhos)==len(loadings) or len(steps)<2:
        raise ValueError('Positive times, at least two trials and equal candidate counts required')
    rows=[];tuning=[];accuracy=[];coverage=[]
    for regime,s in enumerate(noise_levels):
        selected,selected_rows=_tune(seed,regime,s,tau2,checkpoints,validation_trials,steps,rhos,loadings)
        tuning.extend(selected_rows)
        for trial in range(trials):
            x,clean,U,V,initial=_trajectory(seed,regime,trial,max(checkpoints),s,tau2)
            P=np.outer(x,x);R=s*np.eye(3)+tau2*P
            f=BinghamFilter(3,1,s,tau2,beta=1);ew=EWCovariancePCA(3,1,beta=1)
            trackers={v:GROUSE(3,1,step=v,rng=np.random.default_rng(initial)) for v in steps}
            loaded={v:LoadedCovariance(3,beta=1.,loading=v) for v in loadings}
            times={p:0. for p in steps};loaded_times={p:0. for p in loadings};filter_time=ew_time=0.
            for t,u in enumerate(U,1):
                start=perf_counter();f.update(u);filter_time+=perf_counter()-start
                start=perf_counter();ew.update(u);ew_time+=perf_counter()-start
                for param,tracker in trackers.items():
                    start=perf_counter();tracker.update(u);times[param]+=perf_counter()-start
                for param,tracker in loaded.items():
                    start=perf_counter();tracker.update(u);loaded_times[param]+=perf_counter()-start
                if t not in checkpoints:continue
                start=perf_counter();M=mean_projector(f.A,order);moment_seconds=perf_counter()-start
                # Reference calculations validate the evaluator, and are not
                # charged as production posterior decision cost.
                audit_start=perf_counter();low=mean_projector(f.A,24);reference=mean_projector(f.A,reference_order)
                error=float(np.max(abs(M-reference)));used_order=order
                if error>tolerance:
                    used_order=2*reference_order;start=perf_counter();M=mean_projector(f.A,used_order)
                    moment_seconds+=perf_counter()-start
                    high=mean_projector(f.A,2*used_order);error=float(np.max(abs(M-high)))
                audit_seconds=perf_counter()-audit_start
                if error>tolerance:raise ArithmeticError('Quadrature accuracy budget unresolved')
                if abs(np.trace(M)-1)>1e-10 or np.linalg.eigvalsh(M).min() < -1e-12:
                    raise ArithmeticError('Invalid numerical mean projector')
                Pm=projector(f.basis);Pe=projector(ew.basis)
                assert np.max(abs(Pm-Pe))<1e-10
                if trial<coverage_trials and t==max(checkpoints):
                    from .reference import rank_one_regions_gr13
                    start=perf_counter();regions=rank_one_regions_gr13(f.A,.95);integration_seconds=perf_counter()-start
                    ball_distance=float(np.sqrt(max(0.,1-np.trace(Pm@P))));hpd_loss=float(np.trace(f.A@(Pm-P)))
                    coverage.append(dict(regime=regime,s=s,trial=trial,t=t,mass=.95,
                        radius=regions.radius,hpd_threshold=regions.hpd_loss,
                        ball_covered=ball_distance<=regions.radius,hpd_covered=hpd_loss<=regions.hpd_loss,
                        ball_mass=regions.ball_mass,hpd_mass=regions.hpd_mass,integration_error=regions.integration_error,
                        integration_seconds=integration_seconds))
                accuracy.append(dict(regime=regime,s=s,trial=trial,t=t,order=used_order,
                    order24_difference=float(np.max(abs(low-reference))),reference_difference=error,
                    spectral_range=float(np.ptp(np.linalg.eigvalsh(f.A))),moment_seconds=moment_seconds,
                    quadrature_audit_seconds=audit_seconds))
                step_p=selected['Projection','GROUSE',t];step_r=selected['Reconstruction','GROUSE',t];step_b=selected['Beamforming','GROUSE',t]
                rho_p=selected['Projection','Shrunk PCA',t];rho_r=selected['Reconstruction','Shrunk PCA',t];rho_b=selected['Beamforming','Shrunk PCA',t]
                loading=selected['Beamforming','Loaded covariance',t]
                # Tuples give task-specific actions, update costs and selected
                # hyperparameters. Means are evaluated only when requested.
                methods={
                    'Posterior mean':(M,M,M,filter_time+moment_seconds,filter_time+moment_seconds,filter_time+moment_seconds,None,None,None),
                    'Mode':(Pm,Pm,Pm,filter_time,filter_time,filter_time,None,None,None),
                    'EW PCA':(Pe,Pe,Pe,ew_time,ew_time,ew_time,1.,1.,1.),
                    'GROUSE':(projector(trackers[step_p].basis),projector(trackers[step_r].basis),projector(trackers[step_b].basis),times[step_p],times[step_r],times[step_b],step_p,step_r,step_b),
                    'Shrunk PCA':(rho_p*Pe+(1-rho_p)*np.eye(3)/3,rho_r*Pe+(1-rho_r)*np.eye(3)/3,rho_b*Pe+(1-rho_b)*np.eye(3)/3,ew_time,ew_time,ew_time,rho_p,rho_r,rho_b),
                    'Prior mean':(np.eye(3)/3,np.eye(3)/3,np.eye(3)/3,0.,0.,0.,None,None,None),
                    'Oracle':(P,P,P,0.,0.,0.,None,None,None),
                    'Loaded covariance':(None,None,None,np.nan,np.nan,loaded_times[loading],None,None,loading),
                }
                Rbar=s*np.eye(3)+tau2*M;optimal=_beam_weights(Rbar,V)
                optimal_expected_power=_power(optimal,Rbar)
                posterior_risk=float(np.mean(np.einsum('bi,ij,bj->b',V,M-M@M,V)))
                for method,(action,reconstruction_action,beam_action,time_p,time_r,time_b,param_p,param_r,param_b) in methods.items():
                    if action is None:
                        projection_loss=reconstruction_loss=extra=np.nan
                    else:
                        start=perf_counter();response=V@action.T;time_p+=perf_counter()-start
                        projection_loss=float(np.mean(np.sum((response-V@P.T)**2,axis=1)))
                        start=perf_counter()
                        estimate=tau2/(3*s+tau2)*u if method=='Prior mean' else tau2/(s+tau2)*reconstruction_action@u
                        time_r+=perf_counter()-start
                        reconstruction_loss=float(np.sum((estimate-clean[t-1])**2))
                        extra=float(np.mean(np.sum((response-V@M.T)**2,axis=1)))
                    covariance=loaded[loading].covariance if method=='Loaded covariance' else s*np.eye(3)+tau2*beam_action
                    start=perf_counter();weights=_beam_weights(covariance,V);time_b+=perf_counter()-start
                    expected_power=_power(weights,Rbar);beam_excess=expected_power-optimal_expected_power
                    if beam_excess < -1e-10:raise ArithmeticError('Posterior minimum-variance identity failed')
                    gain_error=float(np.max(abs(np.sum(weights.conj()*V,axis=1)-1)))
                    if gain_error>1e-10:raise ArithmeticError('Distortionless gain failed')
                    rows.append(dict(regime=regime,s=s,tau2=tau2,trial=trial,t=t,method=method,
                        projection_loss=projection_loss,reconstruction_loss=reconstruction_loss,
                        beamforming_loss=_power(weights,R),beamforming_expected_loss=expected_power,
                        beamforming_posterior_excess=beam_excess,unit_gain_error=gain_error,
                        posterior_risk=posterior_risk,posterior_excess_risk=extra,
                        decision_seconds=time_p,projection_seconds=time_p,reconstruction_seconds=time_r,beamforming_seconds=time_b,
                        moment_seconds=moment_seconds if method=='Posterior mean' else 0.,
                        projection_parameter=param_p,reconstruction_parameter=param_r,beamforming_parameter=param_b))
        print(f'Decision study: noise variance {s:g}, {trials} trajectories complete',flush=True)
    metrics=pd.DataFrame(rows);coverage=pd.DataFrame(coverage)
    return dict(metrics=metrics,paired=paired_summary(metrics),coverage=coverage,coverage_summary=_coverage_summary(coverage),
        tuning=pd.DataFrame(tuning),accuracy=pd.DataFrame(accuracy),
        config=dict(seed=seed,trials=trials,validation_trials=validation_trials,checkpoints=list(checkpoints),
            noise_levels=list(noise_levels),tau2=tau2,n=3,k=1,query_directions_per_trial=8,
            steps=list(steps),rhos=list(rhos),loadings=list(loadings),candidate_count=len(steps),
            coverage_trials_per_regime=min(trials,coverage_trials),coverage_mass=.95,
            calibration_protocol='Exact joint Haar prior, stationary state and Gaussian likelihood; predeclared first coverage_trials trajectories at terminal time; not robustness under drift',
            quadrature_order=order,reference_order=reference_order,tolerance=tolerance,
            projection_protocol='Same passive observations; fresh independent unit queries supplied to every method at decision time',
            reconstruction_protocol='Same current observation; known noise; Prior mean uses isotropic LMMSE, other methods use conditional Wiener shrinkage',
            beamforming_protocol='Same passive snapshots; independent desired unit queries shared by all methods; average population interference-plus-noise output power, not a mean-SINR optimality claim',
            tuning='Matched nine-candidate families per task/horizon/noise on independent validation trajectories; grids broadened because validation selected previous grid boundaries, not because of test outcomes; constant GROUSE steps only; shrinkage includes both constrained endpoints and loading includes the isotropic limit; boundary selections retained',
            runtime_accounting='Per-task times include cumulative required updates, current requested mean and task action; other-horizon mean evaluations, quadrature audits, calibration integration and validation are excluded. Shrunk PCA uses the cheap EW-PCA implementation; Loaded covariance is beamforming-only; task times are not additive. Bingham/Mode timings include the general filter implementation overhead, not an asymptotic distinction from matching EW PCA.',
            state_process='Haar initial rank-one subspace, constant state, no forgetting',
            uncertainty='Paired 95% Student intervals over trajectories; eight within-trial queries averaged, not counted as independent replicates',
            scope='Controlled low-dimensional real Bayes decision study; not a claim of large-scale complex or drifting tracking superiority'))
