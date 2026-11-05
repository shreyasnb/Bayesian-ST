"""Paired n=8,k=3 conditional tracking, posterior fans, and drift bounds.

Static and exact-frame controls are coupled by deterministic orthogonal
transport. Their posterior parameters and draws transform exactly; sampling is
therefore shared between those controls. Unknown-frame filters receive the same
physical observations on each moving path, but use G=I. Forgetting is an
approximate prediction rule, not a specified stochastic transition model.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import expm
from .geometry import projector,chordal,risk
from .filter import BinghamFilter
from .moments import sample_bingham,chain_diagnostics,effective_sample_size
from .theory import tracking_bound


def configurations(slow_rate=.003,fast_rate=.025,forgetting=.97):
    return [
        dict(protocol='static',label='Static',rate=0.,beta=1.,known=False,exact=True),
        dict(protocol='known_slow',label='Known motion',rate=slow_rate,beta=1.,known=True,exact=True),
        dict(protocol='identity_slow',label='G=I, slow drift',rate=slow_rate,beta=1.,known=False,exact=False),
        dict(protocol='forget_slow',label='G=I, slow drift, forgetting',rate=slow_rate,beta=forgetting,known=False,exact=False),
        dict(protocol='known_fast',label='Known fast motion',rate=fast_rate,beta=1.,known=True,exact=True),
        dict(protocol='identity_fast',label='G=I, fast drift',rate=fast_rate,beta=1.,known=False,exact=False),
        dict(protocol='forget_fast',label='G=I, fast drift, forgetting',rate=fast_rate,beta=forgetting,known=False,exact=False),
    ]


def generator(n=8):
    """A fixed, known skew generator independent of all latent subspaces."""
    K=np.zeros((n,n))
    for (i,j),speed in zip([(0,3),(1,4),(2,5),(6,7)],[1.,.7,.4,.3]):
        if j<n:K[j,i]=speed;K[i,j]=-speed
    return K


def paired_data(seed,trial,T=128,n=8,k=3,s=.03,input_norm=2.,slow_rate=.003,fast_rate=.025):
    """One Haar initial state, common canonical noise, preassigned patterns.

    The physical input at time t is exp(t*rate*K) times a coordinate cycle;
    it uses no hidden-state information. Physical noise is a rotation of the
    common iid Gaussian stream and remains iid N(0,sI).
    """
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial,4801]))
    X0=np.linalg.qr(rng.normal(size=(n,k)))[0]
    base_v=input_norm*np.eye(n)[np.arange(T)%n]
    base_u=base_v@projector(X0)+np.sqrt(s)*rng.normal(size=(T,n))
    P0=projector(X0);K=generator(n)
    data={}
    for rate in [0.,slow_rate,fast_rate]:
        rotations=np.array([expm((t+1)*rate*K) for t in range(T)])
        data[rate]=dict(G=expm(rate*K),rotations=rotations,
            X=rotations@X0,u=np.einsum('tij,tj->ti',rotations,base_u),
            v=np.einsum('tij,tj->ti',rotations,base_v))
    return X0,data


def _scalar_components(bases,rotation,component=1):
    X=np.einsum('ij,bjk->bik',rotation,bases)
    return np.sum(X[:,component,:]*X[:,0,:],axis=1)


def _scalar_diagnostics(chains,rotation):
    # Check both coordinates used in projected Monte Carlo clouds.
    summaries=[]
    for component in (1,2):
        values=[_scalar_components(c.bases,rotation,component) for c in chains]
        half=min(map(len,values))//2
        splits=np.array([v[:half] for v in values]+[v[-half:] for v in values])
        W=splits.var(axis=1,ddof=1).mean();B=half*splits.mean(axis=1).var(ddof=1)
        rhat=np.sqrt(((half-1)*W/half+B/half)/W) if W>1e-16 else 1.
        summaries.append((min(effective_sample_size(v) for v in values),max(1.,rhat)))
    return dict(component_min_ess=float(min(a[0] for a in summaries)),
                component_rhat=float(max(a[1] for a in summaries)))


def _posterior(A,k,seed,rotations,draws,burnin,max_draws,max_burnin):
    start=perf_counter();rng=np.random.default_rng(seed);attempts=0
    while True:
        attempts+=1
        random_initial=np.linalg.qr(rng.normal(size=(len(A),k)))[0]
        chains=[sample_bingham(A,k,rng,draws=draws,burnin=burnin),
                sample_bingham(A,k,rng,draws=draws,burnin=burnin,initial=random_initial)]
        diagnostics=chain_diagnostics(chains)
        components=[_scalar_diagnostics(chains,R) for R in rotations]
        ok=(diagnostics['min_ess']>=50 and diagnostics['max_rhat']<1.1
            and diagnostics['max_standardized_difference']<5
            and all(d['component_min_ess']>=50 and d['component_rhat']<1.1 for d in components))
        if ok or draws>=max_draws:break
        draws=min(max_draws,draws*2);burnin=min(max_burnin,max(1000,burnin*2))
    diagnostics.update(sampling_ok=bool(ok),actual_draws_per_chain=draws,actual_burnin=burnin,
                       sampling_attempts=attempts,sampler_seconds=perf_counter()-start)
    return (chains[0].mean+chains[1].mean)/2,chains,diagnostics


def _summary(A,Xhat,Xtrue,M,chains,rotation,diagnostics,seed,s):
    bases=np.concatenate([np.einsum('ij,bjk->bik',rotation,c.bases) for c in chains])
    component=np.sum(bases[:,1,:]*bases[:,0,:],axis=1)
    Pm=projector(Xhat);Pt=projector(Xtrue);n,k=Xhat.shape
    distances=np.sqrt(np.maximum(0.,k-np.einsum('bni,nm,bmi->b',bases,Pm,bases)))
    losses=np.trace(A@Pm)-np.einsum('bni,nm,bmi->b',bases,A,bases)
    radius=float(np.quantile(distances,.95));h=float(np.quantile(losses,.95))
    quantiles=np.quantile(component,[.025,.10,.25,.50,.75,.90,.975])
    q=np.eye(n)[:,0];w=np.eye(n)[:,1]
    clean=float(w@Pt@q);predicted=float(w@M@q)
    independent=np.random.default_rng(np.random.SeedSequence([seed,2191]))
    noisy_component=component+np.sqrt(s)*independent.normal(size=len(component))
    fresh_observation=clean+np.sqrt(s)*independent.normal()
    pred_low,pred_high=np.quantile(noisy_component,[.025,.975])
    batches=np.array_split(component,8)
    tail_se=np.std([np.quantile(b,[.025,.975]) for b in batches],axis=0,ddof=1)/np.sqrt(8)
    response_draws=np.einsum('bni,bi->bn',bases,bases[:,0,:])
    raw_mean=response_draws.mean(axis=0)
    raw_projector=np.einsum('bni,bmi->nm',bases,bases)/len(bases)
    empirical_risk=np.mean(np.sum((response_draws-raw_mean)**2,axis=1))
    formula_risk=risk(raw_projector,q)
    identity_error=abs(empirical_risk-formula_risk)
    if identity_error>1e-10:raise ArithmeticError('Finite-draw projector risk identity failed')
    scalar_diag=_scalar_diagnostics(chains,rotation)
    return dict(
        true_response=clean,mean_response=predicted,mode_response=float(w@Pm@q),
        q025=float(quantiles[0]),q10=float(quantiles[1]),q25=float(quantiles[2]),
        q50=float(quantiles[3]),q75=float(quantiles[4]),q90=float(quantiles[5]),q975=float(quantiles[6]),
        clean_response_covered=bool(quantiles[0]<=clean<=quantiles[6]),
        future_observation=float(fresh_observation),predictive_q025=float(pred_low),predictive_q975=float(pred_high),
        predictive_covered=bool(pred_low<=fresh_observation<=pred_high),
        tail_mcse=float(max(tail_se)),response_risk=risk(M,q),
        actual_response_squared_error=float(np.linalg.norm((M-Pt)@q)**2),
        posterior_radius=radius,hpd_threshold=h,ball_covered=bool(chordal(Xhat,Xtrue)<=radius),
        hpd_covered=bool(np.trace(A@(Pm-Pt))<=h),sample_risk_identity_error=float(identity_error),
        **diagnostics,**scalar_diag)


def run_trajectory(task):
    """Independent worker; all protocols within a trial share latent/data draws."""
    config,trial=task
    T=config['T'];n=config['n'];k=config['k'];s=config['s'];times=config['report_times']
    X0,data=paired_data(config['seed'],trial,T,n,k,s,config['input_norm'],config['slow_rate'],config['fast_rate'])
    protocols=configurations(config['slow_rate'],config['fast_rate'],config['forgetting'])
    filters={p['protocol']:BinghamFilter(n,k,s,beta=p['beta']) for p in protocols}
    modal=[];bounds=[];posterior=[];cloud=[];max_equivariance=0.
    sample=trial<config['posterior_trials'];all_base_samples={}
    for t in range(1,T+1):
        for p in protocols:
            key=p['protocol'];d=data[p['rate']];G=d['G'] if p['known'] else np.eye(n)
            Y=filters[key].update(d['u'][t-1],v=d['v'][t-1],G=G)
            Pt=projector(d['X'][t-1]);Pm=projector(Y)
            modal.append(dict(trial=trial,t=t,protocol=key,rate=p['rate'],beta=p['beta'],
                true_response=float(Pt[1,0]),mode_response=float(Pm[1,0]),
                chordal_error=float(chordal(Y,d['X'][t-1]))))
        if t not in times:continue
        static_A=filters['static'].A
        for p in protocols:
            key=p['protocol'];f=filters[key];d=data[p['rate']]
            Gs=np.broadcast_to(d['G'] if p['known'] else np.eye(n),(t,n,n))
            bound=tracking_bound(d['X'][:t],d['v'][:t],s,p['beta'],Gs,delta=config['delta'])
            error=float(chordal(f.basis,d['X'][t-1]))
            bounds.append(dict(trial=trial,t=t,protocol=key,rate=p['rate'],beta=p['beta'],
                chordal_error=error,gamma=bound.gamma,epsilon=bound.epsilon,drift_term=bound.drift,
                noise_term=bound.noise,initialization=bound.initialization,valid=bound.valid,
                bound=bound.bound,ceiling=bound.ceiling,
                bound_violated=bool(bound.valid and error>bound.bound)))
            if p['exact']:
                R=d['rotations'][t-1]
                equivariance=float(np.max(abs(f.A-R@static_A@R.T)))
                max_equivariance=max(max_equivariance,equivariance)
                if equivariance>1e-8:raise ArithmeticError('Known-frame parameter equivariance failed')
        if not sample:continue
        # Sample one canonical posterior and transport to both exact-frame controls.
        rotations=[np.eye(n),data[config['slow_rate']]['rotations'][t-1],data[config['fast_rate']]['rotations'][t-1]]
        seed=np.random.SeedSequence([config['seed'],trial,t,0])
        M,chains,diag=_posterior(static_A,k,seed,rotations,config['draws'],config['burnin'],config['max_draws'],config['max_burnin'])
        for p in protocols:
            key=p['protocol'];f=filters[key];d=data[p['rate']]
            if p['exact']:
                R=d['rotations'][t-1];mean=R@M@R.T;draws=chains;diagnostics=diag
                target='static';reused=key!='static'
            else:
                target=['identity_slow','forget_slow','identity_fast','forget_fast'].index(key)+1
                seed=np.random.SeedSequence([config['seed'],trial,t,target])
                mean,draws,diagnostics=_posterior(f.A,k,seed,[np.eye(n)],config['draws'],config['burnin'],config['max_draws'],config['max_burnin'])
                R=np.eye(n);reused=False
            values=_summary(f.A,f.basis,d['X'][t-1],mean,draws,R,diagnostics,
                            [config['seed'],trial,t,round(p['rate']*1e6)],s)
            posterior.append(dict(trial=trial,t=t,protocol=key,rate=p['rate'],beta=p['beta'],
                exact_model=p['exact'],draws_transported=reused,sample_target=str(target),**values))
            if trial==0 and t in (16,64,128):
                retained=np.concatenate([c.bases for c in draws])
                indices=np.linspace(0,len(retained)-1,min(256,len(retained)),dtype=int)
                retained=np.einsum('ij,bjk->bik',R,retained[indices])
                response=np.einsum('bni,bi->bn',retained,retained[:,0,:])
                truth=projector(d['X'][t-1])[:,0];modal_response=projector(f.basis)[:,0]
                for j,row in enumerate(response):
                    cloud.append(dict(protocol=key,t=t,draw=j,x=float(row[1]),y=float(row[2]),
                        true_x=float(truth[1]),true_y=float(truth[2]),mean_x=float(mean[1,0]),
                        mean_y=float(mean[2,0]),mode_x=float(modal_response[1]),mode_y=float(modal_response[2])))
    return dict(modal=modal,bounds=bounds,posterior=posterior,cloud=cloud,max_equivariance=max_equivariance)


def summarize(posterior,bounds):
    """Wilson intervals use independent trajectories; chains are not replicates."""
    from .reporting import wilson
    rows=[]
    if len(posterior):
        for (protocol,t),g in posterior.groupby(['protocol','t']):
            for field in ['ball_covered','hpd_covered','clean_response_covered','predictive_covered']:
                values=g[field].astype(bool);low,high=wilson(int(values.sum()),len(values))
                rows.append(dict(protocol=protocol,t=t,region=field,coverage=float(values.mean()),
                    successes=int(values.sum()),trials=len(values),low=low,high=high,
                    sampling_passes=int(g.sampling_ok.astype(bool).sum())))
    return pd.DataFrame(rows)
