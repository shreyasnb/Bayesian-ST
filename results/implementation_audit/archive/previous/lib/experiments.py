"""Reproducible array and field studies from Sections 4 and H of the draft.

These are finite simulation protocols, not performance guarantees. Training,
validation and evaluation streams are independent. In particular posterior
coverage under forgetting/misspecification is measured, not assumed nominal.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import expm, null_space
from scipy.special import logsumexp

from .geometry import gaussian, orth, projector, chordal, risk
from .filter import BinghamFilter
from .moments import sample_bingham, chain_diagnostics
from .models import increment, constant
from .theory import regional_bound
from benchmarks.case1 import LoadedCovariance, mvdr, output_power, sample_output_power
from benchmarks.case2 import EWCovariancePCA, RollingPCA, GROUSE, reconstruct


def _rng(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([seed, *map(int, keys)]))


def _posterior(A, k, rng, draws, burnin, complex_data=False,
               max_draws=24000, max_burnin=8000):
    start = perf_counter()
    attempts=0
    while True:
        attempts += 1
        # Independent random initialization checks agreement beyond a shared
        # modal start. Retry budgets are fixed in advance; failed accuracy is
        # still reported if the maximum budget is insufficient.
        chains = [sample_bingham(A,k,rng,draws=draws,burnin=burnin,complex_data=complex_data),
                  sample_bingham(A,k,rng,draws=draws,burnin=burnin,complex_data=complex_data,
                                 initial=orth(gaussian(rng,(len(A),k),complex_data)))]
        diagnostics = chain_diagnostics(chains)
        diagnostics['sampling_ok'] = (diagnostics['min_ess'] >= 50 and
                                      diagnostics['max_rhat'] < 1.10 and
                                      diagnostics['max_standardized_difference'] < 5)
        if diagnostics['sampling_ok'] or draws >= max_draws:
            break
        draws=min(max_draws,max(2*draws,6000))
        burnin=min(max_burnin,max(2*burnin,2500))
    diagnostics.update(moment_seconds=perf_counter()-start,actual_draws_per_chain=draws,
                       actual_burnin=burnin,sampling_attempts=attempts)
    return (chains[0].mean+chains[1].mean)/2, np.concatenate([c.bases for c in chains]), diagnostics


def _credible(A, Xhat, Xtrue, bases, mass=.95):
    P = projector(Xhat)
    sqdist = np.maximum(0., Xhat.shape[1]-np.einsum('bnk,nm,bmk->b',bases.conj(),P,bases).real)
    hpd = np.trace(A@P).real-np.einsum('bnk,nm,bmk->b',bases.conj(),A,bases).real
    rho2, h = np.quantile(sqdist,mass), np.quantile(hpd,mass)
    # Quantile errors from contiguous batches retain some chain dependence.
    batches = np.array_split(np.arange(len(bases)),8)
    rho_se = np.std([np.quantile(sqdist[b],mass) for b in batches],ddof=1)/np.sqrt(8)
    h_se = np.std([np.quantile(hpd[b],mass) for b in batches],ddof=1)/np.sqrt(8)
    return dict(radius=float(np.sqrt(rho2)),hpd_threshold=float(h),
                radius_squared_mcse=float(rho_se),hpd_threshold_mcse=float(h_se),
                ball_covered=chordal(Xhat,Xtrue)**2 <= rho2,
                hpd_covered=np.trace(A@(P-projector(Xtrue))).real <= h,
                ball_mass=float(np.mean(sqdist<=rho2)), hpd_mass=float(np.mean(hpd<=h)))


def _score_prior(prior,u,k,s,tau2,rng,draws,burnin,complex_data=False,v=None):
    """Monte Carlo predictive density before correction, retaining constants.

    Log-mean-exp has finite-sample bias. Importance ESS and independent-chain
    disagreement are retained so low-accuracy scores can be identified.
    """
    score_start=perf_counter()
    B=increment(u,s,tau2,v,complex_data)
    estimates=[]; esses=[]; sampler_esses=[]
    for _ in range(2):
        result=sample_bingham(prior,k,rng,draws=draws,burnin=burnin,complex_data=complex_data)
        sampler_esses.append(result.ess)
        e=np.einsum('bnk,nm,bmk->b',result.bases.conj(),B,result.bases).real
        lse=logsumexp(e)
        estimates.append(constant(u,k,s,tau2,v,complex_data)+lse-np.log(len(e)))
        esses.append(float(np.exp(2*lse-logsumexp(2*e))))
    return dict(predictive_log_score=float(np.mean(estimates)),
                score_chain_difference=float(abs(estimates[0]-estimates[1])),
                score_importance_ess=float(min(esses)),score_sampler_ess=float(min(sampler_esses)),score_seconds=perf_counter()-score_start)


def _array_path(rng,T,drift):
    n,k=16,3
    j=np.arange(n)
    steering=np.exp(1j*np.pi*np.outer(j,np.sin([-.4,.05,.45])))/np.sqrt(n)
    X0=orth(steering)
    Q0=null_space(X0.conj().T)[:,0]
    phases=np.exp(1j*drift*np.outer(np.arange(T+1),j-(n-1)/2))
    return phases[:,:,None]*X0, phases*Q0, np.diag(np.exp(1j*.5*drift*(j-(n-1)/2)))


def _array_data(seed,T,drift,overlap,mismatch,s=.3,tau2=2.):
    rng=_rng(seed)
    X,Q,G=_array_path(rng,T,drift)
    a=np.sqrt(overlap)*X[:,:,0]+np.sqrt(1-overlap)*Q
    powers=np.eye(3)
    if mismatch=='Unequal powers': powers=np.diag([.25,1.,1.75])
    if mismatch=='Correlation': powers=np.full((3,3),.55)+.45*np.eye(3)
    L=np.linalg.cholesky(powers)
    U=[]; R=[]
    for t in range(1,T+1):
        u=np.sqrt(tau2)*X[t]@L@gaussian(rng,3,True)+np.sqrt(s)*gaussian(rng,16,True)
        cov=s*np.eye(16)+tau2*X[t]@powers@X[t].conj().T
        if mismatch=='Contamination': u += np.sqrt(1.5)*a[t]*gaussian(rng,(),True)
        U.append(u);R.append(cov)
    return np.asarray(U),X[1:],a[1:],np.asarray(R),G


def _tune_array(seed,T,drift,overlap,mismatch):
    U,X,a,R,G=_array_data(seed,T,drift,overlap,mismatch)
    rng=_rng(seed,90)
    betas=(.94,.98,1.)
    validation=[(np.linalg.cholesky(cov)@gaussian(rng,(16,128),True)).T for cov in R]
    records=[]
    for beta in betas:
        f=BinghamFilter(16,3,.3,2.,beta=beta,complex_data=True)
        score=[]
        for t,u in enumerate(U):
            f.update(u,G=G)
            if t>=max(8,T//3):
                w=mvdr(.3*np.eye(16)+2*projector(f.basis),a[t])
                evaluation=validation[t]
                score.append(sample_output_power(w,evaluation))
        records.append(dict(method='Bingham',beta=beta,loading=np.nan,validation_loss=np.mean(score)))
    for beta,loading in zip(betas,(.05,.2,.8)):
        f=LoadedCovariance(16,beta=beta,loading=loading)
        score=[]
        for t,u in enumerate(U):
            f.transform(G);cov=f.update(u)
            if t>=max(8,T//3):
                w=mvdr(cov,a[t]);evaluation=validation[t]
                score.append(sample_output_power(w,evaluation))
        records.append(dict(method='Loaded covariance',beta=beta,loading=loading,validation_loss=np.mean(score)))
    # Every method receives exactly three predeclared validation candidates.
    table=pd.DataFrame(records)
    chosen={name:g.loc[g.validation_loss.idxmin()].to_dict() for name,g in table.groupby('method')}
    return chosen,table


def _run_case1_repeat(task):
    (seed,case,name,drift,overlap,mismatch,rep,T,draws,burnin,
     max_draws,max_burnin,checkpoints,settings)=task
    rows=[];regions=[]
    U,X,a,R,G=_array_data(seed+100*case+rep,T,drift,overlap,mismatch)
    rng=_rng(seed,case,rep,17)
    beta=settings['Bingham']['beta']
    f=BinghamFilter(16,3,.3,2.,beta=beta,complex_data=True)
    loaded=LoadedCovariance(16,beta=settings['Loaded covariance']['beta'],loading=settings['Loaded covariance']['loading'])
    ew=EWCovariancePCA(16,3,beta=beta)
    update_seconds=loaded_seconds=moment_total=0.
    for t,u in enumerate(U,1):
        start=perf_counter();f.update(u,G=G);update_seconds+=perf_counter()-start
        start=perf_counter();loaded.transform(G);Rload=loaded.update(u);loaded_seconds+=perf_counter()-start
        ew.transform(G);ew.update(u)
        if t not in checkpoints: continue
        M,bases,diag=_posterior(f.A,3,rng,draws,burnin,True,max_draws,max_burnin)
        moment_total+=diag['moment_seconds']
        credible=_credible(f.A,f.basis,X[t-1],bases)
        evaluation=(np.linalg.cholesky(R[t-1])@gaussian(rng,(16,256),True)).T
        covs={'Posterior mean':.3*np.eye(16)+2*M,'Mode':.3*np.eye(16)+2*projector(f.basis),
              'Loaded covariance':Rload,'Oracle':R[t-1]}
        for method,cov in covs.items():
            decision_start=perf_counter()
            w=mvdr(cov,a[t-1]);power=output_power(w,R[t-1])
            decision_time=perf_counter()-decision_start
            posterior_info={**diag,**credible} if method in ('Posterior mean','Mode') else {}
            cumulative=loaded_seconds if method=='Loaded covariance' else 0. if method=='Oracle' else update_seconds
            rows.append(dict(scenario=name,mismatch=mismatch,repeat=rep,t=t,method=method,beta=beta,
                output_power=power,heldout_power=sample_output_power(w,evaluation),sinr_db=-10*np.log10(power),
                subspace_error=chordal(f.basis if method in ('Posterior mean','Mode') else
                      np.linalg.eigh(Rload)[1][:,-3:] if method=='Loaded covariance' else X[t-1],X[t-1]),
                ew_modal_distance=chordal(f.basis,ew.basis),
                update_seconds=cumulative,decision_seconds=cumulative+decision_time+(moment_total if method=='Posterior mean' else 0.),
                **posterior_info))
        # Equal-mass regional leakage bounds for the posterior-mean beamformer.
        # Optimization and sampling errors are recorded separately.
        w=mvdr(covs['Posterior mean'],a[t-1]);B=np.outer(w,w.conj())
        for region,C,threshold in [('Ball',projector(f.basis),3-credible['radius']**2),
                ('HPD',f.A,np.trace(f.A@projector(f.basis)).real-credible['hpd_threshold'])]:
            start=perf_counter();bound=regional_bound(B,C,threshold,3)
            regions.append(dict(scenario=name,repeat=rep,t=t,region=region,
                lower=bound.lower,upper=bound.upper,optimization_gap=bound.gap,
                feasible=bound.feasible,optimization_seconds=perf_counter()-start,
                true_leakage=float(np.vdot(w,projector(X[t-1])@w).real),
                **credible,**diag))
    return rows,regions


def _json_scalars(value):
    """Portable checkpoint values; never deserialize executable objects."""
    if isinstance(value,dict):return {str(k):_json_scalars(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_scalars(v) for v in value]
    if isinstance(value,np.generic):return _json_scalars(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _array_checkpoint_signature(task):
    import hashlib,json,platform,scipy
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for path in sorted([*root.joinpath('lib').glob('*.py'),*root.joinpath('benchmarks').rglob('*.py')]):
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    digest.update(json.dumps(_json_scalars(task),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    digest.update(f'{platform.python_version()}:{np.__version__}:{scipy.__version__}'.encode())
    return digest.hexdigest()


def _array_checkpoint_load(path,signature,task):
    import json,hashlib
    try:
        document=json.loads(path.read_text())
        result=document['result']
        result_hash=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if result_hash!=document.get('result_sha256'):
            raise ValueError('checkpoint result checksum mismatch')
        if document.get('schema')!=1 or document.get('sha256')!=signature:
            raise ValueError('checkpoint signature mismatch')
        rows,regions=result
        expected=len(task[12])
        if len(rows)!=4*expected or len(regions)!=2*expected:
            raise ValueError('checkpoint result length mismatch')
        if any(r.get('scenario')!=task[2] or r.get('repeat')!=task[6] for r in [*rows,*regions]):
            raise ValueError('checkpoint result identity mismatch')
        if any(not np.isfinite(r.get('output_power',np.nan)) for r in rows):
            raise ValueError('checkpoint has invalid output power')
        if any(not np.isfinite(r.get('upper',np.nan)) for r in regions):
            raise ValueError('checkpoint has invalid regional certificate')
        return rows,regions
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(f'Ignoring invalid array checkpoint {path.name}: {error}',flush=True)
        return None


def _array_checkpoint_save(path,signature,task,result):
    import json,os,hashlib
    portable=_json_scalars(result)
    result_hash=hashlib.sha256(json.dumps(portable,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    document=dict(schema=1,sha256=signature,task=_json_scalars(task),result=portable,result_sha256=result_hash)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(document,separators=(',',':'),allow_nan=False))
    temporary.replace(path)


def run_case1(seed=1729,repeats=4,draws=1500,burnin=1000,T=72,max_draws=24000,max_burnin=8000,workers=1,checkpoint_dir=None):
    """Complex n=16,k=3 array study, independent trajectories and snapshots.

    Returns frames ``metrics``, ``regions``, ``tuning``, plus serializable
    ``config``. Checkpoints vary snapshot count. Scenarios separately vary
    desired overlap, unmodelled drift, and likelihood mismatch.
    """
    if T < 12 or repeats < 1: raise ValueError('T >= 12 and repeats >= 1 required')
    scenarios=[('Static',0.,.25,'Matched'),('Near interference',.003,.85,'Matched'),
               ('Far from interference',.003,.05,'Matched'),('Drift',.012,.25,'Matched'),
               ('Unequal powers',.003,.25,'Unequal powers'),('Correlation',.003,.25,'Correlation'),
               ('Contamination',.003,.25,'Contamination')]
    checkpoints=sorted(set([max(4,T//6),T//2,T]))
    if not isinstance(workers,int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    rows=[]; regions=[]; tuning=[];tasks=[]
    for case,(name,drift,overlap,mismatch) in enumerate(scenarios):
        settings,table=_tune_array(seed+10000+case,T,drift,overlap,mismatch)
        table['scenario']=name;tuning.append(table)
        for rep in range(repeats):
            tasks.append((seed,case,name,drift,overlap,mismatch,rep,T,draws,burnin,
                          max_draws,max_burnin,checkpoints,settings))
    from pathlib import Path
    checkpoint_root=None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint_root is not None:checkpoint_root.mkdir(parents=True,exist_ok=True)
    finished={};pending=[];checkpoint_paths={};signatures={}
    for index,task in enumerate(tasks):
        signature=_array_checkpoint_signature(task)
        signatures[index]=signature
        path=None if checkpoint_root is None else checkpoint_root/f'job_{task[1]:02d}_{task[6]:03d}_{signature[:16]}.json'
        checkpoint_paths[index]=path
        cached=None if path is None or not path.is_file() else _array_checkpoint_load(path,signature,task)
        if cached is None:pending.append((index,task))
        else:finished[index]=cached
    if checkpoint_root is not None:
        print(f'Array jobs: {len(finished)}/{len(tasks)} restored from validated checkpoints',flush=True)
    def retain(index,result):
        path=checkpoint_paths[index]
        if path is not None:_array_checkpoint_save(path,signatures[index],tasks[index],result)
        finished[index]=result
        if workers>1 or checkpoint_root is not None:
            print(f'Array jobs: {len(finished)}/{len(tasks)} complete',flush=True)
    if workers == 1:
        for index,task in pending:retain(index,_run_case1_repeat(task))
    elif pending:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        from multiprocessing import get_context
        # Spawn avoids inheriting a notebook kernel's threads and BLAS state.
        # Save completed jobs immediately; flatten results in input order below.
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(_run_case1_repeat,task):index for index,task in pending}
            for future in as_completed(futures):retain(futures[future],future.result())
    for index in range(len(tasks)):
        repeat_rows,repeat_regions=finished[index]
        rows.extend(repeat_rows);regions.extend(repeat_regions)
    return dict(metrics=pd.DataFrame(rows),regions=pd.DataFrame(regions),tuning=pd.concat(tuning,ignore_index=True),
        config=dict(seed=seed,repeats=repeats,T=T,draws=draws,burnin=burnin,n=16,k=3,s=.3,tau2=2.,
                    checkpoints=checkpoints,max_draws=max_draws,max_burnin=max_burnin,coverage_distribution='Repeated independent observations of deterministic moving subspace paths',
                    known_motion_fraction=.5,validation_candidates=3,evaluation_snapshots=256,
                    nominal_mass=.95,coverage_guaranteed=False,workers=workers,
                    checkpoint_dir=None if checkpoint_dir is None else str(checkpoint_dir)))


def _field_data(seed,T,drift=.018,mismatch='Matched',conditional=False,input_rank=8,s=.2,tau2=1.5):
    rng=_rng(seed)
    # Low spatial frequencies on an 8x8 sensor grid form a smooth, known
    # orthonormal basis (separable DCT-II, including the constant field).
    grid=np.arange(8)
    cosine=np.cos(np.pi*np.outer(grid+.5,np.arange(8))/8)
    cosine[:,0]/=np.sqrt(8);cosine[:,1:]*=np.sqrt(2/8)
    frequencies=[(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2)]
    Phi=np.stack([np.outer(cosine[:,p],cosine[:,q]).ravel() for p,q in frequencies],axis=1)
    X0=orth(rng.normal(size=(8,3)))
    K=rng.normal(size=(8,8));K=K-K.T;K*=drift/np.linalg.norm(K,2)
    G=expm(.5*K);step=expm(K);X=X0.copy()
    rows=[]
    for t in range(T):
        X=step@X
        v=np.zeros(8);v[t%input_rank]=2.
        v=np.linalg.matrix_power(G,t+1)@v  # preassigned supplied frame, fixed transported rank
        if conditional: clean=projector(X)@v
        else:
            scales=np.sqrt([.25,1.,1.75]) if mismatch=='Unequal powers' else np.ones(3)
            clean=np.sqrt(tau2)*X@(scales*rng.normal(size=3))
        w=Phi@clean+np.sqrt(s)*rng.normal(size=64)
        if mismatch=='Sparse corruption':
            indices=rng.choice(64,3,replace=False);w[indices]+=rng.normal(scale=3,size=3)
        rows.append((X.copy(),Phi.T@w,clean.copy(),v,w))
    return Phi,rows,G


def _tune_field(seed,T,mismatch):
    Phi,trajectory,G=_field_data(seed,T,mismatch=mismatch)
    options={'Bingham':[('beta',v) for v in (.94,.98,1.)],
             'EW PCA':[('beta',v) for v in (.94,.98,1.)],
             'Rolling PCA':[('window',v) for v in (12,24,48)],
             'GROUSE':[('step',v) for v in (.01,.04,.1)]}
    rows=[]
    for method,candidates in options.items():
        for param,value in candidates:
            if method=='Bingham':f=BinghamFilter(8,3,.2,1.5,beta=value)
            elif method=='EW PCA':f=EWCovariancePCA(8,3,beta=value)
            elif method=='Rolling PCA':f=RollingPCA(8,3,window=value)
            else:f=GROUSE(8,3,step=value,rng=seed)
            errors=[]
            for t,(X,u,clean,v,w) in enumerate(trajectory):
                if method=='Bingham': f.update(u,G=G)
                else:f.transform(G);f.update(u)
                if t>=max(8,T//3):errors.append(np.linalg.norm(reconstruct(u,f.basis,.2,1.5)-clean)**2)
            rows.append(dict(method=method,parameter=param,value=value,validation_loss=np.mean(errors)))
    table=pd.DataFrame(rows)
    chosen={name:g.loc[g.validation_loss.idxmin()].to_dict() for name,g in table.groupby('method')}
    return chosen,table


def _noise_study(seed,T,draws,burnin):
    """Separate n=8 noise-learning runs; never used to validate Theorem 5.

    Frozen priors and a finite E-step budget match Algorithm 1. High-dimensional
    numerical EM ascent is explicitly unverified; the deterministic general
    theory notebook checks exact EM ascent independently.
    """
    rows=[]
    for conditional in (False,True):
        rng=_rng(seed,int(conditional),71)
        _,trajectory,G=_field_data(seed+int(conditional),T,conditional=conditional)
        f=BinghamFilter(8,3,.6,None if conditional else .6,beta=.98,window=8)
        evaluations=[]
        def mean(A,k):
            r=sample_bingham(A,k,rng,draws=max(200,draws//4),burnin=max(150,burnin//3))
            evaluations.append(r)
            return r.mean
        for t,(X,u,clean,v,w) in enumerate(trajectory,1):
            fitting=t>=8 and t%max(8,T//3)==0
            kwargs={}
            if fitting:
                # Score is computed BEFORE fitting or assimilating this observation.
                prior=.98*G@f.A@G.T
                kwargs=_score_prior(prior,u,3,f.s,f.tau2,rng,max(200,draws//3),max(150,burnin//3),v=v if conditional else None)
            start=perf_counter()
            f.update(u,v=v if conditional else None,G=G,fit=fitting,mean=mean if fitting else None,iterations=2)
            if fitting:
                if conditional:
                    naive=uncertainty=0.
                    for rec in f.records:
                        M=mean(rec.prior+increment(rec.u,f.s,None,rec.v),3)
                        naive+=np.linalg.norm(rec.u-M@rec.v)**2/(8*len(f.records))
                        uncertainty+=np.vdot(rec.v,(M-M@M)@rec.v).real/(8*len(f.records))
                else: naive=uncertainty=np.nan
                rows.append(dict(protocol='Calibration' if conditional else 'Passive',t=t,s=f.s,tau2=f.tau2,
                    true_s=.2,true_tau2=np.nan if conditional else 1.5,
                    naive_residual_variance=naive,uncertainty_variance=uncertainty,
                    expected_residual_variance=naive+uncertainty,
                    residual_diagnostic_target='Frozen window evaluated at the accepted final variance; not the preceding EM E-step',fit_seconds=perf_counter()-start,
                    mean_evaluations=f.last_fit.mean_evaluations,accepted=f.last_fit.accepted,
                    numerical_ascent_verified=False,minimum_ess=min(e.ess for e in evaluations),**kwargs))
                evaluations.clear()
    return pd.DataFrame(rows)


def run_case2(seed=2718,repeats=4,draws=1500,burnin=1000,T=72,max_draws=24000,max_burnin=8000):
    """Field n=8,k=3 in 64 sensors: passive reconstruction and calibration.

    ``metrics`` uses Eq.(45) on the same current measurement assimilated by
    each method. ``calibration`` predicts fresh deterministic known patterns
    from identical Bingham parameters. ``noise`` is a separate EM experiment.
    """
    if T < 12 or repeats < 1: raise ValueError('T >= 12 and repeats >= 1 required')
    rows=[];calibration=[];tuning=[];fields=[]
    checkpoints=sorted(set([max(4,T//6),T//2,T]))
    for case,mismatch in enumerate(('Matched','Unequal powers','Sparse corruption')):
        chosen,table=_tune_field(seed+10000+case,T,mismatch);table['scenario']=mismatch;tuning.append(table)
        for rep in range(repeats):
            rng=_rng(seed,case,rep,19)
            Phi,trajectory,G=_field_data(seed+100*case+rep,T,mismatch=mismatch)
            f=BinghamFilter(8,3,.2,1.5,beta=chosen['Bingham']['value'])
            methods={'EW PCA':EWCovariancePCA(8,3,beta=chosen['EW PCA']['value']),
                     'Rolling PCA':RollingPCA(8,3,window=int(chosen['Rolling PCA']['value'])),
                     'GROUSE':GROUSE(8,3,step=chosen['GROUSE']['value'],rng=seed+rep)}
            seconds={m:0. for m in ['Bingham',*methods]};moment_total=0.
            for t,(X,u,clean,v,w) in enumerate(trajectory,1):
                score={}
                if t in checkpoints:
                    score=_score_prior(f.beta*G@f.A@G.T,u,3,f.s,f.tau2,rng,max(250,draws//2),max(150,burnin//2))
                start=perf_counter();f.update(u,G=G);seconds['Bingham']+=perf_counter()-start
                for name,method in methods.items():
                    start=perf_counter();method.transform(G);method.update(u);seconds[name]+=perf_counter()-start
                if t not in checkpoints:continue
                M,bases,diag=_posterior(f.A,3,rng,draws,burnin,False,max_draws,max_burnin)
                moment_total+=diag['moment_seconds']
                credible=_credible(f.A,f.basis,X,bases)
                estimates={'Posterior mean':1.5/1.7*M@u,'Mode':reconstruct(u,f.basis,.2,1.5),
                    **{name:reconstruct(u,method.basis,.2,1.5) for name,method in methods.items()},
                    'Oracle':1.5/1.7*projector(X)@u}
                if mismatch=='Matched' and rep==0 and t==T:
                    fields=pd.DataFrame(dict(sensor=np.arange(64),row=np.repeat(np.arange(8),8),
                        column=np.tile(np.arange(8),8),clean=Phi@clean,observed=w,
                        posterior_mean=Phi@estimates['Posterior mean'],mode=Phi@estimates['Mode']))
                for name,estimate in estimates.items():
                    posterior_info={**diag,**credible,**score} if name in ('Posterior mean','Mode') else {}
                    cumulative=seconds.get(name,0. if name=='Oracle' else seconds['Bingham'])
                    basis=f.basis if name in ('Posterior mean','Mode') else methods[name].basis if name in methods else X
                    rows.append(dict(scenario=mismatch,repeat=rep,t=t,method=name,
                        field_squared_error=float(np.linalg.norm(Phi@(estimate-clean))**2),
                        normalized_error=float(np.linalg.norm(estimate-clean)**2/max(np.linalg.norm(clean)**2,1e-12)),
                        subspace_error=chordal(basis,X),update_seconds=cumulative,
                        decision_seconds=cumulative+(moment_total if name=='Posterior mean' else 0.),
                        **posterior_info))
    # Independent preassigned calibration inputs; never recovered from hidden X.
    for rank in (6,7,8):
        for rep in range(repeats):
            rng=_rng(seed,rank,rep,29)
            _,trajectory,G=_field_data(seed+2000+rep,T,conditional=True,input_rank=rank)
            f=BinghamFilter(8,3,.2,beta=.98)
            for t,(X,u,clean,v,w) in enumerate(trajectory,1):
                f.update(u,v=v,G=G)
                if t not in checkpoints:continue
                M,bases,diag=_posterior(f.A,3,rng,draws,burnin,False,max_draws,max_burnin)
                fresh=np.eye(8)[:,(t+3)%8]*2  # preassigned, independent of X
                mean_response=M@fresh;modal_response=projector(f.basis)@fresh
                true=projector(X)@fresh
                # Eq.30 is an exact posterior identity for the numerical mean.
                calibration.append(dict(input_rank=rank,repeat=rep,t=t,
                    mean_squared_error=float(np.linalg.norm(mean_response-true)**2),
                    modal_squared_error=float(np.linalg.norm(modal_response-true)**2),
                    posterior_risk=risk(M,fresh),
                    modal_excess_risk=float(np.linalg.norm(modal_response-mean_response)**2),
                    subspace_error=chordal(f.basis,X),**diag))
    return dict(metrics=pd.DataFrame(rows),calibration=pd.DataFrame(calibration),fields=fields,
        noise=_noise_study(seed+3000,T,draws,burnin),tuning=pd.concat(tuning,ignore_index=True),
        config=dict(seed=seed,repeats=repeats,T=T,draws=draws,burnin=burnin,n=8,k=3,d=64,s=.2,tau2=1.5,
                    checkpoints=checkpoints,max_draws=max_draws,max_burnin=max_burnin,known_motion_fraction=.5,nominal_mass=.95,validation_candidates=3,spatial_basis='Eight low-frequency separable DCT-II modes on an 8x8 grid',calibration_input_rank='Rank of supplied-frame transported preassigned inputs',
                    coverage_distribution='Independent Gaussian-QR initial subspaces, Gaussian skew drift paths, and Gaussian observations',
                    coverage_guaranteed=False,noise_fit_ascent_verified=False,
                    score_method='Prior Monte Carlo log-mean-exp, with importance ESS and chain discrepancy'))
