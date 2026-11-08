"""Passive complex-array geometry and independently validated decisions.

Only snapshots enter estimation. True spatial signatures enter generation,
validation losses and scoring. Desired steering is supplied only to decisions.
The new experiment is separate from results/beamforming_unified.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

from .geometry import center
from .beamforming_protocol import proper_complex, haar_line, ula
from .beamforming_numerics import complex_moments, posterior_report, sample_complex_bingham
from .beamforming_decisions import isotropized_mean, optimize_scalar, scalar_statistics, conditional_excess
from benchmarks.case1.baselines import mvdr, output_power, sample_output_power
from benchmarks.case2.baselines import GROUSE

LAWS=('haar','ula3','ula10','ula30')
METHODS=('mean','mode','iso','scalar','scm','lw','grouse','oracle')


def configuration(profile='quick'):
    if profile not in ('quick','paper'):
        raise ValueError('Choose quick or paper')
    quick=profile=='quick'
    return dict(schema=1,profile=profile,base_commit='2a8697c7e7fa27b3779f3abeaed32ca8b323a9ef',
        cohorts=[dict(n=8,validation_trials=8 if quick else 128,test_trials=16 if quick else 256),
                 dict(n=16,validation_trials=4 if quick else 64,test_trials=8 if quick else 128)],
        horizons=[1,4,16,64],laws=list(LAWS),noises=[1.,.2],tau2=1.,beta=1.,
        validation_seed=7360811+(10000 if quick else 0),test_seed=7360812+(10000 if quick else 0),
        tail_seed=7360813+(10000 if quick else 0),fixture_seed=7360814+(10000 if quick else 0),
        desired_deg=0.,grouse_steps=np.geomspace(1e-4,10.,9).tolist(),bracket_points=65,
        heldout_snapshots=128,independent_queries=8,
        tail_trials=2 if quick else 8,tail_times={'8':[4,16,64],'16':[16,64]},
        posterior_draws=2048 if quick else 16384,display_draws=2048,fixture_display_draws=4096,
        relative_quantile_mcse=.08 if quick else .025,
        query_degrees=[-30.,-10.,0.,10.,30.],scan_degrees=np.linspace(-60.,60.,121).tolist(),
        story=dict(law='ula10',s=.2,trial=0,dimensions=[8,16],horizons=[4,16,64]),
        fixture=dict(n=8,desired_deg=0.,mode_deg=3.,weak_gap=8.,strong_gap=80.,s=.2,tau2=1.,
                     draws=4096 if quick else 32768),
        model='Proper complex marginal model: u=h*z+e, z~CN(0,tau2), e~CN(0,sI). Uniform complex-line prior; beta=1.',
        separation='Stationary desired0deg; ULA interferer3/10/30deg or independent Haar line. Angles are radians inside numerical functions.',
        validation='Continuous SCM relative loading and point-covariance scalar are optimized separately on independent expected-power validation losses; GROUSE has nine independently validated steps. No test outcomes select any settings.',
        scm='R_alpha=(1-alpha)S+alpha tr(S)I/n, alpha in[0,1]. Equivalent relative diagonal loading alpha/(1-alpha); alpha=1 is the exact isotropic endpoint. alpha=0 is ineligible when S is singular.',
        lw='Uncentered Ledoit-Wolf outer-product Frobenius shrinkage, extended to proper complex data using Hermitian products. One-snapshot convention alpha=1 is explicit; no sample covariance can be inverted there.',
        scalar='R=sI+tau2[rho P_mode+(1-rho)I/n]; one rho per condition/horizon selected continuously on validation. Per-posterior modal-overlap matching is a separate control.',
        uncertainty='First predeclared tail_trials trajectories per law/noise/dimension. Two independent native iid batches; finite Monte Carlo mass/quantile precision flags are separate from mean accuracy.',
        fixture_scope='Selected-data posterior calculation, not a random-training performance claim. Two parameters share spectrum, mode and every centered chordal-distance quantile, but swap complementary axes relative to the desired steering.',
        statistics='Independent trajectories are the replication units. Query losses are averaged within each trajectory; paired Student intervals are pointwise exploratory, without multiplicity correction.')


def cohort(config,n):
    return next(c for c in config['cohorts'] if c['n']==n)


def dataset(config,n,law,s,trial,split):
    if law not in LAWS or split not in ('validation','test'):
        raise ValueError('Unknown generation law or split')
    seed=config['validation_seed'] if split=='validation' else config['test_seed']
    children=np.random.SeedSequence([seed,n,LAWS.index(law),int(round(1000*s)),trial]).spawn(5)
    latent=np.random.default_rng(children[0]);observations=np.random.default_rng(children[1])
    h=haar_line(latent,n) if law=='haar' else ula(n,np.deg2rad(float(law[3:])))
    length=max(config['horizons']);q=config['tau2']
    U=proper_complex(observations,length,q)[:,None]*h+proper_complex(observations,(length,n),s)
    query_rng=np.random.default_rng(children[2]);queries=np.array([haar_line(query_rng,n) for _ in range(config['independent_queries'])])
    initial=haar_line(np.random.default_rng(children[3]),n)
    heldout_rng=np.random.default_rng(children[4])
    heldout=proper_complex(heldout_rng,config['heldout_snapshots'],q)[:,None]*h+proper_complex(heldout_rng,(config['heldout_snapshots'],n),s)
    return dict(U=U,h=h,a=ula(n,np.deg2rad(config['desired_deg'])),queries=queries,initial=initial,heldout=heldout)


def sample_covariance(U):
    U=np.asarray(U,complex)
    if U.ndim!=2 or not len(U) or not np.isfinite(U).all():
        raise ValueError('Expected finite row-wise snapshots')
    S=U.T@U.conj()/len(U)
    return (S+S.conj().T)/2


def covariance_shrinkage(S,alpha):
    """Continuous relative-loading family, including exact infinite loading."""
    S=np.asarray(S,complex);n=len(S)
    if not 0<=alpha<=1:raise ValueError('alpha must be in[0,1]')
    mu=float(np.trace(S).real/n)
    return (1-alpha)*S+alpha*mu*np.eye(n)


def ledoit_wolf(U):
    """Known-zero-mean Ledoit--Wolf shrinkage for Hermitian covariance.

    Let Y_j=u_j u_j^H and S=mean(Y_j). The Frobenius numerator is
    sum_j||Y_j-S||_F^2/m^2=(sum_j||u_j||^4-m||S||_F^2)/m^2.
    This identity uses Hermitian norms and applies directly to complex vectors.
    It is the standard large-sample shrinkage estimate, not a finite-sample
    confidence bound. At m=1 the explicit covariance convention is alpha=1.
    """
    U=np.asarray(U,complex);S=sample_covariance(U);m,n=U.shape
    mu=float(np.trace(S).real/n)
    if mu<=0:raise ValueError('Positive sample energy required')
    if m==1:return mu*np.eye(n),1.
    energy=np.sum(abs(U)**2,axis=1)
    phi=max(0.,float(np.sum(energy**2)-m*np.linalg.norm(S,'fro')**2)/(m*m))
    delta=float(np.linalg.norm(S-mu*np.eye(n),'fro')**2)
    alpha=1. if delta<=np.finfo(float).eps*max(1.,np.linalg.norm(S,'fro')**2) else float(np.clip(phi/delta,0.,1.))
    R=covariance_shrinkage(S,alpha)
    # A nonzero continuous-noise sample has alpha>0 unless S is already HPD.
    if np.linalg.eigvalsh(R)[0]<=0:raise ArithmeticError('Ledoit-Wolf covariance is not positive definite')
    return R,alpha


def scm_statistics(S,a,h):
    values,E=np.linalg.eigh(S)
    b=E.conj().T@a;c=(E.conj().T@h).conj()*b
    return dict(values=values.tolist(),p=(abs(b)**2).tolist(),c_real=c.real.tolist(),c_imag=c.imag.tolist())


def scm_validation_loss(alpha,statistics,s,tau2):
    values=np.asarray([r['values'] for r in statistics]);p=np.asarray([r['p'] for r in statistics])
    c=np.asarray([r['c_real'] for r in statistics])+1j*np.asarray([r['c_imag'] for r in statistics])
    d=(1-alpha)*values+alpha*values.mean(axis=1)[:,None]
    if not np.isfinite(d).all() or np.any(d<=1e-13*np.max(values,axis=1)[:,None]):return np.inf
    denominator=np.sum(p/d,axis=1)
    noise=np.sum(p/d**2,axis=1)
    interference=abs(np.sum(c/d,axis=1))**2
    return float(np.mean((s*noise+tau2*interference)/denominator**2))


def continuous_minimum(function,points=65):
    """Predeclared grid bracketing followed by refinement of every local basin."""
    grid=np.linspace(0.,1.,points);values=np.array([function(x) for x in grid])
    candidates=[(values[0],0.),(values[-1],1.)];evaluations=len(grid)
    if np.ptp(values[np.isfinite(values)])>1e-13:
        for j in range(1,len(grid)-1):
            if np.isfinite(values[j]) and values[j]<=values[j-1] and values[j]<=values[j+1]:
                fit=minimize_scalar(function,bounds=(grid[j-1],grid[j+1]),method='bounded',options=dict(xatol=1e-10,maxiter=128))
                candidates.append((float(fit.fun),float(fit.x)));evaluations+=fit.nfev
    else:
        candidates.append((float(np.nanmin(values)),float(grid[np.nanargmin(values)])))
    loss,value=min(candidates,key=lambda x:x[0])
    if not np.isfinite(loss):raise ArithmeticError('No finite validation setting')
    return dict(value=value,loss=loss,evaluations=evaluations,boundary=value in (0.,1.),grid=grid.tolist(),losses=values.tolist())


def grouse_paths(U,steps,initial,horizons):
    result={}
    for step in steps:
        tracker=GROUSE(U.shape[1],1,step=step,rng=0);tracker.basis=np.asarray(initial)[:,None].copy()
        records={};seconds=0.
        for j,u in enumerate(U,1):
            start=perf_counter();X=tracker.update(u);seconds+=perf_counter()-start
            if j in horizons:records[j]=(X@X.conj().T,seconds)
        result[float(step)]=records
    return result


def validation_task(task):
    config,n,law,s,trial=task;data=dataset(config,n,law,s,trial,'validation');q=config['tau2'];rows=[]
    h,a=data['h'],data['a'];R=s*np.eye(n)+q*np.outer(h,h.conj())
    paths=grouse_paths(data['U'],config['grouse_steps'],data['initial'],config['horizons'])
    for t in config['horizons']:
        S=sample_covariance(data['U'][:t]);e,E=np.linalg.eigh(S);P=np.outer(E[:,-1],E[:,-1].conj())
        rows.append(dict(n=n,law=law,s=s,t=t,trial=trial,scalar=scalar_statistics(P,a,R).tolist(),
            scm=scm_statistics(S,a,h),grouse=[output_power(mvdr(s*np.eye(n)+q*paths[step][t][0],a),R) for step in config['grouse_steps']]))
    return rows


def select_policies(records,config):
    selected=[];curves=[]
    groups={}
    for row in records:groups.setdefault((row['n'],row['law'],row['s'],row['t']),[]).append(row)
    for (n,law,s,t),rows in sorted(groups.items()):
        expected=cohort(config,n)['validation_trials']
        if len(rows)!=expected or len({r['trial'] for r in rows})!=expected:raise ArithmeticError('Incomplete validation cohort')
        scalar=optimize_scalar(np.asarray([r['scalar'] for r in rows]),s,config['tau2'],n,config['bracket_points'])
        scm=continuous_minimum(lambda alpha:scm_validation_loss(alpha,[r['scm'] for r in rows],s,config['tau2']),config['bracket_points'])
        losses=np.mean([r['grouse'] for r in rows],axis=0);i=int(np.argmin(losses));step=float(config['grouse_steps'][i])
        selected.append(dict(n=n,law=law,s=s,t=t,rho=scalar['rho'],alpha=scm['value'],grouse_step=step,
            scalar_loss=scalar['loss'],scm_loss=scm['loss'],grouse_loss=float(losses[i]),
            scalar_evaluations=scalar['evaluations'],scm_evaluations=scm['evaluations'],
            scalar_boundary=scalar['rho']<1e-8 or scalar['rho']>1-1e-8,scm_boundary=scm['boundary'],
            grouse_boundary=i in (0,len(losses)-1),independent_validation_trials=expected))
        curves.extend(dict(n=n,law=law,s=s,t=t,family='SCM',parameter=a,loss=b) for a,b in zip(scm['grid'],scm['losses']))
        curves.extend(dict(n=n,law=law,s=s,t=t,family='GROUSE',parameter=a,loss=float(b)) for a,b in zip(config['grouse_steps'],losses))
    return selected,curves


def _query_loss(M,Ptrue,queries):
    return float(np.mean(np.sum(abs((M-Ptrue)@queries.T)**2,axis=0)))


def test_task(task):
    config,n,law,s,trial,selected=task;data=dataset(config,n,law,s,trial,'test');q=config['tau2'];h,a=data['h'],data['a']
    Ptrue=np.outer(h,h.conj());Rtrue=s*np.eye(n)+q*Ptrue;queries=data['queries']
    policies={p['t']:p for p in selected if p['n']==n and p['law']==law and p['s']==s}
    steps=list(dict.fromkeys(float(p['grouse_step']) for p in policies.values()))
    paths=grouse_paths(data['U'],steps,data['initial'],config['horizons'])
    rows=[];shapes=[];tails=[];scans=[];spectra=[];stories={}
    story=law==config['story']['law'] and s==config['story']['s'] and trial==config['story']['trial']
    scan_vectors=np.array([ula(n,np.deg2rad(d)) for d in config['scan_degrees']])
    if story:
        stories.update({f'n{n}__{key}':data[key] for key in ['U','h','a','queries']})
        stories[f'n{n}__scan_vectors']=scan_vectors;stories['scan_degrees']=np.array(config['scan_degrees'])
    for t in config['horizons']:
        policy=policies[t];U=data['U'][:t];begin=perf_counter();S=sample_covariance(U);scatter_seconds=perf_counter()-begin
        begin=perf_counter();e,E=np.linalg.eigh(S);mode_seconds=perf_counter()-begin;P=np.outer(E[:,-1],E[:,-1].conj())
        A=center(q/(s*(s+q))*t*S);mean=complex_moments(A)
        if not mean.numerical_ok:raise ArithmeticError('Native posterior mean unresolved')
        M=mean.mean;Mi,rho=isotropized_mean(M,P);Pscalar=policy['rho']*P+(1-policy['rho'])*np.eye(n)/n
        Rlw,alpha_lw=ledoit_wolf(U);Pg,tg=paths[policy['grouse_step']][t]
        covariances=dict(mean=s*np.eye(n)+q*M,mode=s*np.eye(n)+q*P,iso=s*np.eye(n)+q*Mi,
            scalar=s*np.eye(n)+q*Pscalar,scm=covariance_shrinkage(S,policy['alpha']),lw=Rlw,
            grouse=s*np.eye(n)+q*Pg,oracle=Rtrue)
        maps=dict(mean=M,mode=P,iso=Mi,scalar=Pscalar,grouse=Pg,oracle=Ptrue)
        weights={method:mvdr(R,a) for method,R in covariances.items()};wmean=weights['mean'];Rbar=covariances['mean']
        risk=float(np.mean(np.einsum('bi,ij,bj->b',queries.conj(),M-M@M,queries).real))
        for method,R in covariances.items():
            begin=perf_counter();w=mvdr(R,a);decision_seconds=perf_counter()-begin
            excess,_,error=conditional_excess(w,wmean,Rbar);mapping=maps.get(method)
            rows.append(dict(n=n,law=law,s=s,tau2=q,t=t,trial=trial,method=method,
                true_power=output_power(w,Rtrue),heldout_power=sample_output_power(w,data['heldout']),
                posterior_power=output_power(w,Rbar),posterior_excess=excess,optimality_residual=error,
                unit_gain_error=abs(np.vdot(w,a)-1),response_loss=_query_loss(mapping,Ptrue,queries) if mapping is not None else np.nan,
                posterior_risk=risk,posterior_response_loss=(risk+float(np.mean(np.sum(abs((mapping-M)@queries.T)**2,axis=0)))) if mapping is not None else np.nan,
                mean_ok=mean.numerical_ok,mean_error_bound=mean.moment_error_bound,mean_order=mean.order,
                scatter_seconds=scatter_seconds,mode_seconds=mode_seconds,mean_seconds=mean.seconds,
                decision_seconds=decision_seconds,grouse_update_seconds=tg if method=='grouse' else np.nan,
                alpha=policy['alpha'] if method=='scm' else alpha_lw if method=='lw' else np.nan,
                rho=policy['rho'] if method=='scalar' else rho if method=='iso' else np.nan,
                grouse_step=policy['grouse_step'] if method=='grouse' else np.nan))
        shapes.append(dict(n=n,law=law,s=s,t=t,trial=trial,rho=rho,anisotropy=float(np.linalg.norm(M-Mi,'fro')),
            mode_overlap=float(np.trace(P@M).real),isotropic_mode_overlap=float(np.trace(P@Mi).real),
            pca_mode_error=float(np.linalg.norm(P-E[:,-1,None]@E[:,-1,None].conj().T,'fro'))))
        if story and t in config['story']['horizons']:
            prefix=f'n{n}_t{t}__'
            stories.update({prefix+key:value for key,value in dict(A=A,Pbar=M,Pmode=P,Piso=Mi,desired=a,queries=queries).items()})
            for method,w in weights.items():stories[prefix+'weight_'+method]=w
            for index,(mm,mi) in enumerate(zip(np.linalg.eigvalsh(M)[::-1],np.linalg.eigvalsh(Mi)[::-1])):
                spectra.append(dict(n=n,law=law,s=s,t=t,trial=trial,index=index,Pbar=mm,Piso=mi))
        tail_time=t in config['tail_times'][str(n)]
        if tail_time and trial<config['tail_trials']:
            W=np.array([ula(n,np.deg2rad(d)) for d in config['query_degrees']])
            seed=np.random.SeedSequence([config['tail_seed'],n,LAWS.index(law),int(s*1000),trial,t])
            report=posterior_report(A,W,truth=h,draws=config['posterior_draws'],rng=np.random.default_rng(seed),
                return_samples=True,relative_mcse_tolerance=config['relative_quantile_mcse'])
            d=report['numerics'];quantiles=s+q*report['quadratic_quantiles'];truth=s+q*abs(W@h.conj())**2
            tails.append(dict(n=n,law=law,s=s,t=t,trial=trial,ball_covered=report['ball_covered'],hpd_covered=report['hpd_covered'],
                ball_radius=report['ball_radius'],hpd_threshold=report['hpd_threshold'],
                mean_ok=d['mean_ok'],state_ok=d['state_ok'],tail_ok=d['tail_ok'],numerical_ok=d['numerical_ok'],
                hpd_mass=d['hpd_independent_mass'],ball_mass=d['ball_independent_mass'],state_mass_error=d['state_mass_error'],
                quantile_mcse_fraction=d['maximum_quantile_mcse_fraction_of_width'],
                draws_per_batch=config['posterior_draws'],report_seconds=d['total_seconds'],
                scalar_coverage=float(np.mean((truth>=quantiles[:,0])&(truth<=quantiles[:,-1]))),
                scalar_width=float(np.mean(quantiles[:,-1]-quantiles[:,0]))))
            if story:
                prefix=f'n{n}_t{t}__';X=report['vectors']
                stories[prefix+'draws']=X[:config['display_draws']];stories[prefix+'posterior_draws_per_batch']=np.array(config['posterior_draws']);stories[prefix+'hpd_threshold']=np.array(report['hpd_threshold'])
                stories[prefix+'ball_radius']=np.array(report['ball_radius'])
                values=s+q*abs(X.conj()@scan_vectors.T)**2
                bands=np.quantile(values,[.025,.25,.5,.75,.975],axis=0)
                for j,angle in enumerate(config['scan_degrees']):
                    scans.append(dict(n=n,law=law,s=s,t=t,trial=trial,angle_deg=angle,
                        truth=s+q*abs(np.vdot(scan_vectors[j],h))**2,
                        mean=s+q*np.vdot(scan_vectors[j],M@scan_vectors[j]).real,
                        q025=bands[0,j],q25=bands[1,j],q50=bands[2,j],q75=bands[3,j],q975=bands[4,j],
                        numerical_ok=d['numerical_ok'],scope='Pointwise iid Monte Carlo scan quantiles; only the five fixed report queries have the declared quantile precision diagnostic.'))
    return dict(metrics=rows,shape=shapes,tails=tails,scans=scans,spectra=spectra,arrays=stories)


def paired_table(metrics):
    import pandas as pd
    frame=pd.DataFrame(metrics);base=frame[frame.method=='mean'];rows=[]
    for (n,law,s,t,method),g in frame[frame.method!='mean'].groupby(['n','law','s','t','method'],sort=False):
        paired=g.merge(base,on=['n','law','s','t','trial'],suffixes=('_reference','_mean'),validate='one_to_one')
        for endpoint in ['true_power','heldout_power','response_loss','posterior_power']:
            values=(paired[endpoint+'_reference']-paired[endpoint+'_mean']).to_numpy(float);finite=np.isfinite(values)
            N=len(values);defined=bool(finite.all());gain=float(values.mean()) if defined else np.nan
            half=float(student_t.ppf(.975,N-1)*values.std(ddof=1)/np.sqrt(N)) if defined and N>1 else np.nan
            rows.append(dict(n=n,law=law,s=s,t=t,reference=method,endpoint=endpoint,trials=N,finite_pairs=int(finite.sum()),
                gain=gain,lower=gain-half,upper=gain+half,numerical_ok=bool(paired.mean_ok_mean.all() and paired.mean_ok_reference.all()),
                defined=defined,analytic_tie=bool(method=='iso' and t==1),
                inference='Paired pointwise Student95% interval across independent trajectories; no multiplicity correction'))
    return pd.DataFrame(rows)


def fixture(config):
    """Two selected-data posteriors with exactly equal radial distributions."""
    f=config['fixture'];n=f['n'];s=f['s'];q=f['tau2']
    h=ula(n,np.deg2rad(f['mode_deg']));a=ula(n,np.deg2rad(f['desired_deg']))
    tangent=(np.eye(n)-np.outer(h,h.conj()))@a;tangent/=np.linalg.norm(tangent)
    rest=null_space(np.column_stack([h,tangent]).conj().T);Q=np.column_stack([h,tangent,rest])
    permutation=np.eye(n);permutation[[1,2]]=permutation[[2,1]];G=Q@permutation@Q.conj().T
    values=np.r_[0.,-f['weak_gap'],np.full(n-2,-f['strong_gap'])]
    A1=center((Q*values)@Q.conj().T);A2=center(G@A1@G.conj().T)
    M1=complex_moments(A1).mean;M2=complex_moments(A2).mean
    X1=sample_complex_bingham(A1,draws=f['draws'],rng=np.random.default_rng(config['fixture_seed'])).vectors;X2=X1@G.T
    P=np.outer(h,h.conj());Mi,rho=isotropized_mean(M1,P)
    radius1=np.sqrt(np.maximum(0,1-abs(X1@h.conj())**2));radius2=np.sqrt(np.maximum(0,1-abs(X2@h.conj())**2))
    check=float(np.max(abs(radius1-radius2)));transform_error=float(np.linalg.norm(M2-G@M1@G.conj().T,'fro'))
    eta=q/(s*(s+q));positive=values+f['strong_gap'];active=np.where(positive>0)[0]
    U1=(Q[:,active]*np.sqrt(positive[active]/eta)).T;U2=U1@G.T
    recovery=max(np.linalg.norm(center(eta*(U.T@U.conj()))-A,'fro') for U,A in [(U1,A1),(U2,A2)])
    angles=np.linspace(-60.,60.,241);scan=np.array([ula(n,np.deg2rad(d)) for d in angles]);records=[];arrays=dict(A1=A1,A2=A2,M1=M1,M2=M2,Pmode=P,Piso=Mi,
        Q=Q,G=G,h=h,desired=a,U1=U1,U2=U2,draws1=X1[:config['fixture_display_draws']],draws2=X2[:config['fixture_display_draws']],radius1=radius1,radius2=radius2,
        angles_deg=angles,scan_vectors=scan,s=np.array(s),tau2=np.array(q))
    for index,(A,M,X) in enumerate([(A1,M1,X1),(A2,M2,X2)],1):
        Rbar=s*np.eye(n)+q*M;wstar=mvdr(Rbar,a)
        for method,R in [('mean',Rbar),('mode',s*np.eye(n)+q*P),('iso',s*np.eye(n)+q*Mi)]:
            w=mvdr(R,a);draw_power=s*np.vdot(w,w).real+q*abs(X.conj()@w)**2
            exact=output_power(w,Rbar)
            records.append(dict(distribution=index,method=method,expected_power=exact,
                excess=exact-output_power(wstar,Rbar),mc_power=float(draw_power.mean()),
                mcse=float(draw_power.std(ddof=1)/np.sqrt(len(draw_power))),rho=rho,
                radius95=float(np.quantile(radius1,.95)),mode_distance=0.,
                scope='Designed conditional posterior calculation; not a randomly generated training benchmark'))
            arrays[f'weight{index}_{method}']=w
    arrays['checks']=np.array([check,transform_error,recovery])
    if max(check,transform_error,recovery)>2e-10:raise ArithmeticError('Equal-radius fixture identity failed')
    return records,arrays,dict(radius_pair_error=check,predictive_mean_rotation_error=transform_error,
        passive_selected_data_recovery_error=float(recovery),same_mode=True,same_spectrum=True,
        equal_entire_radius_distribution='Exact unitary pushforward fixing the mode; numerical pairs verify it',
        fixture_scope=config['fixture_scope'])
