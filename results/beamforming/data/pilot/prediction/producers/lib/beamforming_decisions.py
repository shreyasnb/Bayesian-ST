"""Beamforming decisions from native-complex passive rank-one snapshots.

Only observations enter the sufficient statistics. The true signature and
covariance are used by the simulator, validation losses and evaluation only.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

from .beamforming_protocol import (profile, proper_complex, haar_line, ula,
    physical_angles)
from .beamforming_numerics import complex_moments, posterior_report
from .geometry import center
from .models import increment
from .theory import regional_bound
from benchmarks.case1.baselines import mvdr, output_power, sample_output_power
from benchmarks.case2.baselines import GROUSE

CONDITIONS=('haar','ula_static','ula_moving')
METHODS=('mean','mode','pca','isotropic','scalar','loaded','grouse','oracle')
LABELS=dict(mean='Posterior mean',mode='Posterior mode',pca='EW PCA',
    isotropic='Matched isotropic mean',scalar='Regularized PCA',
    loaded='Loaded covariance',grouse='GROUSE',oracle='Oracle')


def default_config(name='quick'):
    config=profile(name)
    config.update(schema=1,validation_seed=6571911 if name=='quick' else 6571011,
        test_seed=6571912 if name=='quick' else 6571012,
        conditions=list(CONDITIONS),betas=[1.,.995,.98,.95,.9],
        beta_expansion=[.8,.6],
        loadings=[0.,1e-4,1e-3,.01,.1,1.,10.,100.,1e4],
        loading_expansion=[1e-6,1e-8,1e6,1e8],
        steps=[1e-4,3e-4,.001,.003,.01,.03,.1,.3,1.],
        step_expansion=[1e-5,3e-5,3.,10.],
        scalar_bracket_points=65,heldout_snapshots=64,
        tail_times=[8,32],tail_trials=16 if name=='quick' else 256,
        posterior_draws=2048 if name=='quick' else 16384,
        reference_draws=2048 if name=='quick' else 16384,
        relative_quantile_mcse=.08 if name=='quick' else .025,
        scan_angles=np.deg2rad(np.linspace(-70,70,141)).tolist(),
        evaluation_angles=np.deg2rad([-20.,20.,45.]).tolist(),
        story_condition='ula_static',story_s=.2,story_trial=0,story_times=[8,32],
        independent_queries=8,
        validation='Independent trajectories; expected true output power selects each family/horizon. Mean, same-A mode and matched isotropization share the selected mean memory. EW PCA, scalar regularization, loaded covariance and GROUSE are selected separately.',
        stationary='beta=1 fixed for Haar and stationary ULA; uniform ambient complex-line prior',
        moving='Identity transport; each family selects memory from validation, not the true angle path.',
        scalar='rho*P+(1-rho)I/n, optimized continuously on independent validation, separately from per-posterior matched isotropization',
        statistics='Strict trajectory-level paired Student intervals; all snapshots, queries and posterior draws within a trajectory are not independent replications.',
        tail_scope='Exact iid native-complex samples; sample quantiles and independent mass/CDF checks are Monte Carlo approximations, not deterministic tail certificates.',
        primary='Complete SNR/snapshot decision map; comparisons are pointwise exploratory, without choosing a winning test regime.',
        truth_unavailable_to_estimator=True)
    return config


def _condition_index(condition):
    if condition not in CONDITIONS:raise ValueError('Unknown array condition')
    return CONDITIONS.index(condition)


def trajectory(config,condition,s,trial,split='test'):
    n=config['n'];T=max(config['snapshots']);seed=config['validation_seed'] if split=='validation' else config['test_seed']
    root=np.random.SeedSequence([seed,_condition_index(condition),int(round(1000*s)),int(trial)])
    latent,measurements,queries,initial,heldout=root.spawn(5)
    rng=np.random.default_rng(latent)
    if condition=='haar':
        h=np.broadcast_to(haar_line(rng,n),(T,n)).copy();angles=np.full(T,np.nan)
    else:
        angles=physical_angles(T,moving=condition=='ula_moving')
        h=np.array([ula(n,angle) for angle in angles])
    rng=np.random.default_rng(measurements)
    z=proper_complex(rng,T,config['q'])
    U=h*z[:,None]+proper_complex(rng,(T,n),s)
    a=ula(n,config['desired_angle']);qrng=np.random.default_rng(queries)
    V=proper_complex(qrng,(config['independent_queries'],n));V/=np.linalg.norm(V,axis=1)[:,None]
    initial_line=haar_line(np.random.default_rng(initial),n)
    return dict(U=U,h=h,a=a,angles=angles,queries=V,initial=initial_line,
        heldout_seed=heldout,condition=condition,s=float(s),q=config['q'])


def weighted_paths(U,betas):
    """Independent scatter implementation: no Bingham increment is used."""
    U=np.asarray(U,complex);n=U.shape[1];paths={}
    for beta in betas:
        S=np.zeros((n,n),complex);mass=0.;scatter=[];masses=[];modes=[];stat_clocks=[];mode_clocks=[];stat_seconds=mode_seconds=0.
        for u in U:
            begin=perf_counter();S=beta*S+np.outer(u,u.conj());mass=beta*mass+1.;stat_seconds+=perf_counter()-begin
            begin=perf_counter();h=np.linalg.eigh(S)[1][:,-1];mode_seconds+=perf_counter()-begin
            scatter.append(S.copy());masses.append(mass);modes.append(np.outer(h,h.conj()));stat_clocks.append(stat_seconds);mode_clocks.append(mode_seconds)
        paths[float(beta)]=dict(scatter=np.asarray(scatter),mass=np.asarray(masses),mode=np.asarray(modes),scatter_seconds=np.asarray(stat_clocks),eigensystem_seconds=np.asarray(mode_clocks))
    return paths


def isotropized_mean(M,P,tolerance=1e-9):
    n=len(M);rho=float((np.trace(P@M).real-1/n)/(1-1/n))
    if not -tolerance<=rho<=1+tolerance:raise ArithmeticError('Matched isotropization coefficient outside [0,1]')
    rho=float(np.clip(rho,0.,1.));result=rho*P+(1-rho)*np.eye(n)/n
    if abs(np.trace(P@(M-result)).real)>10*tolerance:
        raise ArithmeticError('Matched isotropization changes modal chordal risk')
    return result,rho


def conditional_excess(w,w_star,Rbar):
    left=output_power(w,Rbar)-output_power(w_star,Rbar)
    difference=w-w_star;right=output_power(difference,Rbar)
    error=abs(left-right)
    if error>1e-9*max(1.,abs(left),abs(right)):
        raise ArithmeticError('Expected-power optimality identity failed')
    if right < -1e-12:raise ArithmeticError('Negative posterior excess')
    return float(left),float(right),float(error)


def scalar_statistics(P,a,R):
    """Coefficients of power under one globally chosen covariance shrinkage."""
    b=P@a
    return np.array([output_power(a,R),np.vdot(b,R@a).real,
                     output_power(b,R),np.vdot(a,b).real])


def scalar_validation_loss(rho,statistics,s,q,n):
    x=np.asarray(statistics,float);c=s+q*(1-rho)/n;gamma=q*rho/(c+q*rho)
    value=(x[:,0]-2*gamma*x[:,1]+gamma*gamma*x[:,2])/(1-gamma*x[:,3])**2
    return float(value.mean())


def optimize_scalar(statistics,s,q,n,brackets=65):
    """Continuous minimization from a declared dense bracketing scan.

    This is an optimization diagnostic, not a certified global lower bound.
    Every local basin on the predeclared grid is refined; endpoints remain.
    """
    grid=np.linspace(0,1,int(brackets));values=np.array([scalar_validation_loss(r,statistics,s,q,n) for r in grid])
    candidates=[(values[0],0.),(values[-1],1.)];evaluations=len(grid)
    for j in range(1,len(grid)-1):
        if values[j]<=values[j-1] and values[j]<=values[j+1]:
            fit=minimize_scalar(lambda r:scalar_validation_loss(r,statistics,s,q,n),
                bounds=(grid[j-1],grid[j+1]),method='bounded',options=dict(xatol=1e-9,maxiter=100))
            evaluations+=fit.nfev;candidates.append((float(fit.fun),float(fit.x)))
    best=min(candidates,key=lambda x:x[0])
    return dict(loss=best[0],rho=best[1],evaluations=evaluations,brackets=int(brackets))


def _loaded(scatter,mass,loading,observation_count=None):
    if loading==0 and observation_count is not None and observation_count<len(scatter):
        raise ValueError('Unloaded sample covariance is singular before n snapshots')
    R=scatter/mass
    return R+loading*float(np.trace(R).real/len(R))*np.eye(len(R))


def _grouse_paths(U,steps,initial):
    paths={}
    for step in steps:
        f=GROUSE(U.shape[1],1,step=step,rng=0)
        # The existing geodesic update is complex-Hermitian; use the native
        # complex uniform initialization rather than its real default basis.
        f.basis=np.asarray(initial).reshape(-1,1).copy();P=[];clocks=[];seconds=0.
        for u in U:
            start=perf_counter();X=f.update(u);seconds+=perf_counter()-start
            P.append(X@X.conj().T);clocks.append(seconds)
        paths[float(step)]=dict(mode=np.asarray(P),seconds=np.asarray(clocks))
    return paths


def candidate_betas(config,condition):
    return [1.] if condition!='ula_moving' else list(config['betas'])


def validation_task(task):
    config,condition,s,trial=task;data=trajectory(config,condition,s,trial,'validation')
    U=data['U'];n=config['n'];q=config['q'];a=data['a'];rows=[]
    betas=candidate_betas(config,condition);paths=weighted_paths(U,betas)
    grouse=_grouse_paths(U,config['steps'],data['initial'])
    for t in config['snapshots']:
        j=t-1;Ptrue=np.outer(data['h'][j],data['h'][j].conj());R=s*np.eye(n)+q*Ptrue
        for beta,path in paths.items():
            P=path['mode'][j];base=dict(condition=condition,s=s,trial=trial,t=t,beta=beta)
            A=center(q/(s*(s+q))*path['scatter'][j]);moment=complex_moments(A)
            rows.append(dict(base,family='mean',loss=output_power(mvdr(s*np.eye(n)+q*moment.mean,a),R),mean_ok=moment.numerical_ok))
            rows.append(dict(base,family='pca',loss=output_power(mvdr(s*np.eye(n)+q*P,a),R),mean_ok=True))
            coeff=scalar_statistics(P,a,R)
            rows.append(dict(base,family='scalar',loss=np.nan,mean_ok=True,
                scalar_A=coeff[0],scalar_B=coeff[1],scalar_C=coeff[2],scalar_z=coeff[3]))
            for loading in config['loadings']:
                try:
                    covariance=_loaded(path['scatter'][j],path['mass'][j],loading,t)
                    loss=output_power(mvdr(covariance,a),R)
                except (ValueError,np.linalg.LinAlgError):loss=np.inf
                rows.append(dict(base,family='loaded',loading=float(loading),loss=loss,mean_ok=np.isfinite(loss)))
        for step,path in grouse.items():
            P=path['mode'][j]
            rows.append(dict(condition=condition,s=s,trial=trial,t=t,beta=np.nan,family='grouse',step=float(step),
                loss=output_power(mvdr(s*np.eye(n)+q*P,a),R),mean_ok=True))
    return rows


def choose_policies(rows,config):
    frame=pd.DataFrame(rows);selection=[];curves=[]
    expected=config['validation_trials']
    for (condition,s,t,family),g in frame.groupby(['condition','s','t','family'],sort=False):
        keys=['step'] if family=='grouse' else ['beta','loading'] if family=='loaded' else ['beta']
        options=[]
        for settings,h in g.groupby(keys,sort=False,dropna=False):
            values=settings if isinstance(settings,tuple) else (settings,);spec=dict(zip(keys,map(float,values)))
            complete=bool(len(h)==expected and h.trial.nunique()==expected and h.mean_ok.astype(bool).all())
            if family=='scalar':
                fit=optimize_scalar(h[['scalar_A','scalar_B','scalar_C','scalar_z']].to_numpy(),s,config['q'],config['n'],config['scalar_bracket_points'])
                spec['rho']=fit['rho'];loss=fit['loss'];evals=fit['evaluations']
            else:loss=float(h.loss.mean());evals=len(h)
            resolved=bool(complete and np.isfinite(loss))
            row=dict(condition=condition,s=float(s),t=int(t),family=family,**spec,
                validation_loss=loss if resolved else np.inf,resolved=resolved,independent_trials=len(h),
                evaluations=evals,selected=False)
            curves.append(row);options.append((row['validation_loss'],len(curves)-1))
        score,index=min(options,key=lambda x:x[0]);curves[index]['selected']=True
        selection.append({k:v for k,v in curves[index].items() if k not in ['selected','evaluations','independent_trials']})
    return selection,pd.DataFrame(curves)


def boundary_expansion(selected,config):
    minimum_beta=min(config['betas']);minimum_step=min(config['steps']);maximum_step=max(config['steps'])
    finite_loading=[x for x in config['loadings'] if x>0]
    beta=any(p['condition']=='ula_moving' and p.get('beta')==minimum_beta for p in selected)
    low=any(p.get('loading')==min(finite_loading) for p in selected)
    high=any(p.get('loading')==max(finite_loading) for p in selected)
    small=any(p.get('step')==minimum_step for p in selected);large=any(p.get('step')==maximum_step for p in selected)
    return dict(betas=config['beta_expansion'] if beta else [],
        loadings=([1e-6,1e-8] if low else [])+([1e6,1e8] if high else []),
        steps=([1e-5,3e-5] if small else [])+([3.,10.] if large else []))


def selected_for(selected,condition,s,t):
    return {p['family']:p for p in selected if p['condition']==condition and p['s']==s and p['t']==t}


def _interval_score(lower,upper,value,alpha=.05):
    return upper-lower+2/alpha*(np.maximum(lower-value,0)+np.maximum(value-upper,0))


def posterior_diagnostics(A,M,data,t,config,rng,scan_vectors=None,make_cloud=False):
    """Checked native-complex tails, converted into physical scanning powers."""
    n=len(A);s=data['s'];q=data['q'];j=t-1;h=data['h'][j]
    angles=np.asarray(config['evaluation_angles'])
    W=np.array([ula(n,a) for a in angles]) if scan_vectors is None else np.asarray(scan_vectors)
    report=posterior_report(A,W,truth=h,draws=config['posterior_draws'],rng=rng,
        relative_mcse_tolerance=config['relative_quantile_mcse'],return_samples=True)
    info=report['numerics'];X=report['vectors'];Pm=report['mode_projector']
    quantiles=s+q*report['quadratic_quantiles']
    truth=s+q*abs(W@h.conj())**2
    mean=s+q*np.einsum('bi,ij,bj->b',W.conj(),M,W).real
    rows=[]
    for idx in range(len(W)):
        lo,hi=quantiles[idx,0],quantiles[idx,4]
        rows.append(dict(query=idx,angle=float(angles[idx]) if scan_vectors is None else np.nan,
            truth=float(truth[idx]),mean=float(mean[idx]),lower=float(lo),upper=float(hi),
            q25=float(quantiles[idx,1]),median=float(quantiles[idx,2]),q75=float(quantiles[idx,3]),
            covered=bool(lo<=truth[idx]<=hi),width=float(hi-lo),interval_score=float(_interval_score(lo,hi,truth[idx])),
            cdf_error=float(max(info['scalar_cdf_error'])),numerical_ok=bool(info['numerical_ok'])))
    state=dict(ball_radius=report['ball_radius'],hpd_threshold=report['hpd_threshold'],
        ball_covered=report['ball_covered'],hpd_covered=report['hpd_covered'],
        ball_mass_reference=info['ball_independent_mass'],hpd_mass_reference=info['hpd_independent_mass'],
        mean_ok=info['mean_ok'],state_ok=info['state_ok'],response_ok=info['tail_ok'],
        numerical_ok=info['numerical_ok'],posterior_draws=config['posterior_draws'],
        reference_draws=config['posterior_draws'],state_mass_error=info['state_mass_error'],
        quantile_mcse_fraction=info['maximum_quantile_mcse_fraction_of_width'],
        hpd_tie_kind=info['hpd_tie_kind'],mass_scope=info['error_scope'])
    cloud=[]
    if make_cloud:
        v=data['queries'][0];projection=X*np.einsum('bi,i->b',X.conj(),v)[:,None]
        expected=M@v;point=Pm@v;true=h*np.vdot(h,v)
        for value in projection[:,1]:cloud.append(dict(x=float(value.real),y=float(value.imag),
            mean_x=float(expected[1].real),mean_y=float(expected[1].imag),mode_x=float(point[1].real),mode_y=float(point[1].imag),
            true_x=float(true[1].real),true_y=float(true[1].imag),query=0,
            projection='Real and imaginary parts of sensor-2 component of Pv; not a full-state region'))
    return state,rows,cloud,X,(report['hpd_constraint_matrix'],report['hpd_constraint_threshold'])


def test_task(task):
    config,condition,s,trial,selected=task;data=trajectory(config,condition,s,trial,'test')
    n=config['n'];q=config['q'];U=data['U'];a=data['a'];I=np.eye(n)
    policies={t:selected_for(selected,condition,s,t) for t in config['snapshots']}
    betas=sorted({p['beta'] for bytime in policies.values() for p in bytime.values() if 'beta' in p})
    paths=weighted_paths(U,betas);steps=sorted({p['grouse']['step'] for p in policies.values()})
    grouse=_grouse_paths(U,steps,data['initial'])
    metrics=[];shape=[];spectra=[];coverage=[];components=[];scans=[];patterns=[];cloud=[];story={}
    rng_hold=np.random.default_rng(data['heldout_seed']);rng_tail=np.random.default_rng(
        np.random.SeedSequence([config['test_seed'],_condition_index(condition),int(1000*s),trial,9981]))
    for t in config['snapshots']:
        j=t-1;p=policies[t];Ptrue=np.outer(data['h'][j],data['h'][j].conj());R=s*I+q*Ptrue
        path=paths[p['mean']['beta']];A=center(q/(s*(s+q))*path['scatter'][j]);start=perf_counter()
        result=complex_moments(A);M=result.mean;moment_seconds=perf_counter()-start
        P=path['mode'][j];Miso,rho=isotropized_mean(M,P)
        # Independent original update checks the scatter-to-Bingham equality.
        B=np.zeros((n,n),complex)
        for u in U[:t]:B=center(p['mean']['beta']*B+increment(u,s,q,complex_data=True))
        parameter_error=float(np.max(abs(B-A)))
        Pcheck=np.linalg.eigh(B)[1][:,-1:];projector_error=float(np.linalg.norm(P-Pcheck@Pcheck.conj().T))
        if parameter_error>1e-8 or projector_error>1e-8:raise ArithmeticError('Passive Bingham mode and weighted scatter PCA disagree')
        if t==1 and np.linalg.norm(M-Miso)>1e-8:raise ArithmeticError('One-snapshot isotropic equality failed')
        maps=dict(mean=M,mode=P,isotropic=Miso,pca=paths[p['pca']['beta']]['mode'][j],
            scalar=p['scalar']['rho']*paths[p['scalar']['beta']]['mode'][j]+(1-p['scalar']['rho'])*I/n,
            grouse=grouse[p['grouse']['step']]['mode'][j],oracle=Ptrue)
        covariances={method:s*I+q*D for method,D in maps.items()}
        load=p['loaded'];covariances['loaded']=_loaded(paths[load['beta']]['scatter'][j],paths[load['beta']]['mass'][j],load['loading'],t)
        weights={};decision_clocks={}
        for method,C in covariances.items():
            start=perf_counter();weights[method]=mvdr(C,a);decision_clocks[method]=perf_counter()-start
        Rbar=s*I+q*M;wstar=weights['mean'];heldout=data['h'][j]*proper_complex(rng_hold,(config['heldout_snapshots'],1),q)+proper_complex(rng_hold,(config['heldout_snapshots'],n),s)
        for method in METHODS:
            w=weights[method];D=maps.get(method);excess,identity_gap,error=conditional_excess(w,wstar,Rbar)
            gain_error=float(abs(np.vdot(w,a)-1))
            if gain_error>1e-10:raise ArithmeticError('Distortionless constraint failed')
            projection_loss=(float(np.mean(np.sum(abs(data['queries']@(D-Ptrue).T)**2,axis=1))) if D is not None else np.nan)
            risk=float(np.mean(np.einsum('bi,ij,bj->b',data['queries'].conj(),M-M@M,data['queries']).real)) if method=='mean' else np.nan
            policy=p.get('mean' if method in ['mean','mode','isotropic'] else method,{})
            metrics.append(dict(condition=condition,s=s,q=q,n=n,trial=trial,t=t,method=method,label=LABELS[method],
                true_power=output_power(w,R),heldout_power=sample_output_power(w,heldout),
                posterior_power=output_power(w,Rbar),posterior_excess=excess,optimality_identity_error=error,
                unit_gain_error=gain_error,projection_loss=projection_loss,posterior_risk=risk,
                beta=policy.get('beta',np.nan),rho=rho if method=='isotropic' else policy.get('rho',np.nan),
                loading=policy.get('loading',np.nan),step=policy.get('step',np.nan),
                mean_ok=result.numerical_ok,mean_error_bound=result.moment_error_bound,mean_order=result.order,
                moment_seconds=moment_seconds if method in ['mean','isotropic'] else 0.,decision_seconds=decision_clocks[method],
                update_seconds=(float(grouse[p['grouse']['step']]['seconds'][j]) if method=='grouse' else
                    float(paths[policy.get('beta',p['mean']['beta'])]['scatter_seconds'][j]) if method!='oracle' else 0.),
                mode_seconds=(float(paths[policy.get('beta',p['mean']['beta'])]['eigensystem_seconds'][j]) if method in ['mode','pca','scalar'] else 0.),
                pca_parameter_error=parameter_error,pca_projector_error=projector_error))
        shape.append(dict(condition=condition,s=s,q=q,n=n,trial=trial,t=t,rho=rho,
            anisotropy=float(np.linalg.norm(M-Miso)),modal_risk=float(1-np.trace(P@M).real),
            isotropic_modal_risk=float(1-np.trace(P@Miso).real),
            one_snapshot_equality=t==1 and np.linalg.norm(M-Miso)<1e-8))
        for index,(mean_eigen,isotropic_eigen) in enumerate(zip(np.linalg.eigvalsh(M),np.linalg.eigvalsh(Miso))):
            spectra.append(dict(condition=condition,s=s,trial=trial,t=t,index=index,mean=mean_eigen,isotropic=isotropic_eigen))
        common=dict(condition=condition,s=s,q=q,n=n,trial=trial,t=t)
        if trial<config['tail_trials'] and t in config['tail_times']:
            try:
                state,rows,_,_,_=posterior_diagnostics(A,M,data,t,config,rng_tail)
                coverage.append(dict(common,**state));components.extend(dict(common,**r) for r in rows)
            except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
                coverage.append(dict(common,mean_ok=result.numerical_ok,state_ok=False,response_ok=False,numerical_ok=False,failure=str(error)))
        if condition==config['story_condition'] and s==config['story_s'] and trial==config['story_trial'] and t in config['story_times']:
            angles=np.asarray(config['scan_angles']);W=np.array([ula(n,x) for x in angles])
            state,rows,pcloud,X,region=posterior_diagnostics(A,M,data,t,config,rng_tail,W,make_cloud=t==8)
            region_matrix,threshold=region
            for angle,w,row in zip(angles,W,rows):
                Bquery=np.outer(w,w.conj());upper=regional_bound(Bquery,region_matrix,threshold,1);lower=regional_bound(-Bquery,region_matrix,threshold,1)
                scans.append(dict(common,angle=float(angle),**{k:v for k,v in row.items() if k!='angle'},
                    mode=s+q*np.vdot(w,P@w).real,hpd_lower=s-q*lower.upper,hpd_upper=s+q*upper.upper,
                    region_gap=max(lower.gap,upper.gap),region_feasible=lower.feasible and upper.feasible,
                    hpd_mass_reference=state['hpd_mass_reference'],hpd_state_covered=state['hpd_covered'],
                    ball_state_covered=state['ball_covered'],hpd_ok=state['state_ok'],mean_ok=result.numerical_ok,
                    scalar_quantiles_ok=state['response_ok'],
                    envelope_ok=lower.feasible and upper.feasible and max(lower.gap,upper.gap)<1e-6))
                for method,wgt in weights.items():patterns.append(dict(common,angle=float(angle),method=method,
                    response=float(abs(np.vdot(wgt,w))**2)))
            cloud.extend(dict(common,**r) for r in pcloud)
            for key,value in dict(A=A,M=M,P_mode=P,P_true=Ptrue,a=a,h=data['h'][j],U=U[:t],queries=data['queries'],samples=X).items():story[f't{t}__{key}']=value
            for method,wgt in weights.items():story[f't{t}__w_{method}']=wgt
    return dict(metrics=metrics,shape=shape,spectra=spectra,coverage=coverage,power_components=components,
        scans=scans,weight_beampattern=patterns,projection_cloud=cloud,story=story)


def paired_summary(metrics):
    frame=pd.DataFrame(metrics);rows=[]
    for (condition,s,t),g in frame.groupby(['condition','s','t'],sort=False):
        for endpoint in ['true_power','heldout_power','projection_loss','posterior_power']:
            pivot=g.pivot(index='trial',columns='method',values=endpoint)
            for reference in [m for m in METHODS if m!='mean']:
                d=pivot[reference]-pivot['mean'];finite=np.isfinite(d).all();n=len(d)
                analytic_tie=bool(t==1 and reference=='isotropic')
                if analytic_tie:
                    if np.max(abs(d))>1e-9:raise ArithmeticError('Analytic one-snapshot equality was numerically lost')
                    d=d*0.
                gain=float(d.mean()) if finite else np.nan
                half=float(student_t.ppf(.975,n-1)*d.std(ddof=1)/np.sqrt(n)) if finite and n>1 else np.nan
                rows.append(dict(condition=condition,s=s,t=t,endpoint=endpoint,reference=reference,trials=n,
                    baseline=float(pivot[reference].mean()),mean=float(pivot['mean'].mean()),gain=gain,low=gain-half,high=gain+half,
                    relative_gain=gain/float(pivot[reference].mean()) if finite and float(pivot[reference].mean())>0 else np.nan,
                    win_fraction=float(np.mean(d>0)) if finite else np.nan,resolved=bool(finite),analytic_tie=analytic_tie,
                    inference='Pointwise exploratory paired Student interval; queries and draws are not replicates'))
    return pd.DataFrame(rows)
