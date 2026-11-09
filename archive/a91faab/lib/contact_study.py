"""Frozen mechanical-contact pilot with force-aware and velocity-only controls.

Only known applied force and noisy instantaneous velocity enter estimators.
Positions, true normals and resistance are plant/scoring information. The
anisotropic-resistance stress scores actual KKT query velocities, separately
from the geometric projection target used by the maintained scalar model.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .contact_mechanics import generate_contact,solve_contact_velocity
from .filter import BinghamFilter
from .geometry import center,projector,risk
from .theory import tracking_bound


BETAS=[1.,.98,.95,.9,.8]
RHOS=[0.,.25,.5,.75,1.]
RIDGES=[0.,1e-6,1e-5,1e-4,1e-3,.01,.1,1.]
CURVED=['sphere_large','sphere_small','weak']
MAIN=['plane',*CURVED]
METHODS=['Posterior mean','Mode','Tuned mode','Scalar shrinkage','Ridge map','Velocity PCA']


def default_config():
    return dict(schema=1,validation_seed=6245101,test_seed=6245102,
        validation_trials=16,test_trials=64,T=600,dt=.05,d=10.,v0=.02,sigma=.001,
        betas=BETAS,rhos=RHOS,ridges=RIDGES,report_times=[60,180,600],
        validation_conditions=MAIN,main_conditions=MAIN,stress_conditions=['anisotropic'],
        primary_condition='sphere_small',primary_time=600,
        primary_endpoint='Terminal actual clean-response squared error averaged over nine queries within trajectory',
        primary_references=['Scalar shrinkage','Ridge map'],
        selection='One policy per method on equally weighted three-condition curved validation mixture; plane uses separate beta=1 validation control',
        shrinkage_selection='For each beta, minimize the pooled validation quadratic over continuous rho in[0,1]; one rho shared across trajectories and curved conditions. Coarse rho curves are diagnostics, not the optimization restriction.',
        sensing='Known applied force and noisy instantaneous solver velocity only. Positions are plant/scoring records, not encoder measurements.',
        observation='Known isotropic viscous resistance in matched laws; fixed true sigma=0.001 m/s; no noise learning or motion supplied',
        stress_target='Actual KKT velocity under hypothetical force d*w; geometric P*w scored separately',
        statistics='Independent trajectories are replicates; nine queries and three coordinates averaged within each trajectory. Primary comparisons preselected; other intervals exploratory.',
        posterior='Deterministic complementary normal-line sphere integration with two-resolution diagnostics; no MCMC',
        noisy_intervals='Not evaluated; every interval in this pilot is a clean-response component interval',
        input_program='Plane cycles world axes; curved guide uses fixed world-frame sinusoidal force program',
        query_program='Three axes and normalized pairwise sums/differences, all norm v0',
        reporting='Posterior integration and task decisions at observations60/180/600; corrections every observation',
        delta=.05)


def query_vectors(v0=.02):
    e=np.eye(3);vectors=[e[i] for i in range(3)]
    for i,j in [(0,1),(0,2),(1,2)]:
        vectors.extend([(e[i]+e[j])/np.sqrt(2),(e[i]-e[j])/np.sqrt(2)])
    return float(v0)*np.asarray(vectors)


def plant_data(config,seed,trial,condition):
    return generate_contact(seed,trial,condition,**{k:config[k] for k in ['T','dt','d','v0','sigma']})


def query_truth(data,t,queries,d):
    """Use independently solved physical responses, including the D mismatch."""
    normal=data['normals'][t-1];D=data['resistance']
    physical=np.array([solve_contact_velocity(d*w,normal,D)[0] for w in queries])
    geometric=queries@data['tangent_projectors'][t-1].T
    if data['diagnostics']['model_matched'] and np.max(abs(physical-geometric))>1e-11:
        raise ArithmeticError('Hypothetical KKT query disagrees with the matched projector identity')
    return physical,geometric


def response_loss(D,queries,truth):
    return float(np.mean(np.sum((queries@D.T-truth)**2,axis=1)))


def weighted_projector_ls(U,V,beta=1.):
    """Independent conditional projector least squares, without increment()."""
    weights=beta**np.arange(len(U)-1,-1,-1)
    S=(U.T*weights)@V+(V.T*weights)@U-(V.T*weights)@V
    values,basis=np.linalg.eigh((S+S.T)/2)
    return basis[:,-2:]@basis[:,-2:].T,S,values


def _snapshots(data,betas,s,report_times):
    """Reuse each parameter trajectory for all scalar/ridge candidates.

    Timing for the Bingham correction includes its returned eigenspace every
    observation. Ridge and PCA accumulate sufficient statistics each step and
    form decisions only at report times. The independent LS audit is untimed.
    """
    n=3;state={}
    for beta in sorted(set(map(float,betas))):
        state[beta]=dict(f=BinghamFilter(n,2,s,tau2=None,beta=beta),C=np.zeros((n,n)),
            cross=np.zeros((n,n)),scatter=np.zeros((n,n)),update=0.,ridge_update=0.,pca_update=0.,records={})
    for t,(u,v) in enumerate(zip(data['U'],data['V']),1):
        for beta,a in state.items():
            begin=perf_counter();basis=a['f'].update(u,v=v);a['update']+=perf_counter()-begin
            begin=perf_counter();a['C']=beta*a['C']+np.outer(v,v);a['cross']=beta*a['cross']+np.outer(u,v)
            a['ridge_update']+=perf_counter()-begin
            begin=perf_counter();a['scatter']=beta*a['scatter']+np.outer(u,u);a['pca_update']+=perf_counter()-begin
            if t not in report_times:continue
            P=projector(basis)
            least_squares,S,values=weighted_projector_ls(data['U'][:t],data['V'][:t],beta)
            parameter_difference=float(np.max(abs(center(S/(2*s))-a['f'].A)))
            projector_difference=float(np.linalg.norm(P-least_squares))
            score_difference=float(abs(np.trace(S@(P-least_squares))))
            gap=float(values[1]-values[0])
            if parameter_difference>1e-7 or score_difference>1e-10 or (gap>1e-10 and projector_difference>1e-8):
                raise ArithmeticError('Bingham mode and independent weighted projector LS disagree')
            begin=perf_counter();E=np.linalg.eigh(a['scatter'])[1][:,-2:];Ppca=projector(E)
            pca_decision=perf_counter()-begin
            a['records'][t]=dict(A=a['f'].A.copy(),mode=P,C=a['C'].copy(),cross=a['cross'].copy(),
                pca=Ppca,bingham_seconds=a['update'],ridge_seconds=a['ridge_update'],pca_seconds=a['pca_update'],
                pca_decision_seconds=pca_decision,ls_parameter_error=parameter_difference,
                ls_projector_error=projector_difference,ls_objective_error=score_difference,
                input_eigenvalues=np.linalg.eigvalsh(a['C']))
    return {b:a['records'] for b,a in state.items()}


def ridge_map(snapshot,regularization):
    C=snapshot['C'];scale=max(float(np.trace(C)/3),np.finfo(float).tiny)
    return snapshot['cross']@np.linalg.pinv(C+float(regularization)*scale*np.eye(3),rcond=1e-12)


def _mean(A):
    from .contact_posterior import contact_mean
    return contact_mean(A)


def _diagnostics(summary):
    """Read the posterior module's explicit diagnostics, without hiding flags."""
    return summary['numerics']


def validation_task(task):
    config,trial,condition=task
    data=plant_data(config,config['validation_seed'],trial,condition)
    betas=[1.] if condition=='plane' else config['betas']
    snapshots=_snapshots(data,betas,config['sigma']**2,[config['T']])
    queries=query_vectors(config['v0']);truth,_=query_truth(data,config['T'],queries,config['d'])
    rows=[]
    for beta,by_t in snapshots.items():
        a=by_t[config['T']];begin=perf_counter();M,diagnostics=_mean(a['A']);mean_time=perf_counter()-begin
        ok=bool(diagnostics['numerical_ok'])
        common=dict(trial=trial,condition=condition,scope='plane' if condition=='plane' else 'curved',
                    beta=beta,numerical_ok=ok,mean_seconds=mean_time)
        rows.append(dict(common,method='Posterior mean',rho=np.nan,regularization=np.nan,
                         loss=response_loss(M,queries,truth)))
        rows.append(dict(common,method='Tuned mode',rho=np.nan,regularization=np.nan,
                         numerical_ok=True,loss=response_loss(a['mode'],queries,truth)))
        rows.append(dict(common,method='Velocity PCA',rho=np.nan,regularization=np.nan,
                         numerical_ok=True,loss=response_loss(a['pca'],queries,truth)))
        for rho in config['rhos']:
            D=rho*a['mode']+(1-rho)*(2/3)*np.eye(3)
            rows.append(dict(common,method='Scalar shrinkage',rho=rho,regularization=np.nan,
                             numerical_ok=True,loss=response_loss(D,queries,truth)))
        for regularization in config['ridges']:
            D=ridge_map(a,regularization)
            rows.append(dict(common,method='Ridge map',rho=np.nan,regularization=regularization,
                             numerical_ok=True,loss=response_loss(D,queries,truth)))
    return rows


def select_policies(validation):
    frame=pd.DataFrame(validation);selected=[];curves=[]
    for scope in ['plane','curved']:
        portion=frame[frame.scope==scope]
        for method,g in portion.groupby('method',sort=False):
            keys=['beta']+(['rho'] if method=='Scalar shrinkage' else ['regularization'] if method=='Ridge map' else [])
            choices=[]
            for settings,h in g.groupby(keys,sort=False,dropna=False):
                settings=settings if isinstance(settings,tuple) else (settings,)
                parameters=dict(zip(keys,settings))
                # Average the three curved conditions within each independent
                # validation trial before uncertainty summaries.
                independent=h.groupby('trial').loss.mean();score=float(independent.mean())
                se=float(independent.std(ddof=1)/np.sqrt(len(independent))) if len(independent)>1 else np.nan
                row=dict(scope=scope,method=method,**parameters,validation_loss=score,se=se,
                    independent_trials=len(independent),evaluations=len(h),
                    numerical_failures=int((~h.numerical_ok.astype(bool)).sum()),selected=False,
                    finite_grid_edge=bool(scope=='curved' and parameters['beta']==min(BETAS)
                                          or parameters.get('regularization',0.)==max(RIDGES)),
                    boundary_note='rho endpoints and beta=1/ridge=0 are structural; smallest beta/largest ridge are finite search edges')
                curves.append(row);choices.append((score,len(curves)-1))
            if method=='Scalar shrinkage':
                # The response loss is exactly quadratic in a common rho.
                # Recover its coefficients from fixed grid probes; optimize
                # pooled validation data, never a separate oracle per trial.
                for beta,h in g.groupby('beta',sort=False):
                    grid=h.pivot(index=['trial','condition'],columns='rho',values='loss')
                    c=grid[0.];a=2*(grid[1.]-2*grid[.5]+c);b=(a+c-grid[1.])/2
                    independent_a=a.groupby(level='trial').mean();independent_b=b.groupby(level='trial').mean()
                    if independent_a.mean()<=0:raise ArithmeticError('Degenerate scalar-shrinkage validation quadratic')
                    rho=float(np.clip(independent_b.mean()/independent_a.mean(),0.,1.))
                    individual=(a*rho*rho-2*b*rho+c).groupby(level='trial').mean()
                    score=float(individual.mean())
                    curves.append(dict(scope=scope,method=method,beta=float(beta),rho=rho,
                        validation_loss=score,se=float(individual.std(ddof=1)/np.sqrt(len(individual))) if len(individual)>1 else np.nan,
                        independent_trials=len(individual),evaluations=len(h)//len(RHOS),numerical_failures=0,selected=False,
                        finite_grid_edge=bool(scope=='curved' and beta==min(BETAS)),
                        boundary_note='Continuous pooled rho optimum; rho0/1 are structural constraints',
                        optimization='Analytic common rho from pooled independent validation trajectories'))
                    choices.append((score,len(curves)-1))
            _,index=min(choices,key=lambda x:x[0]);curves[index]['selected']=True
            selected.append({k:v for k,v in curves[index].items() if k in ['scope','method','beta','rho','regularization']})
    return selected,pd.DataFrame(curves)


def _policy(selected,condition,method):
    scope='plane' if condition=='plane' else 'curved'
    return next(p for p in selected if p['scope']==scope and p['method']==method)


def test_task(task):
    config,trial,condition,selected=task
    data=plant_data(config,config['test_seed'],trial,condition)
    policies={m:_policy(selected,condition,m) for m in METHODS if m!='Mode'}
    betas={p['beta'] for p in policies.values()}
    snapshots=_snapshots(data,betas,config['sigma']**2,config['report_times'])
    queries=query_vectors(config['v0']);metrics=[];posteriors=[];query_rows=[];bounds=[];paths=[]
    cumulative_integration=0.;cumulative_decisions={method:0. for method in METHODS}
    from .contact_posterior import contact_summary
    for t in config['report_times']:
        beta=float(policies['Posterior mean']['beta']);a=snapshots[beta][t]
        truth,geometric=query_truth(data,t,queries,config['d'])
        begin=perf_counter();summary=contact_summary(a['A'],queries,true_normal=data['normals'][t-1]);integration=perf_counter()-begin
        M=np.asarray(summary['mean']);P=np.asarray(summary['mode_projector']);diag=_diagnostics(summary)
        ok=bool(diag['numerical_ok'])
        mean_ok=bool(diag['mean_numerical_ok'])
        interval_ok=bool(diag['response_quantiles_ok'])
        state_ok=bool(diag['regions_numerical_ok'])
        cumulative_integration+=integration
        lower=np.asarray(summary['response_lower']);upper=np.asarray(summary['response_upper'])
        radius=float(summary['ball_radius']);threshold=float(summary['hpd_threshold'])
        trueP=data['tangent_projectors'][t-1]
        chordal=float(np.sqrt(max(0.,2-np.trace(P@trueP))))
        hpd_loss=float(np.trace(a['A']@(P-trueP)))
        covered=(truth>=lower)&(truth<=upper);geometric_covered=(geometric>=lower)&(geometric<=upper)
        risks=np.array([risk(M,w) for w in queries])
        post=dict(condition=condition,trial=trial,t=t,beta=beta,modal_chordal_error=chordal,
            ball_radius=radius,hpd_threshold=threshold,hpd_loss=hpd_loss,
            ball_covered=bool(chordal<=radius),hpd_covered=bool(hpd_loss<=threshold),
            component_coverage_mean=float(covered.mean()),geometric_component_coverage=float(geometric_covered.mean()),
            width_normalized=float((upper-lower).mean()/config['v0']),
            mean_component_width=float((upper-lower).mean()),risk_mean=float(risks.mean()),
            empirical_response_error=response_loss(M,queries,truth),geometric_response_error=response_loss(M,queries,geometric),
            integration_seconds=integration,cumulative_integration_seconds=cumulative_integration,
            numerical_ok=ok,mean_numerical_ok=mean_ok,state_regions_ok=state_ok,response_intervals_ok=interval_ok,
            diagnostic_mean_error=float(diag.get('mean_resolution_error',np.nan)),
            diagnostic_threshold_error=float(diag.get('region_integration_error',np.nan)),
            diagnostic_quantile_error=float(diag.get('quantile_scaled_error',np.nan)),
            ls_parameter_error=a['ls_parameter_error'],ls_projector_error=a['ls_projector_error'],
            ls_objective_error=a['ls_objective_error'],minimum_retained_excitation=float(a['input_eigenvalues'][0]),
            maximum_retained_excitation=float(a['input_eigenvalues'][-1]),
            response_target='actual KKT mobility; scalar model mismatched' if condition=='anisotropic' else 'geometric tangent projection equals actual KKT mobility')
        post.update({f'numerics_{key}':value for key,value in diag.items() if np.isscalar(value)})
        normal=data['normals'][t-1];X=data['X'][t-1];C=a['C']
        normal_energy=float(normal@C@normal)
        local_information=(X.T@C@X+normal_energy*np.eye(2))/config['sigma']**2
        post.update(local_information_min=float(np.linalg.eigvalsh(local_information)[0]),
                    local_information_max=float(np.linalg.eigvalsh(local_information)[-1]),
                    normal_excitation_retained=normal_energy,
                    normal_alignment_retained=normal_energy/max(float(np.trace(C)),np.finfo(float).tiny),
                    current_normal_alignment=float((normal@data['V'][t-1])**2/max(float(data['V'][t-1]@data['V'][t-1]),np.finfo(float).tiny)))
        posteriors.append(post)
        for j,w in enumerate(queries):
            mean_y=M@w;mode_y=P@w
            for component in range(3):
                query_rows.append(dict(condition=condition,trial=trial,t=t,query=j,component=component,
                    wx=w[0],wy=w[1],wz=w[2],truth=truth[j,component],geometric_truth=geometric[j,component],
                    mean=mean_y[component],mode=mode_y[component],lower=lower[j,component],upper=upper[j,component],
                    covered=bool(covered[j,component]),geometric_covered=bool(geometric_covered[j,component]),
                    width=upper[j,component]-lower[j,component],numerical_ok=interval_ok,
                    mean_numerical_ok=mean_ok,risk=risks[j]))
        actions={'Posterior mean':(M,a,beta,integration,0.),'Mode':(P,a,beta,0.,0.)}
        for method in ['Tuned mode','Scalar shrinkage','Ridge map','Velocity PCA']:
            p=policies[method];b=float(p['beta']);other=snapshots[b][t]
            begin=perf_counter()
            D=other['mode'] if method=='Tuned mode' else other['pca'] if method=='Velocity PCA' else ridge_map(other,p['regularization']) if method=='Ridge map' else p['rho']*other['mode']+(1-p['rho'])*(2/3)*np.eye(3)
            prep=perf_counter()-begin+(other['pca_decision_seconds'] if method=='Velocity PCA' else 0.)
            actions[method]=(D,other,b,0.,prep)
        for method,(D,snapshot,b,integration_time,prep) in actions.items():
            begin=perf_counter();prediction=queries@D.T;decision=prep+perf_counter()-begin
            cumulative_decisions[method]+=decision
            loss=float(np.mean(np.sum((prediction-truth)**2,axis=1)))
            geometric_loss=float(np.mean(np.sum((prediction-geometric)**2,axis=1)))
            update=snapshot['ridge_seconds'] if method=='Ridge map' else snapshot['pca_seconds'] if method=='Velocity PCA' else snapshot['bingham_seconds']
            state_error=float(np.sqrt(max(0.,2-np.trace(D@trueP)))) if method in ['Mode','Tuned mode','Velocity PCA'] else chordal if method=='Posterior mean' else np.nan
            metrics.append(dict(condition=condition,trial=trial,t=t,method=method,beta=b,loss=loss,
                normalized_loss=loss/config['v0']**2,geometric_loss=geometric_loss,
                modal_chordal_error=state_error,update_seconds=update,integration_seconds=integration_time,
                cumulative_integration_seconds=cumulative_integration if method=='Posterior mean' else 0.,
                decision_seconds=decision,cumulative_decision_seconds=cumulative_decisions[method],
                cumulative_seconds=update+(cumulative_integration if method=='Posterior mean' else 0.)+cumulative_decisions[method],
                numerical_ok=mean_ok if method=='Posterior mean' else True,
                hypothetical_queries=len(queries),statistical_unit='trajectory; query errors averaged before inference'))
        if condition!='anisotropic':
            bound=tracking_bound(data['X'][:t],data['V'][:t],config['sigma']**2,beta=beta,delta=config['delta'])
            bounds.append(dict(condition=condition,trial=trial,t=t,method='Posterior mean modal basis',beta=beta,
                applicable=True,valid=bound.valid,bound=bound.bound,ceiling=bound.ceiling,gamma=bound.gamma,
                noise=bound.noise,drift=bound.drift,epsilon=bound.epsilon,modal_chordal_error=chordal,
                violated=bool(bound.valid and chordal>bound.bound),scope='Fixed time, real conditional model, true variance, identity supplied motion'))
        else:
            bounds.append(dict(condition=condition,trial=trial,t=t,method='Posterior mean modal basis',beta=beta,
                applicable=False,valid=False,bound=np.nan,ceiling=1.,gamma=np.nan,noise=np.nan,drift=np.nan,
                epsilon=np.nan,modal_chordal_error=chordal,violated=False,scope='Not applicable: anisotropic response-law mismatch'))
    if trial==0:
        for j,(p,n,u,v,y) in enumerate(zip(data['positions'],data['normals'],data['U'],data['V'],data['velocities']),1):
            paths.append(dict(condition=condition,trial=trial,t=j,time=data['times'][j-1],
                x=p[0],y=p[1],z=p[2],nx=n[0],ny=n[1],nz=n[2],
                vx=v[0],vy=v[1],vz=v[2],dx=y[0],dy=y[1],dz=y[2],ux=u[0],uy=u[1],uz=u[2]))
    trajectory=dict(data['diagnostics'],condition=condition,trial=trial,
                    condition_label=data['diagnostics']['condition'])
    return dict(metrics=metrics,posterior=posteriors,query_predictions=query_rows,bounds=bounds,paths=paths,trajectories=[trajectory])


def summaries(frames,config):
    from .reporting import wilson
    pairs=[];terminal=frames['metrics'][frames['metrics'].t==config['T']]
    for condition,g in terminal.groupby('condition',sort=False):
        pivot=g.pivot(index='trial',columns='method',values='loss');mean=pivot['Posterior mean']
        numerical_failures=int((~g[g.method=='Posterior mean'].numerical_ok.astype(bool)).sum())
        for reference in METHODS:
            if reference=='Posterior mean':continue
            gain=pivot[reference]-mean;n=len(gain);se=float(gain.std(ddof=1)/np.sqrt(n))
            half=float(student_t.ppf(.975,n-1)*se) if n>1 else np.nan
            primary=condition==config['primary_condition'] and reference in config['primary_references']
            pairs.append(dict(condition=condition,reference=reference,trials=n,gain=float(gain.mean()),
                low=float(gain.mean()-half),high=float(gain.mean()+half),
                mean_loss=float(mean.mean()),reference_loss=float(pivot[reference].mean()),
                normalized_gain=float(gain.mean()/config['v0']**2),normalized_low=float((gain.mean()-half)/config['v0']**2),
                normalized_high=float((gain.mean()+half)/config['v0']**2),primary=primary,numerical_failures=numerical_failures,
                interval='Two preselected references at primary condition; other pointwise intervals exploratory; no simultaneous claim'))
    coverage=[]
    for (condition,t),g in frames['posterior'].groupby(['condition','t'],sort=False):
        for region,flag,width in [('State ball','ball_covered','ball_radius'),('State HPD','hpd_covered','hpd_threshold')]:
            successes=int(g[flag].sum());lo,hi=wilson(successes,len(g))
            coverage.append(dict(condition=condition,t=t,region=region,successes=successes,trials=len(g),
                coverage=successes/len(g),low=lo,high=hi,mean_width=float(g[width].mean()),
                width_quantity=width,numerical_failures=int((~g.state_regions_ok.astype(bool)).sum()),interval='Wilson across independent trajectories'))
        independent=g.component_coverage_mean;n=len(independent)
        half=float(student_t.ppf(.975,n-1)*independent.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
        coverage.append(dict(condition=condition,t=t,region='Clean response',successes=np.nan,trials=n,
            coverage=float(independent.mean()),low=max(0.,float(independent.mean()-half)),high=min(1.,float(independent.mean()+half)),
            mean_width=float(g.width_normalized.mean()),width_quantity='mean component interval width / v0',
            numerical_failures=int((~g.response_intervals_ok.astype(bool)).sum()),interval='Student across trajectory means of27 component coverage flags'))
    query_coverage=[]
    for keys,g in frames['query_predictions'].groupby(['condition','t','query','component'],sort=False):
        successes=int(g.covered.sum());lo,hi=wilson(successes,len(g))
        query_coverage.append(dict(condition=keys[0],t=keys[1],query=keys[2],component=keys[3],
            successes=successes,trials=len(g),coverage=successes/len(g),low=lo,high=hi,
            mean_width=float(g.width.mean()),numerical_failures=int((~g.numerical_ok.astype(bool)).sum())))
    costs=frames['metrics'].groupby(['condition','t','method'],sort=False).agg(
        update_seconds=('update_seconds','mean'),integration_seconds=('integration_seconds','mean'),
        decision_seconds=('decision_seconds','mean'),trials=('trial','nunique')).reset_index()
    return dict(paired=pd.DataFrame(pairs),coverage=pd.DataFrame(coverage),query_coverage=pd.DataFrame(query_coverage),costs=costs)
