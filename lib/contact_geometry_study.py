"""Conditional contact-normal experiments for the draft-5 simulation redesign.

The physical plant independently solves maintained bilateral contact. Estimators
receive applied free velocity, noisy realized velocity, and a measured wrist
orientation only. Surface positions and normals enter generation/scoring, never
correction. All densities are real, with normal parameter C=-A for a tangent
hyperplane parameter A. Numerical reports retain local-Gaussian failures.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t
from .geometry import center
from .models import increment
from .spectral_series import rank_one_bingham
from .contact_mechanics import generate_contact
from .contact_posterior import contact_mean, contact_summary, normal_quadrature
from .contact_process_prediction import fast_moments
from .contact_gaussian import (laplace_state, measurement, measurement_jacobian,
    exponential_normal, parallel_rotation, _MappedGaussian)
from .theory import tracking_bound, fisher_information

METHODS=('mean','mode','isotropic','regularized','ridge','residual_pca','laplace')


def configuration(profile='quick'):
    if profile not in ('quick','paper'):raise ValueError('quick or paper required')
    quick=profile=='quick'
    return dict(schema=1,profile=profile,n=3,k=2,v0=.02,sigma_static=.01,
        static_times=[1,2,4,8,16,32],static_inputs=['balanced','unbalanced'],
        static_validation_trials=4 if quick else 32,static_test_trials=8 if quick else 128,
        dynamic_conditions=['plane','sphere'],dynamic_T=48 if quick else 160,
        dynamic_times=[8,24,48] if quick else [8,40,80,160],dynamic_dt=.05,sigma_dynamic=.004,
        dynamic_validation_trials=4 if quick else 16,dynamic_test_trials=8 if quick else 64,
        betas=[1.,.98,.95,.9,.8,.6],ridge_loadings=[0.,1e-4,1e-3,.01,.1,1.,10.,100.],
        gaussian_variances=[0.,1e-4,4e-4,1.6e-3,6.4e-3],gaussian_warmup=6,
        lambdas=[1.,10.],higher_dimensions=[8,16,32],higher_trials=4 if quick else 32,
        higher_samples_per_dimension=4,certificate_trials=8 if quick else 128,
        certificate_times=[1024,4096],certificate_delta=.05,
        seed_validation=8113101+(10000 if quick else 0),
        seed_test=8113102+(10000 if quick else 0),seed_story=8113103,
        sigma_story=[.01,.004],story_normal=[0.,0.,1.],
        geometry='Real Gr(N-1,N); contact-normal lines are Gr(1,N), with C=-A.',
        mechanics='Maintained ideal bilateral guide with known isotropic resistance; u=P_X v+e.',
        wrist='Known rotation independent of hidden surface normal; R maps wrist to world, G_t=R_t.T@R_(t-1).',
        inputs='Every static command has norm v0. Balanced axes versus directions mostly along e1; this is not a truth-aligned adaptive probe.',
        statistical_units='Independent contact/data trajectories; query coordinates are averaged within each trajectory.',
        primary='Static broad-posterior full-vector response loss; dynamic terminal and time-average losses are separate.',
        gaussian='Local normal-coordinate EKF, six actual-observation conditional-Bingham warmup; independently validated Q.',
        exclusions='No latent geometry, true normal rotations, future observations, learned motion, temperature or stochastic kappa supplied to estimators.',
        higher_scope='Generalized isotropic-resistance one-constraint velocity model, not a physical high-DOF robot with known Jacobians.')


def unit(rng,n):
    value=rng.normal(size=n)
    return value/np.linalg.norm(value)


def projector_from_normal(normal):
    n=np.asarray(normal,float)
    return np.eye(len(n))-np.outer(n,n)


def normal_mode(A):
    values,E=np.linalg.eigh((A+A.T)/2)
    normal=E[:,0]
    return projector_from_normal(normal),normal,float(values[1]-values[0])


def mean_projector(A,fast=False):
    """Validated rank-one complement, preserving native real measure."""
    A=np.asarray(A,float);start=perf_counter()
    if fast and len(A)==3:
        result=fast_moments(A,checked=True)
        return result.mean,dict(numerical_ok=result.numerical_ok,order=result.angular_order,
            error_bound=np.nan,resolution_error=result.resolution_error,seconds=perf_counter()-start,
            log_normalizer=result.log_normalizer,scope='Analytic radial moments with independent angular refinement')
    result=rank_one_bingham(-A,complex_data=False,tol=1e-10,max_order=4096)
    M=np.eye(len(A))-result.mean
    return M,dict(numerical_ok=result.accuracy_met,order=result.order,
        error_bound=result.moment_error_bound,seconds=perf_counter()-start,
        log_normalizer=float(np.trace(A)+result.log_normalizer))


def isotropized_projector_mean(M,P):
    n=len(M);normal_mean=np.eye(n)-M;normal_P=np.eye(n)-P
    rho=float((np.trace(normal_P@normal_mean)-1/n)/(1-1/n))
    if rho < -1e-8 or rho > 1+1e-8:raise ArithmeticError('Isotropization outside its spectral range')
    iso_normal=rho*normal_P+(1-rho)*np.eye(n)/n
    return np.eye(n)-iso_normal,rho


def regularized_projector(P,rho):
    n=len(P)
    return (1-1/n)*np.eye(n)+float(rho)*(P-(1-1/n)*np.eye(n))


def response_mse(D,P,v0):
    """Tight-frame average: N query coordinates are not independent trials."""
    return float(v0*v0/len(P)*np.linalg.norm(D-P,'fro')**2)


def soft_action(D,lam):
    D=np.asarray(D,float)
    if (not np.isfinite(D).all() or not np.allclose(D,D.T,atol=1e-10,rtol=0)
            or np.linalg.eigvalsh(D)[0]<-1e-8 or np.linalg.eigvalsh(D)[-1]>1+1e-8
            or not np.isfinite(lam) or lam<0):
        raise ValueError('Soft action needs a PSD contraction and nonnegative lambda')
    return np.linalg.solve(np.eye(len(D))+float(lam)*(np.eye(len(D))-D),np.eye(len(D)))


def soft_cost(W,P,lam,v0):
    n=len(P)
    return float(v0*v0/n*(np.linalg.norm(W-np.eye(n),'fro')**2+
        float(lam)*np.trace(W.T@(np.eye(n)-P)@W)))


def conditional_soft_excess(D,comparison,lam,v0):
    W=soft_action(D,lam);L=soft_action(comparison,lam);H=np.eye(len(D))+lam*(np.eye(len(D))-D)
    difference=soft_cost(L,D,lam,v0)-soft_cost(W,D,lam,v0)
    identity=float(v0*v0/len(D)*np.trace((L-W).T@H@(L-W)))
    return difference,abs(difference-identity)


def command_program(n,T,kind,v0):
    if kind=='balanced':return v0*np.eye(n)[np.arange(T)%n]
    if kind!='unbalanced':raise ValueError('Unknown deterministic input design')
    values=[]
    for j in range(T):
        value=np.eye(n)[0].copy()
        phase=j%(1+2*(n-1))
        if phase:
            axis=(phase+1)//2
            value[axis]=.2 if phase%2 else -.2
        values.append(v0*value/np.linalg.norm(value))
    return np.asarray(values)


def static_data(cfg,trial,kind,split='test',n=3,T=None,normal=None):
    seed=cfg['seed_validation'] if split=='validation' else cfg['seed_test']
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial,n,31801]))
    h=unit(rng,n) if normal is None else np.asarray(normal,float)
    T=max(cfg['static_times']) if T is None else int(T)
    V=command_program(n,T,kind,cfg['v0']);P=projector_from_normal(h)
    # Same trial root pairs the normal and Gaussian errors across input designs.
    noise=cfg['sigma_static']*rng.normal(size=(T,n));clean=V@P
    return dict(U=clean+noise,V=V,P=P,normal=h,clean=clean)


def static_paths(U,V,s):
    n=U.shape[1];A=np.zeros((n,n));S=np.zeros((n,n));R=np.zeros((n,n));C=np.zeros((n,n));Y=np.zeros((n,n))
    states=[];maximum_audit=0.;update_seconds=0.
    for u,v in zip(U,V):
        begin=perf_counter();A=center(A+increment(u,s,v=v));update_seconds+=perf_counter()-begin
        S+=np.outer(u,v)+np.outer(v,u)-np.outer(v,v)
        maximum_audit=max(maximum_audit,float(np.max(abs(A-center(S/(2*s))))))
        residual=v-u;R+=np.outer(residual,residual);C+=np.outer(v,v);Y+=np.outer(u,v)
        states.append(dict(A=A.copy(),S=S.copy(),residual=R.copy(),C=C.copy(),Y=Y.copy(),update_seconds=update_seconds))
    return states,maximum_audit


def ridge_map(state,loading):
    C=state['C'];n=len(C);scale=max(float(np.trace(C)/n),1e-15)
    strength=(float(loading)+1e-10)*scale
    anchor=(1-1/n)*np.eye(n)
    return np.linalg.solve(C+strength*np.eye(n),(state['Y']+strength*anchor).T).T


def residual_projector(state):
    h=np.linalg.eigh(state['residual'])[1][:,-1]
    return projector_from_normal(h)


def _record(D,P,A,method,cfg,**fields):
    n=len(P);p,normal,gap=normal_mode(A);M=fields.pop('posterior_mean',None)
    estimate=fields.pop('estimator_projector',D if method=='residual_pca' else p)
    state_error=(float(np.sqrt(max(0.,1-np.trace((np.eye(n)-estimate)@(np.eye(n)-P)))))
                 if method!='ridge' else np.nan)
    row=dict(method=method,response_mse=response_mse(D,P,cfg['v0']),
        normalized_response_mse=float(np.linalg.norm(D-P,'fro')**2),
        normal_chordal_error=state_error,
        mode_gap=gap,**fields)
    if M is not None:
        row['posterior_risk']=float(cfg['v0']**2/n*np.trace(M-M@M))
        row['posterior_response_excess']=float(cfg['v0']**2/n*np.linalg.norm(D-M,'fro')**2)
    for lam in cfg['lambdas']:
        key=str(int(lam))
        if method=='ridge':row['cost_'+key]=np.nan;row['oracle_excess_'+key]=np.nan
        else:
            cost=soft_cost(soft_action(D,lam),P,lam,cfg['v0'])
            oracle=soft_cost(soft_action(P,lam),P,lam,cfg['v0'])
            row['cost_'+key]=cost;row['oracle_excess_'+key]=cost-oracle
        if M is not None and method!='ridge':
            value,error=conditional_soft_excess(M,D,lam,cfg['v0'])
            row['posterior_excess_'+key]=value;row['identity_error_'+key]=error
    return row


def static_validation(task):
    cfg,kind,trial=task;data=static_data(cfg,trial,kind,'validation')
    path,_=static_paths(data['U'],data['V'],cfg['sigma_static']**2);rows=[]
    for t in cfg['static_times']:
        state=path[t-1];P,_,_=normal_mode(state['A']);n=len(P)
        D0=(1-1/n)*np.eye(n);D1=P-D0;target=data['P']
        rows.append(dict(kind=kind,trial=trial,t=t,family='regularized',
            quadratic=float(np.sum(D1*D1)),linear=float(np.sum(D1*(D0-target))),loading=np.nan,loss=np.nan))
        for loading in cfg['ridge_loadings']:
            rows.append(dict(kind=kind,trial=trial,t=t,family='ridge',loading=loading,
                loss=response_mse(ridge_map(state,loading),target,cfg['v0']),quadratic=np.nan,linear=np.nan))
    return rows


def select_static(rows,cfg):
    import pandas as pd
    frame=pd.DataFrame(rows);selected=[]
    for kind in cfg['static_inputs']:
        for t in cfg['static_times']:
            data=frame[(frame.kind==kind)&(frame.t==t)]
            scalar=data[data.family=='regularized'];rho=float(np.clip(-scalar.linear.sum()/scalar.quadratic.sum(),0.,1.))
            ridge=data[data.family=='ridge'].groupby('loading').loss.mean();loading=float(ridge.idxmin())
            selected.append(dict(kind=kind,t=t,rho=rho,loading=loading,
                loading_boundary=loading in (min(cfg['ridge_loadings']),max(cfg['ridge_loadings']))))
    return selected


def static_test(task):
    cfg,kind,trial,selection=task;data=static_data(cfg,trial,kind)
    path,audit=static_paths(data['U'],data['V'],cfg['sigma_static']**2);rows=[];fisher=[]
    for t in cfg['static_times']:
        state=path[t-1];A=state['A'];P,_,_=normal_mode(A);M,diag=mean_projector(A);iso,rho=isotropized_projector_mean(M,P)
        chosen=next(x for x in selection if x['kind']==kind and x['t']==t)
        values={'mean':M,'mode':P,'isotropic':iso,'regularized':regularized_projector(P,chosen['rho']),
            'ridge':ridge_map(state,chosen['loading']),'residual_pca':residual_projector(state)}
        try:
            normal_g,cov_g,basis_g,_=laplace_state(A)
            local=TangentGaussian(cfg['sigma_static']**2,0.)
            local.n,local.cov,local.E=normal_g,cov_g,basis_g
            gaussian_mean,gaussian_diag=local.mean(checked=True)
            values['laplace']=gaussian_mean
        except ArithmeticError as error:
            gaussian_diag=dict(numerical_ok=False,local_ok=False,cut_locus_mass=np.nan,reason=str(error))
            values['laplace']=np.full((3,3),np.nan)
        # Independent expected LS modal parameter audit, not another algorithm.
        normal=data['normal'];E=np.linalg.svd(normal[None,:],full_matrices=True)[2][1:].T
        F=fisher_information(E,data['V'][:t].T,cfg['sigma_static']**2);ev=np.linalg.eigvalsh(F)
        fisher.append(dict(kind=kind,trial=trial,t=t,n=3,fisher_min=float(ev[0]),fisher_max=float(ev[-1]),
            input_energy=float(np.sum(data['V'][:t]**2)),input_rank=int(np.linalg.matrix_rank(data['V'][:t]))))
        for method,D in values.items():
            if not np.isfinite(D).all():
                rows.append(dict(method=method,kind=kind,trial=trial,t=t,n=3,numerical_ok=False,
                    local_ok=False,response_mse=np.nan,status=gaussian_diag.get('reason','Unresolved Gaussian mean')))
                continue
            rows.append(_record(D,data['P'],A,method,cfg,kind=kind,trial=trial,t=t,n=3,
                numerical_ok=bool(gaussian_diag['numerical_ok'] if method=='laplace' else diag['numerical_ok']),
                local_ok=bool(gaussian_diag['local_ok']) if method=='laplace' else True,
                cut_locus_mass=gaussian_diag['cut_locus_mass'] if method=='laplace' else np.nan,
                moment_seconds=diag['seconds'] if method in ('mean','isotropic') else 0.,
                update_seconds=state['update_seconds'],rho=rho if method=='isotropic' else chosen['rho'] if method=='regularized' else np.nan,
                anisotropy=float(np.linalg.norm(M-iso,'fro')),modal_audit_error=audit,posterior_mean=M))
    return dict(metrics=rows,fisher=fisher)


def _rotation(axis,angle):
    a=np.asarray(axis,float);a=a/np.linalg.norm(a)
    K=np.array([[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def dynamic_data(cfg,condition,trial,split='test'):
    seed=cfg['seed_validation'] if split=='validation' else cfg['seed_test']
    plant=generate_contact(seed+940,trial,'plane' if condition=='plane' else 'sphere_small',
        T=cfg['dynamic_T'],dt=cfg['dynamic_dt'],v0=cfg['v0'],sigma=cfg['sigma_dynamic'])
    times=plant['times'];angles=.7*np.sin(2*np.pi*np.arange(len(times))/max(1,len(times)-1))
    R=np.array([_rotation([1.,1.,.4],value) for value in angles]);G=np.empty_like(R);G[0]=np.eye(3)
    for j in range(1,len(R)):G[j]=R[j].T@R[j-1]
    data=dict(U=np.einsum('tji,tj->ti',R,plant['U']),V=np.einsum('tji,tj->ti',R,plant['V']),
        P=np.einsum('tji,tjk,tkl->til',R,plant['tangent_projectors'],R),
        normal=np.einsum('tji,tj->ti',R,plant['normals']),G=G,R=R,times=times,
        positions=plant['positions'],world_normal=plant['normals'],world_U=plant['U'],world_V=plant['V'],
        physical_diagnostics=plant['diagnostics'])
    return data


@dataclass
class TangentGaussian:
    """Local Gaussian EKF with explicitly chosen per-step tangent variance Q.

    This baseline does not borrow an invented nonzero-mean passive likelihood.
    The conditional contact Jacobian is exact; six actual data pairs initialize
    its local chart. It is not an invariant-density Bingham approximation.
    """
    s: float
    variance: float
    warmup: int=6

    def __post_init__(self):
        if self.s<=0 or self.variance<0 or not np.isfinite(self.s+self.variance):
            raise ValueError('Positive sensor variance and nonnegative Gaussian variance required')
        self.A=np.zeros((3,3));self.n=None;self.E=None;self.cov=None;self.t=0;self.nonlocal_updates=0

    def update(self,u,v,G):
        if self.n is None:
            self.A=center(G@self.A@G.T+increment(u,self.s,v=v));self.t+=1
            if self.t>=self.warmup:
                try:self.n,self.cov,self.E,_=laplace_state(self.A)
                except ArithmeticError:pass
            return
        self.n=G@self.n;self.E=G@self.E;self.cov=self.cov+self.variance*np.eye(2)
        J=measurement_jacobian(self.n,v,self.E);S=J@self.cov@J.T+self.s*np.eye(3)
        K=np.linalg.solve(S,J@self.cov).T;delta=K@(u-measurement(self.n,v));H=np.eye(2)-K@J
        covariance=H@self.cov@H.T+self.s*K@K.T;tangent=self.E@delta
        rotation=parallel_rotation(self.n,tangent);self.n=exponential_normal(self.n,tangent);self.n/=np.linalg.norm(self.n)
        E=rotation@self.E;E-=np.outer(self.n,self.n@E);Q=np.linalg.qr(E)[0];T=Q.T@E
        self.E=Q;self.cov=T@covariance@T.T;self.cov=(self.cov+self.cov.T)/2;self.t+=1
        self.nonlocal_updates+=int(np.linalg.norm(delta)>=np.pi/2)
        if np.linalg.eigvalsh(self.cov)[0]<-1e-10:raise ArithmeticError('Gaussian covariance lost PSD')

    def mean(self,checked=False):
        if self.n is None:
            M,d=mean_projector(self.A)
            return M,dict(numerical_ok=d['numerical_ok'],local_ok=False,warmup=True,cut_locus_mass=np.nan,resolution_error=0.)
        order=64 if checked else 32;high=_MappedGaussian(self.n,self.cov,self.E,order);M=high.mean()
        error=float(np.linalg.norm(M-_MappedGaussian(self.n,self.cov,self.E,32).mean(),2)) if checked else np.nan
        while checked and error>1e-7 and order<256:
            previous=M;order*=2;high=_MappedGaussian(self.n,self.cov,self.E,order);M=high.mean()
            error=float(np.linalg.norm(M-previous,2))
        mass=float(high.cut_locus_mass())
        return M,dict(numerical_ok=bool(np.isfinite(M).all() and (not checked or error<1e-7)),
            local_ok=bool(mass<=.01),warmup=False,cut_locus_mass=mass,resolution_error=error,angular_order=order)


def dynamic_paths(cfg,data,beta=1.,loading=0.,gaussian_Q=0.,report_times=None,transport=True,
                  time_average_methods=()):
    s=cfg['sigma_dynamic']**2;n=3;A=np.zeros((n,n));S=np.zeros_like(A);C=np.zeros_like(A);Y=np.zeros_like(A);R=np.zeros_like(A)
    output=[];arrays=[];sums={name:0. for name in time_average_methods}
    report_times=set(cfg['dynamic_times'] if report_times is None else report_times)
    max_audit=0.;update_time=0.;eigen_time=0.;moment_time=0.
    for j,(u,v) in enumerate(zip(data['U'],data['V'])):
        G=data['G'][j] if transport else np.eye(3)
        begin=perf_counter();A=center(beta*G@A@G.T+increment(u,s,v=v));update_time+=perf_counter()-begin
        # Independent algebraic reference and estimators share measured G only.
        S=beta*G@S@G.T+np.outer(u,v)+np.outer(v,u)-np.outer(v,v)
        max_audit=max(max_audit,float(np.max(abs(A-center(S/(2*s))))))
        C=beta*G@C@G.T+np.outer(v,v);Y=beta*G@Y@G.T+np.outer(u,v)
        residual=v-u;R=beta*G@R@G.T+np.outer(residual,residual)
        begin=perf_counter();P,normal,gap=normal_mode(A);eigen_time+=perf_counter()-begin
        arrays.append(dict(A=A.copy(),mode=normal.copy(),G=G.copy()))
        state=dict(C=C,Y=Y,residual=R)
        need_mean=j+1 in report_times or bool(set(time_average_methods)&{'mean','isotropic'})
        if need_mean:
            M,diag=mean_projector(A,fast=True);moment_time+=diag['seconds'];iso,rho=isotropized_projector_mean(M,P)
        values={'mode':P,'ridge':ridge_map(state,loading),'residual_pca':residual_projector(state)}
        if need_mean:values.update(mean=M,isotropic=iso)
        for name in time_average_methods:sums[name]+=response_mse(values[name],data['P'][j],cfg['v0'])
        if j+1 not in report_times:continue
        for method,D in values.items():
            rows=_record(D,data['P'][j],A,method,cfg,t=j+1,n=3,beta=beta,loading=loading,
                gaussian_Q=np.nan,transport=transport,numerical_ok=bool(diag['numerical_ok']),
                local_ok=True,cut_locus_mass=np.nan,update_seconds=update_time,
                eigen_seconds=eigen_time if method in ('mean','mode','isotropic') else 0.,
                moment_seconds=moment_time if method in ('mean','isotropic') else 0.,
                time_average_mse=sums[method]/(j+1) if method in sums else np.nan,
                rho=rho if method=='isotropic' else np.nan,modal_audit_error=max_audit,posterior_mean=M)
            output.append(rows)
    return output,arrays


def dynamic_validation(task):
    cfg,condition,trial=task;data=dynamic_data(cfg,condition,trial,'validation');rows=[]
    # Tuning uses the final endpoint of independent validation trajectories.
    for beta in cfg['betas']:
        path,_=dynamic_paths(cfg,data,beta=beta,loading=0.,gaussian_Q=0.,report_times=[len(data['U'])])
        for row in path:
            if row['method'] in ('mean','mode','residual_pca'):
                rows.append(dict(condition=condition,trial=trial,family=row['method'],beta=beta,loading=np.nan,Q=np.nan,loss=row['response_mse']))
        # Ridge tuning recomputes only sufficient statistics, no extra data.
        C=np.zeros((3,3));Y=C.copy()
        for j,(u,v) in enumerate(zip(data['U'],data['V'])):
            G=data['G'][j];C=beta*G@C@G.T+np.outer(v,v);Y=beta*G@Y@G.T+np.outer(u,v)
        for loading in cfg['ridge_loadings']:
            loss=response_mse(ridge_map(dict(C=C,Y=Y),loading),data['P'][-1],cfg['v0'])
            rows.append(dict(condition=condition,trial=trial,family='ridge',beta=beta,loading=loading,Q=np.nan,loss=loss))
    for Q in cfg['gaussian_variances']:
        f=TangentGaussian(cfg['sigma_dynamic']**2,Q,cfg['gaussian_warmup'])
        for j,(u,v) in enumerate(zip(data['U'],data['V'])):f.update(u,v,data['G'][j])
        M,d=f.mean(checked=True)
        rows.append(dict(condition=condition,trial=trial,family='gaussian',beta=np.nan,loading=np.nan,Q=Q,
            loss=response_mse(M,data['P'][-1],cfg['v0']),numerical_ok=d['numerical_ok'],local_ok=d['local_ok']))
    return rows


def select_dynamic(rows,cfg):
    import pandas as pd
    frame=pd.DataFrame(rows);selected=[]
    for condition in cfg['dynamic_conditions']:
        for family in ('mean','mode','residual_pca','ridge','gaussian'):
            f=frame[(frame.condition==condition)&(frame.family==family)]
            columns=['Q'] if family=='gaussian' else ['beta','loading'] if family=='ridge' else ['beta']
            summary=f.groupby(columns,dropna=False).loss.mean();best=summary.idxmin()
            best=(best,) if len(columns)==1 else best
            selected.append(dict(condition=condition,family=family,**{key:float(value) for key,value in zip(columns,best)},
                validation_loss=float(summary.min())))
    return selected


def dynamic_test(task):
    cfg,condition,trial,selection=task;data=dynamic_data(cfg,condition,trial);rows=[];story=None
    chosen={r['family']:r for r in selection if r['condition']==condition}
    # Individual validation-selected beta for each estimator; isotropization
    # deliberately uses the same A as the mean to isolate posterior shape.
    for family in ('mean','mode','residual_pca','ridge'):
        spec=chosen[family];path,arrays=dynamic_paths(cfg,data,beta=spec['beta'],loading=spec.get('loading',0.),
            gaussian_Q=chosen['gaussian']['Q'],time_average_methods=('mean','isotropic') if family=='mean' else (family,))
        keep=(family,'isotropic','mode') if family=='mean' else (family,)
        for row in path:
            if row['method'] in keep:
                if family=='mean' and row['method']=='mode':row=dict(row,method='mode_same_A')
                rows.append(dict(condition=condition,trial=trial,**row))
        if family=='mean':
            if trial==0:story=dict(data=data,arrays=arrays,beta=spec['beta'])
            identity,_=dynamic_paths(cfg,data,beta=spec['beta'],loading=spec.get('loading',0.),
                gaussian_Q=chosen['gaussian']['Q'],transport=False,time_average_methods=('mean','mode'))
            for row in identity:
                if row['method'] in ('mean','mode'):rows.append(dict(condition=condition,trial=trial,**dict(row,method=row['method']+'_identity')))
    # Gaussian selected independently; shared known measured motion, same data.
    f=TangentGaussian(cfg['sigma_dynamic']**2,chosen['gaussian']['Q'],cfg['gaussian_warmup'])
    average=0.;gaussian_seconds=0.
    for j,(u,v) in enumerate(zip(data['U'],data['V'])):
        begin=perf_counter()
        f.update(u,v,data['G'][j])
        gm,gd=f.mean(checked=j+1 in cfg['dynamic_times']);average+=response_mse(gm,data['P'][j],cfg['v0'])
        gaussian_seconds+=perf_counter()-begin
        if j+1 in cfg['dynamic_times']:
            M,d=gm,gd;A=story['arrays'][j]['A'] if story is not None else np.eye(3)*0
            row=_record(M,data['P'][j],A,'gaussian',cfg,condition=condition,trial=trial,t=j+1,n=3,
                gaussian_Q=chosen['gaussian']['Q'],numerical_ok=d['numerical_ok'],local_ok=d['local_ok'],
                cut_locus_mass=d['cut_locus_mass'],transport=True,update_seconds=gaussian_seconds,
                time_average_mse=average/(j+1),
                estimator_projector=projector_from_normal(f.n) if f.n is not None else normal_mode(f.A)[0],
                nonlocal_updates=f.nonlocal_updates,posterior_mean=M)
            row['mode_gap']=np.nan  # A Gaussian has no Bingham parameter gap.
            rows.append(row)
    return dict(metrics=rows,story=story)


def higher_test(task):
    cfg,n,trial=task;T=cfg['higher_samples_per_dimension']*n
    data=static_data(cfg,trial,'balanced',n=n,T=T);path,audit=static_paths(data['U'],data['V'],cfg['sigma_static']**2)
    A=path[-1]['A'];P,_,_=normal_mode(A);M,d=mean_projector(A);iso,rho=isotropized_projector_mean(M,P)
    values={'mean':M,'mode':P,'isotropic':iso,'residual_pca':residual_projector(path[-1])};rows=[]
    for method,D in values.items():
        rows.append(_record(D,data['P'],A,method,cfg,n=n,trial=trial,t=T,numerical_ok=d['numerical_ok'],
            moment_seconds=d['seconds'] if method in ('mean','isotropic') else 0.,order=d['order'],
            moment_error_bound=d['error_bound'],rho=rho if method=='isotropic' else np.nan,
            modal_audit_error=audit,posterior_mean=M,scope='Generalized single linear maintained constraint'))
    return rows


def certificate_test(task):
    cfg,trial=task;n=3;T=max(cfg['certificate_times']);s=cfg['sigma_static']**2
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed_test'],trial,49071]))
    normal=unit(rng,n);X=np.linalg.svd(normal[None,:],full_matrices=True)[2][1:].T
    V=command_program(3,T,'balanced',cfg['v0']);P=projector_from_normal(normal)
    U=V@P+np.sqrt(s)*rng.normal(size=(T,n));S=np.zeros((n,n));rows=[]
    for j,(u,v) in enumerate(zip(U,V)):
        S+=np.outer(u,v)+np.outer(v,u)-np.outer(v,v)
        if j+1 not in cfg['certificate_times']:continue
        A=center(S/(2*s));_,estimate,_=normal_mode(A)
        # Complement dual: r=v-u=(nn^T)v-e has the same sensor variance.
        # Its modal normal equals the complementary tangent-plane mode. Using
        # k=1 here avoids the looser sqrt(N-1) prefactor for a hyperplane.
        b=paper_certificate(np.broadcast_to(normal[None,:,None],(j+1,3,1)),V[:j+1],s,beta=1.,delta=cfg['certificate_delta'])
        actual=float(np.sqrt(max(0.,1-(normal@estimate)**2)))
        rows.append(dict(trial=trial,t=j+1,n=3,**b,actual_error=actual,
            violated=bool(b['valid'] and actual>b['bound']+1e-10),fixed_time=True))
    return rows


def paired_metrics(frame,group_columns,endpoint='response_mse'):
    rows=[]
    for keys,g in frame.groupby(group_columns,dropna=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        mean=g[g.method=='mean'].set_index('trial')
        for method,h in g.groupby('method'):
            if method=='mean':continue
            other=h.set_index('trial');ids=mean.index.intersection(other.index)
            delta=other.loc[ids,endpoint]-mean.loc[ids,endpoint];finite=np.isfinite(delta)
            values=delta[finite].to_numpy();N=len(values)
            gain=float(values.mean()) if N else np.nan
            radius=float(student_t.ppf(.975,N-1)*values.std(ddof=1)/np.sqrt(N)) if N>1 else np.nan
            rows.append(dict(zip(group_columns,keys),reference=method,endpoint=endpoint,trials=len(ids),finite_pairs=N,
                gain=gain,lower=gain-radius,upper=gain+radius,
                positive='reference minus posterior mean; independent trajectories',
                numerical_flags_retained=True))
    return rows


def posterior_story(A,queries,truth,rng):
    """Precompute every density/region/quantile used by the plotter.

    Resampled quadrature nodes are visual points only; credible thresholds and
    response quantiles use the independent continuous integration routines.
    """
    from .contact_posterior import response_quantiles
    report=contact_summary(A,queries,true_normal=truth)
    lower50,upper50,diagnostic50=response_quantiles(A,queries,mass=.5)
    quad=normal_quadrature(-A,64,192)
    index=rng.choice(len(quad['weights']),size=1024,p=quad['weights'])
    samples=quad['normal_nodes'][index]*rng.choice([-1.,1.],size=(1024,1))
    M,diagnostic=mean_projector(A,fast=len(A)==3)
    difference=float(np.linalg.norm(M-report['mean'],2))
    values,E=np.linalg.eigh(-A);order=np.argsort(values)[::-1];values=values[order];E=E[:,order]
    return dict(A=A,normal_parameter=-A,mean=report['mean'],normal_mean=np.eye(3)-report['mean'],
        normal_mode=report['mode_normal'],normal_logz=quad['log_normalizer'],
        ball_radius=report['ball_radius'],hpd_threshold=report['hpd_threshold'],
        lower95=report['response_lower'],upper95=report['response_upper'],lower50=lower50,upper50=upper50,
        response_truth=queries@projector_from_normal(truth),response_mean=queries@report['mean'],
        normal_samples=samples,posterior_axes=E[:,1:],posterior_curvature=2*(values[0]-values[1:]),
        numerical_ok=bool(report['numerics']['numerical_ok'] and diagnostic50['response_quantiles_ok']
                          and diagnostic['numerical_ok'] and difference<1e-7),
        mean_reference_error=difference,state_ball_covered=report['ball_covered'],state_hpd_covered=report['hpd_covered'])


def build_stories(cfg,selection):
    """Exactly preassigned normal/trajectories and times, independent of wins."""
    queries=cfg['v0']*np.eye(3);arrays=dict(static_times=np.array(cfg['static_times']),
        static_kinds=np.array(cfg['static_inputs']),static_sigmas=np.array(cfg['sigma_story']),
        static_true_normal=np.asarray(cfg['story_normal']),queries=queries)
    reports=[]
    for ik,kind in enumerate(cfg['static_inputs']):
        V=command_program(3,max(cfg['static_times']),kind,cfg['v0']);h=np.asarray(cfg['story_normal'])
        rng_noise=np.random.default_rng(cfg['seed_story']);standard=rng_noise.normal(size=V.shape)
        for iz,sigma in enumerate(cfg['sigma_story']):
            U=V@projector_from_normal(h)+sigma*standard
            path,_=static_paths(U,V,sigma*sigma);prefix=f'static_{ik}_{iz}_'
            arrays[prefix+'U']=U;arrays[prefix+'V']=V
            values=[]
            for t in cfg['static_times']:
                result=posterior_story(path[t-1]['A'],queries,h,np.random.default_rng(cfg['seed_story']+100*ik+10*iz+t))
                E=np.linalg.svd(h[None,:],full_matrices=True)[2][1:].T
                F=fisher_information(E,V[:t].T,sigma*sigma);f,axes=np.linalg.eigh(F)
                result.update(fisher_truth_eigenvalues=f,fisher_truth_axes=E@axes)
                values.append(result)
                reports.append(dict(kind=kind,sigma=sigma,t=t,source='preassigned stationary density',
                    numerical_ok=result['numerical_ok'],mean_reference_error=result['mean_reference_error']))
            for name in values[0]:arrays[prefix+name]=np.asarray([r[name] for r in values])
    arrays['dynamic_conditions']=np.array(cfg['dynamic_conditions']);arrays['dynamic_report_times']=np.asarray(cfg['dynamic_times'])
    for ic,condition in enumerate(cfg['dynamic_conditions']):
        data=dynamic_data(cfg,condition,0);beta=next(r['beta'] for r in selection if r['condition']==condition and r['family']=='mean')
        _,path=dynamic_paths(cfg,data,beta=beta);prefix=f'dynamic_{ic}_'
        for name in ('U','V','P','normal','G','R','times','positions','world_normal','world_U','world_V'):
            arrays[prefix+name]=data[name]
        arrays[prefix+'A']=np.asarray([p['A'] for p in path]);arrays[prefix+'mode_normal']=np.asarray([p['mode'] for p in path])
        arrays[prefix+'beta']=np.asarray(beta)
        values=[]
        for t in cfg['dynamic_times']:
            result=posterior_story(path[t-1]['A'],queries,data['normal'][t-1],
                np.random.default_rng(cfg['seed_story']+10000+100*ic+t));values.append(result)
            reports.append(dict(kind=condition,sigma=cfg['sigma_dynamic'],t=t,source='preassigned physical trajectory zero',
                numerical_ok=result['numerical_ok'],mean_reference_error=result['mean_reference_error']))
        for name in values[0]:arrays[prefix+'report_'+name]=np.asarray([r[name] for r in values])
    return arrays,reports


def dynamic_certificates(cfg,condition,trial,selection):
    data=dynamic_data(cfg,condition,trial);beta=next(r['beta'] for r in selection if r['condition']==condition and r['family']=='mean')
    rows=[]
    for transport in (True,False):
        G=data['G'] if transport else np.broadcast_to(np.eye(3),data['G'].shape)
        A=np.zeros((3,3))
        for j,(u,v) in enumerate(zip(data['U'],data['V'])):
            A=center(beta*G[j]@A@G[j].T+increment(u,cfg['sigma_dynamic']**2,v=v))
            if j+1 not in cfg['dynamic_times']:continue
            b=paper_certificate(data['normal'][:j+1,:,None],data['V'][:j+1],cfg['sigma_dynamic']**2,
                beta=beta,G=G[:j+1],delta=cfg['certificate_delta'])
            _,estimate,_=normal_mode(A);actual=float(np.sqrt(max(0.,1-(data['normal'][j]@estimate)**2)))
            rows.append(dict(condition=condition,trial=trial,t=j+1,beta=beta,transport=transport,
                **b,actual_error=actual,violated=bool(b['valid'] and actual>b['bound']+1e-10),
                evaluation_only=True,normal_residual_equivalence=True))
    return rows


def paper_certificate(X_history,V,s,beta=1.,G=None,delta=.05):
    """Printed draft-5 Eq.32 sum-of-norms budget and Eq.33 bound.

    The existing helper also computes a sharper aggregate-operator variant.
    We retain that only as a separately named diagnostic. Main fields below
    are exactly the printed paper expressions, without clipping invalid bounds.
    """
    original=tracking_bound(X_history,V,s,beta=beta,G=G,delta=delta)
    X=np.asarray(X_history);V=np.asarray(V);T,n,k=X.shape
    target=X[-1]@X[-1].T;drift=0.
    for j in range(T):
        transported=original.transports[j]@X[j]
        difference=transported@transported.T-target
        drift+=beta**(T-1-j)*np.linalg.norm(difference,2)*np.dot(V[j],V[j])/s
    epsilon=original.initialization+drift+original.noise
    valid=bool(original.gamma>0 and epsilon<original.gamma/2)
    bound=float(2*np.sqrt(k)*epsilon/original.gamma) if valid else np.nan
    return dict(gamma=original.gamma,noise=original.noise,drift=float(drift),epsilon=float(epsilon),
        valid=valid,bound=bound,aggregate_operator_drift=original.drift,
        aggregate_operator_bound=original.bound,certificate_formula='Draft5 Eq32 and Eq33')
