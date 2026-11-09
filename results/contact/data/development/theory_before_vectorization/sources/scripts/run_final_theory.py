"""Verify, pilot, and run final-guide mathematical and local-accuracy studies."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import hashlib,json,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.linalg import expm,null_space
from scipy.stats import beta as beta_distribution,t as student_t
from lib.final_theory import (
    configuration,exact_aggregate,sequential_parameter,align_block,
    sharp_marginal_certificate,sharp_conditional_certificate,natural_curvature,
    tangent_coordinate,global_kl_identity,independent_nodes,native_response_covariance)
from lib.guide_tracking import certificate_observation
from lib.guide_information import conditional_fisher,finite_hessian,chart_basis
from lib.beamforming_theory import passive_fisher_information,passive_scores,passive_tangent_basis
from lib.models import increment,constant
from lib.geometry import center
from lib.block_em import moments
from lib.beamforming_numerics import sample_complex_bingham

SOURCES=['lib/final_theory.py','scripts/run_final_theory.py','tests/test_final_theory.py',
    'lib/guide_tracking.py','lib/guide_information.py','lib/beamforming_theory.py',
    'lib/models.py','lib/geometry.py','lib/block_em.py','lib/spectral_series.py',
    'lib/contact_posterior.py','lib/beamforming_numerics.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(path,value):path.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n')
def gaussian(rng,shape,cx=False):
    z=rng.normal(size=shape)
    return (z+1j*rng.normal(size=shape))/np.sqrt(2) if cx else z
def interval(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return np.nan,np.nan,np.nan
    mean=float(values.mean())
    half=float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else np.nan
    return mean,mean-half,mean+half
def binomial(successes,n):
    if not n:return np.nan,np.nan
    return (0. if not successes else float(beta_distribution.ppf(.025,successes,n-successes+1)),
            1. if successes==n else float(beta_distribution.ppf(.975,successes+1,n-successes)))
def mode_parameter(A,k):
    e,E=np.linalg.eigh(A);P=E[:,-k:]@E[:,-k:].conj().T
    return P,bool(e[-k]-e[-k-1]>1e-10*max(1.,np.ptp(e)))


def algebra(cfg,example):
    rows=[];conditional=example=='contact';k=2 if conditional else 1
    for seed in cfg['algebra_seeds']:
        for cx in (False,True):
            rng=np.random.default_rng(seed)
            for L in cfg['block_lengths']:
                for beta in cfg['block_beta']:
                    for moving in (False,True):
                        U=gaussian(rng,(L,3),cx);V=gaussian(rng,(L,3),cx) if conditional else None
                        A0=np.diag([1.5,-.2,-1.3])+2*np.eye(3)
                        ss=np.linspace(.3,.8,L);qq=np.linspace(.7,1.3,L)
                        G=None
                        if moving:
                            G=[]
                            for j in range(L):
                                Z=gaussian(rng,(3,3),cx)
                                G.append(expm(.2*(Z-Z.conj().T)))
                            G=np.array(G)
                        result=exact_aggregate(A0,U,ss,qq,V,beta,G,cx,True)
                        A=result['A'];seq=sequential_parameter(A0,U,ss,qq,V,beta,G,cx)
                        M,z,diag=moments(A,k,cx);M0,z0,d0=moments(seq,k,cx)
                        P,unique=mode_parameter(A,k);P0,_=mode_parameter(seq,k)
                        scale=max(1.,float(np.linalg.norm(A,'fro')))
                        aligned=result['aligned'];raw=beta**L*aligned['A0']+result['increment']
                        _,zraw,_=moments(raw,k,cx)
                        gauge_error=abs(zraw-z-k*np.trace(raw).real/3)
                        v=np.array([.2,.7,-.4])
                        errors=dict(parameter_error=float(np.linalg.norm(A-seq,'fro')/scale),
                            mean_error=float(np.linalg.norm(M-M0,'fro')),
                            normalizer_error=abs(z-z0),trace_restoration_error=gauge_error,
                            mode_error=float(np.linalg.norm(P-P0,'fro')) if unique else np.nan,
                            predictive_mean_error=float(np.linalg.norm((M-M0)@v)),
                            predictive_covariance_error=float(np.linalg.norm((ss[-1]*np.eye(3)+qq[-1]*M)
                                -(ss[-1]*np.eye(3)+qq[-1]*M0),'fro')) if not conditional else np.nan)
                        rows.append(dict(seed=seed,example=example,n=3,k=k,L=L,beta=beta,
                            complex_data=cx,known_frame_transport=moving,boundary_factor=beta**L,
                            mode_unique=unique,numerical_ok=bool(diag['numerical_ok'] and d0['numerical_ok']),
                            **errors,tolerance=cfg['tolerances']['algebra_relative'],
                            status='passed' if max(v for v in errors.values() if np.isfinite(v))<1e-9 else 'failed'))
    # A fixed nonuniform prior makes even the mode change under accidental /L.
    U=np.tile([0.,2.,0.],(4,1));A0=np.diag([1.5,0.,-1.5])
    A=exact_aggregate(A0,U,1.,1.);bad=center(A0+sum(increment(u,1.,1.) for u in U)/len(U))
    P,_=mode_parameter(A,1);Q,_=mode_parameter(bad,1)
    M,_,_=moments(A,1);K,_,_=moments(bad,1)
    rows.append(dict(seed=0,example=example,n=3,k=1,L=4,beta=1.,
        complex_data=False,known_frame_transport=False,boundary_factor=1.,
        fixture='nonuniform_prior_covariance_division_by_L',
        normalization_parameter_difference=float(np.linalg.norm(A-bad,'fro')),
        normalization_mode_difference=float(np.linalg.norm(P-Q,'fro')),
        normalization_mean_difference=float(np.linalg.norm(M-K,'fro')),
        numerical_ok=True,status='passed' if np.linalg.norm(M-K)>.2 else 'failed'))
    # A rank-two posterior after one marginal observation can have a mode tie.
    _,unique=mode_parameter(increment(np.array([1.,0.,0.,0.,0.]),.2,1.),2)
    rows.append(dict(seed=0,example=example,n=5,k=2,L=1,beta=1.,
        fixture='uniform_rank_two_one_observation',mode_unique=unique,
        numerical_ok=True,status='passed' if not unique else 'failed',
        scope='No native general-rank moment or inverse curvature is requested for this tie'))
    return pd.DataFrame(rows)


def information(cfg,example):
    from lib.implementation_validation import verify_component
    old=verify_component(ROOT,'paper','information')
    oldcounts=verify_component(ROOT,'paper','information_counts')
    paths=[ROOT/'results/implementation_audit/paper/information/case_checks.csv',
           ROOT/'results/implementation_audit/paper/information/finite_differences.csv',
           ROOT/'results/implementation_audit/paper/information_counts/summary.csv']
    channel='conditional' if example=='contact' else 'marginal'
    frame=pd.read_csv(paths[0]);frame=frame[frame.channel.eq(channel)].copy()
    frame=frame.rename(columns={'s':'sigma2'});frame['source_scope']='Reused source/content-verified stationary information check'
    fd=pd.read_csv(paths[1]);fd=fd[fd.channel.eq(channel)].rename(columns={'s':'sigma2'})
    counts=pd.read_csv(paths[2]).rename(columns={'s':'sigma2'})
    count=8192 if cfg['stage']=='paper' else 512
    lossrows=[];klrows=[]
    for cx in (False,True):
        rng=np.random.default_rng(cfg['seed']+int(cx)+100)
        X=np.eye(3,dtype=complex if cx else float)[:,:1]
        for sigma2 in [.2,1.]:
            q=1.3;L=8
            V=np.sqrt(q)*gaussian(rng,(count,L,3),cx)
            E=np.sqrt(sigma2)*gaussian(rng,(count,L,3),cx)
            P=X@X.conj().T;U=np.einsum('ij,btj->bti',P,V)+E
            derivatives=passive_tangent_basis(X,cx)
            J=np.einsum('dij,btj->btdi',derivatives,V)
            scores=(2. if cx else 1.)/sigma2*np.einsum('bti,btdi->bd',E.conj(),J).real
            ms=passive_scores(U.reshape(-1,3),X,sigma2,q,cx).reshape(count,L,-1).sum(axis=1)
            empirical=np.mean(scores*scores,axis=0)
            marginal=np.mean(ms*ms,axis=0)
            reference=(4. if cx else 2.)*L*q/sigma2
            mref=(2. if cx else 1.)*L*q*q/(sigma2*(sigma2+q))
            for i in range(scores.shape[1]):
                lossrows.append(dict(complex_data=cx,n=3,k=1,L=L,sigma2=sigma2,tau2=q,
                    coordinate=i,episodes=count,expected_conditional=reference,expected_marginal=mref,
                    empirical_conditional=float(empirical[i]),empirical_marginal=float(marginal[i]),
                    conditional_mcse=float((scores[:,i]**2).std(ddof=1)/np.sqrt(count)),
                    marginal_mcse=float((ms[:,i]**2).std(ddof=1)/np.sqrt(count)),
                    exact_loss_factor=q/(2*(sigma2+q)),
                    identity_error=abs(mref-reference*q/(2*(sigma2+q))),
                    input_law='Recorded/unobserved v ~ N(0,tau2 I), or proper CN, identical law'))
            Y=np.linalg.qr(gaussian(rng,(3,1),cx))[0]
            for recorded in (False,True):
                direct,formula=global_kl_identity(X,Y,sigma2,q,
                    V[0] if recorded else None,cx)
                klrows.append(dict(n=3,k=1,complex_data=cx,channel='conditional' if recorded else 'marginal',
                    sigma2=sigma2,tau2=q,observations=L if recorded else 1,
                    direct_gaussian_kl=direct,global_formula_kl=formula,error=abs(direct-formula),
                    tolerance=1e-10,status='passed' if abs(direct-formula)<1e-10 else 'failed'))
    reused={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for folder,meta in [('information',old),('information_counts',oldcounts)]:
        p=ROOT/'results/implementation_audit/paper'/folder/'manifest.json';reused[str(p.relative_to(ROOT))]=sha(p)
    return dict(information=frame,information_derivatives=fd,information_counts=counts,
                information_loss=pd.DataFrame(lossrows),global_kl=pd.DataFrame(klrows)),reused


def inverse_fisher(cfg,example):
    cx=example=='beamforming';truth=np.array([0.,0.,1.],complex if cx else float)
    Pn=np.outer(truth,truth.conj());P=Pn if cx else np.eye(3)-Pn
    rows=[];summaries=[]
    for sigma2 in cfg['local_sigma2']:
        for L in cfg['local_lengths']:
            V=np.eye(3)[np.arange(L)%3]
            for beta in cfg['local_beta']:
                w=beta**np.arange(L-1,-1,-1);m,b=w.sum(),w@w
                if cx:
                    one=passive_fisher_information(3,1,sigma2,1.,complex_data=True)
                    J=m*one;K=b*one
                else:
                    # Normal residual v-u has the dual rank-one conditional law.
                    J=conditional_fisher(truth[:,None],V*np.sqrt(w)[:,None],sigma2)
                    K=conditional_fisher(truth[:,None],V*w[:,None],sigma2)
                inv=np.linalg.pinv(J);sandwich=inv@K@inv
                coordinates=[];squared=[];locals_=[];curvature=[]
                for trial in range(cfg['local_episodes']):
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],305,int(cx),trial,L,int(sigma2*100),int(beta*100)]))
                    if cx:
                        U=gaussian(rng,(L,1),True)*truth+np.sqrt(sigma2)*gaussian(rng,(L,3),True)
                        A=exact_aggregate(np.zeros((3,3)),U,sigma2,1.,beta=beta,complex_data=True)
                    else:
                        U=V@P.T+np.sqrt(sigma2)*gaussian(rng,(L,3))
                        A=-exact_aggregate(np.zeros((3,3)),U,sigma2,V=V,beta=beta)
                    e,E=np.linalg.eigh(A);h=E[:,-1]
                    z,local=tangent_coordinate(h,truth,cx,cfg['local_radius'])
                    unique=bool(e[-1]-e[-2]>1e-10)
                    H=natural_curvature(A,1,cx)
                    invH=float(np.sum(1/H['values'])) if H['isolated'] else np.nan
                    coordinates.append(z);squared.append(float(z@z));locals_.append(local);curvature.append(invH)
                    row=dict(example=example,n=3,application_k=1 if cx else 2,sigma2=sigma2,tau2=1.,
                        L=L,beta=beta,trial=trial,local=local,mode_unique=unique,
                        tangent_squared_error=float(z@z),posterior_inverse_curvature_trace=invH,
                        inverse_fisher_trace=float(np.trace(inv)),weighted_sampling_reference_trace=float(np.trace(sandwich)),
                        m=float(m),b=float(b),local_radius=cfg['local_radius'])
                    row.update({f'z{i}':float(v) for i,v in enumerate(z)});rows.append(row)
                Z=np.array(coordinates);mse,lo,hi=interval(squared)
                bias=Z.mean(axis=0);cov=np.cov(Z,rowvar=False,ddof=1)
                local_fraction=float(np.mean(locals_))
                local_lower,local_upper=binomial(int(np.sum(locals_)),len(locals_))
                summaries.append(dict(example=example,n=3,sigma2=sigma2,L=L,beta=beta,
                    episodes=len(Z),tangent_bias_norm=float(np.linalg.norm(bias)),
                    tangent_bias_se=float(np.sqrt(np.trace(cov)/len(Z))),
                    tangent_mse=mse,tangent_mse_lower=lo,tangent_mse_upper=hi,
                    sampling_covariance_trace=float(np.trace(cov)),
                    inverse_fisher_trace=float(np.trace(inv)),weighted_sampling_reference_trace=float(np.trace(sandwich)),
                    covariance_reference_ratio=float(np.trace(cov)/np.trace(sandwich)),
                    posterior_inverse_curvature_trace=float(np.nanmean(curvature)),local_fraction=local_fraction,
                    local_fraction_lower=local_lower,local_fraction_upper=local_upper,
                    local_reference_eligible=bool(local_fraction>=.95),
                    scope='Local stationary diagnostic with bias/locality reported; no global CR equality or posterior covariance bound'))
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def curvature(cfg,example):
    cx=example=='beamforming';rows=[]
    low_order,high_order=cfg['quadrature_orders']
    for case,gaps in enumerate(cfg['curvature_gaps']):
        A=-np.diag(gaps);ref=independent_nodes(A,1,cx,high_order)
        low=independent_nodes(A,1,cx,low_order)
        H=natural_curvature(A,1,cx)
        points=ref['nodes'];z=[]
        for point in points:
            # The independent nodes already use the modal phase convention.
            overlap=abs(point[0]);theta=np.arccos(np.clip(overlap,0.,1.))
            fixed=point*np.conj(point[0])/max(abs(point[0]),1e-15)
            coordinate=fixed[1:]*theta/max(np.sqrt(1-overlap*overlap),1e-15)
            z.append(np.r_[coordinate.real,coordinate.imag] if cx else coordinate.real)
        Z=np.array(z);weights=ref['weights'];mean=weights@Z
        C=np.einsum('b,bi,bj->ij',weights,Z-mean,(Z-mean))
        theta=np.arccos(np.clip(abs(points[:,0]),0.,1.))
        mass=float(weights@(theta<cfg['local_radius']))
        reference_error=float(np.linalg.norm(ref['mean']-low['mean'],2))
        for i,h in enumerate(H['values']):
            rows.append(dict(example=example,case=case,n=3,k=1,complex_data=cx,
                gap_1=gaps[1],gap_2=gaps[2],span=gaps[2],coordinate=i,
                hessian_eigenvalue=h,inverse_curvature_variance=1/h,
                empirical_log_coordinate_variance=float(C[i,i]),local_fraction=mass,
                local_approximation_eligible=bool(mass>=cfg['tolerances']['local_probability']),
                isolated=H['isolated'],anisotropy_ratio=float(max(H['values'])/min(H['values'])),
                real_graph_volume_curvature=3. if not cx else np.nan,
                graph_density_hessian_eigenvalue=h+3 if not cx else np.nan,
                quadrature_mean_resolution=reference_error,numerical_ok=reference_error<2e-8,
                coordinate_scope='Bounded projective log coordinate; no claim of finite unbounded graph-coordinate moments'))
    return pd.DataFrame(rows)


def path(T,condition,rate):
    theta=np.arange(T)*rate
    if condition=='abrupt':theta=np.r_[np.zeros(T//2),np.full(T-T//2,np.pi/4)]
    h=np.c_[np.sin(theta),np.zeros(T),np.cos(theta)]
    return h


def tracking(cfg,example):
    cx=example=='beamforming';scenarios=[];n=3
    def add(condition,T,beta,rate=0.,nonuniform=False,zero=False,design='balanced'):
        h=path(T,condition,rate);line=np.einsum('ti,tj->tij',h,h)
        Ps=line.astype(complex) if cx else np.eye(3)-line
        A0=np.diag(cfg['nonuniform_prior']) if nonuniform else np.zeros((3,3))
        V=None if cx else np.eye(3)[np.arange(T)%3]
        if design=='concentrated':V=np.tile([1.,0.,0.],(T,1))
        c=(sharp_marginal_certificate(Ps,cfg['sigma2'],0. if zero else cfg['tau2'],beta,cfg['delta'],A0,complex_data=True)
           if cx else sharp_conditional_certificate(Ps,V,cfg['sigma2'],beta,cfg['delta'],A0))
        scenarios.append(dict(condition=condition,T=T,beta=beta,rate=rate,
            nonuniform_prior=nonuniform,zero_signal=zero,design=design,h=h,V=V,c=c))
    for T in cfg['static_times']:
        for beta in [1.,.99]:add('static',T,beta)
    for rate in cfg['drift_rates'][1:]:
        for beta in cfg['bound_beta']:add('smooth',cfg['dynamic_T'],beta,rate)
    for beta in [1.,.99]:add('abrupt',cfg['dynamic_T'],beta)
    add('static',cfg['static_times'][-1],1.,nonuniform=True)
    if cx:add('static',256,1.,zero=True)
    else:add('static',256,1.,design='concentrated')
    phase=[]
    for beta in [1.,.9999,.999,.995,.99,.98]:
        for rate in [0.,.00001,.0001,.001]:
            h=path(cfg['dynamic_T'],'smooth',rate);line=np.einsum('ti,tj->tij',h,h)
            Ps=line.astype(complex) if cx else np.eye(3)-line
            V=np.eye(3)[np.arange(len(h))%3]
            c=sharp_marginal_certificate(Ps,cfg['sigma2'],1.,beta,complex_data=True) if cx else sharp_conditional_certificate(Ps,V,cfg['sigma2'],beta)
            phase.append(dict(example=example,beta=beta,drift_rate=rate,t=len(h),
                gamma=c['gamma'],epsilon=c['epsilon'],bound=c['bound'],ceiling=c['ceiling'],
                valid=c['valid'],informative=c['informative'],initialization=c['initialization'],
                drift=c['drift'],noise=c['noise']))
    rows=[]
    for index,spec in enumerate(scenarios):
        c=spec['c'];T=spec['T'];V=spec['V'];h=spec['h']
        for trial in range(cfg['episodes']):
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],913,index,trial,int(cx)]))
            noise=np.sqrt(cfg['sigma2'])*gaussian(rng,(T,3),cx)
            if cx:U=h*np.sqrt(c['tau2'])*gaussian(rng,(T,1),True)+noise
            else:U=np.einsum('tij,tj->ti',c['Ps'],V)+noise
            actual=certificate_observation(c,U,V)
            row=dict(example=example,condition=spec['condition'],n=n,k=c['k'],t=T,
                beta=spec['beta'],drift_rate=spec['rate'],nonuniform_prior=spec['nonuniform_prior'],
                zero_signal=spec['zero_signal'],design=spec['design'],trial=trial,
                sigma2=cfg['sigma2'],tau2=c.get('tau2',np.nan),
                gamma=c['gamma'],epsilon=c['epsilon'],initialization=c['initialization'],
                drift=c['drift'],noise=c['noise'],bound=c['bound'],loose_bound=c['loose_bound'],
                separate_norm_drift=c.get('separate_norm_drift',np.nan),
                valid=c['valid'],informative=c['informative'],ceiling=c['ceiling'],
                reference_fisher_min=c['reference_fisher_min'],information_gap_error=c['gap_information_error'],
                perturbation_scale=max(1.,c['gamma'],c['epsilon']),**actual)
            rows.append(row)
        print(example,'sharp tracking cell',index+1,'/',len(scenarios),flush=True)
    frame=pd.DataFrame(rows);summary=[]
    keys=['example','condition','t','beta','drift_rate','nonuniform_prior','zero_signal','design']
    for values,g in frame.groupby(keys,sort=False):
        eligible=int(g.valid.sum());violations=int(g.loc[g.valid,'violation'].sum())
        lo,hi=binomial(violations,eligible);mean,lower,upper=interval(g.modal_error)
        summary.append(dict(zip(keys,values),episodes=len(g),eligible_episodes=eligible,
            informative_episodes=int(g.informative.sum()),violations=violations if eligible else np.nan,
            violation_lower=lo,violation_upper=hi,mean_error=mean,error_lower=lower,error_upper=upper,
            bound=float(g.bound.iloc[0]),ceiling=float(g.ceiling.iloc[0]),
            mean_initialization=float(g.initialization.mean()),mean_drift=float(g.drift.mean()),mean_noise=float(g.noise.mean()),
            actual_perturbation=float(g.actual_perturbation.mean()),
            maximum_scaled_decomposition=float((g.decomposition_error/g.perturbation_scale).max()),
            maximum_information_gap_error=float(g.information_gap_error.max())))
    return frame,pd.DataFrame(summary),pd.DataFrame(phase)


def numerical(cfg,example):
    cx=example=='beamforming';k=1 if cx else 2;rows=[]
    from lib.guide_reliability import real_region_thresholds
    low_order,high_order=cfg['numerical_orders']
    for case,gaps in enumerate(cfg['numerical_gaps']):
        normal_A=-np.diag(gaps);A=normal_A if k==1 else -normal_A
        base=dict(example=example,case=case,n=3,k=k,complex_data=cx,span=gaps[-1],
            native_tolerance=cfg['native_tolerance'],native_max_order=cfg['native_max_order'],
            reference_order_low=low_order,reference_order_high=high_order)
        try:
            low=independent_nodes(A,k,cx,low_order);high=independent_nodes(A,k,cx,high_order)
            M,psi,diag=moments(A,k,cx,tol=cfg['native_tolerance'],max_order=cfg['native_max_order'])
            v=np.array([.3,.7,-.2],complex if cx else float)
            D=np.array([[.3,.1,0.],[.1,-.1,.2],[0.,.2,-.2]])
            step=.003
            _,plus,_=moments(A+step*D,k,cx,tol=1e-12)
            _,minus,_=moments(A-step*D,k,cx,tol=1e-12)
            first=(plus-minus)/(2*step);second=(plus-2*psi+minus)/step**2
            response=np.einsum('bij,j->bi',high['projectors'],v);mu=high['mean']@v
            covariance=np.einsum('b,bi,bj->ij',high['weights'],response-mu,(response-mu).conj())
            lowresponse=np.einsum('bij,j->bi',low['projectors'],v);lm=low['mean']@v
            lowcov=np.einsum('b,bi,bj->ij',low['weights'],lowresponse-lm,(lowresponse-lm).conj())
            native_cov,_,_=native_response_covariance(A,k,v,cx)
            values=np.einsum('bij,ji->b',high['projectors'],D).real
            var=float(high['weights']@(values-float(high['weights']@values))**2)
            if cx:
                rng=np.random.default_rng(cfg['seed']+case+488)
                draws=8192 if cfg['stage']=='paper' else 1024
                one=sample_complex_bingham(normal_A,draws,rng);two=sample_complex_bingham(normal_A,draws,rng)
                a=1-abs(one.vectors[:,0])**2;b=1-abs(two.vectors[:,0])**2
                quant=np.quantile(a,[.025,.975]);mass=np.array([(b<=q).mean() for q in quant])
                tail_error=float(np.max(abs(mass-[.025,.975])))
                tail_tolerance=float(4*np.sqrt(.975*.025/draws)+.003)
                tail_ok=tail_error<tail_tolerance
                tail_scope='Two independent exact-native IID batches; finite binomial precision, not deterministic tail enclosure'
            else:
                q0,_,_,_=real_region_thresholds(normal_A,[.025,.975],64)
                q1,_,_,_=real_region_thresholds(normal_A,[.025,.975],192)
                tail_error=float(np.max(abs(q1-q0)));tail_tolerance=1e-6
                tail_ok=tail_error<tail_tolerance
                tail_scope='Continuous radial CDF; independent angular-resolution comparison'
            mean_error=float(np.linalg.norm(M-high['mean'],2))
            logz_error=float(abs(psi-high['log_normalizer']))
            resolution=max(float(np.linalg.norm(low['mean']-high['mean'],2)),
                           float(abs(low['log_normalizer']-high['log_normalizer'])))
            cov_error=float(np.linalg.norm(native_cov-covariance,2))
            risk_error=float(abs(np.trace(covariance).real-np.vdot(v,(M-M@M)@v).real))
            cov_resolution=float(np.linalg.norm(covariance-lowcov,2))
            first_error=float(abs(first-np.trace(M@D).real))
            second_error=float(abs(second-var))
            mean_ok=bool(diag['numerical_ok'] and mean_error<2e-8)
            normalizer_ok=bool(diag['numerical_ok'] and logz_error<2e-8)
            reference_ok=bool(resolution<2e-8)
            second_ok=bool(cov_error<cfg['tolerances']['second_moment_absolute']
                           and cov_resolution<cfg['tolerances']['covariance_resolution']
                           and risk_error<2e-8)
            derivative_ok=bool(max(first_error,second_error)<cfg['tolerances']['normalizer_derivative'])
            rows.append(dict(base,mean_error=mean_error,log_normalizer_error=logz_error,
                reference_resolution_error=resolution,second_moment_error=cov_error,
                covariance_resolution_error=cov_resolution,
                response_risk_identity_error=risk_error,
                normalizer_first_derivative_error=first_error,
                normalizer_second_derivative_error=second_error,
                quantile_error=tail_error,quantile_tolerance=tail_tolerance,quantile_ok=tail_ok,
                native_mean_declared_ok=bool(diag['numerical_ok']),mean_ok=mean_ok,
                normalizer_ok=normalizer_ok,reference_ok=reference_ok,
                second_moment_ok=second_ok,derivative_ok=derivative_ok,
                numerical_ok=bool(mean_ok and normalizer_ok and reference_ok and second_ok and derivative_ok),
                tail_scope=tail_scope,failure_reason=''))
        except (ArithmeticError,FloatingPointError,ValueError) as error:
            rows.append(dict(base,numerical_ok=False,quantile_ok=False,
                native_mean_declared_ok=False,mean_ok=False,normalizer_ok=False,
                reference_ok=False,second_moment_ok=False,derivative_ok=False,
                failure_reason=type(error).__name__+': '+str(error),
                scope='Unresolved bounded numerical target retained; no substitute visually plausible approximation'))
    return pd.DataFrame(rows)


def run(example,cfg,output_root):
    data=output_root/example/'data'/cfg['stage']/'theory';tables=output_root/example/'tables'/cfg['stage']/'theory'
    data.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    sources={name:sha(ROOT/name) for name in SOURCES}
    manifest=data/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta['config']!=cfg or meta['sources']!=sources:
            raise RuntimeError('Preserve obsolete theory results and use a fresh output root')
        for name,expected in meta['artifacts'].items():
            if sha(ROOT/name)!=expected:raise RuntimeError('Changed cached theory artifact: '+name)
        print('Verified completed',example,cfg['stage'],'theory');return
    save(data/'preregistered.json',dict(example=example,stage=cfg['stage'],component='theory',
        config=cfg,sources=sources,registered_utc=datetime.now(timezone.utc).isoformat()))
    begin=perf_counter()
    frames,reused=information(cfg,example)
    frames['block_checks']=algebra(cfg,example)
    frames['inverse_fisher'],frames['inverse_fisher_summary']=inverse_fisher(cfg,example)
    frames['curvature']=curvature(cfg,example)
    frames['tracking'],frames['tracking_summary'],frames['tracking_phase']=tracking(cfg,example)
    frames['numerical_accuracy']=numerical(cfg,example)
    for name,frame in frames.items():frame.to_csv(tables/(name+'.csv'),index=False)
    if any(sha(ROOT/name)!=expected for name,expected in sources.items()):
        raise RuntimeError('Theory scientific source changed during execution')
    checks=dict(algebra_failures=int(frames['block_checks'].status.eq('failed').sum()),
        information_reused_checks=len(frames['information']),
        global_kl_failures=int(frames['global_kl'].status.eq('failed').sum()),
        information_loss_identity_error=float(frames['information_loss'].identity_error.max()),
        tracking_reports=len(frames['tracking']),valid_bound_reports=int(frames['tracking'].valid.sum()),
        informative_bound_reports=int(frames['tracking'].informative.sum()),
        eligible_violations=int(frames['tracking'].violation.sum()),
        maximum_scaled_decomposition=float((frames['tracking'].decomposition_error/frames['tracking'].perturbation_scale).max()),
        maximum_information_gap_error=float(frames['tracking'].information_gap_error.max()),
        inverse_fisher_episodes=cfg['local_episodes'],
        inverse_fisher_local_eligible_cells=int(frames['inverse_fisher_summary'].local_reference_eligible.sum()),
        numerical_targets=len(frames['numerical_accuracy']),
        unresolved_numerical_targets=int((~frames['numerical_accuracy'].numerical_ok).sum()),
        unresolved_quantile_targets=int((~frames['numerical_accuracy'].quantile_ok).sum()),
        seconds=perf_counter()-begin)
    save(data/'checks.json',checks)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in list(data.glob('*.json'))+list(tables.glob('*.csv')) if p!=manifest}
    save(manifest,dict(status='completed',example=example,stage=cfg['stage'],component='theory',
        config=cfg,sources=sources,artifacts=artifacts,reused_inputs=reused,checks=checks,
        specification='Supplied v5.tex and supplementary(14).tex theory/appendices, hashes in config. Historical numerical section not used to define experiments.'))
    print(example,cfg['stage'],json.dumps(portable(checks),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['verify','pilot','paper'],default='verify')
    parser.add_argument('--example',choices=['beamforming','contact','both'],default='both')
    parser.add_argument('--output-root',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    if not args.output_root.resolve().is_relative_to(ROOT):
        raise ValueError('Output root must remain inside the repository for portable artifact paths')
    if args.stage=='paper':
        from lib.final_validation import require_gate
        require_gate(ROOT)
    cfg=configuration(args.stage)
    for example in (['beamforming','contact'] if args.example=='both' else [args.example]):
        run(example,cfg,args.output_root)


if __name__=='__main__':main()
