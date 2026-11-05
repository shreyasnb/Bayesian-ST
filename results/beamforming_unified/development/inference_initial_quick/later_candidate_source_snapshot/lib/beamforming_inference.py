"""Ordered-power inference for the native complex passive rank-one model.

The variance prior is a joint product of inverse-gamma densities restricted to
``a>b>0`` with one global normalizer. It is not an independently specified
prior on sensor power ``s=b`` and interference power ``q=a-b``. A stationary
experiment has one common projector; a moving experiment has one state per
observation and exactly T-1 process factors. Integrated state laws are mixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import (betainc, gammainc, gammaincc, gammaincinv,
    gammainccinv, gammaln, logsumexp)

from .geometry import center
from .models import increment, constant
from .em import Record, fit_noise
from .moments import effective_sample_size


@dataclass(frozen=True)
class PowerPrior:
    a_shape: float = 3.
    a_scale: float = 3.
    b_shape: float = 3.
    b_scale: float = 1.
    kappa_shape: float = 2.
    kappa_rate: float = .04
    label: str = 'Default'

    def __post_init__(self):
        if any(not np.isfinite(x) or x<=0 for x in
               [self.a_shape,self.a_scale,self.b_shape,self.b_scale,
                self.kappa_shape,self.kappa_rate]):
            raise ValueError('Proper inverse-gamma and gamma priors need positive parameters')


def sensitivity_prior():
    return PowerPrior(a_scale=1.5,b_scale=.5,kappa_rate=.02,label='Sensitivity')


def ordered_prior_normalizer(prior=PowerPrior()):
    return float(betainc(prior.a_shape,prior.b_shape,
        prior.a_scale/(prior.a_scale+prior.b_scale)))


def ordered_prior_log_density(a,b,prior=PowerPrior()):
    if not np.isfinite(a+b) or not a>b>0:return -np.inf
    value=0.
    for x,shape,scale in [(a,prior.a_shape,prior.a_scale),(b,prior.b_shape,prior.b_scale)]:
        value+=shape*np.log(scale)-gammaln(shape)-(shape+1)*np.log(x)-scale/x
    return float(value-np.log(ordered_prior_normalizer(prior)))


def truncated_inverse_gamma(shape,scale,rng=None,lower=0.,upper=np.inf):
    """Exact inverse-CDF draw on lower<x<upper; no confidence-inducing floor."""
    if not np.isfinite(shape+scale) or min(shape,scale)<=0 or not 0<=lower<upper:
        raise ValueError('Invalid inverse-gamma shape, scale, or truncation')
    rng=np.random.default_rng(rng)
    zl=0. if np.isposinf(upper) else scale/upper
    zu=np.inf if lower==0 else scale/lower
    pl,pu=gammainc(shape,zl),gammainc(shape,zu)
    ql,qu=gammaincc(shape,zl),gammaincc(shape,zu)
    uniform=float(rng.random())
    uniform=min(max(uniform,np.nextafter(0.,1.)),np.nextafter(1.,0.))
    if pu<=.9 and pu>pl:
        z=gammaincinv(shape,pl+uniform*(pu-pl))
    elif ql>qu:
        z=gammainccinv(shape,qu+uniform*(ql-qu))
    elif pu>pl:
        z=gammaincinv(shape,pl+uniform*(pu-pl))
    else:raise ArithmeticError('Truncated inverse-gamma probability is numerically unresolved')
    value=float(scale/z)
    if not lower<value<upper or not np.isfinite(value):
        raise ArithmeticError('Truncated inverse-gamma inversion hit a numerical boundary')
    return value


def _observations(U):
    U=np.asarray(U,dtype=complex)
    if U.ndim!=2 or U.shape[0]<1 or U.shape[1]<2 or not np.isfinite(U).all():
        raise ValueError('Observations must be a finite nonempty T by n matrix')
    return U


def variance_conditionals(U,states,prior=PowerPrior()):
    """Return complex-model posterior shape/scale pairs, before order truncation."""
    U=_observations(U);h=np.asarray(states,dtype=complex)
    if h.ndim==1:h=np.broadcast_to(h,U.shape)
    if h.shape!=U.shape or not np.allclose(np.sum(abs(h)**2,axis=1),1.,rtol=0,atol=1e-9):
        raise ValueError('Need one unit state or a T by n unit-state path')
    parallel=np.abs(np.einsum('ti,ti->t',h.conj(),U))**2
    total=np.sum(abs(U)**2,axis=1)
    if np.any(parallel>total+1e-10):raise ArithmeticError('Projection energy exceeds total energy')
    perpendicular=np.maximum(0.,total-parallel)
    T,n=U.shape
    return dict(a_shape=prior.a_shape+T,a_scale=prior.a_scale+float(parallel.sum()),
        b_shape=prior.b_shape+(n-1)*T,b_scale=prior.b_scale+float(perpendicular.sum()),
        parallel_energy=float(parallel.sum()),perpendicular_energy=float(perpendicular.sum()))


def passive_log_likelihood(U,states,a,b):
    U=_observations(U);h=np.asarray(states,dtype=complex)
    if not a>b>0:return -np.inf
    if h.ndim==1:h=np.broadcast_to(h,U.shape)
    if h.shape!=U.shape:raise ValueError('State/observation dimensions differ')
    parallel=np.abs(np.einsum('ti,ti->t',h.conj(),U))**2
    perpendicular=np.sum(abs(U)**2,axis=1)-parallel
    n=U.shape[1]
    return float(-len(U)*(n*np.log(np.pi)+np.log(a)+(n-1)*np.log(b))
        -parallel.sum()/a-perpendicular.sum()/b)


def _transports(G,T,n):
    G=np.broadcast_to(np.eye(n,dtype=complex),(T,n,n)).copy() if G is None else np.asarray(G,dtype=complex)
    if G.shape!=(T,n,n) or not np.isfinite(G).all() or not np.allclose(
        G.conj().transpose(0,2,1)@G,np.eye(n),atol=1e-10,rtol=0):
        raise ValueError('Need T supplied unitary transports; G[0] is unused')
    return G


def path_state_parameter(U,states,index,a,b,kappa,G=None):
    U=_observations(U);h=np.asarray(states,dtype=complex);T,n=U.shape
    if h.shape!=U.shape or not 0<=index<T or not a>b>0 or not np.isfinite(kappa) or kappa<0 or not np.allclose(np.sum(abs(h)**2,axis=1),1.,atol=1e-9,rtol=0):
        raise ValueError('Invalid path conditional')
    G=_transports(G,T,n)
    A=(1/b-1/a)*np.outer(U[index],U[index].conj())
    if index:
        previous=G[index]@h[index-1];A+=kappa*np.outer(previous,previous.conj())
    if index+1<T:
        following=G[index+1].conj().T@h[index+1];A+=kappa*np.outer(following,following.conj())
    return center(A)


def transition_overlap(states,G=None):
    h=np.asarray(states,dtype=complex)
    if h.ndim!=2 or len(h)<1:raise ValueError('Need a unit-state path')
    T,n=h.shape;G=_transports(G,T,n)
    if not np.allclose(np.sum(abs(h)**2,axis=1),1.,atol=1e-9,rtol=0):raise ValueError('Nonunit states')
    value=np.abs(np.einsum('ti,tij,tj->t',h[1:].conj(),G[1:],h[:-1]))**2
    return float(np.clip(value,0,1).sum())


def kappa_log_density(kappa,overlap_sum,transitions,n,prior=PowerPrior()):
    """Normalized-kernel conditional, stable even for concentrated transitions."""
    if transitions<0 or n<2 or not -1e-10<=overlap_sum<=transitions+1e-10:
        raise ValueError('Invalid overlap sum or transition count')
    if not np.isfinite(kappa) or kappa<=0:return -np.inf
    from .beamforming_numerics import transition_log_normalizer
    if kappa>.001:
        probability=float(gammainc(n-1,kappa))
        value=(overlap_sum-transitions-prior.kappa_rate)*kappa
        value+=((n-1)*transitions+prior.kappa_shape-1)*np.log(kappa)
        value-=transitions*(gammaln(n)+np.log(probability))
    else:
        value=(prior.kappa_shape-1)*np.log(kappa)-prior.kappa_rate*kappa
        value+=kappa*overlap_sum-transitions*transition_log_normalizer(n,kappa)
    return float(value)


def sample_kappa(current,overlap_sum,transitions,n,prior=PowerPrior(),rng=None,width=.8,max_steps=32):
    """Slice update in log kappa, including the log-coordinate Jacobian."""
    rng=np.random.default_rng(rng)
    if not np.isfinite(current) or current<=0 or width<=0:raise ValueError('Positive slice start/width required')
    def target(x):
        if not -700<x<700:return -np.inf
        return kappa_log_density(np.exp(x),overlap_sum,transitions,n,prior)+x
    x=np.log(current);level=target(x)+np.log(rng.random())
    left=x-width*rng.random();right=left+width
    J=int(rng.integers(max_steps));K=max_steps-1-J
    while J and target(left)>level:left-=width;J-=1
    while K and target(right)>level:right+=width;K-=1
    for _ in range(200):
        proposal=float(rng.uniform(left,right))
        if target(proposal)>=level:return float(np.exp(proposal))
        if proposal<x:left=proposal
        else:right=proposal
    raise ArithmeticError('Scalar kappa slice shrink budget exhausted')


def joint_covariance_mean(a,b,states):
    a,b=np.asarray(a,float).ravel(),np.asarray(b,float).ravel();h=np.asarray(states,complex)
    h=h.reshape(-1,h.shape[-1])
    if h.shape[0]!=len(a) or a.shape!=b.shape or not np.isfinite(a+b).all() or not np.all(a>b) or not np.all(b>0) or not np.allclose(np.sum(abs(h)**2,axis=1),1.,atol=1e-9,rtol=0):
        raise ValueError('Need paired ordered powers and unit states')
    return np.mean(b)*np.eye(h.shape[1])+np.einsum('d,di,dj->ij',a-b,h,h.conj())/len(a)


def _split_rhat(values):
    x=np.asarray(values,float);chains,draws=x.shape;half=draws//2
    if chains<2 or half<2:return np.nan
    x=np.concatenate([x[:,:half],x[:,-half:]],axis=0)
    W=float(np.var(x,axis=1,ddof=1).mean());B=float(half*np.var(x.mean(axis=1),ddof=1))
    return float(np.sqrt(((half-1)/half*W+B/half)/W)) if W>0 else (1. if B==0 else np.inf)


def _diagnostics(a,b,kappa,states,seconds,failures):
    terminal=states[:,:,-1,:] if states.ndim==4 else states
    P=np.einsum('cdi,cdj->cdij',terminal,terminal.conj());C,D,n,_=P.shape
    values={'a':a,'b':b,'q':a-b}
    if np.isfinite(kappa).all():values['kappa']=kappa
    for i in range(n):
        values[f'P_{i}{i}']=P[:,:,i,i].real
        for j in range(i):
            values[f'P_{i}{j}_real']=P[:,:,i,j].real
            values[f'P_{i}{j}_imag']=P[:,:,i,j].imag
    details=[]
    for name,x in values.items():
        if not np.isfinite(x).all():continue
        ess=min(effective_sample_size(row) for row in x)
        rhat=_split_rhat(x)
        chain_quantiles=np.quantile(x,[.025,.975],axis=1)
        width=max(float(np.quantile(x,.975)-np.quantile(x,.025)),1e-12)
        tail_difference=float(np.max(np.ptp(chain_quantiles,axis=1))/width)
        chain_mcse=float(np.std(x.mean(axis=1),ddof=1)/np.sqrt(C)) if C>1 else np.nan
        details.append(dict(quantity=name,min_chain_ess=float(ess),rhat=rhat,
            chain_mean_mcse=chain_mcse,tail_difference_fraction=tail_difference))
    minimum=min((x['min_chain_ess'] for x in details),default=0.)
    maxr=max((x['rhat'] for x in details),default=np.inf)
    maxt=max((x['tail_difference_fraction'] for x in details),default=np.inf)
    return dict(completed=not failures,failures=failures,seconds=seconds,chains=C,draws_per_chain=D,
        min_chain_ess=minimum,max_rhat=maxr,max_between_chain_tail_fraction=maxt,
        mixing_ok=bool(not failures and minimum>=50 and maxr<1.1 and maxt<.15),
        diagnostics_scope='Finite independent-chain diagnostics; not a proof of mixing or exact credible coverage',details=details)


@dataclass
class JointResult:
    a: np.ndarray
    b: np.ndarray
    kappa: np.ndarray
    states: np.ndarray
    covariance_mean: np.ndarray
    projector_mean: np.ndarray
    diagnostics: dict


def _gibbs(U,prior,stationary,draws,burnin,chains,seed,G=None,known_variances=None,kappa_fixed=None):
    from .beamforming_numerics import sample_complex_bingham
    U=_observations(U);T,n=U.shape
    if min(draws,chains)<1 or burnin<0:raise ValueError('Invalid Gibbs budgets')
    G=_transports(G,T,n)
    if stationary and kappa_fixed is not None:raise ValueError('Stationary common state has no kappa')
    if kappa_fixed is not None and (not np.isfinite(kappa_fixed) or kappa_fixed<0):
        raise ValueError('Path Gibbs requires finite nonnegative fixed concentration')
    if known_variances is not None:
        if not known_variances[0]>known_variances[1]>0:raise ValueError('Known variances must be ordered a>b')
    state_shape=(chains,draws,n) if stationary else (chains,draws,T,n)
    retained=np.full(state_shape,np.nan+0j);aa=np.full((chains,draws),np.nan);bb=aa.copy();kk=aa.copy()
    scatter=U.T@U.conj();eigenvalues,basis=np.linalg.eigh(scatter)
    failures=[];start=perf_counter()
    for chain in range(chains):
        rng=np.random.default_rng(np.random.SeedSequence([seed,chain,4613]))
        factor=[.6,1.,1.8][chain%3]
        b=prior.b_scale/max(prior.b_shape-1,1)*factor
        a=max(prior.a_scale/max(prior.a_shape-1,1)*factor,b*1.5)
        if known_variances is not None:a,b=known_variances
        kappa=prior.kappa_shape/prior.kappa_rate*factor if kappa_fixed is None else float(kappa_fixed)
        if chain%3==0:h=basis[:,-1].copy()
        elif chain%3==1:
            h=rng.normal(size=n)+1j*rng.normal(size=n);h/=np.linalg.norm(h)
        else:h=basis[:,0].copy()
        path=np.broadcast_to(h,(T,n)).copy()
        try:
            for sweep in range(burnin+draws):
                if stationary:
                    A=center((1/b-1/a)*scatter)
                    sample=sample_complex_bingham(A,draws=1,rng=rng)
                    h=sample.vectors[0]
                    if not sample.diagnostics['numerical_ok']:raise ArithmeticError('State conditional sampler unresolved')
                else:
                    # Alternating sweep direction reduces a systematic endpoint update order.
                    indices=range(T) if sweep%2==0 else range(T-1,-1,-1)
                    for j in indices:
                        A=(1/b-1/a)*np.outer(U[j],U[j].conj())
                        if j:
                            v=G[j]@path[j-1];A+=kappa*np.outer(v,v.conj())
                        if j+1<T:
                            v=G[j+1].conj().T@path[j+1];A+=kappa*np.outer(v,v.conj())
                        sample=sample_complex_bingham(center(A),draws=1,rng=rng)
                        if not sample.diagnostics['numerical_ok']:raise ArithmeticError('Path conditional sampler unresolved')
                        path[j]=sample.vectors[0]
                if known_variances is None:
                    powers=variance_conditionals(U,h if stationary else path,prior)
                    a=truncated_inverse_gamma(powers['a_shape'],powers['a_scale'],rng,lower=b)
                    b=truncated_inverse_gamma(powers['b_shape'],powers['b_scale'],rng,upper=a)
                if not stationary and kappa_fixed is None:
                    overlap=float(np.sum(np.abs(np.einsum('ti,tij,tj->t',path[1:].conj(),G[1:],path[:-1]))**2))
                    kappa=sample_kappa(kappa,overlap,T-1,n,prior,rng)
                if sweep>=burnin:
                    index=sweep-burnin;aa[chain,index]=a;bb[chain,index]=b
                    kk[chain,index]=np.nan if stationary else kappa
                    retained[chain,index]=h if stationary else path
        except (ArithmeticError,np.linalg.LinAlgError) as error:
            failures.append(dict(chain=chain,sweep=sweep,reason=str(error)))
    terminal=retained[:,:,-1,:] if not stationary else retained
    if not np.isfinite(terminal).all() or not np.isfinite(aa+bb).all():
        R=M=np.full((n,n),np.nan+0j)
    else:
        R=joint_covariance_mean(aa,bb,terminal)
        M=np.einsum('cdi,cdj->ij',terminal,terminal.conj())/(chains*draws)
    return JointResult(aa,bb,kk,retained,R,M,
        _diagnostics(aa,bb,kk,retained,perf_counter()-start,failures))


def stationary_gibbs(U,prior=PowerPrior(),draws=512,burnin=256,chains=3,seed=6571042,known_variances=None):
    return _gibbs(U,prior,True,draws,burnin,chains,seed,known_variances=known_variances)


def full_path_gibbs(U,prior=PowerPrior(),draws=512,burnin=256,chains=3,seed=6571042,G=None,known_variances=None,kappa_fixed=None):
    return _gibbs(U,prior,False,draws,burnin,chains,seed,G,known_variances,kappa_fixed)


def joint_predictive_score(result,u,G=None,seed=6571043,max_draws=2048):
    """Prospective mixture score from a fit on past data only.

    Moving fits simulate one new transition per retained joint parameter/state
    draw. This is a Monte Carlo predictive density, not a Gaussian based only
    on the posterior covariance mean. Gaussian constants are included.
    """
    from .beamforming_numerics import sample_complex_transition
    h=result.states[:,:,-1,:] if result.states.ndim==4 else result.states
    h=h.reshape(-1,h.shape[-1]);a=result.a.ravel();b=result.b.ravel();k=result.kappa.ravel()
    if not np.isfinite(h).all():raise ArithmeticError('Cannot score an incomplete posterior')
    chosen=np.linspace(0,len(a)-1,min(max_draws,len(a)),dtype=int)
    h,a,b,k=h[chosen],a[chosen],b[chosen],k[chosen]
    if result.states.ndim==4:
        rng=np.random.default_rng(seed);G=np.eye(h.shape[1]) if G is None else np.asarray(G)
        h=np.asarray([sample_complex_transition(G@v,concentration,rng=rng,size=1)[0]
            for v,concentration in zip(h,k)])
    u=np.asarray(u,complex);parallel=abs(h.conj()@u)**2;perpendicular=np.vdot(u,u).real-parallel
    logweights=-len(u)*np.log(np.pi)-np.log(a)-(len(u)-1)*np.log(b)-parallel/a-perpendicular/b
    score=float(logsumexp(logweights)-np.log(len(logweights)))
    chain_ids=chosen//result.a.shape[1]
    chain_scores=[float(logsumexp(logweights[chain_ids==c])-np.log(np.sum(chain_ids==c)))
        for c in np.unique(chain_ids)]
    chain_mcse=float(np.std(chain_scores,ddof=1)/np.sqrt(len(chain_scores))) if len(chain_scores)>1 else np.nan
    shifted=np.exp(logweights-logweights.max());denominator=shifted.mean()
    relative_se=float(shifted.std(ddof=1)/np.sqrt(len(shifted))/denominator) if len(shifted)>1 else np.nan
    return dict(log_score=score,draws=len(chosen),iid_relative_mcse=relative_se,
        chain_log_score_mcse=chain_mcse,
        mcse_scope='IID formula is descriptive only: outer joint posterior draws are Gibbs-correlated',
        scoring='Held-out next observation only; prefix fit is never updated with this observation')


def joint_forecast_covariance(result,G=None):
    """Average the joint next covariance, preserving q, kappa and state dependence."""
    if result.states.ndim==3:return result.covariance_mean.copy()
    from .beamforming_numerics import transition_coefficients_complex
    h=result.states[:,:,-1,:].reshape(-1,result.states.shape[-1])
    n=h.shape[1];G=np.eye(n) if G is None else np.asarray(G)
    h=h@G.T;q=(result.a-result.b).ravel()
    rho=np.array([transition_coefficients_complex(n,k).rho for k in result.kappa.ravel()])
    return (np.mean(result.b)+np.mean(q*(1-rho))/n)*np.eye(n)+np.einsum('d,di,dj->ij',q*rho,h,h.conj())/len(q)


def recorded_prior_em(U,s0=.5,q0=1.,iterations=5,kappa=None,G=None):
    """Existing frozen-prior EM, with its own fixed objective and final prior.

    Priors are accumulated once using s0,q0, then held fixed for EM. The final
    decision uses the final recorded prior plus one correction at fitted powers.
    It does not refit the full stationary history or claim ascent of that law.
    """
    from .beamforming_numerics import complex_moments,predict_complex
    U=_observations(U);n=U.shape[1];A=np.zeros((n,n),complex);records=[]
    G=_transports(G,len(U),n)
    for j,u in enumerate(U):
        if kappa is not None and j:A=predict_complex(A,kappa,G=G[j]).prior
        records.append(Record(A.copy(),u.copy()))
        A=center(A+increment(u,s0,tau2=q0,complex_data=True))
    normalizer=lambda value:float(complex_moments(value).log_normalizer)
    mean=lambda value,k:complex_moments(value).mean if k==1 else (_ for _ in ()).throw(ValueError('Rank one only'))
    fit=fit_noise(records,1,s0,q0,mean,iterations=iterations,
        log_normalizer=normalizer,complex_data=True)
    last=center(records[-1].prior+increment(U[-1],fit.s,tau2=fit.tau2,complex_data=True))
    M=complex_moments(last).mean
    return dict(s=fit.s,q=fit.tau2,a=fit.s+fit.tau2,b=fit.s,A=last,mean=M,
        covariance=fit.s*np.eye(n)+fit.tau2*M,objectives=fit.objectives,
        accepted=fit.accepted,rejected=fit.rejected,mean_evaluations=fit.mean_evaluations,
        ascent_verified=bool(np.all(np.diff(fit.objectives)>=-1e-8)),
        objective='Sum of normalized one-observation predictive densities with every recorded prior frozen')


def stationary_reference_n2(U,prior=PowerPrior(),orders=(160,240)):
    """Independent scalar-grid integration after exact CP1 state marginalization."""
    U=_observations(U)
    if U.shape[1]!=2:raise ValueError('This reference is native complex n=2 only, not the array')
    S=U.T@U.conj();lam,V=np.linalg.eigh(S);gap=float(lam[1]-lam[0]);P=np.outer(V[:,1],V[:,1].conj())
    answers=[]
    for order in orders:
        lb=np.linspace(-8,4,int(order));lq=np.linspace(-12,6,int(order))
        B,Q=np.meshgrid(np.exp(lb),np.exp(lq),indexing='ij');A=B+Q;c=1/B-1/A;d=c*gap
        correction=np.where(d<1e-4,d/2+d*d/24,
            d+np.log(-np.expm1(-d))-np.log(d))
        logZ=c*lam[0]+correction
        logw=-(prior.a_shape+1)*np.log(A)-prior.a_scale/A
        logw+=-(prior.b_shape+1)*np.log(B)-prior.b_scale/B
        logw+=-len(U)*(np.log(A)+np.log(B))-np.trace(S).real/B+logZ
        logw+=np.log(B)+np.log(Q)
        weight=np.exp(logw-logw.max());weight[[0,-1],:]*=.5;weight[:,[0,-1]]*=.5;weight/=weight.sum()
        r=np.empty_like(d);small=d<1e-3;r[small]=.5+d[small]/12-d[small]**3/720
        r[~small]=1/(-np.expm1(-d[~small]))-1/d[~small]
        pweight=float(np.sum(weight*r));M=(1-pweight)*np.eye(2)+(2*pweight-1)*P
        qmean=float(np.sum(weight*Q));qr=float(np.sum(weight*Q*r))
        R=float(np.sum(weight*B))*np.eye(2)+(qmean-qr)*np.eye(2)+(2*qr-qmean)*P
        answers.append(dict(order=int(order),a=float(np.sum(weight*A)),b=float(np.sum(weight*B)),q=qmean,
            mean=M,covariance=R,log_integral=float(logw.max()+np.log(np.exp(logw-logw.max()).sum())+np.log(lb[1]-lb[0])+np.log(lq[1]-lq[0]))))
    low,high=answers[-2],answers[-1]
    error=max(abs(high[x]-low[x]) for x in ['a','b','q'])
    error=max(error,float(np.max(abs(high['covariance']-low['covariance']))))
    return dict(**high,refinement_error=error,numerical_ok=bool(error<2e-4),
        scope='Native complex CP1 analytical state marginalization, log b/log q quadrature; not a native n=8 reference')
