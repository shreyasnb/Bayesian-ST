"""Native complex rank-one calculations for passive array inference.

Squared eigen-coordinates have a Dirichlet(1,...,1) distribution.  The iid
sampler below is exact rejection sampling on that simplex, not an unrestricted
real Bingham sampler.  Moment error and sampling error have separate scopes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.linalg import null_space
from scipy.special import gammainc, gammaincinv, gammaln, hyp1f1
from scipy.integrate import quad

from .geometry import center
from .spectral_series import rank_one_bingham
from .process_prediction import MeanFit, ProcessPrediction, TransitionCoefficients


def _matrix(A):
    A=np.asarray(A,dtype=complex)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or len(A)<2:
        raise ValueError('A must be a square matrix of dimension at least two')
    if not np.isfinite(A).all() or not np.allclose(A,A.conj().T,atol=1e-11,rtol=0):
        raise ValueError('A must be finite and Hermitian')
    return (A+A.conj().T)/2


@dataclass(frozen=True)
class ComplexMoments:
    mean: np.ndarray
    log_normalizer: float
    order: int
    moment_error_bound: float
    log_normalizer_error_bound: float
    numerical_ok: bool
    seconds: float
    backend: str

    @property
    def accuracy_met(self):return self.numerical_ok

    @property
    def diagnostics(self):
        return dict(order=self.order,moment_error_bound=self.moment_error_bound,
            log_normalizer_error_bound=self.log_normalizer_error_bound,
            numerical_ok=self.numerical_ok,seconds=self.seconds,backend=self.backend,
            scope='Analytic/truncation control; floating-point rounding is not enclosed')


def _log_truncated_exponential_integral(gaps):
    d=np.asarray(gaps,float);small=d<1e-4
    output=np.empty_like(d)
    output[small]=-d[small]/2+d[small]**2/24-d[small]**4/2880
    output[~small]=np.log(-np.expm1(-d[~small]))-np.log(d[~small])
    return output


def complex_moments(A,tol=1e-10,max_order=4096):
    """Mean/log normalizer under normalized invariant complex-line measure."""
    start=perf_counter();A=_matrix(A)
    if not 0<tol<.1 or max_order<1:raise ValueError('Invalid numerical budget')
    if len(A)==2:
        values,U=np.linalg.eigh(A);gap=float(values[1]-values[0])
        if gap<1e-3:
            top=.5+gap/12-gap**3/720+gap**5/30240
        else:top=float(1/(-np.expm1(-gap))-1/gap)
        M=(U*np.array([1-top,top]))@U.conj().T
        psi=float(values[1]+_log_truncated_exponential_integral(np.array([gap]))[0])
        return ComplexMoments((M+M.conj().T)/2,psi,0,0.,0.,True,
            perf_counter()-start,'analytic complex CP1 integral')
    result=rank_one_bingham(A,complex_data=True,tol=tol,max_order=max_order)
    return ComplexMoments(result.mean,result.log_normalizer,result.order,
        result.moment_error_bound,result.log_normalizer_error_bound,result.accuracy_met,
        perf_counter()-start,result.backend+' (native complex)')


mean_bingham_complex=complex_moments


@dataclass(frozen=True)
class ComplexSamples:
    vectors: np.ndarray
    diagnostics: dict

    @property
    def bases(self):return self.vectors[:,:,None]

    @property
    def mean(self):
        return self.vectors.T@self.vectors.conj()/len(self.vectors)


def sample_complex_bingham(A,draws=1,rng=None,max_proposals=None,strict=True):
    """Draw iid complex lines by two exactly normalized rejection envelopes.

    With eigenvalues lambda_0>=... and d_i=lambda_0-lambda_i, one proposal is
    uniform Dirichlet.  The other draws independent truncated exponentials for
    all nonleading squared coordinates and accepts their sum <=1. Their ratio
    of acceptance probabilities is known without evaluating the normalizer:
    p_box/p_uniform = 1 / ((n-1)! prod_i integral_0^1 exp(-d_i x) dx).
    Thus even a one-vector Gibbs conditional needs only an eigendecomposition.
    Uniform independent phases restore the native complex invariant measure.
    """
    begin=perf_counter();A=_matrix(A);n=len(A)
    if not isinstance(draws,(int,np.integer)) or draws<1:raise ValueError('draws must be positive')
    if rng is None:rng=np.random.default_rng()
    budget=max(20000,1000*int(draws)) if max_proposals is None else int(max_proposals)
    if budget<draws:raise ValueError('proposal budget is smaller than requested draws')
    values,U=np.linalg.eigh(A);values=values[::-1];U=U[:,::-1]
    gaps=np.maximum(0.,values[0]-values)
    comparison=float(-gammaln(n)-_log_truncated_exponential_integral(gaps[1:]).sum())
    proposal='truncated exponential box' if comparison>0 else 'uniform simplex'
    output=[];accepted=proposed=accepted_candidates=0
    while accepted<draws and proposed<budget:
        count=min(budget-proposed,16384,max(2,int(np.ceil(1.5*(draws-accepted)))))
        if proposal=='uniform simplex':
            squared=rng.exponential(size=(count,n));squared/=squared.sum(axis=1)[:,None]
            mask=np.log(rng.random(count))<=-(squared@gaps)
            squared=squared[mask]
        else:
            d=gaps[1:];uniform=rng.random((count,n-1))
            tail=np.empty_like(uniform);flat=d<1e-12
            tail[:,flat]=uniform[:,flat]
            tail[:,~flat]=-np.log1p(uniform[:,~flat]*np.expm1(-d[~flat]))/d[~flat]
            total=tail.sum(axis=1);mask=total<=1.
            squared=np.column_stack([1-total[mask],tail[mask]])
        proposed+=count
        accepted_candidates+=len(squared)
        take=min(draws-accepted,len(squared))
        if take:
            phase=np.exp(2j*np.pi*rng.random((take,n)))
            output.append((np.sqrt(squared[:take])*phase)@U.T);accepted+=take
    vectors=np.concatenate(output,axis=0) if output else np.empty((0,n),complex)
    ok=bool(accepted==draws and np.isfinite(vectors).all()
            and np.allclose(np.sum(abs(vectors)**2,axis=1),1.,atol=2e-12,rtol=0))
    info=dict(numerical_ok=ok,requested_draws=int(draws),draws=accepted,
        proposals=proposed,proposal_budget=budget,proposal=proposal,
        acceptance=accepted_candidates/proposed if proposed else 0.,
        accepted_candidates=accepted_candidates,
        log_box_to_uniform_acceptance_ratio=comparison,seconds=perf_counter()-begin,
        independent=True,measure='complex line / Dirichlet(1,...,1) squared coordinates',
        error_scope='Exact rejection law; empirical moment and tail estimates retain Monte Carlo error')
    if strict and not ok:raise ArithmeticError(f'Complex iid rejection budget exhausted: {info}')
    return ComplexSamples(vectors,info)


@lru_cache(maxsize=256)
def transition_log_normalizer(n,kappa):
    if not isinstance(n,(int,np.integer)) or n<2 or not np.isfinite(kappa) or kappa<0:
        raise ValueError('Need dimension >=2 and finite kappa >=0')
    if kappa<50:return float(np.log(hyp1f1(1,n,kappa)))
    return float(kappa+gammaln(n)-(n-1)*np.log(kappa)+np.log(gammainc(n-1,kappa)))


@lru_cache(maxsize=256)
def transition_coefficients_complex(n,kappa):
    if n<2 or np.isnan(kappa) or kappa<0:raise ValueError('Invalid transition parameter')
    if np.isposinf(kappa):return TransitionCoefficients(1.,0.,1.,float(kappa),n,1,True,0.)
    if kappa<50:
        a=float(hyp1f1(2,n+1,kappa)/(n*hyp1f1(1,n,kappa)))
    else:a=float(1-(n-1)*gammainc(n,kappa)/(kappa*gammainc(n-1,kappa)))
    b=(1-a)/(n-1)
    return TransitionCoefficients(a,b,a-b,float(kappa),n,1,True,0.)


def sample_complex_transition(h,kappa,rng=None,size=1):
    """Independent exact exp(kappa*|x^H h|^2) complex-line transitions."""
    h=np.asarray(h,dtype=complex)
    if h.ndim!=1 or len(h)<2 or not np.isfinite(h).all() or not np.isclose(np.vdot(h,h).real,1.,atol=1e-10,rtol=0):
        raise ValueError('h must be a finite unit complex vector')
    if size<1 or np.isnan(kappa) or kappa<0:raise ValueError('Invalid transition request')
    if rng is None:rng=np.random.default_rng()
    n=len(h)
    if np.isposinf(kappa):return np.broadcast_to(h,(size,n)).copy()
    if kappa<1e-5:
        # Beta proposal is stable all the way to zero. The tiny rejection
        # correction is essential: small positive kappa is not set to zero.
        radial=[]
        while len(radial)<size:
            r=rng.beta(1,n-1,size=max(2,size-len(radial)))
            radial.extend(r[np.log(rng.random(len(r)))<=kappa*(r-1)].tolist())
        r=np.array(radial[:size])
    else:
        probability=gammainc(n-1,kappa)
        y=gammaincinv(n-1,rng.random(size)*probability)/kappa
        r=1-y
    if np.any(r < -1e-12) or np.any(r>1+1e-12):raise ArithmeticError('Transition radial inversion left [0,1]')
    r=np.clip(r,0.,1.)
    z=(rng.normal(size=(size,n))+1j*rng.normal(size=(size,n)))/np.sqrt(2)
    z-=np.outer(z@h.conj(),h);norm=np.linalg.norm(z,axis=1)
    if np.any(norm==0):raise ArithmeticError('Degenerate orthogonal Gaussian draw')
    z/=norm[:,None]
    return np.sqrt(r)[:,None]*h+np.sqrt(1-r)[:,None]*z


def sample_complex_transitions(centres,kappa,rng=None):
    """One independent transition for each row, for predictive-mixture checks."""
    H=np.asarray(centres,complex)
    if H.ndim!=2 or H.shape[1]<2 or not np.allclose(np.sum(abs(H)**2,axis=1),1.,atol=1e-10,rtol=0):
        raise ValueError('Centres must be row-wise unit complex vectors')
    if rng is None:rng=np.random.default_rng()
    if np.isnan(kappa) or kappa<0:raise ValueError('kappa must be nonnegative')
    size,n=H.shape
    if np.isposinf(kappa):return H.copy()
    if kappa<1e-5:
        radial=[]
        while len(radial)<size:
            values=rng.beta(1,n-1,size=max(2,size-len(radial)))
            radial.extend(values[np.log(rng.random(len(values)))<=kappa*(values-1)].tolist())
        r=np.array(radial[:size])
    else:r=1-gammaincinv(n-1,rng.random(size)*gammainc(n-1,kappa))/kappa
    if np.any(r < -1e-12) or np.any(r>1+1e-12):raise ArithmeticError('Radial inversion left [0,1]')
    r=np.clip(r,0.,1.)
    Z=(rng.normal(size=(size,n))+1j*rng.normal(size=(size,n)))/np.sqrt(2)
    Z-=np.sum(Z*H.conj(),axis=1)[:,None]*H
    Z/=np.linalg.norm(Z,axis=1)[:,None]
    return np.sqrt(r)[:,None]*H+np.sqrt(1-r)[:,None]*Z


def sample_complex_laplace(A,draws=8192,rng=None,cut_locus_tolerance=.01):
    """Same-parameter local complex Gaussian mapped by the CP exponential.

    Complex tangent variances are 1/gap, or 1/(2*gap) per real coordinate.
    A singular cutoff is rejected; mass beyond pi/2 is flagged instead of
    clipping samples or pretending the local ellipsoid is a global HPD set.
    """
    start=perf_counter();A=_matrix(A)
    if rng is None:rng=np.random.default_rng()
    values,U=np.linalg.eigh(A);gaps=values[-1]-values[:-1]
    if draws<1 or np.min(gaps)<=1e-12:raise ArithmeticError('Singular complex Laplace cutoff')
    xi=(rng.normal(size=(draws,len(gaps)))+1j*rng.normal(size=(draws,len(gaps))))/np.sqrt(2*gaps)
    radius=np.linalg.norm(xi,axis=1)
    vectors=np.cos(radius)[:,None]*U[:,-1]+np.sinc(radius/np.pi)[:,None]*(xi@U[:,:-1].T)
    mass=float(np.mean(radius>=np.pi/2));se=float(np.sqrt(mass*(1-mass)/draws))
    local_ok=bool(mass+3*se+3/draws<=cut_locus_tolerance)
    return ComplexSamples(vectors,dict(numerical_ok=local_ok,local_approximation_ok=local_ok,
        cut_locus_mass=mass,cut_locus_mass_mcse=se,draws=draws,independent=True,
        seconds=perf_counter()-start,measure='complex tangent Gaussian mapped to CP',
        scope='Same-A Laplace shape control; not a passive observation-mean EKF or invariant-density HPD region'))


def _spectral_covariance(values,mean):
    """Covariance of simplex coordinates from complex spectral identities.

    For distinct eigenvalues E[x_i*x_j]=(m_i-m_j)/(lambda_i-lambda_j).
    Within an equal-eigenvalue block, conditional uniform Dirichlet coordinates
    give E[x_i^2]=2E[x_i*x_j]; row sums equal m_i. Near ties are grouped only
    for the Newton preconditioner; final fits use the checked forward mean.
    """
    values=np.asarray(values,float);m=np.asarray(mean,float);n=len(m)
    tolerance=1e-7*max(1.,float(np.ptp(values)))
    order=np.argsort(values);groups=[]
    for i in order:
        if not groups or abs(values[i]-values[groups[-1][0]])>tolerance:groups.append([int(i)])
        else:groups[-1].append(int(i))
    second=np.zeros((n,n))
    for g,first in enumerate(groups):
        for second_group in groups[g+1:]:
            value=(m[first].mean()-m[second_group].mean())/(values[first].mean()-values[second_group].mean())
            second[np.ix_(first,second_group)]=value
            second[np.ix_(second_group,first)]=value
    for group in groups:
        other=[j for j in range(n) if j not in group]
        common=(m[group].mean()-second[np.ix_(group,other)].sum()/len(group))/(len(group)+1)
        second[np.ix_(group,group)]=common
        second[group,group]=2*common
    return (second-np.outer(m,m)+second.T-np.outer(m,m))/2


def invert_complex_mean(M,initial=None,tol=1e-8,maxiter=60,max_order=4096):
    """Trace-gauged complex rank-one moment fit with a checked Newton solve."""
    begin=perf_counter();M=_matrix(M);n=len(M)
    target,U=np.linalg.eigh(M);target=target[::-1];U=U[:,::-1]
    if abs(target.sum()-1)>1e-9 or target.min()<=0 or target.max()>=1:
        raise ValueError('Finite rank-one moment must have trace one and eigenvalues inside (0,1)')
    initial_A=np.zeros_like(M) if initial is None else center(_matrix(initial))
    initial_forward=complex_moments(initial_A,tol=min(1e-12,tol*.01),max_order=max_order)
    objective0=float(initial_forward.log_normalizer-np.trace(initial_A@M).real)
    if np.max(abs(target-1/n))<tol*.01:
        return MeanFit(np.zeros_like(M),float(np.max(abs(target-1/n))),0.,objective0,0,1,
            perf_counter()-begin,True,0.,'uniform complex target')
    Q=null_space(np.ones((1,n)));evaluations=1
    def evaluate(x):
        nonlocal evaluations
        lam=Q@x;forward=complex_moments(np.diag(lam),tol=min(1e-12,tol*.01),max_order=max_order)
        m=forward.mean.diagonal().real;evaluations+=1
        return float(forward.log_normalizer-lam@target),Q.T@(m-target),Q.T@_spectral_covariance(lam,m)@Q,m,forward
    broad=n*(n+1)*(target-1/n);sharp=-1/target;sharp-=sharp.mean()
    fraction=np.clip((target[0]-1/n-.1)/.6,0.,1.)
    x=Q.T@((1-fraction)*broad+fraction*sharp);current=evaluate(x)
    warm=Q.T@np.diag(U.conj().T@initial_A@U).real;candidate=evaluate(warm)
    if candidate[0]<current[0]:x=warm;current=candidate
    iterations=0;fallbacks=0
    for iterations in range(maxiter+1):
        objective,gradient,H,m,forward=current
        if np.max(abs(m-target))+forward.moment_error_bound<=tol*.5:break
        if iterations==maxiter:break
        if np.linalg.eigvalsh(H)[0]>0:step=np.linalg.solve(H,gradient)
        else:step=gradient;fallbacks+=1
        accepted=False;rate=1.
        for _ in range(30):
            trial=x-rate*step;updated=evaluate(trial)
            if updated[0]<=objective-1e-4*rate*float(gradient@step)+2e-13:
                x=trial;current=updated;accepted=True;break
            rate*=.5
        if not accepted:break
    objective,gradient,H,m,forward=current;A=center((U*(Q@x))@U.conj().T)
    checked=complex_moments(A,tol=min(1e-12,tol*.01),max_order=max_order)
    residual=float(np.linalg.norm(checked.mean-M,2));objective=float(checked.log_normalizer-np.trace(A@M).real)
    ok=bool(residual+checked.moment_error_bound<=tol and objective<=objective0+5e-8)
    fit=MeanFit(A,residual,objective,objective0,int(iterations),int(evaluations),
        perf_counter()-begin,ok,checked.moment_error_bound,
        f'complex simplex covariance Newton; gradient fallbacks={fallbacks}')
    if not ok:raise ArithmeticError(f'Complex inverse unresolved: residual={residual:.4g}, moment={checked.moment_error_bound:.4g}, iterations={iterations}')
    return fit


def predict_complex(A,kappa,G=None,initial=None,tol=1e-8,maxiter=60):
    """Exact predicted first moment, approximate single-Bingham density."""
    begin=perf_counter();A=center(_matrix(A));n=len(A)
    G=np.eye(n) if G is None else np.asarray(G,complex)
    if G.shape!=(n,n) or not np.allclose(G.conj().T@G,np.eye(n),atol=1e-10,rtol=0):
        raise ValueError('G must be unitary')
    coefficients=transition_coefficients_complex(n,float(kappa));forward=complex_moments(A)
    rotated=center(G@A@G.conj().T)
    target=coefficients.rho*(G@forward.mean@G.conj().T)+(1-coefficients.rho)*np.eye(n)/n
    if np.isposinf(kappa):
        result=complex_moments(rotated);objective=float(result.log_normalizer-np.trace(rotated@target).real)
        fit=MeanFit(rotated,0.,objective,objective,0,0,perf_counter()-begin,True,
            result.moment_error_bound,'exact deterministic unitary transport')
        return ProcessPrediction(rotated,target,coefficients,fit,False)
    fit=invert_complex_mean(target,rotated if initial is None else initial,tol,maxiter)
    return ProcessPrediction(fit.A,target,coefficients,fit,bool(kappa>0))


def cp1_regions(A,mass=.95):
    """Exact complex CP1 ball and HPD-energy thresholds; uniform ties need a centre."""
    A=_matrix(A)
    if len(A)!=2 or not 0<mass<1:raise ValueError('Need complex dimension two and mass in (0,1)')
    gap=float(np.ptp(np.linalg.eigvalsh(A)))
    distance2=mass if gap==0 else float(-np.log1p(-mass*(-np.expm1(-gap)))/gap)
    return dict(ball_radius=np.sqrt(distance2),hpd_threshold=gap*distance2,
        uniform_tie=gap==0,exact_mass=mass)


def cp1_power_cdf(A,query,value,epsabs=1e-11):
    """Independent continuous CP1 CDF with a resolved radial boundary layer.

    Condition on r=|h^H h_mode|^2 and integrate the exact uniform relative-phase
    CDF. Its outer variable is the *normalized radial CDF*, so a sharp radial
    boundary layer cannot be silently missed by fixed physical coordinates.
    This is also the spherical vMF law on the Bloch sphere, not a real Watson
    law. Aligned queries use the analytic truncated-exponential CDF directly.
    """
    A=_matrix(A);w=np.asarray(query,complex)
    if len(A)!=2 or w.shape!=(2,) or not np.isfinite(w).all() or np.vdot(w,w).real<=0:
        raise ValueError('Need a nonzero complex two-vector query')
    norm=float(np.vdot(w,w).real);p=float(value)/norm
    if p<=0:return 0.
    if p>=1:return 1.
    values,U=np.linalg.eigh(A);gap=float(values[1]-values[0])
    if gap==0:return p
    overlap=float(np.clip(abs(np.vdot(U[:,1],w))**2/norm,0.,1.))
    denominator=float(-np.expm1(-gap))
    def radial_cdf(r):
        return float(np.exp(gap*(r-1))*(-np.expm1(-gap*r))/denominator)
    if overlap==1:return radial_cdf(p)
    if overlap==0:return float(-np.expm1(-gap*p)/denominator)
    def integrand(v):
        if gap<50:
            distance=-np.log1p((1-v)*np.expm1(-gap))/gap
        else:
            distance=-np.logaddexp(np.log(v),np.log1p(-v)-gap)/gap
        location=overlap+(1-2*overlap)*distance
        amplitude=2*np.sqrt(max(0.,overlap*(1-overlap)*distance*(1-distance)))
        if amplitude==0:return float(location<=p)
        z=(p-location)/amplitude
        if z<=-1:return 0.
        if z>=1:return 1.
        return float(1-np.arccos(z)/np.pi)
    roots=[(np.sqrt(overlap*p)+sign*np.sqrt((1-overlap)*(1-p)))**2 for sign in [-1,1]]
    points=sorted(set(v for r in roots if 0<r<1 for v in [radial_cdf(r)] if 0<v<1))
    integral,error=quad(integrand,0.,1.,points=points,epsabs=epsabs,
        epsrel=epsabs,limit=300)
    if error>max(20*epsabs,1e-9):raise ArithmeticError('Independent CP1 CDF unresolved')
    return float(integral)


def _batch_quantile_mcse(values,probabilities,batches=16):
    blocks=np.array_split(np.asarray(values),min(batches,len(values)//16))
    estimates=np.array([np.quantile(block,probabilities,axis=0).T for block in blocks])
    return np.std(estimates,axis=0,ddof=1)/np.sqrt(len(blocks))


def posterior_report(A,queries=None,truth=None,draws=8192,rng=None,seed=76131,
                     mass=.95,relative_mcse_tolerance=.025,return_samples=False,
                     max_proposals=None):
    """Native-complex mean, iid scalar quantiles and state-region diagnostics.

    ``draws`` iid vectors train thresholds and a second independent batch of the
    same size checks their masses. Scalar columns are .025,.25,.5,.75,.975.
    Reported MCSE is a batch estimate, not a deterministic integration bound.
    State coverage uses scoring truth only; geometry never changes inference.
    """
    begin=perf_counter();A=_matrix(A);n=len(A)
    if draws<256 or not 0<mass<1:raise ValueError('At least 256 iid draws and valid mass required')
    caller_rng=rng is not None
    if rng is None:rng=np.random.default_rng(seed)
    W=np.empty((0,n),complex) if queries is None else np.atleast_2d(np.asarray(queries,complex))
    if W.shape[1]!=n or not np.isfinite(W).all():raise ValueError('Queries must be finite row vectors')
    mean=complex_moments(A);values,U=np.linalg.eigh(A);maximum=float(values[-1])
    top=np.isclose(values,maximum,rtol=0,atol=1e-12*max(1.,float(np.ptp(values))))
    if np.count_nonzero(top)>1:
        auxiliary=rng.normal(size=np.count_nonzero(top))+1j*rng.normal(size=np.count_nonzero(top))
        auxiliary/=np.linalg.norm(auxiliary);mode_vector=U[:,top]@auxiliary
    else:mode_vector=U[:,-1]
    Pmode=np.outer(mode_vector,mode_vector.conj());uniform=bool(np.ptp(values)==0)
    start=perf_counter();train=sample_complex_bingham(A,draws,rng,max_proposals=max_proposals)
    check=sample_complex_bingham(A,draws,rng,max_proposals=max_proposals)
    sampling_seconds=perf_counter()-start
    start=perf_counter()
    train_distance=np.maximum(0.,1-abs(train.vectors@mode_vector.conj())**2)
    check_distance=np.maximum(0.,1-abs(check.vectors@mode_vector.conj())**2)
    train_loss=maximum-np.einsum('bi,ij,bj->b',train.vectors.conj(),A,train.vectors).real
    check_loss=maximum-np.einsum('bi,ij,bj->b',check.vectors.conj(),A,check.vectors).real
    if uniform:
        radius=float(mass**(1/(2*(n-1))));threshold=0.
    elif n==2:
        exact=cp1_regions(A,mass);radius=exact['ball_radius'];threshold=exact['hpd_threshold']
    else:
        radius=float(np.sqrt(np.quantile(train_distance,mass)));threshold=float(np.quantile(train_loss,mass))
    ball_mass=float(np.mean(check_distance<=radius*radius))
    hpd_mass=ball_mass if uniform else float(np.mean(check_loss<=threshold))
    state_error=max(abs(ball_mass-mass),abs(hpd_mass-mass))
    state_se=float(np.sqrt(mass*(1-mass)*2/draws))
    state_ok=bool(state_error<=5*state_se+2/draws)
    region_seconds=perf_counter()-start
    probabilities=np.array([.025,.25,.5,.75,.975]);start=perf_counter()
    if len(W):
        power=abs(train.vectors.conj()@W.T)**2
        check_power=abs(check.vectors.conj()@W.T)**2
        quantiles=np.quantile(power,probabilities,axis=0).T
        mcse=_batch_quantile_mcse(power,probabilities)
        checked_mass=np.mean(check_power[:,:,None]<=quantiles[None,:,:],axis=0)
        cdf_error=np.max(abs(checked_mass-probabilities),axis=0)
        cdf_se=np.sqrt(probabilities*(1-probabilities)*2/draws)
        widths=quantiles[:,-1]-quantiles[:,0]
        max_mcse_fraction=float(np.max(mcse/np.maximum(widths[:,None],1e-14)))
        tail_ok=bool(np.all(cdf_error<=5*cdf_se+2/draws) and max_mcse_fraction<=relative_mcse_tolerance)
    else:
        quantiles=np.empty((0,5));mcse=quantiles.copy();cdf_error=np.zeros(5)
        cdf_se=np.sqrt(probabilities*(1-probabilities)*2/draws);max_mcse_fraction=0.;tail_ok=True
    quantile_seconds=perf_counter()-start
    ball_covered=hpd_covered=None
    if truth is not None:
        truth=np.asarray(truth,complex)
        Ptrue=np.outer(truth,truth.conj()) if truth.ndim==1 else truth
        if Ptrue.shape!=(n,n):raise ValueError('Scoring truth must be a vector or projector')
        distance=max(0.,1-float(np.trace(Pmode@Ptrue).real))
        loss=maximum-float(np.trace(A@Ptrue).real)
        ball_covered=bool(distance<=radius*radius)
        hpd_covered=ball_covered if uniform else bool(loss<=threshold)
    risk=np.einsum('qi,ij,qj->q',W.conj(),mean.mean-mean.mean@mean.mean,W).real
    info=dict(mean_ok=mean.numerical_ok,sampling_ok=train.diagnostics['numerical_ok'] and check.diagnostics['numerical_ok'],
        state_ok=state_ok,tail_ok=tail_ok,numerical_ok=bool(mean.numerical_ok and state_ok and tail_ok),
        mean_error_bound=mean.moment_error_bound,mean_order=mean.order,
        iid_draws_per_batch=draws,total_iid_draws=2*draws,
        ball_independent_mass=ball_mass,hpd_independent_mass=hpd_mass,
        state_mass_error=state_error,state_mass_standard_error=state_se,
        scalar_cdf_error=cdf_error.tolist(),scalar_cdf_standard_error=cdf_se.tolist(),
        maximum_quantile_mcse_fraction_of_width=max_mcse_fraction,
        relative_mcse_tolerance=relative_mcse_tolerance,
        hpd_tie_kind='auxiliary Haar-centred ball within uniform-density ties' if uniform else 'continuous density-loss threshold',
        auxiliary_seed=seed if np.count_nonzero(top)>1 and not caller_rng else None,
        auxiliary_rng='caller-provided Generator' if np.count_nonzero(top)>1 and caller_rng else None,
        threshold_kind='analytic CP1' if n==2 else 'analytic uniform ball' if uniform else 'iid empirical quantile',
        error_scope='Independent iid mass checks and approximate batch quantile MCSE; mean truncation does not certify tails',
        mean_seconds=mean.seconds,sampling_seconds=sampling_seconds,region_seconds=region_seconds,
        quantile_seconds=quantile_seconds,total_seconds=perf_counter()-begin)
    output=dict(mean=mean.mean,mode=mode_vector,mode_vector=mode_vector,mode_projector=Pmode,
        log_normalizer=mean.log_normalizer,quadratic_mean=np.einsum('qi,ij,qj->q',W.conj(),mean.mean,W).real,
        quadratic_quantiles=quantiles,power_quantiles=quantiles,quantile_probabilities=probabilities,
        quantile_mcse=mcse,ball_radius=radius,hpd_threshold=threshold,
        ball_covered=ball_covered,hpd_covered=hpd_covered,risks=risk,
        hpd_constraint_matrix=Pmode if uniform else A,
        hpd_constraint_threshold=1-radius*radius if uniform else maximum-threshold,
        numerics=info,diagnostics=info)
    if return_samples:output['vectors']=train.vectors
    return output
