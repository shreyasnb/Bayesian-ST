"""Efficient real Gr(2,3) specialization of the existing Bingham predictor.

The transition first moment is exact conditional on the maintained preceding
Bingham law. Its forward-KL projection back into one Bingham density remains an
assumed-density approximation. Analytic radial integrals and periodic angular
quadrature avoid extending the positive Taylor series to contact concentrations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from scipy.special import dawsn, hyp1f1, erf

from .contact_posterior import _matrix, _angular_geometry, _radial_rule
from .geometry import center
from .process_prediction import TransitionCoefficients, MeanFit, ProcessPrediction


@dataclass(frozen=True)
class KernelParameters:
    """Normal-line coefficients; plane coefficients are supplied separately."""
    kappa: float
    a: float
    b: float
    rho: float
    chordal_rms: float
    angle_variance: float
    integration_error: float


def _watson_integrals(kappa,order):
    q,w=_radial_rule(order)
    if kappa>96:
        y=48*q
        x=np.sqrt(1-y/kappa)
        weights=48*w*np.exp(-y)/(2*kappa*x)
    else:
        x=q;weights=w*np.exp(kappa*(x*x-1))
    weights=weights/weights.sum()
    return float(weights@(x*x)),float(weights@(np.arccos(x)**2)/2)


@lru_cache(maxsize=128)
def kernel_parameters(kappa):
    """Exact-kernel moment and angular second moment, with resolution checks.

    A normal-line draw has q=|n dot m| on [0,1] with density proportional to
    exp(kappa*q*q). The angular variance is E[acos(q)^2]/2, not the small-angle
    surrogate E[sin(theta)^2]/2. Positive infinity is deterministic transport.
    """
    kappa=float(kappa)
    if np.isnan(kappa) or kappa<0:raise ValueError('kappa must be nonnegative')
    if np.isposinf(kappa):return KernelParameters(kappa,1.,0.,1.,0.,0.,0.)
    if kappa==0:
        # Integral_0^1 acos(q)^2 dq = pi-2.
        return KernelParameters(0.,1/3,1/3,0.,np.sqrt(2/3),(np.pi-2)/2,0.)
    if kappa<.1:
        a=float(hyp1f1(1.5,2.5,kappa)/(3*hyp1f1(.5,1.5,kappa)))
    else:
        root=np.sqrt(kappa)
        a=float(1/(2*root*dawsn(root))-1/(2*kappa))
    low,varlow=_watson_integrals(kappa,64)
    high,varhigh=_watson_integrals(kappa,128)
    error=max(abs(low-high),abs(varlow-varhigh),abs(high-a))
    if error>2e-9:raise ArithmeticError(f'Kernel integration unresolved: {error:.3g}')
    b=(1-a)/2;rho=a-b
    return KernelParameters(kappa,a,b,float(rho),float(np.sqrt(max(0.,1-a))),varhigh,float(error))


def kernel_tangent_variance(kappa):
    return kernel_parameters(kappa).angle_variance


@lru_cache(maxsize=128)
def kappa_from_chordal_rms(rms):
    """Invert E[1-(n dot m)^2] exactly; zero denotes positive infinity."""
    rms=float(rms)
    maximum=np.sqrt(2/3)
    if not np.isfinite(rms) or not 0<=rms<=maximum:
        raise ValueError('rms must lie in [0,sqrt(2/3)]')
    if rms==0:return np.inf
    if rms==maximum:return 0.
    target=rms*rms
    def difference(k):
        if k==0:return 2/3-target
        if k<.1:a=float(hyp1f1(1.5,2.5,k)/(3*hyp1f1(.5,1.5,k)))
        else:
            root=np.sqrt(k);a=float(1/(2*root*dawsn(root))-1/(2*k))
        return 1-a-target
    upper=max(2.,2/target)
    while difference(upper)>0:upper*=2
    return float(brentq(difference,0.,upper,xtol=1e-10,rtol=2e-13))


def _radial(g):
    """I_m = integral (1-q^2)^m exp[-g(1-q^2)] dq, m=0,1,2."""
    g=np.asarray(g,dtype=float)
    root=np.sqrt(g)
    r0=np.divide(dawsn(root),root,out=np.ones_like(root),where=root>0)
    # Negative-argument hypergeometric integrals stay positive and avoid
    # subtracting almost equal q^0/q^2/q^4 moments at large concentration.
    r1=(2/3)*hyp1f1(2.,2.5,-g)
    r2=(8/15)*hyp1f1(3.,3.5,-g)
    return r0,r1,r2


def _spectral(gaps,order=64,covariance=True):
    gaps=np.asarray(gaps,dtype=float)
    cosine,sine,g,weights=_angular_geometry(gaps,order)
    r0,r1,r2=_radial(g)
    Z=float(weights@r0)
    if not np.isfinite(Z) or Z<=0:raise ArithmeticError('Nonpositive angular normalizer')
    c=cosine*cosine;s=sine*sine
    mu=np.array([weights@(r0-r1),weights@(c*r1),weights@(s*r1)])/Z
    # The three exact sufficient statistics sum to one.
    mu[0]=1-mu[1]-mu[2]
    if gaps[0]<1e-11:mu[:2]=mu[:2].mean()
    if abs(gaps[1]-gaps[0])<1e-11:mu[1:]=mu[1:].mean()
    cov=None
    if covariance:
        second=np.array([[weights@(c*c*r2),weights@(c*s*r2)],
                         [weights@(c*s*r2),weights@(s*s*r2)]])/Z
        cov=(second-np.outer(mu[1:],mu[1:]));cov=(cov+cov.T)/2
    return mu,float(np.log(Z)),cov


@dataclass(frozen=True)
class FastMoments:
    mean: np.ndarray
    log_normalizer: float
    numerical_ok: bool
    resolution_error: float
    logz_resolution_error: float
    seconds: float
    normal_eigenvalues: np.ndarray
    normal_basis: np.ndarray
    angular_order: int
    resolution_checked: bool


def fast_moments(A,checked=False,angular_order=64):
    """Plane mean and normalized log integral from exact radial moments.

    Unchecked per-step evaluations use a development-validated fixed angular
    budget. ``checked=True`` doubles that budget and records the discrepancy;
    unperformed numerical checks are NaN rather than a claimed zero error.
    """
    start=perf_counter();A=_matrix(A)
    values,U=np.linalg.eigh(-A);values=values[::-1];U=U[:,::-1]
    gaps=np.maximum(0.,values[0]-values[1:])
    # The nearly uniform ridge needs a finer angular grid than a pole. This
    # depends on the parameter, never on a hidden state or coverage outcome.
    if gaps[0]<1. and gaps[1]>100:angular_order=max(128,angular_order)
    mu,logz,_=_spectral(gaps,angular_order,False)
    error=logerror=np.nan;order=angular_order
    if checked:
        refined,logref,_=_spectral(gaps,2*angular_order,False)
        error=float(np.max(abs(mu-refined)));logerror=float(abs(logz-logref))
        mu,logz,order=refined,logref,2*angular_order
    M=np.eye(3)-(U*mu)@U.T
    ok=bool(np.isfinite(M).all() and np.isfinite(logz) and np.min(mu)>=-1e-12
            and (not checked or (error<5e-8 and logerror<5e-8)))
    return FastMoments((M+M.T)/2,float(np.trace(A)+values[0]+logz),ok,error,logerror,
                       perf_counter()-start,mu,U,order,bool(checked))


def _target_eigenbasis(M):
    values,U=np.linalg.eigh(np.eye(3)-M)
    return values[::-1],U[:,::-1]


def invert_contact_mean(M,initial=None,tol=1e-8,maxiter=30,angular_order=64,max_angular_order=256):
    """Trace-zero two-variable inverse moment map; unresolved fits raise.

    The Newton Hessian is the covariance of squared normal coordinates under
    the same normalized angular integration used for the objective gradient.
    Initial gap estimates are only warm starts; no rescaling is accepted in
    place of an evaluated inverse-moment residual. An unresolved doubled
    angular check triggers a bounded refit at a finer angular order. Every
    refit recomputes the objective, gradient and Hessian at that order; the
    tolerance remains unchanged and all attempted work is timed.
    """
    start=perf_counter();M=_matrix(M)
    if not 0<tol<.1 or maxiter<1:raise ValueError('Invalid inverse fit budget')
    if not isinstance(angular_order,(int,np.integer)) or not isinstance(max_angular_order,(int,np.integer)) or not 16<=angular_order<=max_angular_order:
        raise ValueError('Invalid bounded angular fit budget')
    target,U=_target_eigenbasis(M)
    if abs(target.sum()-1)>1e-9 or target.min()<=0 or target.max()>=1:
        raise ValueError('Moment must have trace two and eigenvalues strictly between zero and one')
    initial_A=np.zeros((3,3)) if initial is None else center(_matrix(initial))
    # Initial and final objective values require resolved normalizers. A
    # coarse initial value must not veto a valid higher-resolution fit.
    initial_evaluation=fast_moments(initial_A,checked=True,angular_order=angular_order)
    if not initial_evaluation.numerical_ok and angular_order<max_angular_order:
        initial_evaluation=fast_moments(initial_A,checked=True,angular_order=max_angular_order)
    if not initial_evaluation.numerical_ok:
        raise ArithmeticError('Initial inverse objective unresolved at bounded angular budget')
    initial_objective=float(initial_evaluation.log_normalizer-np.trace(initial_A@M))
    def checked_analytic(A):
        order=angular_order;attempts=0
        while True:
            evaluated=fast_moments(A,checked=True,angular_order=order)
            residual=float(np.max(abs(evaluated.normal_eigenvalues-target)))
            passed=bool(evaluated.numerical_ok and residual+evaluated.resolution_error<=tol)
            if passed or order>=max_angular_order:return evaluated,residual,passed,attempts,order
            order=min(2*order,max_angular_order);attempts+=1
    if np.max(abs(target-1/3))<tol*.01:
        return MeanFit(np.zeros((3,3)),float(np.max(abs(target-1/3))),0.,initial_objective,0,1,
                       perf_counter()-start,True,0.,'uniform target; no inversion')
    if abs(target[1]-target[2])<1e-12:
        # An axisymmetric pole is exactly the Watson transition family.
        kappa=kappa_from_chordal_rms(float(np.sqrt(1-target[0])))
        diagonal=np.array([0.,-kappa,-kappa])
        A=center(-(U*diagonal)@U.T);evaluated,residual,passed,refinements,final_order=checked_analytic(A)
        objective=float(evaluated.log_normalizer-np.trace(A@M))
        if not passed:raise ArithmeticError('Axisymmetric pole inverse unresolved')
        return MeanFit(A,residual,objective,initial_objective,0,refinements+1,perf_counter()-start,
                       True,evaluated.resolution_error,f'analytic Watson pole inverse; angular checks through {final_order}')
    if abs(target[0]-target[1])<1e-12:
        def bottom(b):
            if b<.01:
                return float(hyp1f1(1.5,2.5,-b)/(3*hyp1f1(.5,1.5,-b)))
            root=np.sqrt(b)
            return float(1/(2*b)-np.exp(-b)/(np.sqrt(np.pi)*root*erf(root)))
        upper=max(2.,1/target[2])
        b=brentq(lambda x:bottom(x)-target[2],0.,upper,xtol=1e-10,rtol=2e-13)
        diagonal=np.array([0.,0.,-b]);A=center(-(U*diagonal)@U.T)
        evaluated,residual,passed,refinements,final_order=checked_analytic(A)
        objective=float(evaluated.log_normalizer-np.trace(A@M))
        if not passed:raise ArithmeticError('Axisymmetric ridge inverse unresolved')
        return MeanFit(A,residual,objective,initial_objective,0,refinements+1,perf_counter()-start,
                       True,evaluated.resolution_error,f'analytic truncated-Gaussian ridge inverse; angular checks through {final_order}')
    broad=7.5*(target[0]-target[1:])
    concentrated=.5*(1/target[1:]-1/target[0])
    blend=np.clip((target[0]-.4)/.4,0.,1.)
    gaps=np.maximum(0.,(1-blend)*broad+blend*concentrated)
    if gaps[0]<1. and gaps[1]>100:angular_order=min(max_angular_order,max(128,angular_order))
    evaluations=1
    def evaluate(x):
        nonlocal evaluations
        evaluations+=1
        mean,logz,cov=_spectral(x,angular_order)
        objective=float(logz+x@target[1:])
        gradient=target[1:]-mean[1:]
        return objective,gradient,cov,mean
    if initial is not None:
        C=U.T@(-initial_A)@U;candidate=np.maximum(0.,C[0,0]-np.diag(C)[1:])
        # Harmonic gap correction is an initialization only, useful after a
        # finite process step substantially contracts a very sharp posterior.
        if np.max(abs(initial_evaluation.normal_eigenvalues-target))<1e-5:
            if evaluate(candidate)[0]<evaluate(gaps)[0]:gaps=candidate
    objective,gradient,H,mean=evaluate(gaps);iterations=0
    for iterations in range(maxiter+1):
        residual=float(np.max(abs(mean-target)))
        if residual<=tol*.2:break
        if iterations==maxiter:break
        eigen=np.linalg.eigvalsh(H)
        if eigen[0]<=0 or not np.isfinite(H).all():
            raise ArithmeticError('Inverse Hessian covariance unresolved')
        step=np.linalg.solve(H,gradient)
        rate=1.
        accepted=False
        for _ in range(24):
            proposal=gaps-rate*step
            if np.min(proposal)<-1e-10 or proposal[1]<proposal[0]-1e-10:
                rate*=.5;continue
            proposal=np.maximum(proposal,0.)
            new=evaluate(proposal)
            if new[0]<=objective-1e-4*rate*float(gradient@step)+2e-14:
                gaps=proposal;objective,gradient,H,mean=new;accepted=True;break
            rate*=.5
        if not accepted:break
    refined,ref_logz,_=_spectral(gaps,2*angular_order,False)
    uncertainty=float(np.max(abs(refined-mean)))
    residual=float(np.max(abs(refined-target)))
    diagonal=np.r_[0.,-gaps]
    A=center(-(U*diagonal)@U.T)
    final_objective=float(ref_logz+gaps@target[1:])
    converged=bool(residual+uncertainty<=tol and final_objective<=initial_objective+5e-8)
    fit=MeanFit(A,residual,final_objective,initial_objective,int(iterations),int(evaluations),
                perf_counter()-start,converged,uncertainty,'two-variable covariance Newton; doubled angular check')
    if not converged:
        if uncertainty>tol*.05 and angular_order<max_angular_order:
            next_order=min(2*angular_order,max_angular_order)
            refined_fit=invert_contact_mean(M,initial=A,tol=tol,maxiter=maxiter,
                angular_order=next_order,max_angular_order=max_angular_order)
            # Compare objectives at resolved budgets. The original coarse
            # initial normalizer may itself be biased near a narrow ridge.
            resolved_initial=fast_moments(initial_A,checked=True,angular_order=max_angular_order)
            if not resolved_initial.numerical_ok:
                raise ArithmeticError('Original inverse objective unresolved after angular refinement')
            resolved_initial_objective=float(resolved_initial.log_normalizer-np.trace(initial_A@M))
            if refined_fit.objective>resolved_initial_objective+5e-8:
                raise ArithmeticError('Angular-refined inverse did not improve the resolved original objective')
            return replace(refined_fit,initial_objective=resolved_initial_objective,
                iterations=fit.iterations+refined_fit.iterations,
                evaluations=fit.evaluations+refined_fit.evaluations,
                seconds=perf_counter()-start,
                message=(f'angular refinement {angular_order}->{next_order}; '
                    f'initial residual={residual:.6g}, backend={uncertainty:.6g}; '+refined_fit.message))
        raise ArithmeticError(f'Contact inverse unresolved: residual={residual:.3g}, backend={uncertainty:.3g}, iterations={iterations}, angular_order={angular_order}, max_angular_order={max_angular_order}')
    return fit


def predict_contact(A,kappa,G=None,previous_mean=None,initial=None,tol=1e-8):
    """Exact transition mean and approximate Bingham density on real Gr(2,3)."""
    start=perf_counter();A=center(_matrix(A))
    G=np.eye(3) if G is None else np.asarray(G,dtype=float)
    if G.shape!=(3,3) or not np.allclose(G.T@G,np.eye(3),atol=1e-10,rtol=0):
        raise ValueError('Nominal motion must be orthogonal')
    kernel=kernel_parameters(kappa)
    # Plane directions have diagonal coefficient 1-b_normal and complement
    # coefficient 1-a_normal; their difference is the same rho.
    coefficients=TransitionCoefficients(1-kernel.b,1-kernel.a,kernel.rho,float(kappa),3,2,False,kernel.integration_error)
    rotated=center(G@A@G.T)
    if np.isposinf(kappa):
        M=fast_moments(A).mean if previous_mean is None else _matrix(previous_mean)
        target=G@M@G.T
        objective=float(fast_moments(rotated).log_normalizer-np.trace(rotated@target))
        fit=MeanFit(rotated,0.,objective,objective,0,0,perf_counter()-start,True,0.,'exact deterministic transport; no inversion')
        return ProcessPrediction(rotated,target,coefficients,fit,False)
    M=fast_moments(A).mean if previous_mean is None else _matrix(previous_mean)
    target=kernel.rho*(G@M@G.T)+(1-kernel.rho)*(2/3)*np.eye(3)
    fit=invert_contact_mean(target,rotated if initial is None else initial,tol=tol)
    return ProcessPrediction(fit.A,target,coefficients,fit,bool(kappa>0))


def predictive_log_score(A,u,v,s):
    """Proper pre-correction Gaussian score; generalized-posterior eta is absent."""
    from .models import increment,constant
    B=increment(np.asarray(u),s,v=np.asarray(v))
    return float(constant(np.asarray(u),2,s,v=np.asarray(v))+
                 fast_moments(A+B).log_normalizer-fast_moments(A).log_normalizer)
