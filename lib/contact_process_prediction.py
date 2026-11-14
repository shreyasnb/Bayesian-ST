from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.special import dawsn, hyp1f1
from .contact_posterior import _matrix, _angular_geometry, _radial_rule


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
