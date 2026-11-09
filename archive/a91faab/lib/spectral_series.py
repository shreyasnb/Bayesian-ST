"""Positive spectral series for real/complex rank-one Bingham moments.

Appendices C, F and H of Cazzella et al., *On the Latent Information Geometry
of the Grassmann Manifold*, motivate evaluating a normalizer and differentiating
it. Here the rank-one specialization avoids general Jack polynomial enumeration:
squared coordinates have a Dirichlet(a,...,a) law, a=1/2 real or a=1 complex.

The finite approximation uses a smooth identity shift and includes its derivative.
Thus ``mean`` is the exact gradient of the finite-order approximate log normalizer
(at fixed order), is PSD, and has trace one without numerical renormalization.
Reported bounds control series truncation in exact arithmetic; they do not enclose
floating-point rounding. No Hessian/Fisher accuracy guarantee is supplied, and
no general-rank or unrestricted complex-to-real reduction is claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import gammaln, logsumexp


@dataclass(frozen=True)
class SpectralSeriesResult:
    mean: np.ndarray
    log_normalizer: float
    order: int
    relative_tail_bound: float
    moment_error_bound: float
    log_normalizer_error_bound: float
    boundary_mass: float
    trace_error: float
    complex_data: bool
    accuracy_met: bool
    shift: float
    backend: str = 'rank-one positive spectral series'

    @property
    def diagnostics(self):
        return dict(backend=self.backend,order=self.order,
                    relative_tail_bound=self.relative_tail_bound,
                    moment_error_bound=self.moment_error_bound,
                    log_normalizer_error_bound=self.log_normalizer_error_bound,
                    trace_error=self.trace_error,accuracy_met=self.accuracy_met,
                    complex_data=self.complex_data,
                    bound_scope='Series truncation in exact arithmetic; floating-point rounding excluded')


def _input(A, k, complex_data):
    A=np.asarray(A)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or len(A)<2:
        raise ValueError('A must be a square matrix of dimension at least two')
    if k!=1:
        raise ValueError('The spectral series backend supports rank one only')
    if not np.isfinite(A).all() or not np.allclose(A,A.conj().T,rtol=1e-12,atol=1e-12):
        raise ValueError('A must be finite and symmetric/Hermitian')
    return (A+A.conj().T)/2, bool(complex_data or np.iscomplexobj(A))


def rank_one_bingham(A, k=1, *, complex_data=False, tol=1e-10,
                     max_order=4096, min_order=0, fixed_order=None):
    """Return rank-one mean and normalized-invariant-measure log normalizer.

    ``tol`` is an absolute spectral mean-entry error budget and bounds the log
    normalizer error due to truncation. Orders adapt until a positive tail bound
    meets it; an exhausted budget raises ArithmeticError. ``fixed_order`` is a
    diagnostic option: it returns the requested polynomial and its bounds even
    if ``accuracy_met`` is false, useful for finite-difference derivative checks.

    If λ are eigenvalues, use c=-logsumexp(-λ) and x=λ-c>=0. With b=na,
    H(t)=prod_i(1-x_i*t)^(-a)=sum_m h_m t^m and
        F_M=sum_{m=0}^M h_m/(b)_m,    ψ_M=c+log(F_M).
    The derivative includes w_i*t_M/F_M, w=softmax(-λ), in addition to the
    ordinary polynomial derivative. This boundary correction is essential for
    trace one and identity-shift consistency. A fixed-order ψ_M need not have
    a positive Hessian; do not substitute its Hessian for Fisher information.
    """
    A,complex_data=_input(A,k,complex_data)
    if not np.isfinite(tol) or not 0<tol<.1:
        raise ValueError('tol must lie strictly between zero and 0.1')
    if not isinstance(max_order,(int,np.integer)) or max_order<1:
        raise ValueError('max_order must be a positive integer')
    if not isinstance(min_order,(int,np.integer)) or not 0<=min_order<=max_order:
        raise ValueError('min_order must be between zero and max_order')
    if fixed_order is not None:
        if not isinstance(fixed_order,(int,np.integer)) or not 0<=fixed_order<=max_order:
            raise ValueError('fixed_order must be between zero and max_order')
        limit=int(fixed_order)
    else:
        limit=int(max_order)
    n=len(A);a=1. if complex_data else .5;b=n*a
    identity_shift=float(np.trace(A).real/n)
    lam,U=np.linalg.eigh(A-identity_shift*np.eye(n))
    if np.ptp(lam)==0 and fixed_order is None:
        # An exact uniform target has exact moments and normalizer.
        return SpectralSeriesResult(np.eye(n,dtype=A.dtype)/n,
            float(identity_shift+lam[0]),0,0.,0.,0.,0.,0.,complex_data,True,
            float(identity_shift+lam[0]))
    c=-float(logsumexp(-lam))
    x=lam-c;R=float(np.max(x));y=x/R
    # Softmax is stable even at large concentration and respects tied spectra.
    w=np.exp(-lam-logsumexp(-lam))
    logR=np.log(R);log_a=np.log(a)
    orders=np.arange(limit+2,dtype=float)
    log_poch=gammaln(b+orders)-gammaln(b)
    log_factorial=gammaln(orders+1)
    # Positive normalized coefficients avoid cancellation. Direct coefficients
    # are bounded by (b)_m/m!; switch to log convolution before overflow risk.
    use_logs=bool(log_poch[min(limit,len(log_poch)-1)]-log_factorial[min(limit,len(log_factorial)-1)]>650)
    h=np.zeros(limit+1);h[0]=1.
    log_h=np.full(limit+1,-np.inf);log_h[0]=0.
    powers=np.ones(n);power_sums=np.zeros(limit+1)
    g=np.ones(n);log_g=np.zeros(n)
    log_D=np.full(n,-np.inf);log_F=0.;log_last=0.
    relative_tail=np.inf;boundary=1.;moment_bound=np.inf
    best_order=0
    if fixed_order==0:
        # ψ_0=c; its exact derivative is the soft-min weight vector.
        mean=(U*w)@U.conj().T
        return SpectralSeriesResult((mean+mean.conj().T)/2,float(identity_shift+c),
            0,np.expm1(R),np.expm1(R)+1.,R,1.,float(abs(np.trace(mean).real-1)),
            complex_data,False,float(identity_shift+c))
    for m in range(1,limit+1):
        # D_i uses g_{m-1,i}, before the recurrence advances g to degree m.
        log_derivative=log_a+(m-1)*logR+log_g-log_poch[m]
        log_D=np.logaddexp(log_D,log_derivative)
        powers*=y;power_sums[m]=powers.sum()
        if use_logs:
            log_h[m]=log_a+logsumexp(np.log(power_sums[1:m+1])+log_h[m-1::-1])-np.log(m)
        else:
            h[m]=(a/m)*np.dot(power_sums[1:m+1],h[m-1::-1])
            if not np.isfinite(h[m]) or h[m]<=0:
                raise ArithmeticError('Positive coefficient recurrence lost floating-point range')
            log_h[m]=np.log(h[m])
        log_last=log_h[m]+m*logR-log_poch[m]
        log_F=np.logaddexp(log_F,log_last)
        log_g=np.logaddexp(log_h[m],np.log(y,where=y>0,out=np.full(n,-np.inf))+log_g)
        boundary=float(np.exp(log_last-log_F))
        if m+2>R:
            # Each later Taylor-term ratio is no greater than R/(m+2).
            log_tail=(m+1)*logR-log_factorial[m+1]-np.log1p(-R/(m+2))
            log_ratio=log_tail-log_F
            relative_tail=float(np.exp(log_ratio)) if log_ratio<709 else np.inf
            moment_bound=relative_tail+boundary
        best_order=m
        if fixed_order is None and m>=min_order and moment_bound<=tol:
            break
    accuracy_met=bool(moment_bound<=tol)
    if fixed_order is None and not accuracy_met:
        raise ArithmeticError(f'Series accuracy budget unresolved at order {best_order}: '
                              f'mean truncation bound {moment_bound:.3g}; increase max_order')
    spectral_mean=np.exp(log_D-log_F)+w*boundary
    mean=(U*spectral_mean)@U.conj().T
    mean=(mean+mean.conj().T)/2
    return SpectralSeriesResult(mean,float(identity_shift+c+log_F),best_order,
        relative_tail,moment_bound,float(np.log1p(relative_tail)),boundary,
        float(abs(np.trace(mean).real-1)),complex_data,accuracy_met,float(identity_shift+c))
