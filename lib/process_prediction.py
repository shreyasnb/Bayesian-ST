"""Moment-matched Bingham prediction for an explicit isotropic transition.

The transition is exp(kappa*tr(P C)) relative to invariant Grassmann measure.
Its first moment is exact; projection of the predicted mixture back to one
Bingham law is an assumed-density approximation. Only rank one and codimension
one have a deterministic inversion backend here. No general-rank calibration
claim or exact Bingham closure is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize, brentq
from scipy.special import i0e, i1e
from .geometry import center
from .spectral_series import rank_one_bingham


@dataclass(frozen=True)
class TransitionCoefficients:
    a: float
    b: float
    rho: float
    kappa: float
    n: int
    k: int
    complex_data: bool
    truncation_bound: float


@dataclass(frozen=True)
class MeanFit:
    A: np.ndarray
    residual: float
    objective: float
    initial_objective: float
    iterations: int
    evaluations: int
    seconds: float
    converged: bool
    moment_uncertainty: float
    message: str


@dataclass(frozen=True)
class ProcessPrediction:
    prior: np.ndarray
    target_mean: np.ndarray
    coefficients: TransitionCoefficients
    fit: MeanFit
    approximate_density: bool = True


def _supported(n,k):
    if not isinstance(k,(int,np.integer)) or not 0<k<n:
        raise ValueError('rank must satisfy 0 < k < n')
    if k not in (1,n-1):
        raise NotImplementedError('General-rank moment inversion is unresolved; rank/codimension one only')


def _matrix(A):
    A=np.asarray(A)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or len(A)<2:
        raise ValueError('Expected a square matrix of dimension at least two')
    if not np.isfinite(A).all() or not np.allclose(A,A.conj().T,atol=1e-12,rtol=1e-12):
        raise ValueError('Expected a finite symmetric/Hermitian matrix')
    return (A+A.conj().T)/2


def moment_and_normalizer(A,k=1,complex_data=False,tol=1e-12):
    """Return mean, log integral and truncation budget for the same backend.

    On real Gr(1,2), stable Bessel formulas are exact analytical expressions.
    Other supported dimensions use the existing adaptive positive series.
    """
    A=_matrix(A);n=len(A);_supported(n,k)
    complex_data=bool(complex_data or np.iscomplexobj(A))
    if n==2 and not complex_data:
        shift=float(np.trace(A)/2);H=A-shift*np.eye(2)
        r=float(np.hypot(H[0,0],H[0,1]))
        rho=float(i1e(r)/i0e(r))
        M=np.eye(2)/2+(rho/(2*r))*H if r else np.eye(2)/2
        return M,float(shift+r+np.log(i0e(r))),0.
    if k==1:
        r=rank_one_bingham(A,complex_data=complex_data,tol=tol)
        return r.mean,r.log_normalizer,r.moment_error_bound
    r=rank_one_bingham(-A,complex_data=complex_data,tol=tol)
    return np.eye(n)-r.mean,float(np.trace(A).real)+r.log_normalizer,r.moment_error_bound


def transition_coefficients(n,k,kappa,complex_data=False,tol=1e-12):
    """Determine rho from kappa, dimension, rank and invariant measure.

    Positive infinity denotes the deterministic zero-process-noise limit.
    No independently fitted contraction factor is accepted.
    """
    _supported(n,k)
    if np.isnan(kappa) or kappa<0:
        raise ValueError('kappa must be nonnegative, optionally positive infinity')
    if np.isposinf(kappa):
        return TransitionCoefficients(1.,0.,1.,float(kappa),n,k,bool(complex_data),0.)
    C=np.diag(np.r_[np.ones(k),np.zeros(n-k)])
    M,_,uncertainty=moment_and_normalizer(kappa*C,k,complex_data,tol)
    a=float(np.trace(M[:k,:k]).real/k)
    b=float(np.trace(M[k:,k:]).real/(n-k))
    rho=a-b
    if not -10*tol<=rho<=1+10*tol:
        raise ArithmeticError('Transition mean violates the contraction range')
    return TransitionCoefficients(a,b,float(rho),float(kappa),n,k,bool(complex_data),float(uncertainty))


def invert_bingham_mean(M,k=1,initial=None,complex_data=False,tol=1e-8,maxiter=300):
    """Solve the trace-gauged forward-KL projection, rejecting unresolved fits.

    Eigenvectors coincide with the target moment. Optimization is therefore
    over n-1 trace-zero spectral coordinates, warm-started from ``initial``.
    The objective and its gradient use the same moment/normalizer evaluator.
    No scalar concentration rescaling substitutes for the inverse moment map.
    """
    begin=perf_counter();M=_matrix(M);n=len(M);_supported(n,k)
    complex_data=bool(complex_data or np.iscomplexobj(M))
    if not np.isfinite(tol) or tol<=0 or maxiter<1:
        raise ValueError('positive tolerance and iteration budget required')
    values,U=np.linalg.eigh(M)
    if abs(values.sum()-k)>max(1e-10,tol/10):
        raise ValueError('target projector moment must have trace k')
    if values.min()<=0 or values.max()>=1:
        raise ValueError('a finite Bingham fit requires moment eigenvalues strictly inside (0,1)')
    A0=np.zeros((n,n),dtype=M.dtype) if initial is None else center(_matrix(initial))
    if A0.shape!=M.shape:raise ValueError('initial parameter dimension mismatch')
    backend_tol=min(1e-12,tol*.01)
    _,psi0,_=moment_and_normalizer(A0,k,complex_data,backend_tol)
    objective0=float(psi0-np.trace(A0@M).real)
    if np.max(abs(values-k/n))<tol*.01:
        A=np.zeros_like(M);residual=float(np.linalg.norm(M-k*np.eye(n)/n,2))
        return MeanFit(A,residual,0.,objective0,0,1,perf_counter()-begin,True,0.,'uniform target')
    if n==2 and not complex_data:
        # M=(I+rho*H/r)/2 and psi(H)=log I0(r), H trace zero.
        target=float(values[-1]-values[0]);upper=max(1.,1/(1-target))
        r,info=brentq(lambda x:float(i1e(x)/i0e(x))-target,0.,upper,
                      xtol=1e-12,rtol=1e-14,full_output=True)
        A=center((r/target)*(2*M-np.eye(2)))
        fitted,psi,uncertainty=moment_and_normalizer(A,k,False,backend_tol)
        objective=float(psi-np.trace(A@M).real)
        residual=float(np.linalg.norm(fitted-M,2))
        iterations=info.iterations;evaluations=info.function_calls;message='analytic circle inverse'
    else:
        Q=null_space(np.ones((1,n)))
        diagonal=np.diag(U.conj().T@A0@U).real
        start=Q.T@diagonal
        evaluations=0
        def objective_gradient(x):
            nonlocal evaluations
            evaluations+=1
            spectral=Q@x
            mean,psi,_=moment_and_normalizer(np.diag(spectral),k,complex_data,backend_tol)
            return float(psi-spectral@values),Q.T@(mean.diagonal().real-values)
        fit=minimize(objective_gradient,start,method='BFGS',jac=True,
                     options=dict(gtol=tol*.03,maxiter=maxiter))
        A=center((U*(Q@fit.x))@U.conj().T)
        fitted,psi,uncertainty=moment_and_normalizer(A,k,complex_data,backend_tol)
        objective=float(psi-np.trace(A@M).real)
        residual=float(np.linalg.norm(fitted-M,2))
        iterations=fit.nit;message=str(fit.message)
    converged=bool(residual+uncertainty<=tol and objective<=objective0+10*tol)
    result=MeanFit(A,residual,objective,objective0,int(iterations),int(evaluations),
                   perf_counter()-begin,converged,float(uncertainty),message)
    if not converged:
        raise ArithmeticError(f'Unresolved inverse moment fit: residual={residual:.3g}, '
                              f'uncertainty={uncertainty:.3g}; objective={objective:.8g}, start={objective0:.8g}')
    return result


def stochastic_predict(A,k,kappa,G=None,complex_data=False,initial=None,tol=1e-8):
    """Exact predicted first moment, followed by an approximate Bingham density.

    ``G`` is a nominal rotation supplied before the current observation. A
    causal learned rotation makes this a feedback state-model specification,
    not a deterministic-transformation instance of the tracking theorem.
    """
    A=center(_matrix(A));n=len(A);_supported(n,k)
    complex_data=bool(complex_data or np.iscomplexobj(A))
    G=np.eye(n) if G is None else np.asarray(G)
    if G.shape!=(n,n) or not np.isfinite(G).all() or not np.allclose(G.conj().T@G,np.eye(n),atol=1e-10):
        raise ValueError('nominal motion must be orthogonal/unitary')
    coefficients=transition_coefficients(n,k,kappa,complex_data)
    M,_,uncertainty=moment_and_normalizer(A,k,complex_data)
    rotated=center(G@A@G.conj().T)
    target=coefficients.rho*(G@M@G.conj().T)+(1-coefficients.rho)*k*np.eye(n)/n
    if np.isposinf(kappa):
        _,psi,_=moment_and_normalizer(rotated,k,complex_data)
        objective=float(psi-np.trace(rotated@target).real)
        fit=MeanFit(rotated,0.,objective,objective,0,1,0.,True,uncertainty,'deterministic transition limit')
        return ProcessPrediction(rotated,target,coefficients,fit,False)
    fit=invert_bingham_mean(target,k,rotated if initial is None else initial,complex_data,tol)
    return ProcessPrediction(fit.A,target,coefficients,fit,True)
