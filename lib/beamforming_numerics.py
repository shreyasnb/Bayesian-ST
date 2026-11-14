from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .spectral_series import rank_one_bingham

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
