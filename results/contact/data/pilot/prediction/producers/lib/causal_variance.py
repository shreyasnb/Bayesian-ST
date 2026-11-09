"""Future-only variance learning from completed nonoverlapping blocks.

The streaming parameter receives each singleton increment exactly once.
Stationary-block EM uses one saved boundary prior in the endpoint frame.  Its
fitted posterior never replaces or augments the streaming parameter.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .geometry import center, mode
from .models import increment, constant
from .block_em import BlockData, fit_block_em, moments


@dataclass(frozen=True)
class UpdateRecord:
    time: int
    sigma2_used: float
    tau2_used: float | None
    sigma2_next: float
    tau2_next: float | None
    fit_performed: bool
    fit_numerical_ok: bool
    correction_seconds: float
    fitting_seconds: float
    state_change_from_fit: float


def align_completed_block(U, V, transforms, boundary_prior):
    """Express rows and their pre-block prior in the final measurement frame.

    transforms[j] maps the previous measurement frame to observation j.
    Therefore observation j is moved by G_last ... G_(j+1), whereas the
    preceding boundary prior is moved by G_last ... G_first.
    """
    U = np.asarray(U)
    G = np.asarray(transforms)
    if U.ndim != 2 or G.shape != (len(U), U.shape[1], U.shape[1]) or not len(U):
        raise ValueError('Expected U=(L,n) and transforms=(L,n,n)')
    if V is not None and np.asarray(V).shape != U.shape:
        raise ValueError('Recorded inputs must match the response block')
    n = U.shape[1]
    if np.array_equal(G,np.broadcast_to(np.eye(n),(len(U),n,n))):
        return U.copy(),None if V is None else np.asarray(V).copy(),center(boundary_prior.copy()),np.eye(n,dtype=U.dtype)
    product = np.eye(n, dtype=np.result_type(U, G, boundary_prior))
    aligned_u = np.empty_like(U, dtype=product.dtype)
    aligned_v = None if V is None else np.empty_like(np.asarray(V), dtype=product.dtype)
    for j in range(len(U)-1, -1, -1):
        aligned_u[j] = product @ U[j]
        if aligned_v is not None:
            aligned_v[j] = product @ np.asarray(V)[j]
        product = product @ G[j]
    return aligned_u, aligned_v, center(product @ boundary_prior @ product.conj().T), product


class CausalVarianceTracker:
    """Discounted conjugate tracker with future-only stationary-block fits.

    The EM prior is the transported saved pre-block belief, without an added
    likelihood or a replacement streaming state.  Its unweighted stationary
    block objective is deliberately distinct from discounted assimilation.
    Unknown physical drift is not removed by coordinate alignment.
    """
    def __init__(self, n, k, sigma2, tau2=None, *, beta=1., block_length=16,
                 em_iterations=8, starts=None, floor=1e-8, A0=None,
                 complex_data=False, learn=True, moment_tolerance=1e-9,
                 max_order=4096, ascent_tolerance=1e-7):
        if (not np.isfinite(sigma2) or not np.isfinite(floor)
                or not 0 < k < n or not 0 < beta <= 1 or sigma2 < floor
                or floor <= 0 or block_length < 1 or int(block_length) != block_length
                or em_iterations < 0 or int(em_iterations) != em_iterations):
            raise ValueError('Invalid rank, variance, memory, block length or iteration budget')
        if tau2 is not None and (not np.isfinite(tau2) or tau2 < 0):
            raise ValueError('Signal variance must be nonnegative')
        self.n, self.k = int(n), int(k)
        self.sigma2, self.tau2 = float(sigma2), None if tau2 is None else float(tau2)
        self.beta, self.block_length = float(beta), int(block_length)
        self.em_iterations, self.floor = int(em_iterations), float(floor)
        self.complex_data, self.learn = bool(complex_data), bool(learn)
        dtype = complex if self.complex_data else float
        initial = np.zeros((n,n), dtype=dtype) if A0 is None else np.asarray(A0)
        if (initial.shape != (n,n) or not np.isfinite(initial).all()
                or not np.allclose(initial, initial.conj().T, atol=1e-11)
                or (not self.complex_data and np.iscomplexobj(initial))):
            raise ValueError('Invalid initial parameter')
        self.A = center(np.array(initial, dtype=dtype, copy=True))
        self.t = 0
        self.starts = None if starts is None else tuple(tuple(x) for x in starts)
        self.moment_tolerance, self.max_order = float(moment_tolerance), int(max_order)
        self.ascent_tolerance = float(ascent_tolerance)
        self.fit_history, self.records = [], []
        self._U, self._V, self._G = [], [], []
        self._boundary = self.A.copy()
        self.last_fit_data = None

    def _transform(self, G):
        if G is None:
            return np.eye(self.n, dtype=self.A.dtype)
        G = np.asarray(G)
        if (G.shape != (self.n, self.n) or not np.isfinite(G).all()
                or not np.allclose(G.conj().T @ G, np.eye(self.n), atol=1e-10)
                or (not self.complex_data and np.iscomplexobj(G))):
            raise ValueError('A known transform must be orthogonal/unitary')
        return G

    def predict_parameter(self, G=None):
        """The pre-correction parameter; this does not alter state or fits."""
        if G is None:
            return center(self.beta*self.A)
        G = self._transform(G)
        return center(self.beta * G @ self.A @ G.conj().T)

    def posterior(self, G=None, *, predict=False):
        A = self.predict_parameter(G) if predict else self.A
        return moments(A, self.k, self.complex_data,
                       self.moment_tolerance, self.max_order)

    def logscore(self, u, v=None, G=None):
        """Full normalized density of an excluded next observation."""
        prior = self.predict_parameter(G)
        B = increment(np.asarray(u), self.sigma2, self.tau2,
                      None if v is None else np.asarray(v), self.complex_data)
        _, psi0, d0 = moments(prior, self.k, self.complex_data,
                             self.moment_tolerance, self.max_order)
        _, psi1, d1 = moments(prior+B, self.k, self.complex_data,
                             self.moment_tolerance, self.max_order)
        value = constant(np.asarray(u), self.k, self.sigma2, self.tau2,
                         None if v is None else np.asarray(v), self.complex_data)+psi1-psi0
        return float(value), bool(d0['numerical_ok'] and d1['numerical_ok'])

    @property
    def basis(self):
        return mode(self.A, self.k)

    def _fit_completed(self):
        U, V, prior, product = align_completed_block(
            np.asarray(self._U), None if self.tau2 is not None else np.asarray(self._V),
            np.asarray(self._G), self._boundary)
        data = BlockData(U, self.k, V=V, complex_data=self.complex_data,
                         boundary_prior=prior)
        self.last_fit_data = data
        starts = [(self.sigma2, self.tau2)]
        if self.starts is not None:
            starts.extend(self.starts)
        # A zero warm start need not escape a uniform boundary prior.  An
        # interior alternative is explicit and uses no hidden true variance.
        if self.tau2 is not None and self.tau2 == 0:
            starts.append((self.sigma2, max(self.floor, .1*self.sigma2)))
        unique = []
        for s, q in starts:
            q = None if self.tau2 is None else q
            value = (max(float(s), self.floor), None if q is None else max(0.,float(q)))
            if value not in unique:
                unique.append(value)
        histories, candidates = [], []
        for start_index, (s, q) in enumerate(unique):
            try:
                fit = fit_block_em(data, s, q, iterations=self.em_iterations,
                    floor=self.floor, tol=self.moment_tolerance, max_order=self.max_order,
                    ascent_tolerance=self.ascent_tolerance)
                ok = bool(fit.posterior.numerical_ok and np.isfinite(fit.objectives[-1]))
                histories.append(dict(start=start_index, sigma2_start=s, tau2_start=q,
                    sigma2=fit.s, tau2=fit.tau2, numerical_ok=ok, converged=fit.converged,
                    accepted=fit.accepted, rejected=fit.rejected,
                    log_evidence=fit.objectives[-1], iterations=fit.history,
                    moment_diagnostics=fit.posterior.diagnostics))
                if ok:
                    candidates.append((fit.objectives[-1], start_index, fit))
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
                histories.append(dict(start=start_index, sigma2_start=s, tau2_start=q,
                    numerical_ok=False, error=type(error).__name__+': '+str(error)))
        success = bool(candidates)
        selected = None
        if candidates:
            _, selected, fit = max(candidates, key=lambda x:x[0])
            self.sigma2, self.tau2 = float(fit.s), fit.tau2
        self.fit_history.append(dict(time=self.t, block_start=self.t-self.block_length+1,
            block_end=self.t, selected_start=selected, numerical_ok=success,
            sigma2_next=self.sigma2, tau2_next=self.tau2, candidates=histories,
            boundary_prior_count=1, recorded_observations=len(U),
            frame_alignment_error=float(np.linalg.norm(product.conj().T @ product-np.eye(self.n))),
            objective='Unweighted constant-state block from transported pre-block prior',
            streaming_state_replaced=False))
        return success

    def update(self, u, v=None, G=None):
        """Assimilate the present datum once, then fit a completed past block."""
        u = np.asarray(u)
        if (u.shape != (self.n,) or not np.isfinite(u).all()
                or (not self.complex_data and np.iscomplexobj(u))):
            raise ValueError('Invalid measured response')
        if (v is None) != (self.tau2 is not None):
            raise ValueError('Conditional tracker requires recorded v; marginal tracker requires tau2')
        if v is not None and (np.asarray(v).shape != u.shape or not np.isfinite(v).all()
                              or (not self.complex_data and np.iscomplexobj(v))):
            raise ValueError('Invalid recorded input')
        identity_transport=G is None
        G = self._transform(G)
        if not self._U:
            self._boundary = self.A.copy()
        used_s, used_q = self.sigma2, self.tau2
        start = perf_counter()
        prior=center(self.beta*self.A) if identity_transport else self.predict_parameter(G)
        self.A = center(prior+increment(
            u, used_s, used_q, None if v is None else np.asarray(v), self.complex_data))
        correction_seconds = perf_counter()-start
        self.t += 1
        self._U.append(u.copy())
        self._V.append(None if v is None else np.asarray(v).copy())
        self._G.append(G.copy())
        fit_performed = bool(self.learn and len(self._U) == self.block_length)
        ok, fitting_seconds, difference = True, 0., 0.
        if len(self._U) == self.block_length:
            if fit_performed:
                frozen_state = self.A.copy()
                start = perf_counter()
                ok = self._fit_completed()
                fitting_seconds = perf_counter()-start
                difference = float(np.linalg.norm(self.A-frozen_state))
            self._U, self._V, self._G = [], [], []
        record = UpdateRecord(self.t, used_s, used_q, self.sigma2, self.tau2,
            fit_performed, bool(ok), correction_seconds, fitting_seconds, difference)
        self.records.append(record)
        return record
