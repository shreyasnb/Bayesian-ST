"""Stationary-block EM with one fixed boundary Bingham prior.

Every E-step rebuilds the posterior from the same boundary prior and the whole
block.  This is a different objective from ``lib.em``'s sum over independently
recorded one-observation priors.  Noise conventions are per-real-coordinate
variance or proper-complex covariance, as in ``lib.models``.
"""
from dataclasses import dataclass
import numpy as np

from .geometry import center
from .models import constant, increment
from .spectral_series import rank_one_bingham


@dataclass(frozen=True)
class BlockData:
    U: np.ndarray
    k: int
    V: np.ndarray | None = None
    complex_data: bool = False
    boundary_prior: np.ndarray | None = None

    def __post_init__(self):
        dtype = complex if self.complex_data else float
        if not self.complex_data and (np.iscomplexobj(self.U) or np.iscomplexobj(self.V)):
            raise ValueError('Complex observations require complex_data=True')
        U = np.array(self.U, dtype=dtype, copy=True)
        if U.ndim != 2 or not len(U) or U.shape[1] < 2 or not np.isfinite(U).all():
            raise ValueError('U must contain nonempty row-wise finite observations')
        n = U.shape[1]
        if not isinstance(self.k, (int, np.integer)) or not 0 < self.k < n:
            raise ValueError('Require 0 < k < N')
        if self.k not in (1, n - 1):
            raise ValueError('Block EM backend supports rank one and codimension one')
        V = None if self.V is None else np.array(self.V, dtype=dtype, copy=True)
        if V is not None and (V.shape != U.shape or not np.isfinite(V).all()):
            raise ValueError('V must have the same shape as U')
        if not self.complex_data and np.iscomplexobj(self.boundary_prior):
            raise ValueError('A real model requires a real boundary prior')
        A = (np.zeros((n, n), dtype=dtype) if self.boundary_prior is None
             else np.array(self.boundary_prior, dtype=dtype, copy=True))
        if (A.shape != (n, n) or not np.isfinite(A).all()
                or not np.allclose(A, A.conj().T, atol=1e-12, rtol=1e-12)):
            raise ValueError('Boundary prior must be finite and Hermitian')
        if not self.complex_data and np.iscomplexobj(A):
            raise ValueError('A real model requires a real boundary prior')
        for value in (U, V, A):
            if value is not None:
                value.setflags(write=False)
        object.__setattr__(self, 'U', U)
        object.__setattr__(self, 'V', V)
        object.__setattr__(self, 'boundary_prior', A)

    @property
    def n(self):
        return self.U.shape[1]

    @property
    def conditional(self):
        return self.V is not None


@dataclass(frozen=True)
class BlockPosterior:
    A: np.ndarray
    mean: np.ndarray
    log_evidence: float
    numerical_ok: bool
    diagnostics: dict


@dataclass(frozen=True)
class BlockFit:
    s: float
    tau2: float | None
    objectives: list
    history: list
    accepted: int
    rejected: int
    posterior: BlockPosterior
    converged: bool


def moments(A, k, complex_data=False, tol=1e-10, max_order=4096):
    """Mean and normalized log integral, restoring identity shifts by duality."""
    A = np.asarray(A)
    n = len(A)
    if k not in (1, n - 1):
        raise ValueError('Unsupported rank')
    dual = k != 1
    target = -A if dual else A
    try:
        result = rank_one_bingham(target, complex_data=complex_data, tol=tol, max_order=max_order)
        mean = result.mean
        logz = result.log_normalizer
        diag = dict(result.diagnostics)
        ok = result.accuracy_met
    except ArithmeticError:
        # Independent deterministic contact quadrature resolves large real
        # three-dimensional concentrations without silently relaxing a cap.
        if complex_data or n != 3:
            raise
        from .contact_posterior import normal_quadrature
        low = normal_quadrature(target, 64, 192)
        high = normal_quadrature(target, 96, 384)
        error = max(np.linalg.norm(high['mean'] - low['mean'], 2),
                    abs(high['log_normalizer'] - low['log_normalizer']))
        ok = bool(error < max(100 * tol, 1e-8)
                  and high['omitted_radial_relative_bound'] < max(tol, 1e-12))
        mean, logz = high['mean'], high['log_normalizer']
        diag = dict(backend='resolution-checked real sphere quadrature',
                    resolution_error=float(error), accuracy_met=ok,
                    bound_scope='Resolution comparison, not a rigorous error enclosure')
    if dual:
        mean = np.eye(n) - mean
        logz = np.trace(A).real + logz
    diag.update(numerical_ok=bool(ok), rank=int(k), complex_data=bool(complex_data),
                complement_duality=bool(dual))
    return mean, float(logz), diag


def block_posterior(data, s, tau2=None, *, tol=1e-10, max_order=4096):
    """Evaluate one constant-state block under one boundary prior.

    The returned parameter is trace zero for subsequent inference.  Evidence
    uses the uncentered parameter; its discarded scalar is recorded explicitly.
    No data are assimilated into ``data.boundary_prior``.
    """
    if not isinstance(data, BlockData):
        raise TypeError('Use BlockData')
    if data.conditional and tau2 is not None:
        raise ValueError('Conditional blocks have no latent signal variance')
    A = np.array(data.boundary_prior, copy=True)
    constants = 0.
    for j, u in enumerate(data.U):
        v = None if data.V is None else data.V[j]
        A += increment(u, s, tau2, v, data.complex_data)
        constants += constant(u, data.k, s, tau2, v, data.complex_data)
    M, psi, diagnostics = moments(A, data.k, data.complex_data, tol, max_order)
    _, psi0, prior_diagnostics = moments(data.boundary_prior, data.k, data.complex_data, tol, max_order)
    evidence = float(constants + psi - psi0)
    diagnostics.update(boundary_log_normalizer=psi0, posterior_log_normalizer=psi,
        singleton_constants=float(constants), discarded_trace_scalar=float(np.trace(A).real / data.n),
        evidence_trace_restoration=float(data.k * np.trace(A).real / data.n),
        observations=len(data.U), boundary_prior_count=1,
        numerical_ok=bool(diagnostics['numerical_ok'] and prior_diagnostics['numerical_ok']
                          and np.isfinite(evidence)))
    return BlockPosterior(center(A), M, evidence, diagnostics['numerical_ok'], diagnostics)


def expected_residual(data, M):
    """Sum of E||u-Pv||², including uncertainty in P rather than only E[P]."""
    if not data.conditional:
        raise ValueError('Expected conditional residual requires recorded inputs')
    M = np.asarray(M)
    residual = data.U - data.V @ M.T
    uncertainty = np.einsum('bi,ij,bj->', data.V.conj(), M - M @ M, data.V).real
    return float(np.vdot(residual, residual).real + uncertainty)


def ordered_variance_update(parallel, perpendicular, n, k, floor=1e-6):
    """Maximize the variance Q function under signal variance >= 0 and s >= floor."""
    if not np.isfinite([parallel, perpendicular, floor]).all() or floor <= 0:
        raise ValueError('Finite statistics and positive variance floor required')
    if parallel >= max(perpendicular, floor):
        s = max(perpendicular, floor)
        return float(s), float(parallel - s)
    return float(max((k * parallel + (n - k) * perpendicular) / n, floor)), 0.


def fit_block_em(data, s, tau2=None, *, iterations=20, floor=1e-6,
                 tol=1e-10, max_order=4096, ascent_tolerance=1e-8):
    """Fit disjoint stationary-block noise with a fixed normalized-evidence guard.

    Every candidate recomputes all likelihood increments from the boundary
    prior, then checks the same block objective.  A capped/unresolved integral
    raises instead of being accepted as evidence.  ``converged`` concerns the
    declared finite iteration tolerance; EM need not find a global optimum.
    """
    if (not isinstance(iterations, (int, np.integer)) or iterations < 0
            or not np.isfinite(floor) or floor <= 0 or s < floor):
        raise ValueError('Nonnegative iterations and positive floor required')
    if ascent_tolerance < 0 or not np.isfinite(ascent_tolerance):
        raise ValueError('Finite nonnegative ascent tolerance required')
    post = block_posterior(data, s, tau2, tol=tol, max_order=max_order)
    if not post.numerical_ok:
        raise ArithmeticError('Initial block integration unresolved')
    s = float(s)
    tau2 = None if data.conditional else float(tau2)
    objectives = [post.log_evidence]
    history = [dict(iteration=0, s=s, tau2=tau2, log_evidence=post.log_evidence,
                    numerical_ok=True, accepted=True)]
    rejected = accepted = 0
    converged = False
    L, n, k = len(data.U), data.n, data.k
    for iteration in range(1, iterations + 1):
        M = post.mean
        if data.conditional:
            expected = expected_residual(data, M)
            candidate = (max(expected / (n * L), floor), None)
        else:
            parallel_total = float(np.einsum('bi,ij,bj->', data.U.conj(), M, data.U).real)
            total = float(np.vdot(data.U, data.U).real)
            hp = parallel_total / (L * k)
            hn = (total - parallel_total) / (L * (n - k))
            # At zero signal with a uniform prior, both sufficient statistics
            # coincide exactly.  Preserve that mathematically absorbing face.
            uniform = np.linalg.norm(center(data.boundary_prior), 2) <= 1e-13
            candidate = ((max(total / (n * L), floor), 0.) if tau2 == 0 and uniform
                         else ordered_variance_update(hp, hn, n, k, floor))
        try:
            candidate_post = block_posterior(data, *candidate, tol=tol, max_order=max_order)
        except ArithmeticError as error:
            history.append(dict(iteration=iteration, s=float(candidate[0]), tau2=candidate[1],
                log_evidence=None, numerical_ok=False, accepted=False, error=str(error)))
            rejected += 1
            break
        ok = candidate_post.numerical_ok and candidate_post.log_evidence >= objectives[-1] - ascent_tolerance
        history.append(dict(iteration=iteration, s=float(candidate[0]), tau2=candidate[1],
            log_evidence=candidate_post.log_evidence, numerical_ok=candidate_post.numerical_ok,
            accepted=bool(ok), evidence_increment=float(candidate_post.log_evidence - objectives[-1])))
        if not ok:
            rejected += 1
            break
        old_s, old_tau = s, tau2
        s, tau2 = float(candidate[0]), candidate[1]
        post = candidate_post
        accepted += 1
        objectives.append(post.log_evidence)
        scale_change = abs(s - old_s) + (0. if tau2 is None else abs(tau2 - old_tau))
        if scale_change <= tol * (1 + s + (0. if tau2 is None else tau2)):
            converged = True
            break
    return BlockFit(s, tau2, objectives, history, accepted, rejected, post, converged)
