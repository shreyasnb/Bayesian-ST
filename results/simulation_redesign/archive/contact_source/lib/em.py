"""Frozen-prior window EM. Numerical ascent requires an explicit score check."""
from dataclasses import dataclass
import numpy as np
from .models import increment, constant


@dataclass
class Record:
    prior: np.ndarray
    u: np.ndarray
    v: np.ndarray | None = None


@dataclass
class Fit:
    s: float
    tau2: float | None
    objectives: list
    accepted: int
    rejected: int
    mean_evaluations: int


def marginal_mstep(h_parallel, h_perp, n, k, floor=1e-6):
    """Eq. S32, including the ordered-variance boundary and zero statistics."""
    if h_parallel >= max(h_perp, floor):
        s, p = max(h_perp, floor), h_parallel
    else:
        s = p = max((k*h_parallel+(n-k)*h_perp)/n, floor)
    return float(s), float(p-s)


def predictive_score(records, k, s, tau2, log_normalizer, complex_data=False):
    """Do not center increments here: trace shifts belong to the density."""
    total = 0.
    for rec in records:
        B = increment(rec.u,s,tau2,rec.v,complex_data)
        total += constant(rec.u,k,s,tau2,rec.v,complex_data)
        total += log_normalizer(rec.prior+B) - log_normalizer(rec.prior)
    return float(total)


def fit_noise(records, k, s, tau2, mean, iterations=5, floor=1e-6,
              log_normalizer=None, complex_data=False):
    """EM for fixed stored priors; optional likelihood safeguard rejects decreases.

    `mean(A,k)` must return a numerical mean projector. Without a normalizer,
    reported ascent is deliberately unverified (objectives remains empty).
    No historical parameter or observation is reinserted into the recursion.
    """
    if not records or iterations < 0 or floor <= 0:
        raise ValueError('nonempty window, nonnegative iterations, positive floor required')
    n, L = len(records[0].u), len(records)
    conditional = records[0].v is not None
    if any((r.v is not None) != conditional for r in records):
        raise ValueError('window must use one observation model')
    objectives=[]
    if log_normalizer is not None:
        objectives.append(predictive_score(records,k,s,tau2,log_normalizer,complex_data))
    accepted=rejected=evaluations=0
    for _ in range(iterations):
        means=[mean(r.prior+increment(r.u,s,tau2,r.v,complex_data),k) for r in records]
        evaluations += L
        if conditional:
            # Expected residual, not residual of expected response alone (Eq.38).
            total=sum(np.vdot(r.u-M@r.v,r.u-M@r.v).real +
                      np.vdot(r.v,(M-M@M)@r.v).real for r,M in zip(records,means))
            candidate=(max(float(total/(n*L)),floor),None)
        else:
            hp=sum(np.vdot(r.u,M@r.u).real for r,M in zip(records,means))/(L*k)
            hn=sum(np.vdot(r.u,(np.eye(n)-M)@r.u).real for r,M in zip(records,means))/(L*(n-k))
            candidate=marginal_mstep(hp,hn,n,k,floor)
        if log_normalizer is not None:
            score=predictive_score(records,k,*candidate,log_normalizer,complex_data)
            if score < objectives[-1]-1e-8:
                rejected += 1
                break
            objectives.append(score)
        s,tau2=candidate
        accepted += 1
    return Fit(float(s),tau2,objectives,accepted,rejected,evaluations)
