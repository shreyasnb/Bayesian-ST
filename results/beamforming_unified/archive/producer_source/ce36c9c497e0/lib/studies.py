"""Controlled theory experiments, independent of case-study outcomes."""
import time
import numpy as np
import pandas as pd
from scipy.linalg import expm, null_space
from .geometry import projector, chordal, mode
from .filter import BinghamFilter
from .theory import tracking_bound, fisher_information


def information_study(seed=20,replicates=12000):
    rng=np.random.default_rng(seed);X=np.eye(5)[:,:2];Q=null_space(X.T);s=.4
    out=[]
    for label,V in [('parallel',np.eye(5)[:,:2]),('complement',np.eye(5)[:,2:]),
                    ('rank one',np.eye(5)[:,:1]),('full rank',np.eye(5))]:
        J=[]
        for i in range(2):
            for a in range(3):
                D=np.outer(Q[:,a],X[:,i])+np.outer(X[:,i],Q[:,a])
                J.append((D@V).ravel())
        J=np.column_stack(J)
        errors=np.sqrt(s)*rng.normal(size=(replicates,J.shape[0]))
        scores=errors@J/s
        empirical=scores.T@scores/replicates
        F=fisher_information(X,V,s)
        relative=np.linalg.norm(empirical-F)/max(np.linalg.norm(F),1)
        out.append(dict(input=label,lambda_min=np.linalg.eigvalsh(F)[0],relative_error=relative))
    return pd.DataFrame(out)


def tracking_study(seed=21,T=480,repeats=128,s=.01):
    """Predeclared terminal time for repeated fixed-time probability check.

    Drift/forgetting trajectories share preassigned equal-norm patterns; no EM.
    Confidence across an entire curve is not asserted.
    """
    rng=np.random.default_rng(seed);n,k=3,1
    V=np.tile(np.eye(n),(int(np.ceil(T/n)),1))[:T]
    out=[];paths=[]
    H=np.zeros((n,n));H[1,0]=1;H[0,1]=-1
    for drift in [0.,.002,.015]:
        Xs=np.array([expm(H*drift*t)@np.eye(n)[:,:k] for t in range(T)])
        for beta in [1.,.97]:
            bounds={t:tracking_bound(Xs[:t],V[:t],s,beta) for t in sorted(set([12,48,120,T])) if t<=T}
            errors={t:[] for t in bounds}
            for rep in range(repeats):
                f=BinghamFilter(n,k,s,beta=beta)
                for j in range(T):
                    u=projector(Xs[j])@V[j]+np.sqrt(s)*rng.normal(size=n)
                    Y=f.update(u,V[j])
                    if j+1 in bounds:errors[j+1].append(chordal(Y,Xs[j]))
            for t,b in bounds.items():
                err=np.array(errors[t]);violations=int(np.sum(err>b.bound)) if b.valid else None
                out.append(dict(drift=drift,beta=beta,t=t,error=float(np.mean(err)),
                                error_se=float(np.std(err,ddof=1)/np.sqrt(repeats)),
                                bound=b.bound,valid=b.valid,gamma=b.gamma,drift_term=b.drift,
                                noise_term=b.noise,initialization=b.initialization,
                                ceiling=b.ceiling,violations=violations,repeats=repeats))
    return pd.DataFrame(out)


def scalability_study(seed=22,dimensions=(8,16,32,64,128),steps=8):
    """Dense modal recursion scaling; moment cost is reported separately."""
    rng=np.random.default_rng(seed);rows=[]
    for n in dimensions:
        f=BinghamFilter(n,3,.4,tau2=1.,beta=.98)
        us=rng.normal(size=(steps,n));start=time.perf_counter()
        for u in us:f.update(u)
        elapsed=(time.perf_counter()-start)/steps
        rows.append(dict(n=n,k=3,seconds=elapsed,parameter_bytes=f.A.nbytes,
                         finite=bool(np.isfinite(f.A).all()),trace=float(np.trace(f.A))))
    return pd.DataFrame(rows)
