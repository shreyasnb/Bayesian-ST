"""Revised fixed-time certificates and snapshot-only tracking controls.

These are evaluation quantities using the deterministic true path.  They are
not online posterior radii.  Legacy certificates remain unchanged.  The native
complex conditional expression below is a separately derived conservative
extension; supplementary.tex is unavailable, so manuscript matching is not
claimed for that expression.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from .geometry import center
from .models import increment


def _hermitian(A):
    A=np.asarray(A)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or not np.isfinite(A).all():
        raise ValueError('Expected a finite square matrix')
    if not np.allclose(A,A.conj().T,rtol=0,atol=1e-9):
        raise ValueError('Expected a Hermitian matrix')
    return (A+A.conj().T)/2


def _path(Ps,complex_data):
    Ps=np.asarray(Ps)
    if Ps.ndim!=3 or not len(Ps) or Ps.shape[1]!=Ps.shape[2]:
        raise ValueError('Expected projector history (T,N,N)')
    if not complex_data and np.iscomplexobj(Ps):
        raise ValueError('Real certificates require real arrays')
    if not np.isfinite(Ps).all() or not np.allclose(Ps,Ps.conj().transpose(0,2,1),rtol=0,atol=1e-9):
        raise ValueError('History must contain finite Hermitian matrices')
    if not np.allclose(Ps@Ps,Ps,rtol=0,atol=1e-8):raise ValueError('History must contain projectors')
    ranks=np.trace(Ps,axis1=1,axis2=2).real
    k=int(round(ranks[0]));n=Ps.shape[1]
    if not 0<k<n or not np.allclose(ranks,k,rtol=0,atol=1e-8):
        raise ValueError('All projectors must have the same nontrivial rank')
    return Ps,n,k


def _setup(Ps,s,beta,delta,A0,G,complex_data):
    Ps,n,k=_path(Ps,complex_data);T=len(Ps)
    if not np.isfinite(s) or s<=0 or not 0<beta<=1 or not 0<delta<1:
        raise ValueError('Require positive variance, 0<beta<=1 and 0<delta<1')
    A0=np.zeros((n,n),dtype=Ps.dtype) if A0 is None else _hermitian(A0)
    if A0.shape!=(n,n) or (not complex_data and np.iscomplexobj(A0)):
        raise ValueError('Initialization has the wrong shape or field')
    eigenvalues=np.linalg.eigvalsh(A0)
    # This is min_c ||A0-cI||_2, not trace centering in general.
    initial_center=(eigenvalues[0]+eigenvalues[-1])/2
    Acenter=A0-initial_center*np.eye(n)
    weights=beta**np.arange(T-1,-1,-1)
    if G is None:
        transports=np.broadcast_to(np.eye(n,dtype=Ps.dtype),(T,n,n))
        initial_transport=np.eye(n,dtype=Ps.dtype);aligned=Ps
    else:
        G=np.asarray(G)
        if G.shape!=(T,n,n) or not np.isfinite(G).all() or (not complex_data and np.iscomplexobj(G)):
            raise ValueError('Invalid deterministic transport array')
        if not np.allclose(G.conj().transpose(0,2,1)@G,np.eye(n),rtol=0,atol=1e-9):
            raise ValueError('Transports must be orthogonal or unitary')
        transports=np.empty_like(G);product=np.eye(n,dtype=G.dtype)
        for j in range(T-1,-1,-1):
            transports[j]=product;product=product@G[j]
        initial_transport=product
        aligned=transports@Ps@transports.conj().transpose(0,2,1)
    initial_matrix=beta**T*initial_transport@Acenter@initial_transport.conj().T
    return dict(Ps=Ps,n=n,k=k,T=T,s=float(s),beta=float(beta),delta=float(delta),
        weights=weights,transports=transports,aligned=aligned,initial_matrix=initial_matrix,
        initialization=float(beta**T*(eigenvalues[-1]-eigenvalues[0])/2),
        initialization_gauge=float(initial_center),complex_data=bool(complex_data))


def _finish(data,gamma,drift_matrix,drift,noise,reference,formula_name,manuscript_verified=True):
    epsilon=data['initialization']+float(drift)+float(noise)
    valid=bool(gamma>0 and epsilon<gamma/2)
    ceiling=float(np.sqrt(min(data['k'],data['n']-data['k'])))
    bound=float(2*np.sqrt(data['k'])*epsilon/gamma) if valid else np.nan
    return dict(data,gamma=float(gamma),drift=float(drift),noise=float(noise),epsilon=float(epsilon),
        valid=valid,bound=bound,ceiling=ceiling,informative=bool(valid and bound<ceiling),
        drift_matrix=drift_matrix,reference=reference,formula_name=formula_name,
        manuscript_verified=bool(manuscript_verified),
        probability_scope='One preassigned time, deterministic path/design/transport, fixed true scales and memory; evaluation only')


def marginal_certificate(P_history,s,tau2,beta=1.,delta=.05,A0=None,G=None,complex_data=False):
    """The exact marginal formula written in implementation_prompt.md.

    w is unnormalized, h=N log(9)+log(2/delta) over R, and uses 2N over C.
    In proper-complex coordinates eta has no factor one half.  The fluctuation
    constant is four in both fields.  The drift norm is taken after summation.
    """
    if not np.isfinite(tau2) or tau2<=0:raise ValueError('A positive signal variance is required')
    d=_setup(P_history,s,beta,delta,A0,G,complex_data)
    w=d['weights'];n=d['n'];q=float(tau2);m=float(w.sum());b=float(w@w)
    eta=(1. if complex_data else .5)*q/(s*(s+q))
    h=(2 if complex_data else 1)*n*np.log(9)+np.log(2/delta)
    gamma=eta*q*m
    drift_matrix=eta*q*np.einsum('t,tij->ij',w,d['aligned']-d['Ps'][-1])
    drift=float(np.linalg.norm(drift_matrix,2))
    noise=4*eta*(s+q)*(np.sqrt(b*h)+h)
    reference=eta*s*m*np.eye(n)+gamma*d['Ps'][-1]
    result=_finish(d,gamma,drift_matrix,drift,noise,reference,
        'Guide marginal complex' if complex_data else 'Guide marginal real')
    result.update(tau2=q,eta=float(eta),h=float(h),m=m,b=b)
    return result


def conditional_certificate(P_history,V,s,beta=1.,delta=.05,A0=None,G=None,complex_data=False):
    """Printed real conditional theorem and a conservative complex extension.

    For complex data, write the standardized Gaussian matrix as (G_R+iG_I)/
    sqrt(2). A union bound gives ||G|| <= sqrt(2)[2 sqrt(N)+sqrt(2log(2/delta))].
    Since the conditional perturbation is (GD^.5+D^.5 G^H)/sqrt(s), multiplying
    by two yields the reported conservative noise constant. No realification
    of complex arrays through the old real-only routine is performed.
    """
    d=_setup(P_history,s,beta,delta,A0,G,complex_data);V=np.asarray(V)
    if V.shape!=(d['T'],d['n']) or not np.isfinite(V).all() or (not complex_data and np.iscomplexobj(V)):
        raise ValueError('Recorded inputs must have shape (T,N) in the stated field')
    aligned=np.einsum('tij,tj->ti',d['transports'],V);w=d['weights'];f=1. if complex_data else .5
    C=np.einsum('t,ti,tj->ij',w,aligned,aligned.conj())
    D=np.einsum('t,ti,tj->ij',w*w,aligned,aligned.conj())
    P=d['Ps'][-1];e,E=np.linalg.eigh(P);X=E[:,-d['k']:];Q=E[:,:-d['k']]
    gamma=f/s*(np.linalg.eigvalsh(X.conj().T@C@X)[0]+np.linalg.eigvalsh(Q.conj().T@C@Q)[0])
    difference=d['aligned']-P
    outer=np.einsum('ti,tj->tij',aligned,aligned.conj())
    drift_matrix=f/s*np.einsum('t,tij->ij',w,difference@outer+outer@difference)
    norms=np.linalg.norm(difference,ord=2,axis=(-2,-1));energy=np.sum(abs(aligned)**2,axis=1)
    drift=float(2*f/s*np.dot(w,norms*energy))
    top=max(0.,float(np.linalg.eigvalsh(D)[-1]))
    if complex_data:
        noise=2*np.sqrt(2)*(2*np.sqrt(d['n'])+np.sqrt(2*np.log(2/delta)))*np.sqrt(top/s)
    else:
        noise=(2*np.sqrt(d['n'])+np.sqrt(2*np.log(1/delta)))*np.sqrt(top/s)
    reference=f/s*(P@C+C@P-C)
    result=_finish(d,gamma,drift_matrix,drift,noise,reference,
        'Derived conservative conditional complex' if complex_data else 'Draft5 Theorem4 conditional real',
        manuscript_verified=not complex_data)
    result.update(C=C,D=D,V=V,aligned_inputs=aligned,eta=f/s,
        aggregate_operator_drift=float(np.linalg.norm(drift_matrix,2)))
    return result


def certificate_observation(certificate,U,V=None):
    """Check the exact perturbation decomposition and the realized modal error."""
    c=certificate;U=np.asarray(U);n=c['n'];T=c['T'];complex_data=c['complex_data']
    if U.shape!=(T,n) or not np.isfinite(U).all() or (not complex_data and np.iscomplexobj(U)):
        raise ValueError('Observation array does not match the certificate')
    aligned=np.einsum('tij,tj->ti',c['transports'],U);w=c['weights']
    if V is None:
        scatter=np.einsum('t,ti,tj->ij',w,aligned,aligned.conj())
        A=c['initial_matrix']+c['eta']*scatter
    else:
        V=np.asarray(V)
        if V.shape!=(T,n):raise ValueError('Recorded inputs do not match')
        v=np.einsum('tij,tj->ti',c['transports'],V)
        statistic=np.einsum('ti,tj->tij',aligned,v.conj())+np.einsum('ti,tj->tij',v,aligned.conj())-np.einsum('ti,tj->tij',v,v.conj())
        A=c['initial_matrix']+c['eta']*np.einsum('t,tij->ij',w,statistic)
    perturbation=A-c['reference']
    noise=perturbation-c['initial_matrix']-c['drift_matrix']
    if V is None:
        population=c['s']*np.eye(n)+c['tau2']*c['aligned']
        direct=c['eta']*np.einsum('t,tij->ij',w,np.einsum('ti,tj->tij',aligned,aligned.conj())-population)
    else:
        clean=np.einsum('tij,tj->ti',c['aligned'],v);errors=aligned-clean
        direct=c['eta']*np.einsum('t,tij->ij',w,
            np.einsum('ti,tj->tij',errors,v.conj())+np.einsum('ti,tj->tij',v,errors.conj()))
    eigenvalues,E=np.linalg.eigh(A);X=E[:,-c['k']:];estimate=X@X.conj().T
    error=float(np.linalg.norm(estimate-c['Ps'][-1],'fro')/np.sqrt(2))
    return dict(modal_error=error,actual_perturbation=float(np.linalg.norm(perturbation,2)),
        actual_noise=float(np.linalg.norm(noise,2)),
        decomposition_error=float(np.linalg.norm(noise-direct,'fro')),
        perturbation_event=bool(np.linalg.norm(perturbation,2)<=c['epsilon']),
        noise_event=bool(np.linalg.norm(noise,2)<=c['noise']),
        modal_violation_tested=bool(c['valid']),
        violation=bool(c['valid'] and error>c['bound']),
        empirical_gap=float(eigenvalues[-c['k']]-eigenvalues[-c['k']-1]))


def marginal_preflight(n,s,tau2,t,beta=1.,delta=.05,complex_data=False):
    """Static A0=0 allowance/gap ratio and the fixed-memory infinite-horizon limit."""
    if n<2 or t<1 or s<=0 or tau2<=0 or not 0<beta<=1 or not 0<delta<1:raise ValueError('Invalid preflight settings')
    w=beta**np.arange(t);m=w.sum();b=w@w
    h=(2 if complex_data else 1)*n*np.log(9)+np.log(2/delta)
    ratio=4*(1+s/tau2)*(np.sqrt(b*h)+h)/m
    limit=0. if beta==1 else 4*(1+s/tau2)*(np.sqrt(h*(1-beta)/(1+beta))+h*(1-beta))
    return dict(ratio=float(ratio),limit=float(limit),h=float(h),m=float(m),b=float(b),
        valid=bool(ratio<.5),permanently_vacuous=bool(limit>=.5))


def configuration(profile='smoke'):
    if profile not in ('smoke','paper'):raise ValueError('Choose smoke or paper')
    smoke=profile=='smoke'
    return dict(schema=1,profile=profile,source_commit='6e78cc9911e7e78452633f13dd947afa669c3470',
        test_seed=2026110903+(10000 if smoke else 0),validation_seed=2026110904+(10000 if smoke else 0),
        episodes=3 if smoke else 128,validation_episodes=3 if smoke else 32,
        dimension=3,rank=1,s=.2,tau2=1.,delta=.05,
        static_T=1024 if smoke else 16384,static_times=[64,256,1024] if smoke else [64,256,1024,4096,16384],
        dynamic_T=256 if smoke else 4096,dynamic_rates=[.0001,.001],bound_betas=[1.,.999,.99],
        conditional_T=256 if smoke else 4096,conditional_times=[64,256] if smoke else [64,256,1024,4096],
        abrupt_T=256 if smoke else 1024,
        beam_n=6,beam_T=64 if smoke else 256,beam_times=[16,64] if smoke else [16,64,128,256],
        beam_betas=[1.,.99,.95,.9,.8],beam_grouse_steps=[.001,.01,.1,1.],
        beam_conditions=[dict(name='static',motion='static',s=.2,tau2=1.),
            dict(name='drift',motion='drift',s=.2,tau2=1.),
            dict(name='abrupt',motion='abrupt',s=.2,tau2=1.),
            dict(name='noise',motion='drift',s=1.,tau2=1.),
            dict(name='power',motion='drift',s=.2,tau2=4.)],
        heldout=64 if smoke else 256,
        precision='128 independent episodes give a 0.0231 one-sided exact 95% upper bound after zero violations; actual intervals are recorded. No multiplicity or trajectory-wide claim.',
        variance_convention='Real e~N(0,sI); proper-complex e~CN(0,sI), real and imaginary component variances s/2.',
        assumptions='Fixed true scales and fixed memory per theorem cell; deterministic paths/designs/transports shared across repeated noise realizations. Primary tracking G=I; hidden coefficient/geometry used only for simulation/evaluation.',
        manuscript_sources='Explicit marginal equations in implementation_prompt.md and real conditional Theorem4 in supplied draft5. main.tex/supplementary.tex absent; complex conditional conservative derivation is separate and manuscript matching is not applicable.',
        tolerances=dict(decomposition_relative=1e-10,algebra=2e-10,mode_pca=2e-9),
        selection='Independent validation; one beta per method/condition minimizes average held-out-time true covariance power. SCM loading continuously selected including exact isotropic endpoint. Test data never select settings.')
