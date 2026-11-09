"""Final-guide algebra, sharp modal certificates, and local accuracy studies.

Public variance names are sigma2 and tau2. Historical routines retain their
original names and certificates; this module does not rewrite their evidence.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp

from .geometry import center
from .models import increment, constant
from .guide_tracking import _setup, certificate_observation
from .guide_information import conditional_fisher, gaussian_kl, independent_moments
from .beamforming_theory import passive_fisher_information
from .block_em import moments


def _rows(U, complex_data):
    U = np.asarray(U)
    if U.ndim != 2 or not len(U) or not np.isfinite(U).all():
        raise ValueError('Observations must have shape (L,n) and finite entries')
    if not complex_data and np.iscomplexobj(U):
        raise ValueError('Complex observations require complex_data=True')
    return U


def align_block(A0, U, V=None, G=None, *, complex_data=False):
    """Express a completed block and its boundary prior in its endpoint frame.

    G[j] maps the preceding frame to observation j's frame. Intermediate
    rotations must already be recorded. This algebra does not imply that a
    delayed method could learn those rotations without intermediate outputs.
    """
    U = _rows(U, complex_data)
    L, n = U.shape
    A0 = np.asarray(A0)
    if (A0.shape != (n, n) or not np.isfinite(A0).all()
            or not np.allclose(A0, A0.conj().T, atol=1e-11)):
        raise ValueError('Boundary prior must be Hermitian with shape (n,n)')
    if not complex_data and np.iscomplexobj(A0):
        raise ValueError('A real block requires a real boundary prior')
    V = None if V is None else _rows(V, complex_data)
    if V is not None and V.shape != U.shape:
        raise ValueError('Recorded inputs must have the same shape as U')
    if G is None:
        transports = np.broadcast_to(np.eye(n, dtype=U.dtype), (L,n,n))
        initial = np.eye(n, dtype=U.dtype)
        return dict(U=U.copy(), V=None if V is None else V.copy(),
            A0=A0.copy(), transports=transports, initial_transport=initial)
    G = np.asarray(G)
    if (G.shape != (L,n,n) or not np.isfinite(G).all()
            or (not complex_data and np.iscomplexobj(G))
            or not np.allclose(G.conj().transpose(0,2,1)@G, np.eye(n), atol=1e-10)):
        raise ValueError('G must be recorded orthogonal/unitary frame changes')
    dtype = np.result_type(U.dtype, A0.dtype, G.dtype)
    transports = np.empty((L,n,n), dtype=dtype)
    initial = np.eye(n, dtype=dtype)
    for j in range(L-1,-1,-1):
        transports[j] = initial
        initial = initial @ G[j]
    return dict(U=np.einsum('tij,tj->ti',transports,U),
        V=None if V is None else np.einsum('tij,tj->ti',transports,V),
        A0=initial@A0@initial.conj().T, transports=transports,
        initial_transport=initial)


def exact_aggregate(A0, U, sigma2, tau2=None, V=None, beta=1., G=None,
                    complex_data=False, return_details=False):
    """Exactly aggregate L singleton corrections, without covariance averaging.

    Scalars or length-L variance arrays are allowed. Both recorded responses
    and inputs are aligned before their per-observation coefficients are used.
    """
    U = _rows(U,complex_data)
    L,n = U.shape
    if not 0 < beta <= 1:
        raise ValueError('Require 0 < beta <= 1')
    s = np.broadcast_to(np.asarray(sigma2,float), (L,))
    if not np.isfinite(s).all() or np.any(s<=0):
        raise ValueError('sigma2 must contain positive finite variances')
    q = None if V is not None else np.broadcast_to(np.asarray(tau2,float),(L,))
    if q is not None and (not np.isfinite(q).all() or np.any(q<0)):
        raise ValueError('tau2 must contain nonnegative finite variances')
    aligned = align_block(A0,U,V,G,complex_data=complex_data)
    u,v = aligned['U'],aligned['V']
    w = beta**np.arange(L-1,-1,-1)
    f = 1. if complex_data else .5
    if v is None:
        coefficients = f*w*q/(s*(s+q))
        B = np.einsum('t,ti,tj->ij',coefficients,u,u.conj())
    else:
        coefficients = f*w/s
        B = (np.einsum('t,ti,tj->ij',coefficients,u,v.conj())
             +np.einsum('t,ti,tj->ij',coefficients,v,u.conj())
             -np.einsum('t,ti,tj->ij',coefficients,v,v.conj()))
    A = center(beta**L*aligned['A0']+B)
    if not return_details:
        return A
    return dict(A=A,increment=B,weights=w,coefficients=coefficients,
        boundary_factor=beta**L,aligned=aligned,observations=L,
        scope='Exact endpoint aggregation; no action is available before this block ends')


def sequential_parameter(A0,U,sigma2,tau2=None,V=None,beta=1.,G=None,
                         complex_data=False):
    """Independent loop through individual updates, including trace centering."""
    U = _rows(U,complex_data)
    L,n = U.shape
    s=np.broadcast_to(np.asarray(sigma2,float),(L,))
    q=None if V is not None else np.broadcast_to(np.asarray(tau2,float),(L,))
    if G is None:G=np.broadcast_to(np.eye(n),(L,n,n))
    A=np.array(A0,copy=True,dtype=np.result_type(A0,U))
    for j,u in enumerate(U):
        A=center(beta*G[j]@A@G[j].conj().T
            +increment(u,s[j],None if q is None else q[j],
                       None if V is None else V[j],complex_data))
    return A


def sharp_marginal_certificate(P_history,sigma2,tau2,beta=1.,delta=.05,
                               A0=None,G=None,complex_data=False):
    """Final guide's covering allowance and sharp Davis--Kahan envelope."""
    if not np.isfinite(tau2) or tau2<0:
        raise ValueError('tau2 must be nonnegative')
    d=_setup(P_history,sigma2,beta,delta,A0,G,complex_data)
    w,n,q=d['weights'],d['n'],float(tau2)
    m,b=float(w.sum()),float(w@w)
    eta=(1. if complex_data else .5)*q/(sigma2*(sigma2+q))
    h=(2 if complex_data else 1)*n*np.log(9)+np.log(2/delta)
    gamma=eta*q*m
    drift_matrix=eta*q*np.einsum('t,tij->ij',w,d['aligned']-d['Ps'][-1])
    drift=float(np.linalg.norm(drift_matrix,2))
    noise=float(4*eta*(sigma2+q)*(np.sqrt(b*h)+h))
    reference=eta*sigma2*m*np.eye(n)+gamma*d['Ps'][-1]
    result=_sharp_finish(d,gamma,drift_matrix,drift,noise,reference)
    F=passive_fisher_information(n,d['k'],sigma2,q,observations=m,
                                complex_data=complex_data)
    result.update(sigma2=float(sigma2),tau2=q,eta=eta,h=h,m=m,b=b,
        reference_fisher_min=float(np.linalg.eigvalsh(F)[0]),
        gap_information_error=float(abs(2*gamma-np.linalg.eigvalsh(F)[0])),
        formula_name='Final guide marginal, proper complex' if complex_data else 'Final guide marginal, real')
    return result


def sharp_conditional_certificate(P_history,V,sigma2,beta=1.,delta=.05,
                                  A0=None,G=None,complex_data=False):
    """Exact aggregate conditional drift, and the sharp final-guide envelope."""
    from .guide_tracking import conditional_certificate
    c=conditional_certificate(P_history,V,sigma2,beta,delta,A0,G,complex_data)
    aggregate=float(np.linalg.norm(c['drift_matrix'],2))
    result=_sharp_finish(c,c['gamma'],c['drift_matrix'],aggregate,c['noise'],c['reference'])
    e,E=np.linalg.eigh(c['Ps'][-1]);X=E[:,-c['k']:];Q=E[:,:-c['k']]
    F=conditional_fisher(X,c['aligned_inputs']*np.sqrt(c['weights'])[:,None],
                         sigma2,complex_data,Q)
    result.update(C=c['C'],D=c['D'],V=c['V'],aligned_inputs=c['aligned_inputs'],
        eta=c['eta'],sigma2=float(sigma2),separate_norm_drift=c['drift'],
        reference_fisher_min=float(np.linalg.eigvalsh(F)[0]),
        gap_information_error=float(abs(2*c['gamma']-np.linalg.eigvalsh(F)[0])),
        formula_name='Final guide conditional, real' if not complex_data else
                     'Derived conservative complex conditional extension',
        complex_conditional_source_verified=not complex_data)
    return result


def _sharp_finish(data,gamma,drift_matrix,drift,noise,reference):
    eps=float(data['initialization']+drift+noise)
    valid=bool(gamma>0 and eps<gamma/2)
    ceiling=float(np.sqrt(min(data['k'],data['n']-data['k'])))
    bound=float(ceiling*eps/(gamma-eps)) if valid else np.nan
    return dict(data,gamma=float(gamma),drift=float(drift),noise=float(noise),epsilon=eps,
        drift_matrix=drift_matrix,reference=reference,valid=valid,bound=bound,
        ceiling=ceiling,informative=bool(valid and bound<ceiling),
        loose_bound=float(2*np.sqrt(data['k'])*eps/gamma) if valid else np.nan,
        probability_scope='Fixed reporting time, true fixed variances, deterministic path/design/transport; no trajectory-wide or adaptive-variance guarantee')


def natural_curvature(A,k=1,complex_data=False):
    """Invariant-density Hessian in the aligned modal chart, not graph density."""
    values,U=np.linalg.eigh(A);values,U=values[::-1],U[:,::-1]
    gaps=values[:k,None]-values[None,k:]
    eigenvalues=(2*gaps.T).ravel(order='F')
    if complex_data:eigenvalues=np.r_[eigenvalues,eigenvalues]
    isolated=bool(np.min(gaps)>1e-10*max(1.,np.ptp(values)))
    return dict(values=eigenvalues,H=np.diag(eigenvalues),basis=U[:,:k],
        complement=U[:,k:],isolated=isolated,minimum_gap=float(np.min(gaps)),
        real_graph_volume_curvature=float(len(A)) if not complex_data else np.nan)


def tangent_coordinate(h,truth,complex_data=False,local_radius=.45):
    """Projective logarithm in a fixed truth frame with a declared local radius."""
    truth=np.asarray(truth);h=np.asarray(h)
    overlap=np.vdot(truth,h)
    if abs(overlap)<1e-14:return np.full((2 if complex_data else 1)*(len(h)-1),np.nan),False
    h=h*np.conj(overlap)/abs(overlap)
    theta=float(np.arccos(np.clip(np.vdot(truth,h).real,0.,1.)))
    Q=null_space(truth.conj()[None,:])
    coordinate=Q.conj().T@h
    coordinate*=theta/max(np.sin(theta),1e-15)
    real=np.r_[coordinate.real,coordinate.imag] if complex_data else coordinate.real
    return real,bool(theta<local_radius)


def global_kl_identity(X,Y,sigma2,tau2=1.,V=None,complex_data=False):
    """Closed-form global Gaussian KL and a direct determinant/solve reference."""
    P,Q=X@X.conj().T,Y@Y.conj().T
    f=1. if complex_data else .5
    if V is None:
        R=sigma2*np.eye(len(X))+tau2*P
        S=sigma2*np.eye(len(X))+tau2*Q
        direct=f*(np.linalg.slogdet(S)[1]-np.linalg.slogdet(R)[1]
                  +np.trace(np.linalg.solve(S,R)).real-len(X))
        formula=f*tau2**2/(sigma2*(sigma2+tau2))*(X.shape[1]-np.trace(P@Q).real)
    else:
        delta=(P-Q)@np.asarray(V).T
        direct=f*np.vdot(delta,delta).real/sigma2
        C=np.asarray(V).T@np.asarray(V).conj()
        formula=f*np.trace((P-Q)@C@(P-Q)).real/sigma2
    return float(direct),float(formula)


def independent_nodes(A,k=1,complex_data=False,order=64):
    """Area-correct N=3 real sphere or complex simplex/phase quadrature.

    Four phases in each of two relative phase coordinates integrate projector
    polynomials through degree two exactly; they do not resolve arbitrary
    response quantiles. Continuous real angular quadrature has its own
    resolution comparison. The reference is independent of the native series.
    """
    A=np.asarray(A)
    if A.shape!=(3,3) or k not in (1,2):
        raise ValueError('Independent nodes support n=3, k=1 or k=2')
    target=A if k==1 else -A
    lam,E=np.linalg.eigh(target);lam,E=lam[::-1],E[:,::-1]
    z,w=np.polynomial.legendre.leggauss(order)
    if complex_data:
        x=(z+1)/2
        xx,yy=np.meshgrid(x,x,indexing='ij')
        squared=np.stack([xx,(1-xx)*yy,(1-xx)*(1-yy)],axis=-1).reshape(-1,3)
        prior=(np.outer(w,w)/4*2*(1-xx)).ravel()
        phases=np.array([[1.,np.exp(2j*np.pi*a/4),np.exp(2j*np.pi*b/4)]
                         for a in range(4) for b in range(4)])
        local=(np.sqrt(squared)[:,None,:]*phases[None,:,:]).reshape(-1,3)
        prior=np.repeat(prior/16,16)
    else:
        q=(z+1)/2
        phi=2*np.pi*(np.arange(4*order)+.5)/(4*order)
        rr=np.sqrt(1-q*q)
        local=np.stack(np.broadcast_arrays(q[:,None],rr[:,None]*np.cos(phi),
                                           rr[:,None]*np.sin(phi)),axis=-1).reshape(-1,3)
        prior=np.broadcast_to(w[:,None]/(8*order),(order,4*order)).ravel()
    logits=(abs(local)**2)@lam+np.log(prior)
    psi=float(logsumexp(logits));weights=np.exp(logits-psi)
    nodes=local@E.T
    P=np.einsum('bi,bj->bij',nodes,nodes.conj())
    if k==2:
        P=np.eye(3)-P
        psi+=np.trace(A).real
    M=np.einsum('b,bij->ij',weights,P)
    return dict(nodes=nodes,projectors=P,weights=weights,mean=M,
                log_normalizer=psi,order=order,k=k,complex_data=complex_data)


def hermitian_coordinates(n,complex_data=False):
    """A real basis H and trace-dual D for Hermitian/symmetric matrices."""
    H=[]
    for i in range(n):
        A=np.zeros((n,n),complex if complex_data else float);A[i,i]=1;H.append(A)
    for i in range(n):
        for j in range(i+1,n):
            A=np.zeros((n,n),complex if complex_data else float)
            A[i,j]=A[j,i]=1;H.append(A)
            if complex_data:
                A=np.zeros((n,n),complex);A[i,j]=1j;A[j,i]=-1j;H.append(A)
    H=np.array(H)
    D=H/np.einsum('aij,aji->a',H,H).real[:,None,None]
    return H,D


def native_response_covariance(A,k,v,complex_data=False,step=.003):
    """Full response covariance from finite derivatives of the native mean.

    This numerical second-derivative construction has a separate finite-step
    budget. The native first-moment truncation bound does not certify it.
    """
    M,_,diagnostics=moments(A,k,complex_data,tol=1e-12)
    H,D=hermitian_coordinates(len(A),complex_data)
    derivatives=[]
    for direction in D:
        plus,_,dp=moments(A+step*direction,k,complex_data,tol=1e-12)
        minus,_,dm=moments(A-step*direction,k,complex_data,tol=1e-12)
        if not dp['numerical_ok'] or not dm['numerical_ok']:
            raise ArithmeticError('Native mean derivative unresolved')
        derivatives.append((plus-minus)/(2*step))
    cov=np.einsum('aij,bji->ab',D,np.array(derivatives)).real
    cov=(cov+cov.T)/2
    responses=np.einsum('aij,j->ai',H,v)
    response_cov=np.einsum('ab,ai,bj->ij',cov,responses,responses.conj())
    return (response_cov+response_cov.conj().T)/2,M,dict(diagnostics,
        finite_difference_step=step,second_moment_scope='Finite derivative of native means; independently checked against quadrature')


def configuration(stage='verify'):
    if stage not in ('verify','pilot','paper'):
        raise ValueError('Use verify, pilot or paper')
    paper=stage=='paper';verify=stage=='verify'
    return dict(stage=stage,source_commit='a91faabeb10a63b1d1c613d75cc32503cfc0cac6',
        seed=2026111007,algebra_seeds=[2026111001,2026111002,2026111003],
        episodes=128 if paper else 12 if not verify else 3,
        local_episodes=256 if paper else 24 if not verify else 3,
        local_lengths=[4,16,64,256,1024] if paper else [4,64,256],
        local_sigma2=[.2,1.],local_beta=[1.,.9],local_radius=.45,
        block_lengths=[1,4,16,32],block_beta=[1.,.87],
        curvature_gaps=[[0.,1.,1.],[0.,1.,10.],[0.,8.,8.],[0.,8.,80.],
                        [0.,80.,80.],[0.,80.,800.]],
        quadrature_orders=[48,96] if verify else [96,192],
        numerical_gaps=[[0.,1.,8.],[0.,8.,80.],[0.,80.,800.],[0.,4000.,40000.]],
        numerical_orders=[64,128] if paper else [32,64],
        native_max_order=4096,native_tolerance=1e-11,
        static_times=[64,256,1024,4096,16384] if paper else [64,256,1024],
        dynamic_T=4096 if paper else 256,drift_rates=[0.,.0001,.001],
        bound_beta=[1.,.999,.99],sigma2=.2,tau2=1.,delta=.05,
        nonuniform_prior=[-4.,0.,1.],
        tolerances=dict(algebra_relative=1e-10,reference_moment=2e-8,
            second_moment_absolute=3e-5,normalizer_derivative=3e-5,
            covariance_resolution=3e-5,local_probability=.99),
        variance_convention='Real noise covariance sigma2 I; proper-complex covariance sigma2 I and real/imag component variance sigma2/2.',
        specification='Supplied v5.tex main and supplementary(14).tex, theory and appendices; final_simulation_guide.md. Global KL, information loss, invariant versus graph curvature, aggregate drift, sharp bound and information-gap identity reconciled against source text.',
        tex_hashes={'v5.tex':'31aad1eb4ff0f6e6b694eb63a5529decf264073c268f841681533af08346df4b',
                    'supplementary(14).tex':'e678649d45e7dfdf0039e63e02041d9cfac884f9229ef902f174f2db035f311b'},
        repetition_unit='Independent signal/noise episode; no posterior node or query is an independent performance episode.',
        inference_scope='Numerical native rank one/codimension one; information algebra also checks other ranks. No deterministic general-rank moment scaling claim.')
