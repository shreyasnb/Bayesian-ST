"""Deterministic directional Bingham geometry from draft 5.

The state here is a real line in R^3. For contact it represents the NORMAL:
the admissible plane has parameter -A and mean I-E[n n.T]. This module does
not turn selected quadrature nodes into independent Monte Carlo repetitions.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq
from scipy.special import logsumexp
from .spectral_series import rank_one_bingham
from .reference import rank_one_regions_gr13
from .theory import regional_bound


PRIMARY_PARAMETERS=np.asarray([np.diag([0.,-8.,-80.]),np.diag([0.,-80.,-8.])])
CASE_PARAMETERS=np.asarray([*PRIMARY_PARAMETERS,np.diag([0.,-.5,-5.]),np.diag([0.,-8.,-8.])])
CASE_LABELS=(r'$q_{A^{(1)}}$',r'$q_{A^{(2)}}$',r'$q_{\operatorname{diag}(0,-0.5,-5)}$',r'$q_{\operatorname{diag}(0,-8,-8)}$')


def _matrix(A):
    A=np.asarray(A)
    if (A.shape!=(3,3) or np.iscomplexobj(A) or not np.isfinite(A).all()
            or not np.allclose(A,A.T,atol=1e-12,rtol=1e-12)):
        raise ValueError('Expected a finite real symmetric 3 by 3 parameter')
    return (A+A.T)/2


def _spectral(A):
    lam,U=np.linalg.eigh(_matrix(A));order=np.argsort(lam)[::-1]
    return lam[order],U[:,order]


def sphere_rule(A,radial_order=96,angular_order=192):
    """Weighted hemisphere rule for projector functionals under real q_A.

    The normal coordinate q=cos(theta) is uniform under invariant measure;
    phi is uniform on [0,2pi). Antipodal projectors agree, so q>=0 represents
    every projective line once. Weights sum to one after posterior weighting.
    """
    A=_matrix(A)
    if radial_order<16 or angular_order<32:raise ValueError('Insufficient quadrature order')
    lam,U=_spectral(A);x,w=leggauss(int(radial_order));q=(x+1)/2;w=w/2
    phi=2*np.pi*(np.arange(angular_order)+.5)/angular_order
    radial=np.sqrt(np.maximum(0.,1-q*q))
    local=np.stack([np.broadcast_to(q[:,None],(len(q),len(phi))),
        radial[:,None]*np.cos(phi)[None,:],radial[:,None]*np.sin(phi)[None,:]],axis=-1).reshape(-1,3)
    nodes=local@U.T
    logw=np.repeat(np.log(w),angular_order)-np.log(angular_order)
    energy=np.einsum('bi,ij,bj->b',nodes,A,nodes)
    logz=float(logsumexp(logw+energy));weights=np.exp(logw+energy-logz)
    mean=np.einsum('b,bi,bj->ij',weights,nodes,nodes)
    return dict(nodes=nodes,weights=weights,mean=mean,log_normalizer=logz,
        radial_order=int(radial_order),angular_order=int(angular_order),
        measure='Normalized invariant real projective measure via one hemisphere')


@lru_cache(maxsize=64)
def _continuous_spectrum(gap1,gap2,epsabs):
    """Nested continuous integration, independent of the positive series.

    No Dawson primitive, fixed-node quantile or posterior sample is used here.
    Returns the raw hemisphere integral and its normalized spectral mean.
    Error estimates are QUADPACK resolution estimates, not rigorous enclosures.
    """
    a,b=float(gap1),float(gap2);errors=[]
    def angular(phi):
        co,si=np.cos(phi),np.sin(phi);g=a*co*co+b*si*si
        values,error=quad_vec(lambda q: np.exp(-g*(1-q*q))*np.asarray([1.,q*q,(1-q*q)*co*co,(1-q*q)*si*si]),
            0.,1.,epsabs=epsabs,epsrel=epsabs)
        errors.append(float(error));return values
    integral,error=quad_vec(angular,0.,np.pi/2,epsabs=epsabs,epsrel=epsabs)
    normalizer=float(2*integral[0]/np.pi)
    if normalizer<=0:raise ArithmeticError('Continuous normalizer is not positive')
    return normalizer,np.asarray(integral[1:]/integral[0]),float(max([float(error),*errors])/integral[0])


def continuous_moments(A,epsabs=1e-10):
    """Normalizer and mean from direct continuous real-sphere integration."""
    lam,U=_spectral(A);z,m,error=_continuous_spectrum(float(lam[0]-lam[1]),float(lam[0]-lam[2]),float(epsabs))
    return dict(mean=(U*m)@U.T,log_normalizer=float(lam[0]+np.log(z)),integration_error=error)


@lru_cache(maxsize=2048)
def _continuous_mass(gap1,gap2,kind,value,epsabs):
    a,b=float(gap1),float(gap2);value=float(value)
    z,_,_= _continuous_spectrum(a,b,float(epsabs));points=[]
    if kind=='ball':
        if not 0<=value<=1:raise ValueError('A chordal radius lies in [0,1]')
        lower=np.sqrt(max(0.,1-value*value))
    elif kind=='hpd':
        if value<0:return 0.
        if a<value<b and b>a:points=[float(np.arcsin(np.sqrt((value-a)/(b-a))))]
    else:raise ValueError('kind must be ball or hpd')
    def angular(phi):
        g=a*np.cos(phi)**2+b*np.sin(phi)**2
        q0=lower if kind=='ball' else np.sqrt(max(0.,1-value/g)) if g>0 else 0.
        result,_=quad(lambda q:np.exp(-g*(1-q*q)),q0,1.,epsabs=epsabs,epsrel=epsabs)
        return result
    integral,_=quad(angular,0.,np.pi/2,points=points,epsabs=epsabs,epsrel=epsabs,limit=250)
    return float(2*integral/(np.pi*z))


def continuous_region_mass(A,kind,value,epsabs=1e-10):
    lam,_=_spectral(A)
    return _continuous_mass(float(lam[0]-lam[1]),float(lam[0]-lam[2]),kind,float(value),float(epsabs))


def continuous_regions(A,mass=.95,epsabs=1e-10):
    """Independently solve continuous credible-mass equations for unique modes."""
    lam,_=_spectral(A)
    if not 0<mass<1:raise ValueError('Require a posterior mass strictly between zero and one')
    if lam[0]-lam[1]<1e-12:raise ValueError('A unique mode is required; uniform HPD ties need a separate convention')
    radius=brentq(lambda x:continuous_region_mass(A,'ball',x,epsabs)-mass,0.,1.,xtol=1e-11)
    hpd=brentq(lambda x:continuous_region_mass(A,'hpd',x,epsabs)-mass,0.,float(lam[0]-lam[-1]),xtol=1e-10)
    return dict(radius=float(radius),hpd_loss=float(hpd),mass=mass,
        ball_mass=continuous_region_mass(A,'ball',radius,epsabs),
        hpd_mass=continuous_region_mass(A,'hpd',hpd,epsabs))


def axial_parameter(mean,mode_vector):
    """Real axial q_{kappa P_mode} matching the mean modal overlap/r_rms.

    This matches one moment, not the entire centered distance CDF or its 95%
    quantile. The primary coordinate-swap pair provides the stronger exact
    all-distance-quantile equality control without fitting kappa.
    """
    mean=np.asarray(mean,float);h=np.asarray(mode_vector,float);P=np.outer(h,h)
    target=float(np.trace(P@mean))
    if not 1/3<=target<1:raise ValueError('Expected a finite real-line mean modal overlap in [1/3,1)')
    if abs(target-1/3)<1e-12:return 0.,np.eye(3)/3
    def overlap(k):return float(np.trace(P@rank_one_bingham(k*P,tol=1e-11).mean))
    upper=8.
    while overlap(upper)<target:
        upper*=2
        if upper>2048:raise ArithmeticError('Axial matching exceeded its declared spectral range')
    kappa=float(brentq(lambda k:overlap(k)-target,0.,upper,xtol=1e-10))
    return kappa,rank_one_bingham(kappa*P,tol=1e-11).mean


def response_summary(mean_normal,queries):
    """Line and complementary-plane response risks (the same total risk)."""
    M=np.asarray(mean_normal,float);queries=np.atleast_2d(np.asarray(queries,float));plane=np.eye(3)-M
    return dict(mean_normal=queries@M,mean_plane=queries@plane,
        risk=np.einsum('bi,ij,bj->b',queries,M-M@M,queries))


def soft_action(mean_normal,desired,penalty=20.):
    """Paper Eq36 for the admissible plane: I-Pbar=E[n n.T]."""
    M=np.asarray(mean_normal,float);v=np.asarray(desired,float)
    if penalty<0 or not np.isfinite(penalty):raise ValueError('A finite nonnegative soft penalty is required')
    return np.linalg.solve(np.eye(3)+penalty*M,v)


def soft_cost(mean_normal,desired,action,penalty=20.):
    delta=np.asarray(action)-np.asarray(desired)
    return float(delta@delta+penalty*np.asarray(action)@mean_normal@np.asarray(action))


def query_bounds(A,radius,hpd_loss):
    """Bounds for ||n n.T e_i||, NOT for admissible-plane speed."""
    A=_matrix(A);lam,U=_spectral(A);h=U[:,0];P=np.outer(h,h)
    rows=[]
    for i in range(3):
        e=np.eye(3)[i];Q=np.outer(e,e)
        ball=regional_bound(Q,P,1-radius*radius,1)
        hpd=regional_bound(Q,A,float(lam[0]-hpd_loss),1)
        rows.append(dict(query=i,ball_bound=float(np.sqrt(max(0.,ball.upper))),
            hpd_bound=float(np.sqrt(max(0.,hpd.upper))),
            ball_gap=float(ball.gap),hpd_gap=float(hpd.gap),
            feasible=bool(ball.feasible and hpd.feasible),
            quantity='Normal/line projection norm; blocked component in the contact interpretation'))
    return rows
