from __future__ import annotations
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import dawsn, logsumexp


def _matrix(A):
    A=np.asarray(A)
    if (A.shape!=(3,3) or np.iscomplexobj(A) or not np.isfinite(A).all()
            or not np.allclose(A,A.T,atol=1e-12,rtol=1e-12)):
        raise ValueError('Contact summaries require a finite real symmetric 3x3 parameter')
    return (A+A.T)/2


def _spectrum(C):
    values,U=np.linalg.eigh(C)
    values,U=values[::-1],U[:,::-1]
    return values,U,np.maximum(0.,values[0]-values[1:])


@lru_cache(maxsize=16)
def _radial_rule(order):
    q,w=leggauss(order)
    return (q+1)/2,w/2


@lru_cache(maxsize=16)
def _angular_rule(order):
    # A half-step avoids assigning a special node to a principal-vector axis.
    psi=2*np.pi*(np.arange(order)+.5)/order
    return np.cos(psi),np.sin(psi)


def _angular_geometry(gaps,order):
    a,b=gaps;co,si=_angular_rule(order)
    c=np.sqrt((1+a)/(1+b))
    denominator=np.hypot(co,c*si)
    cosine,sine=co/denominator,c*si/denominator
    jacobian=c/denominator**2
    g=a*cosine**2+b*sine**2
    return cosine,sine,g,jacobian/order


def _polar_integral(g,q):
    """Integral_q^1 exp[-g(1-r^2)] dr, vectorized stable Dawson formula."""
    g,q=np.broadcast_arrays(np.asarray(g,dtype=float),np.asarray(q,dtype=float))
    q=np.clip(q,0.,1.);out=np.empty(q.shape,dtype=float)
    small=g<1e-5
    x=q[small];h=g[small]
    out[small]=(1-x)-h*((1-x)-(1-x**3)/3)+h*h*((1-x)-2*(1-x**3)/3+(1-x**5)/5)/2
    root=np.sqrt(g[~small]);x=q[~small]
    out[~small]=(dawsn(root)-np.exp(-g[~small]*(1-x*x))*dawsn(root*x))/root
    return np.maximum(0.,out)


def normal_quadrature(C,radial_order=64,angular_order=192):
    """Normalized hemisphere quadrature, including high and repeated spectra.

    For g>96 use y=g*sin(theta)^2 on [0,48], with exact sphere Jacobian
    1/[2g sqrt(1-y/g)]. The omitted radial mass is at most exp(-48)
    absolutely before normalization. The returned relative bound uses a
    continuous angular normalizer, rather than claiming a uniform relative
    bound for arbitrary concentrations. Angular warping balances a+b anisotropy
    and retains all four azimuthal quadrants, including repeated top spectra.
    """
    C=_matrix(C)
    if radial_order<16 or angular_order<32:
        raise ValueError('Require radial_order>=16 and angular_order>=32')
    values,U,gaps=_spectrum(C)
    t,w=_radial_rule(int(radial_order))
    cosine,sine,g,angular_weights=_angular_geometry(gaps,int(angular_order))
    q=np.broadcast_to(t[:,None],(radial_order,angular_order)).copy()
    weights=np.broadcast_to(w[:,None],q.shape).copy()
    high=g>96.
    if high.any():
        y=48*t[:,None]
        q[:,high]=np.sqrt(1-y/g[high])
        weights[:,high]=48*w[:,None]/(2*g[high]*q[:,high])
    energy=-g[None,:]*(1-q*q)
    lw=np.log(weights)+np.log(angular_weights)[None,:]+energy
    log_integral=float(logsumexp(lw));posterior_weights=np.exp(lw-log_integral).ravel()
    r=np.sqrt(np.maximum(0.,1-q*q))
    local=np.stack([q,r*cosine,r*sine],axis=-1).reshape(-1,3)
    spectral=np.einsum('b,bi,bi->i',posterior_weights,local,local)
    # Repeated eigenvalues have equal means. Keep this invariant explicit;
    # unlike renormalization, averaging a commutant block is an exact symmetry.
    for value in np.unique(values):
        block=np.isclose(values,value,atol=1e-11,rtol=1e-13)
        if block.sum()>1:spectral[block]=spectral[block].mean()
    mean=(U*spectral)@U.T
    continuous_normalizer=float(np.sum(angular_weights*_polar_integral(g,0.)))
    tail_bound=float(np.exp(-48)/continuous_normalizer) if high.any() else 0.
    return dict(mean=(mean+mean.T)/2,log_normalizer=float(values[0]+log_integral),
                normal_nodes=local@U.T,weights=posterior_weights,
                radial_order=int(radial_order),angular_order=int(angular_order),
                omitted_radial_relative_bound=tail_bound,span=float(gaps[-1]))
