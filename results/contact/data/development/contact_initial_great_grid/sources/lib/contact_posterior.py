"""Deterministic real Gr(2,3) summaries for the constrained-contact pilot.

Complement duality turns the tangent plane into a normal line with parameter
-A. The existing spectral backend is used within its validated concentration
range. At the much larger contact concentrations a local, concentration-adapted
sphere quadrature supplies independently checked moments. Scalar response CDFs
integrate the radial coordinate continuously: they are not discrete-node
quantiles. All error estimates here are resolution comparisons, not rigorous
floating-point or integration enclosures.
"""
from __future__ import annotations

from functools import lru_cache
from time import perf_counter
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import dawsn, logsumexp

from .moments import mean_bingham
from .reference import rank_one_regions_gr13


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


def contact_mean(A,strict=False):
    """Tangent-plane mean plus numerical diagnostics, with no MCMC fallback."""
    A=_matrix(A);begin=perf_counter()
    low=normal_quadrature(-A,48,96);high=normal_quadrature(-A,64,192)
    quadrature_error=float(np.linalg.norm(low['mean']-high['mean'],2))
    logz_error=float(abs(low['log_normalizer']-high['log_normalizer']))
    span=high['span'];backend='concentration-adapted sphere quadrature'
    normal_mean=high['mean'];series_error=np.nan
    if span<=600.:
        normal_mean=mean_bingham(-A,1,backend='series',tol=1e-11)
        series_error=float(np.linalg.norm(normal_mean-high['mean'],2))
        backend='existing spectral series, checked by sphere quadrature'
    passed=bool(quadrature_error<1e-7 and logz_error<1e-7
                and (not np.isfinite(series_error) or series_error<1e-7)
                and high['omitted_radial_relative_bound']<1e-10)
    M=np.eye(3)-normal_mean
    diagnostics=dict(numerical_ok=passed,mean_backend=backend,span=span,
        mean_resolution_error=quadrature_error,log_normalizer_resolution_error=logz_error,
        series_reference_error=series_error,radial_relative_tail_bound=high['omitted_radial_relative_bound'],
        mean_seconds=perf_counter()-begin,mean_radial_order=64,mean_angular_order=192,
        log_normalizer=float(np.trace(A)+high['log_normalizer']),
        trace_error=float(abs(np.trace(M)-2)))
    if strict and not passed:raise ArithmeticError('Contact mean quadrature did not resolve its numerical budget')
    return M,diagnostics


class _ResponseCDF:
    """Continuous radial response CDF with numerical angular integration only."""
    def __init__(self,C,angular_order):
        self.C=_matrix(C);values,self.U,gaps=_spectrum(C)
        cosine,sine,self.g,self.weights=_angular_geometry(gaps,angular_order)
        self.V=cosine[:,None]*self.U[:,1]+sine[:,None]*self.U[:,2]
        self.full_radial=_polar_integral(self.g,0.)
        self.normalizer=float(self.weights@self.full_radial)

    def quantiles(self,queries,probabilities=(.025,.975)):
        """All query/component probabilities, using vectorized bisection.

        At each azimuth, response_i=C+A*cos(2theta)+B*sin(2theta).
        Its threshold has at most two roots on theta in [0,pi/2]. Each
        resulting radial interval is integrated by the Dawson primitive.
        """
        W=np.asarray(queries,dtype=float);probabilities=np.asarray(probabilities,dtype=float)
        if W.ndim!=2 or W.shape[1]!=3 or not np.isfinite(W).all() or np.any(np.linalg.norm(W,axis=1)<=0):
            raise ValueError('queries must be nonzero finite row-wise 3-vectors')
        if probabilities.ndim!=1 or np.any(probabilities<=0) or np.any(probabilities>=1):
            raise ValueError('probabilities must lie strictly inside (0,1)')
        count=len(probabilities);u=self.U[:,0]
        component=np.tile(np.arange(3),len(W));w=np.repeat(W,3,axis=0)
        wu=w@u;wv=w@self.V.T
        a=u[component]*wu;b=self.V.T[component]*wv
        c=u[component,None]*wv+self.V.T[component]*wu[:,None]
        A=-(a[:,None]-b)/2;B=-c/2
        C=w[np.arange(len(w)),component,None]-(a[:,None]+b)/2
        R=np.hypot(A,B);delta=np.arctan2(B,A)
        A,B,C,R,delta=[np.repeat(x,count,axis=0) for x in (A,B,C,R,delta)]
        scale=max(float(np.linalg.norm(W,axis=1).max()),np.finfo(float).tiny)
        tiny=scale*1e-14

        def cdf(x):
            z=np.divide(x[:,None]-C,R,out=np.zeros_like(R),where=R>tiny)
            angle=np.arccos(np.clip(z,-1.,1.))
            r1=np.mod(delta+angle,2*np.pi)/2;r2=np.mod(delta-angle,2*np.pi)/2
            r1=np.minimum(r1,np.pi/2);r2=np.minimum(r2,np.pi/2)
            lower,upper=np.minimum(r1,r2),np.maximum(r1,r2)
            roots=[np.zeros_like(lower),lower,upper,np.full_like(lower,np.pi/2)]
            primitive=[np.zeros_like(lower),_polar_integral(self.g,np.cos(lower)),
                       _polar_integral(self.g,np.cos(upper)),np.broadcast_to(self.full_radial,lower.shape)]
            mass=np.zeros_like(R)
            for j in range(3):
                mid=roots[j]+roots[j+1]
                included=C+A*np.cos(mid)+B*np.sin(mid)<=x[:,None]
                mass+=(primitive[j+1]-primitive[j])*included
            mass=np.where(x[:,None]-C>=R,self.full_radial,
                          np.where(x[:,None]-C<=-R,0.,mass))
            mass=np.where(R<=tiny,np.where(C<=x[:,None],self.full_radial,0.),mass)
            return np.clip((mass*self.weights).sum(axis=1)/self.normalizer,0.,1.)

        lengths=np.linalg.norm(w,axis=1);component_value=w[np.arange(len(w)),component]
        lower=np.repeat((component_value-lengths)/2,count)
        upper=np.repeat((component_value+lengths)/2,count)
        target=np.tile(probabilities,len(w))
        for _ in range(36):
            midpoint=(lower+upper)/2;below=cdf(midpoint)<target
            lower=np.where(below,midpoint,lower);upper=np.where(below,upper,midpoint)
        return ((lower+upper)/2).reshape(len(W),3,count)


def response_quantiles(A,queries,mass=.95,angular_orders=(64,128),max_order=512):
    """Component intervals with absolute and interval-relative resolution gates.

    Adaptive refinement uses no ground truth. The endpoint change must be
    below 2% of the refined interval width, with a 5e-6*query-norm floor for
    nearly constant functionals, and below 0.25% of query norm absolutely.
    Failure at the declared cap remains a numerical failure, not a wider band.
    """
    A=_matrix(A);W=np.asarray(queries,dtype=float)
    if not 0<mass<1:raise ValueError('mass must lie in (0,1)')
    low_order,order=map(int,angular_orders)
    if low_order<32 or order<=low_order or max_order<order:raise ValueError('invalid angular resolution budgets')
    probabilities=((1-mass)/2,(1+mass)/2);begin=perf_counter()
    low=_ResponseCDF(-A,low_order).quantiles(W,probabilities)
    attempts=0
    while True:
        attempts+=1;high=_ResponseCDF(-A,order).quantiles(W,probabilities)
        delta=np.max(abs(high-low),axis=2);width=high[:,:,1]-high[:,:,0]
        scales=np.linalg.norm(W,axis=1)[:,None]
        absolute=float(np.max(delta/scales))
        relative=float(np.max(delta/np.maximum(width,2.5e-4*scales)))
        passed=bool(absolute<=.0025 and np.all(delta<=np.maximum(.02*width,5e-6*scales)))
        if passed or order>=max_order:break
        low=high;low_order=order;order=min(max_order,2*order)
    return high[:,:,0],high[:,:,1],dict(response_quantiles_ok=passed,
        quantile_scaled_error=absolute,quantile_relative_error=relative,
        quantile_angular_order=order,quantile_previous_order=low_order,
        quantile_refinements=attempts,quantile_seconds=perf_counter()-begin,
        quantile_method='Continuous radial CDF; warped azimuth quadrature and bisection')


def contact_summary(A,queries,true_normal=None,noise_variance=None,mass=.95,strict=False):
    """Mean, clean response intervals, state regions, risk and numerical checks.

    No noisy-observation interval is inferred from a Gaussian moment shortcut.
    The pilot uses clean intervals; a non-None noise_variance is rejected so a
    caller cannot mistake clean uncertainty for observation prediction.
    """
    if noise_variance is not None:
        raise ValueError('This bounded pilot computes clean intervals only; noisy intervals require a separate Gaussian-mixture CDF')
    A=_matrix(A);W=np.asarray(queries,dtype=float);begin=perf_counter()
    M,diagnostics=contact_mean(A)
    diagnostics['mean_numerical_ok']=diagnostics['numerical_ok']
    values,U,gaps=_spectrum(-A);normal=U[:,0];Pm=np.eye(3)-np.outer(normal,normal)
    coarse_regions=rank_one_regions_gr13(-A,mass=mass,epsabs=1e-11)
    regions=rank_one_regions_gr13(-A,mass=mass,epsabs=1e-13)
    lower,upper,quantile_diagnostics=response_quantiles(A,W,mass=mass)
    diagnostics.update(quantile_diagnostics)
    radius_delta=abs(regions.radius-coarse_regions.radius)
    hpd_delta=abs(regions.hpd_loss-coarse_regions.hpd_loss)
    diagnostics.update(region_integration_error=regions.integration_error,
        ball_probability=regions.ball_mass,hpd_probability=regions.hpd_mass,
        moment_region_logz_error=abs(diagnostics['log_normalizer']-(np.trace(A)+regions.log_normalizer)),
        ball_radius_resolution_error=radius_delta,hpd_threshold_resolution_error=hpd_delta,
        region_radius_scaled_error=radius_delta/max(regions.radius,1e-3),
        region_hpd_scaled_error=hpd_delta/max(regions.hpd_loss,1.),
        region_previous_epsabs=1e-11,region_epsabs=1e-13,
        mode_unique=bool(gaps[0]>1e-10),integration_seconds=perf_counter()-begin)
    diagnostics['regions_numerical_ok']=bool(regions.integration_error<1e-6
        and diagnostics['moment_region_logz_error']<1e-7
        and diagnostics['region_radius_scaled_error']<1e-6
        and diagnostics['region_hpd_scaled_error']<1e-5)
    passed=bool(diagnostics['mean_numerical_ok'] and diagnostics['response_quantiles_ok']
                and diagnostics['regions_numerical_ok'])
    diagnostics['numerical_ok']=passed
    result=dict(mean=M,mode_projector=Pm,mode_normal=normal,
        ball_radius=regions.radius,hpd_threshold=regions.hpd_loss,
        response_lower=lower,response_upper=upper,
        risks=np.einsum('qi,ij,qj->q',W,M-M@M,W),numerics=diagnostics)
    if true_normal is not None:
        n=np.asarray(true_normal,dtype=float)
        if n.shape!=(3,) or not np.isclose(n@n,1.,atol=1e-9):raise ValueError('true_normal must be a unit 3-vector')
        distance=float(np.sqrt(max(0.,1-(n@normal)**2)))
        loss=float(values[0]-n@(-A)@n)
        result.update(ball_covered=bool(distance<=regions.radius),hpd_covered=bool(loss<=regions.hpd_loss),
                      chordal_error=distance,true_hpd_loss=loss)
    if strict and not passed:raise ArithmeticError('Contact posterior summary did not resolve its numerical budget')
    return result
