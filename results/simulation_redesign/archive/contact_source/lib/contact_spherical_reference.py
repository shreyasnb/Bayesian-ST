"""Independent even-spherical-harmonic reference for normal-line filtering.

The transition is a zonal Watson kernel on the unoriented sphere. Funk--Hecke
convolution multiplies each even-degree coefficient, avoiding an all-pairs
spherical kernel matrix. Observations are multiplied in physical coordinates.
The maintained density is not projected onto a Bingham family.
"""
from __future__ import annotations

from functools import lru_cache
from time import perf_counter
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.special import eval_legendre, sph_harm_y
from .models import increment,constant


def _unit(x):
    x=np.asarray(x,dtype=float)
    return x/np.linalg.norm(x)


def tangent_basis(n):
    n=_unit(n);axis=np.eye(3)[:,np.argmin(abs(n))]
    first=np.cross(n,axis);first/=np.linalg.norm(first)
    return np.column_stack([first,np.cross(n,first)])


def sphere_exp(n,coordinates,basis=None):
    n=_unit(n);E=tangent_basis(n) if basis is None else np.asarray(basis)
    x=np.asarray(coordinates);radius=float(np.linalg.norm(x))
    return np.cos(radius)*n+np.sinc(radius/np.pi)*(E@x)


def sample_watson(previous,kappa,rng):
    """Independent exact rejection draw, antipodal sign immaterial.

    |n^T m| has density proportional to exp(kappa*q^2), q in[0,1].
    The accepted q and a uniform azimuth produce the invariant transition.
    """
    m=_unit(previous)
    if not np.isfinite(kappa) or kappa<0:raise ValueError('Finite nonnegative kappa required')
    if kappa==0:return _unit(rng.normal(size=3))
    while True:
        q=rng.random(64);accept=rng.random(64)<np.exp(kappa*(q*q-1))
        if accept.any():q=float(q[np.flatnonzero(accept)[0]]);break
    phi=rng.uniform(0,2*np.pi);E=tangent_basis(m)
    n=q*m+np.sqrt(max(0.,1-q*q))*(E@np.array([np.cos(phi),np.sin(phi)]))
    return n if rng.random()<.5 else -n


@lru_cache(maxsize=16)
def sphere_grid(z_order,phi_order):
    z,w=leggauss(int(z_order));phi=2*np.pi*np.arange(int(phi_order))/phi_order
    Z,PHI=np.meshgrid(z,phi,indexing='ij');R=np.sqrt(1-Z*Z)
    nodes=np.column_stack([R.ravel()*np.cos(PHI).ravel(),R.ravel()*np.sin(PHI).ravel(),Z.ravel()])
    weights=np.repeat(w/(2*phi_order),phi_order)
    return nodes,weights


def harmonic_basis(nodes,degree):
    """Real even harmonics orthonormal for normalized sphere measure."""
    nodes=np.asarray(nodes);theta=np.arccos(np.clip(nodes[:,2],-1,1));phi=np.arctan2(nodes[:,1],nodes[:,0])
    columns=[];degrees=[]
    for ell in range(0,int(degree)+1,2):
        y=sph_harm_y(ell,0,theta,phi)
        columns.append(np.sqrt(4*np.pi)*y.real);degrees.append(ell)
        for m in range(1,ell+1):
            y=sph_harm_y(ell,m,theta,phi)*((-1.)**m)*np.sqrt(8*np.pi)
            columns.extend([y.real,y.imag]);degrees.extend([ell,ell])
    return np.asarray(columns).T,np.asarray(degrees)


@lru_cache(maxsize=8)
def grid_basis(degree,z_order,phi_order):
    nodes,weights=sphere_grid(z_order,phi_order);basis,degrees=harmonic_basis(nodes,degree)
    return nodes,weights,basis,degrees


@lru_cache(maxsize=64)
def kernel_multipliers(kappa,degree):
    if not np.isfinite(kappa) or kappa<0:raise ValueError('This smoothing reference requires finite kappa>=0')
    q,w=leggauss(max(128,2*degree+8));weighted=w*np.exp(kappa*(q*q-1))
    denominator=weighted.sum()
    result={ell:float(weighted@eval_legendre(ell,q)/denominator) for ell in range(0,degree+1,2)}
    result[0]=1.
    return result


def weighted_quantile(values,weights,probabilities):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    v,cdf=cdf_knots(v,w)
    return np.interp(probabilities,cdf,v,left=v[0],right=v[-1])


def cdf_knots(sorted_values,sorted_weights):
    """Coalesce equal function values before midpoint-mass interpolation.

    Axis queries can have an entire azimuth ring at one function value. Treating
    each quadrature node as a separate atom creates spurious CDF staircases.
    Positive grouped quadrature weights define a monotone interpolated CDF;
    its independent-grid refinement remains an empirical numerical check.
    """
    values=np.asarray(sorted_values);weights=np.asarray(sorted_weights)
    tolerance=1e-12*max(float(np.max(abs(values))),1e-12)
    starts=np.r_[0,np.flatnonzero(np.diff(values)>tolerance)+1]
    grouped=np.add.reduceat(weights,starts)
    representatives=values[starts]
    cdf=(np.cumsum(grouped)-.5*grouped)/grouped.sum()
    return representatives,cdf


def supported_knots(values,weights,lower,upper):
    order=np.argsort(values);x,cdf=cdf_knots(np.asarray(values)[order],np.asarray(weights)[order])
    interior=(x>lower+1e-14*max(1.,abs(lower)))&(x<upper-1e-14*max(1.,abs(upper)))
    return np.r_[lower,x[interior],upper],np.r_[0.,cdf[interior],1.]


def functional_quantiles(z_order,phi_order,queries,posterior_weights,probabilities,diagonal_marginals=None):
    nodes,_=sphere_grid(z_order,phi_order);W=np.asarray(queries)
    flattened,indices,sorted_values=query_grid(z_order,phi_order,tuple(map(tuple,W)))
    results=[]
    for j in range(flattened.shape[1]):
        query,component=divmod(j,3);w=W[query];length=np.linalg.norm(w)
        if np.count_nonzero(abs(w)>1e-13*length)==1 and abs(w[component])>.99*length:
            # A repeated functional eigenvalue creates a square-root CDF at
            # y=a. Integrate its regular coordinate |n_i| and transform back.
            x,cdf=(diagonal_marginals[component] if diagonal_marginals is not None
                   else supported_knots(abs(nodes[:,component]),posterior_weights,0.,1.))
            target=1-np.asarray(probabilities) if w[component]>0 else np.asarray(probabilities)
            h=np.interp(target,cdf,x)
            results.append(w[component]*(1-h*h))
        else:
            lo,hi=(w[component]-length)/2,(w[component]+length)/2
            x,cdf=supported_knots(sorted_values[:,j],np.asarray(posterior_weights)[indices[:,j]],lo,hi)
            results.append(np.interp(probabilities,cdf,x))
    return np.asarray(results).reshape(len(W),3,len(probabilities))


def grid_cdf(z_order,phi_order,queries,posterior_weights,endpoints,diagonal_marginals=None):
    nodes,_=sphere_grid(z_order,phi_order);W=np.asarray(queries)
    _,indices,sorted_values=query_grid(z_order,phi_order,tuple(map(tuple,W)))
    values=[]
    for j,target in enumerate(np.asarray(endpoints).ravel()):
        query,component=divmod(j,3);w=W[query];length=np.linalg.norm(w)
        if np.count_nonzero(abs(w)>1e-13*length)==1 and abs(w[component])>.99*length:
            x,cdf=(diagonal_marginals[component] if diagonal_marginals is not None
                   else supported_knots(abs(nodes[:,component]),posterior_weights,0.,1.))
            h=np.sqrt(np.clip(1-target/w[component],0.,1.));prob=float(np.interp(h,x,cdf))
            values.append(1-prob if w[component]>0 else prob)
        else:
            lo,hi=(w[component]-length)/2,(w[component]+length)/2
            x,cdf=supported_knots(sorted_values[:,j],np.asarray(posterior_weights)[indices[:,j]],lo,hi)
            values.append(float(np.interp(target,x,cdf,left=0.,right=1.)))
    return np.asarray(values)


@lru_cache(maxsize=8)
def query_grid(z_order,phi_order,query_tuple):
    nodes,_=sphere_grid(z_order,phi_order);W=np.asarray(query_tuple)
    responses=W[None,:,:]-nodes[:,None,:]*np.einsum('ni,qi->nq',nodes,W)[:,:,None]
    flattened=responses.reshape(len(nodes),-1)
    indices=np.argsort(flattened,axis=0).astype(np.int32)
    sorted_values=np.take_along_axis(flattened,indices,axis=0)
    return flattened,indices,sorted_values


@lru_cache(maxsize=16)
def polar_axis_rotation(degree,component):
    """Exact harmonic coordinate change taking physical axis i to polar z.

    Each even-degree block is integrated on a polynomial-exact analysis grid.
    The returned map changes coefficients, not the density approximation order.
    """
    Q=np.eye(3)
    Q[:,[component,2]]=Q[:,[2,component]]
    if component==2:return Q,np.eye((degree+1)*(degree+2)//2)
    nodes,w,Y,degrees=grid_basis(degree,degree+8,2*(degree+8))
    rotated,_=harmonic_basis(nodes@Q.T,degree)
    transform=np.zeros((Y.shape[1],Y.shape[1]))
    for ell in range(0,degree+1,2):
        selected=np.flatnonzero(degrees==ell)
        transform[np.ix_(selected,selected)]=Y[:,selected].T@(w[:,None]*rotated[:,selected])
    return Q,transform


class SphericalReference:
    """Full recursive density with a fixed declared harmonic/integration budget.

    A finite Watson transition smooths each posterior before the next
    correction. The correction itself is retained as harmonic-prior times the
    exact Gaussian likelihood, rather than projecting the posterior density.
    Small negative spectral leakage is reported; a substantial correction
    marks the reference unresolved.
    """
    def __init__(self,kappa,degree=32,z_order=None,phi_order=None):
        if degree<4 or degree%2:raise ValueError('Even degree>=4 required')
        self.kappa=float(kappa);self.degree=int(degree)
        self.z_order=int(z_order or degree+8);self.phi_order=int(phi_order or 2*self.z_order)
        self.nodes,self.weights,self.Y,self.degrees=grid_basis(self.degree,self.z_order,self.phi_order)
        multipliers=kernel_multipliers(self.kappa,self.degree)
        self.multipliers=np.array([multipliers[int(ell)] for ell in self.degrees])
        self.posterior=np.ones(len(self.nodes));self.prior=np.ones(len(self.nodes))
        self.prior_coefficients=np.zeros(self.Y.shape[1]);self.prior_coefficients[0]=1.
        self.B=np.zeros((3,3));self.energy_shift=0.;self.log_increment=0.
        self.prepared=False;self.t=0;self.prediction_seconds=0.;self.update_seconds=0.
        self.maximum_negative_mass=0.;self.maximum_normalization_error=0.
        self.minimum_prediction_density=1.;self.numerical_ok=True
        self._marginal_cache={}

    def mean(self):
        normal=np.einsum('n,ni,nj->ij',self.weights*self.posterior,self.nodes,self.nodes)
        return np.eye(3)-(normal+normal.T)/2

    def predict(self):
        if self.prepared:return self.prior
        start=perf_counter()
        coefficients=self.Y.T@(self.weights*self.posterior)
        coefficients*=self.multipliers
        values=self.Y@coefficients
        negative=float(self.weights@np.maximum(-values,0.))
        mass=float(self.weights@values)
        self.maximum_negative_mass=max(self.maximum_negative_mass,negative)
        self.maximum_normalization_error=max(self.maximum_normalization_error,abs(mass-1))
        self.minimum_prediction_density=min(self.minimum_prediction_density,float(values.min()))
        self.numerical_ok &= negative<1e-9 and abs(mass-1)<1e-7
        self.prior_coefficients=coefficients
        self.prior=np.maximum(values,0.)/float(self.weights@np.maximum(values,0.))
        self.prepared=True;self.prediction_seconds+=perf_counter()-start
        return self.prior

    def update(self,u,v,s):
        self.predict();start=perf_counter();self.B=increment(np.asarray(u),s,v=np.asarray(v))
        self.energy_shift=float(np.trace(self.B)-np.linalg.eigvalsh(self.B)[0])
        energy=np.trace(self.B)-np.einsum('ni,ij,nj->n',self.nodes,self.B,self.nodes)
        values=self.prior*np.exp(energy-self.energy_shift)
        normalizer=float(self.weights@values)
        if not normalizer>0:raise ArithmeticError('Reference likelihood normalization failed')
        self.log_increment=float(np.log(normalizer))
        self.posterior=values/normalizer;self.prepared=False;self.t+=1
        self._marginal_cache.clear()
        self.update_seconds+=perf_counter()-start
        return float(constant(np.asarray(u),2,s,v=np.asarray(v))+self.energy_shift+self.log_increment)

    def density(self,nodes):
        Y,_=harmonic_basis(np.asarray(nodes),self.degree)
        return self._density_from_basis(np.asarray(nodes),Y)

    def _density_from_basis(self,nodes,Y):
        prior=Y@self.prior_coefficients
        energy=np.trace(self.B)-np.einsum('ni,ij,nj->n',nodes,self.B,nodes)
        return prior*np.exp(energy-self.energy_shift-self.log_increment)

    def diagonal_marginals(self,z_order,phi_order):
        """Regular |n_i| CDFs without artificial azimuth-ring atoms at n_i=0.

        Rotating the polar integration axis resolves the square-root response
        endpoint y=w_i(1-n_i^2). Midpoint interpolation is applied to the smooth
        |n_i| marginal, and its error is checked on independent polar orders.
        """
        key=(int(z_order),int(phi_order))
        if key in self._marginal_cache:return self._marginal_cache[key]
        nodes,w,Y,_=grid_basis(self.degree,*key)
        result=[]
        for component in range(3):
            Q,transform=polar_axis_rotation(self.degree,component)
            prior=Y@(transform@self.prior_coefficients)
            B=Q.T@self.B@Q
            energy=np.trace(B)-np.einsum('ni,ij,nj->n',nodes,B,nodes)
            density=np.maximum(prior,0.)*np.exp(energy-self.energy_shift-self.log_increment)
            weights=w*density;weights/=weights.sum()
            result.append(supported_knots(abs(nodes[:,2]),weights,0.,1.))
        self._marginal_cache[key]=result
        return result

    def mode(self):
        initial=self.nodes[int(np.argmax(self.posterior))];E=tangent_basis(initial)
        def objective(x):
            value=float(self.density(sphere_exp(initial,x,E)[None,:])[0])
            return -np.log(max(value,1e-300))
        result=minimize(objective,np.zeros(2),method='BFGS',options={'gtol':1e-7,'maxiter':60})
        self.mode_diagnostics=dict(optimizer_success=bool(result.success),
            gradient_norm=float(np.linalg.norm(result.jac)),iterations=int(result.nit),
            interpretation='Numerically located maximum initialized at the largest analysis-grid density')
        return sphere_exp(initial,result.x,E)

    def summary(self,queries,true_normal=None,z_order=96,phi_order=192,mass=.95):
        start=perf_counter();W=np.asarray(queries,dtype=float)
        nodes,w,Y,_=grid_basis(self.degree,int(z_order),int(phi_order))
        raw=self._density_from_basis(nodes,Y);raw_mass=float(w@raw)
        negative=float(w@np.maximum(-raw,0.));p=np.maximum(raw,0.)
        p/=float(w@p);posterior_weights=w*p
        N=np.einsum('n,ni,nj->ij',posterior_weights,nodes,nodes);M=np.eye(3)-(N+N.T)/2
        normal=self.mode();modeP=np.eye(3)-np.outer(normal,normal)
        distance=np.sqrt(np.maximum(0.,1-(nodes@normal)**2))
        radius=float(weighted_quantile(distance,posterior_weights,[mass])[0])
        level=float(weighted_quantile(p,posterior_weights,[1-mass])[0])
        probabilities=[(1-mass)/2,(1+mass)/2]
        marginals=self.diagonal_marginals(int(z_order),int(phi_order))
        quantiles=functional_quantiles(int(z_order),int(phi_order),W,posterior_weights,probabilities,marginals)
        ball_mass=float(posterior_weights[distance<=radius].sum())
        hpd_mass=float(posterior_weights[p>=level].sum())
        result=dict(mean=M,mode_projector=modeP,mode_normal=normal,ball_radius=radius,
            hpd_density_level=level,hpd_threshold=level,response_lower=quantiles[:,:,0],response_upper=quantiles[:,:,1],
            risks=np.einsum('qi,ij,qj->q',W,M-M@M,W),
            numerics=dict(numerical_ok=bool(self.numerical_ok and negative<1e-9 and abs(raw_mass-1)<1e-5),
                          normalization_error=abs(raw_mass-1),negative_mass=negative,
                          cumulative_negative_mass=self.maximum_negative_mass,
                          ball_probability=ball_mass,hpd_probability=hpd_mass,
                          mean_analysis_error=float(np.linalg.norm(M-self.mean(),2)),
                          harmonic_degree=self.degree,z_order=z_order,phi_order=phi_order,
                          summary_seconds=perf_counter()-start,
                          mode_gradient_norm=self.mode_diagnostics['gradient_norm'],
                          mode_optimizer_success=self.mode_diagnostics['optimizer_success'],
                          scalar_cdf='Coalesced midpoint-mass interpolation on physical support; diagonal responses use axis-rotated polar |n_i| marginal; grid-refined',
                          region_kind='Full-density superlevel set; threshold is density, not Bingham loss'))
        if true_normal is not None:
            n=_unit(true_normal);distance=float(np.sqrt(max(0.,1-(normal@n)**2)))
            density=float(self.density(n[None,:])[0]/max(raw_mass,1e-300))
            result.update(ball_covered=bool(distance<=radius),hpd_covered=bool(density>=level),
                          chordal_error=distance,true_density=density)
        return result


def compare_reference(coarse,fine,queries,true_normal=None,coarse_order=128,fine_order=192):
    """Resolution audit; numerical targets are retained even when unresolved."""
    a=coarse.summary(queries,true_normal,z_order=coarse_order,phi_order=2*coarse_order)
    b=fine.summary(queries,true_normal,z_order=fine_order,phi_order=2*fine_order)
    nodes,w,Yfine,_=grid_basis(fine.degree,fine_order,2*fine_order)
    _,_,Ycoarse,_=grid_basis(coarse.degree,fine_order,2*fine_order)
    pa=coarse._density_from_basis(nodes,Ycoarse);pb=fine._density_from_basis(nodes,Yfine)
    normalizer_a=float(w@pa);normalizer_b=float(w@pb)
    pa=np.maximum(pa,0.)/float(w@np.maximum(pa,0.));pb=np.maximum(pb,0.)/float(w@np.maximum(pb,0.))
    tv=float(.5*w@abs(pa-pb));mean_error=float(np.linalg.norm(a['mean']-b['mean'],2))
    endpoints=max(float(np.max(abs(a[key]-b[key]))) for key in ['response_lower','response_upper'])
    width=b['response_upper']-b['response_lower'];scales=np.linalg.norm(queries,axis=1)[:,None]
    relative=float(max(np.max(abs(a[key]-b[key])/np.maximum(width,1e-3*scales)) for key in ['response_lower','response_upper']))
    distance=np.sqrt(np.maximum(0.,1-(nodes@b['mode_normal'])**2))
    coarse_nodes,coarse_w,coarse_Y,_=grid_basis(coarse.degree,coarse_order,2*coarse_order)
    coarse_density=coarse._density_from_basis(coarse_nodes,coarse_Y)
    coarse_density=np.maximum(coarse_density,0.)/float(coarse_w@np.maximum(coarse_density,0.))
    coarse_distances=np.sqrt(np.maximum(0.,1-(coarse_nodes@b['mode_normal'])**2))
    ball_mass_a=float(coarse_w@((coarse_distances<=b['ball_radius'])*coarse_density))
    level=b['hpd_density_level'];hpd_mass_a=float(coarse_w@((coarse_density>=level)*coarse_density))
    refined_ball_mass=float(w@((distance<=b['ball_radius'])*pb))
    refined_hpd_mass=float(w@((pb>=level)*pb))
    region_error=max(abs(ball_mass_a-refined_ball_mass),abs(hpd_mass_a-refined_hpd_mass))
    # Report CDF differences at the fine endpoints, not only quantile positions.
    coarse_marginals=coarse.diagonal_marginals(coarse_order,2*coarse_order)
    fine_marginals=fine.diagonal_marginals(fine_order,2*fine_order)
    cdf_error=0.
    for endpoint in ['response_lower','response_upper']:
        coarse_cdf=grid_cdf(coarse_order,2*coarse_order,queries,coarse_w*coarse_density,b[endpoint],coarse_marginals)
        fine_cdf=grid_cdf(fine_order,2*fine_order,queries,w*pb,b[endpoint],fine_marginals)
        cdf_error=max(cdf_error,float(np.max(abs(coarse_cdf-fine_cdf))))
    ok=bool(a['numerics']['numerical_ok'] and b['numerics']['numerical_ok'] and mean_error<2e-4
            and tv<2e-3 and region_error<.005 and cdf_error<.005 and relative<.02)
    audit=dict(reference_numerical_ok=ok,coarse_degree=coarse.degree,fine_degree=fine.degree,
        density_tv=tv,mean_error=mean_error,coarse_summary_order=coarse_order,fine_summary_order=fine_order,
        quantile_max_absolute_error=endpoints,
        quantile_max_width_relative_error=relative,region_mass_error=region_error,
        scalar_cdf_error=cdf_error,coarse_normalization_error=abs(normalizer_a-1),
        fine_normalization_error=abs(normalizer_b-1),
        integration_scope='Even harmonic cutoff plus two independent sphere-grid resolutions; empirical convergence, not a rigorous enclosure')
    return b,audit
