"""Independent full-density filter on complex CP(1), using its Bloch sphere.

All spherical harmonic degrees are retained.  A complex line projector is
P=(I+r.sigma)/2 and tr(P_r P_v)=(1+r.v)/2; its invariant transition is therefore
a von Mises--Fisher kernel exp(kappa*r.v/2), not the real Watson kernel.
Quadratic-functional CDFs are integrated analytically in Legendre coordinates.
"""
from functools import lru_cache
from time import perf_counter
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import sph_harm_y, eval_legendre, ive
from .models import increment, constant


def bloch(vector):
    h=np.asarray(vector,dtype=complex)
    if h.shape!=(2,) or not np.isfinite(h).all() or np.linalg.norm(h)==0:
        raise ValueError('A nonzero complex two-vector is required')
    h=h/np.linalg.norm(h);cross=np.conj(h[0])*h[1]
    return np.array([2*cross.real,2*cross.imag,abs(h[0])**2-abs(h[1])**2])


def bloch_projectors(nodes):
    r=np.asarray(nodes,float);P=np.empty(r.shape[:-1]+(2,2),complex)
    P[...,0,0]=(1+r[...,2])/2;P[...,1,1]=(1-r[...,2])/2
    P[...,0,1]=(r[...,0]-1j*r[...,1])/2;P[...,1,0]=np.conj(P[...,0,1])
    return P


def harmonic_basis(nodes,degree):
    """Real harmonics orthonormal under probability measure dOmega/(4*pi)."""
    r=np.atleast_2d(np.asarray(nodes,float))
    theta=np.arccos(np.clip(r[:,2],-1,1));phi=np.arctan2(r[:,1],r[:,0])
    columns=[];degrees=[]
    for ell in range(int(degree)+1):
        columns.append(np.sqrt(4*np.pi)*sph_harm_y(ell,0,theta,phi).real);degrees.append(ell)
        for m in range(1,ell+1):
            y=(-1.)**m*np.sqrt(8*np.pi)*sph_harm_y(ell,m,theta,phi)
            columns.extend([y.real,y.imag]);degrees.extend([ell,ell])
    return np.asarray(columns).T,np.asarray(degrees)


@lru_cache(maxsize=6)
def grid_basis(degree,z_order,phi_order):
    z,w=leggauss(z_order);phi=2*np.pi*np.arange(phi_order)/phi_order
    Z,F=np.meshgrid(z,phi,indexing='ij');R=np.sqrt(1-Z*Z)
    nodes=np.column_stack([R.ravel()*np.cos(F).ravel(),R.ravel()*np.sin(F).ravel(),Z.ravel()])
    weights=np.repeat(w/(2*phi_order),phi_order)
    Y,degrees=harmonic_basis(nodes,degree)
    return nodes,weights,Y,(Y.T*weights),degrees


@lru_cache(maxsize=64)
def kernel_multipliers(kappa,degree):
    if np.isnan(kappa) or kappa<0:raise ValueError('Nonnegative transition concentration required')
    if np.isposinf(kappa):return np.ones(degree+1)
    if kappa==0:return np.r_[1.,np.zeros(degree)]
    c=kappa/2
    values=ive(np.arange(degree+1)+.5,c)/ive(.5,c)
    values[0]=1.
    return values


def weighted_quantile(values,weights,probability):
    order=np.argsort(values);v=np.asarray(values)[order];w=np.asarray(weights)[order]
    cumulative=(np.cumsum(w)-w/2)/w.sum()
    return float(np.interp(probability,cumulative,v))


class ComplexLineReference:
    """Full-density harmonic recursion with fixed finite numerical budgets.

    ``update`` predicts from the second state onward. Initial state 1 is
    uniform, so a T-state sequence has T-1 nontrivial transition predictions.
    Spectral negatives are measured and retained; propagation does not clip
    the signed truncated expansion. Positive weights used for HPD reporting
    are explicitly flagged if the removed mass exceeds the stated budget.
    """
    def __init__(self,kappa,degree=24,z_order=48,phi_order=96):
        if degree<2 or z_order<=degree or phi_order<=2*degree:
            raise ValueError('Grid must resolve the requested harmonic degree')
        self.kappa=float(kappa);self.degree=int(degree)
        self.z_order=int(z_order);self.phi_order=int(phi_order)
        self.nodes,self.weights,self.Y,self.Yw,self.degrees=grid_basis(degree,z_order,phi_order)
        self.coefficients=np.zeros(len(self.degrees));self.coefficients[0]=1.
        self.multipliers=kernel_multipliers(self.kappa,self.degree)[self.degrees]
        self.t=0;self.predicted=False;self.prediction_seconds=0.;self.update_seconds=0.
        self.minimum_density=1.;self.maximum_negative_mass=0.;self.maximum_normalization_error=0.
        self.last_log_score=np.nan

    def density(self,nodes=None):
        if nodes is None:return self.Y@self.coefficients
        return harmonic_basis(nodes,self.degree)[0]@self.coefficients

    def predict(self):
        if self.predicted:return
        begin=perf_counter();self.coefficients*=self.multipliers
        self.prediction_seconds+=perf_counter()-begin;self.predicted=True

    def update(self,u,s,q):
        if self.t and not self.predicted:self.predict()
        begin=perf_counter();u=np.asarray(u,dtype=complex)
        if u.shape!=(2,):raise ValueError('CP1 reference requires two complex channels')
        prior=self.density();normalization=float(self.weights@prior)
        self.maximum_normalization_error=max(self.maximum_normalization_error,abs(normalization-1))
        self.minimum_density=min(self.minimum_density,float(prior.min()))
        negative=float(self.weights@np.maximum(-prior,0))
        self.maximum_negative_mass=max(self.maximum_negative_mass,negative)
        B=increment(u,s,q,complex_data=True)
        potential=np.einsum('ij,bji->b',B,bloch_projectors(self.nodes)).real
        shift=float(potential.max());raw=prior*np.exp(potential-shift)
        mass=float(self.weights@raw)
        if mass<=0 or not np.isfinite(mass):raise ArithmeticError('Reference update normalization unresolved')
        self.last_log_score=float(constant(u,1,s,q,complex_data=True)+shift+np.log(mass))
        self.coefficients=self.Yw@(raw/mass);self.coefficients[0]=1.
        self.t+=1;self.predicted=False;self.update_seconds+=perf_counter()-begin
        return self.last_log_score

    def mean(self):
        density=self.density();r=np.einsum('b,bi->i',self.weights*density,self.nodes)
        return bloch_projectors(r)

    def _marginal_coefficients(self,queries):
        queries=np.atleast_2d(np.asarray(queries,complex))
        directions=np.asarray([bloch(q) for q in queries]);Y,_=harmonic_basis(directions,self.degree)
        return np.asarray([np.sum(Y[:,self.degrees==ell]*self.coefficients[self.degrees==ell],axis=1)
                           for ell in range(self.degree+1)]).T

    def quadratic_cdf(self,queries,values):
        """Continuous CDF of a^H P a, valid for nonzero row-wise queries."""
        Q=np.atleast_2d(np.asarray(queries,complex));v=np.asarray(values,float)
        if v.shape!=(len(Q),):raise ValueError('One scalar threshold per query required')
        norms=np.sum(abs(Q)**2,axis=1);B=self._marginal_coefficients(Q);result=[]
        for norm,coeff,value in zip(norms,B,v):
            p=float(value/norm)
            if p<=0:result.append(0.);continue
            if p>=1:result.append(1.);continue
            z=2*p-1;ell=np.arange(1,self.degree+1)
            correction=np.sum(coeff[1:]*(eval_legendre(ell+1,z)-eval_legendre(ell-1,z))/(2*(2*ell+1)))
            result.append(float(p+correction))
        return np.asarray(result)

    def quadratic_quantiles(self,queries,probabilities=(.025,.25,.5,.75,.975)):
        Q=np.atleast_2d(np.asarray(queries,complex));norms=np.sum(abs(Q)**2,axis=1)
        coefficients=self._marginal_coefficients(Q);result=np.empty((len(Q),len(probabilities)))
        ell=np.arange(1,self.degree+1)
        for j,(norm,coeff) in enumerate(zip(norms,coefficients)):
            def cdf(p):
                z=2*p-1
                return p+np.sum(coeff[1:]*(eval_legendre(ell+1,z)-eval_legendre(ell-1,z))/(2*(2*ell+1)))
            for k,p in enumerate(probabilities):result[j,k]=norm*brentq(lambda x:cdf(x)-p,0.,1.,xtol=1e-12)
        return result

    def summary(self,queries,true_h=None,mass=.95,tie_seed=72531):
        begin=perf_counter();M=self.mean();eigen,E=np.linalg.eigh(M)
        tied=np.isclose(eigen,eigen[-1],atol=1e-12,rtol=0)
        if tied.sum()>1:
            rng=np.random.default_rng(tie_seed);a=rng.normal(size=tied.sum())+1j*rng.normal(size=tied.sum())
            mode=E[:,tied]@(a/np.linalg.norm(a))
        else:mode=E[:,-1]
        uniform=bool(np.max(abs(self.coefficients[1:]))<1e-13)
        radius2=float(1-self.quadratic_quantiles(mode[None,:],[1-mass])[0,0])
        density=self.density();negative=float(self.weights@np.maximum(-density,0))
        positive=np.maximum(density,0);posterior_weights=self.weights*positive
        hpd_threshold=weighted_quantile(positive,posterior_weights,1-mass)
        mass_hpd=(mass if uniform else float(posterior_weights[density>=hpd_threshold].sum()/posterior_weights.sum()))
        quantiles=self.quadratic_quantiles(queries)
        numerical_ok=bool(max(negative,self.maximum_negative_mass)<1e-9 and self.maximum_normalization_error<1e-9)
        result=dict(mean=M,mode=mode,mode_projector=np.outer(mode,mode.conj()),ball_radius=np.sqrt(max(0.,radius2)),
            hpd_threshold=hpd_threshold,quadratic_quantiles=quantiles,uniform_density=uniform,
            numerics=dict(numerical_ok=numerical_ok,negative_mass=negative,
                maximum_negative_mass=self.maximum_negative_mass,minimum_density=self.minimum_density,
                normalization_error=self.maximum_normalization_error,hpd_mass=mass_hpd,
                hpd_mass_error=abs(mass_hpd-mass),summary_seconds=perf_counter()-begin,
                hpd_tie_kind='Auxiliary Haar-centred ball within uniform-density ties' if uniform else 'Continuous density threshold',
                density_representation='All-degree real spherical harmonics on the complex-line Bloch sphere',
                ball_center='Leading eigenspace of the full-density mean; not claimed as the density mode'))
        if true_h is not None:
            r=bloch(true_h);truth_density=float(self.density(r[None,:])[0])
            ball_covered=bool(1-abs(np.vdot(mode,true_h))**2<=radius2)
            result.update(ball_covered=ball_covered,
                hpd_covered=ball_covered if uniform else bool(truth_density>=hpd_threshold))
        return result


def compare_reference(coarse,fine,queries,true_h=None):
    low=coarse.summary(queries,true_h);high=fine.summary(queries,true_h)
    low_density=coarse.density(fine.nodes);high_density=fine.density()
    density_tv=float(.5*fine.weights@abs(low_density-high_density))
    mean_error=float(np.linalg.norm(low['mean']-high['mean'],2))
    scalar_error=0.
    for k,p in enumerate([.025,.25,.5,.75,.975]):
        scalar_error=max(scalar_error,float(np.max(abs(fine.quadratic_cdf(queries,low['quadratic_quantiles'][:,k])-p))))
    width=high['quadratic_quantiles'][:,-1]-high['quadratic_quantiles'][:,0]
    endpoint_error=float(np.max(abs(low['quadratic_quantiles']-high['quadratic_quantiles'])/np.maximum(width[:,None],1e-12)))
    weights=fine.weights*np.maximum(high_density,0);weights/=weights.sum()
    hpd_mass=(float(1-fine.quadratic_cdf(low['mode'][None,:],np.array([1-low['ball_radius']**2]))[0])
              if low['uniform_density'] else float(weights[low_density>=low['hpd_threshold']].sum()))
    ball_mass=float(1-fine.quadratic_cdf(low['mode'][None,:],np.array([1-low['ball_radius']**2]))[0])
    region_error=max(abs(hpd_mass-.95),abs(ball_mass-.95),high['numerics']['hpd_mass_error'])
    ok=bool(low['numerics']['numerical_ok'] and high['numerics']['numerical_ok'] and
        density_tv<.002 and mean_error<2e-4 and scalar_error<.005 and endpoint_error<.02 and region_error<.005)
    audit=dict(reference_numerical_ok=ok,density_tv=density_tv,mean_error=mean_error,
        scalar_cdf_error=scalar_error,quantile_width_relative_error=endpoint_error,
        region_mass_error=region_error,coarse_negative_mass=low['numerics']['maximum_negative_mass'],
        fine_negative_mass=high['numerics']['maximum_negative_mass'])
    high['numerics']['numerical_ok']=ok
    return high,audit
