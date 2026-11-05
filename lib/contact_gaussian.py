"""Local uncertainty controls for an unoriented contact normal in R^3.

The same-parameter Laplace law and the recursive filter are distinct controls.
Both use an exponential-map Gaussian on the modal tangent plane. Its mapped
response distribution is integrated, not clipped to physical bounds. A local
Gaussian ellipsoid is not an invariant-density Grassmann HPD region. Nonlocal
mass beyond the projective cut locus is measured and explicitly flagged.
"""
from __future__ import annotations

from functools import lru_cache
from time import perf_counter
import numpy as np
from scipy.special import dawsn,ndtri
from .geometry import center
from .models import increment,constant


def tangent_basis(normal):
    n=np.asarray(normal,dtype=float)
    if n.shape!=(3,) or not np.isfinite(n).all() or not np.isclose(n@n,1.,atol=1e-10):
        raise ValueError('normal must be a finite unit three-vector')
    axis=np.eye(3)[:,np.argmin(abs(n))]
    e=np.cross(n,axis);e/=np.linalg.norm(e)
    return np.column_stack([e,np.cross(n,e)])


def measurement(normal,v):
    n=np.asarray(normal);v=np.asarray(v)
    return v-n*(n@v)


def measurement_jacobian(normal,v,basis=None):
    n=np.asarray(normal,dtype=float);v=np.asarray(v,dtype=float)
    E=tangent_basis(n) if basis is None else np.asarray(basis,dtype=float)
    return -((n@v)*E+np.outer(n,v@E))


def exponential_normal(normal,tangent):
    """Sphere exponential; normals differing by sign represent the same plane."""
    n=np.asarray(normal,dtype=float);x=np.asarray(tangent,dtype=float)
    r=float(np.linalg.norm(x))
    return np.cos(r)*n+np.sinc(r/np.pi)*x


def parallel_rotation(normal,tangent):
    """Minimal ambient rotation carrying n along the update geodesic."""
    n=np.asarray(normal,dtype=float);x=np.asarray(tangent,dtype=float)
    r=float(np.linalg.norm(x))
    if r<1e-15:return np.eye(3)
    t=x/r
    return np.eye(3)+(np.cos(r)-1)*(np.outer(n,n)+np.outer(t,t))+np.sin(r)*(np.outer(t,n)-np.outer(n,t))


def normal_coordinates(normal,target,basis=None):
    """Shortest projective logarithm, explicitly choosing the antipodal sign."""
    n=np.asarray(normal,dtype=float);m=np.asarray(target,dtype=float)
    if m.shape!=(3,) or not np.isfinite(m).all() or not np.isclose(m@m,1.,atol=1e-10):
        raise ValueError('target normal must be a finite unit three-vector')
    E=tangent_basis(n) if basis is None else np.asarray(basis,dtype=float)
    dot=float(n@m)
    if dot<0:m=-m;dot=-dot
    dot=np.clip(dot,0.,1.);theta=float(np.arccos(dot))
    residual=m-dot*n;sine=float(np.linalg.norm(residual))
    tangent=residual*(theta/sine) if sine>1e-14 else np.zeros(3)
    return E.T@tangent,theta


def _state(normal,covariance,basis=None):
    n=np.asarray(normal,dtype=float);canonical=tangent_basis(n)
    E=canonical if basis is None else np.asarray(basis,dtype=float)
    P=np.asarray(covariance,dtype=float)
    if E.shape!=(3,2) or not np.allclose(E.T@E,np.eye(2),atol=1e-10) or not np.allclose(E.T@n,0,atol=1e-10):
        raise ValueError('basis must be an orthonormal tangent basis')
    if P.shape!=(2,2) or not np.isfinite(P).all() or not np.allclose(P,P.T,atol=1e-12):
        raise ValueError('covariance must be a finite symmetric 2x2 matrix')
    values,U=np.linalg.eigh((P+P.T)/2)
    if values.min()<-1e-14:raise ValueError('covariance must be positive semidefinite')
    # Zero eigenvalues remain zero; no confidence-inducing variance floor.
    values=np.maximum(values,0.)
    return n,E,(P+P.T)/2,values,U


@lru_cache(maxsize=16)
def _angles(order):
    if order<16:raise ValueError('angular order must be at least 16')
    angle=2*np.pi*(np.arange(order)+.5)/order
    return np.vstack([np.cos(angle),np.sin(angle)])


class _MappedGaussian:
    """Whitened polar Gaussian integration with exact radial primitives.

    xi=Sigma^(1/2) r u(phi), phi uniform and r Rayleigh(1). At each phi,
    the response is C+A*cos(2theta)+B*sin(2theta), theta=r*s(phi).
    All periodic radial intervals are integrated analytically. The omitted
    whitened radial mass is 1e-13. No interval is clipped after calculation.
    """
    def __init__(self,normal,covariance,basis=None,order=64):
        self.n,self.E,self.P,values,U=_state(normal,covariance,basis)
        direction=(self.E@(U*np.sqrt(values)))@_angles(int(order))
        self.variances=np.sum(direction*direction,axis=0)
        sigma=np.sqrt(self.variances)
        self.directions=np.divide(direction,sigma,out=np.zeros_like(direction),where=sigma>0).T
        self.order=int(order);self.radial_tail=1e-13
        self.periods=max(1,int(np.ceil(np.sqrt(-2*np.log(self.radial_tail))*sigma.max(initial=0)/np.pi)))
        if self.periods>64:raise ArithmeticError('Mapped Gaussian radial budget exceeds 64 periods; local law unresolved')

    def mean(self):
        sigma=np.sqrt(self.variances)
        sin2=np.sqrt(2)*sigma*dawsn(np.sqrt(2)*sigma)
        normal_mean=(1-sin2.mean())*np.outer(self.n,self.n)+np.einsum('p,pi,pj->ij',sin2/self.order,self.directions,self.directions)
        return np.eye(3)-(normal_mean+normal_mean.T)/2

    def cut_locus_mass(self):
        return float(np.exp(np.divide(-.5*(np.pi/2)**2,self.variances,
                    out=np.full_like(self.variances,-np.inf),where=self.variances>0)).mean())

    def _cdf(self,x,C,A,B):
        R=np.hypot(A,B);delta=np.arctan2(B,A)
        tiny=1e-14*max(1.,float(np.max(abs(C))),float(np.max(R)))
        z=np.divide(x[:,None]-C,R,out=np.zeros_like(R),where=R>tiny)
        angle=np.arccos(np.clip(z,-1.,1.))
        r1=np.mod(delta+angle,2*np.pi)/2;r2=np.mod(delta-angle,2*np.pi)/2
        lower,upper=np.minimum(r1,r2),np.maximum(r1,r2)
        roots=[np.zeros_like(R),lower,upper,np.full_like(R,np.pi)]
        mass=np.zeros_like(R)
        inv=np.divide(.5,self.variances,out=np.zeros_like(self.variances),where=self.variances>0)
        for period in range(self.periods):
            for j in range(3):
                a=roots[j]+period*np.pi;b=roots[j+1]+period*np.pi
                middle=roots[j]+roots[j+1]
                included=C+A*np.cos(middle)+B*np.sin(middle)<=x[:,None]
                primitive_a=np.exp(-a*a*inv);primitive_b=np.exp(-b*b*inv)
                primitive_a=np.where(a==0,1.,primitive_a)
                mass+=(primitive_a-primitive_b)*included
        mass=np.where(x[:,None]-C>=R,1.,np.where(x[:,None]-C<=-R,0.,mass))
        mass=np.where(R<=tiny,np.where(C<=x[:,None],1.,0.),mass)
        # An exactly deterministic direction has an atom at theta=0.
        mass=np.where(self.variances[None,:]==0,(C+A<=x[:,None]).astype(float),mass)
        return mass.mean(axis=1)

    def quantiles(self,queries,probabilities=(.025,.975)):
        W=np.asarray(queries,dtype=float);probs=np.asarray(probabilities,dtype=float)
        if W.ndim!=2 or W.shape[1]!=3 or np.any(np.linalg.norm(W,axis=1)<=0) or not np.isfinite(W).all():
            raise ValueError('queries must be finite nonzero row-wise three-vectors')
        if np.any(probs<=0) or np.any(probs>=1):raise ValueError('probabilities must be in (0,1)')
        component=np.tile(np.arange(3),len(W));w=np.repeat(W,3,axis=0)
        wn=w@self.n;wt=w@self.directions.T
        a=self.n[component]*wn;b=self.directions.T[component]*wt
        cross=self.n[component,None]*wt+self.directions.T[component]*wn[:,None]
        C=w[np.arange(len(w)),component,None]-(a[:,None]+b)/2
        A=-(a[:,None]-b)/2;B=-cross/2
        count=len(probs);C,A,B=[np.repeat(x,count,axis=0) for x in (C,A,B)]
        length=np.linalg.norm(w,axis=1);value=w[np.arange(len(w)),component]
        lo=np.repeat((value-length)/2,count);hi=np.repeat((value+length)/2,count)
        target=np.tile(probs,len(w))
        for _ in range(34):
            mid=(lo+hi)/2;below=self._cdf(mid,C,A,B)<target
            lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        return ((lo+hi)/2).reshape(len(W),3,count)

    def ball_radius(self,mass=.95):
        if np.max(self.variances)==0:return 0.
        C=np.full((1,self.order),.5);A=-C;B=np.zeros_like(C)
        lo,hi=0.,1.
        for _ in range(38):
            mid=(lo+hi)/2
            if self._cdf(np.array([mid]),C,A,B)[0]<mass:lo=mid
            else:hi=mid
        return float(np.sqrt((lo+hi)/2))


def _unresolved_summary(queries,reason,singular=False,seconds=0.):
    W=np.asarray(queries);nan=np.full((len(W),3),np.nan)
    return dict(mean=np.full((3,3),np.nan),mode_projector=np.full((3,3),np.nan),
        response_lower=nan.copy(),response_upper=nan.copy(),risks=np.full(len(W),np.nan),
        ball_radius=np.nan,hpd_threshold=np.nan,ball_covered=False,hpd_covered=False,
        numerics=dict(numerical_ok=False,mean_numerical_ok=False,response_quantiles_ok=False,
            regions_numerical_ok=False,integration_numerical_ok=False,local_approximation_ok=False,
            singular_precision=singular,unresolved_reason=str(reason),integration_seconds=seconds,
            hpd_kind='Local tangent Gaussian ellipsoid; not invariant-density Grassmann HPD'))


def gaussian_summary(normal,covariance,queries,true_normal=None,basis=None,mass=.95,
                     angular_orders=(32,64),max_order=256,cut_locus_tolerance=.01):
    """Mapped moments/intervals and a local Gaussian ellipsoid inclusion.

    The ellipsoid is HPD relative to tangent-coordinate Lebesgue measure; it
    is not an invariant-density Grassmann HPD region. Integration flags and
    the local-branch approximation flag are separate. Broad mass is retained.
    """
    start=perf_counter();W=np.asarray(queries,dtype=float)
    n,E,P,values,_=_state(normal,covariance,basis)
    low_order,order=map(int,angular_orders)
    if not 0<mass<1 or order<=low_order or max_order<order:raise ValueError('invalid mass or angular budgets')
    try:lower=_MappedGaussian(n,P,E,low_order)
    except ArithmeticError as error:
        return _unresolved_summary(W,error,seconds=perf_counter()-start)
    qlo=lower.quantiles(W,((1-mass)/2,(1+mass)/2))
    attempts=0
    while True:
        attempts+=1;high=_MappedGaussian(n,P,E,order)
        begin=perf_counter();M=high.mean();mean_seconds=perf_counter()-begin
        begin=perf_counter();q=high.quantiles(W,((1-mass)/2,(1+mass)/2));interval_seconds=perf_counter()-begin
        widths=q[:,:,1]-q[:,:,0];delta=np.max(abs(q-qlo),axis=2)
        scale=np.linalg.norm(W,axis=1)[:,None]
        mean_error=float(np.linalg.norm(M-lower.mean(),2))
        quantile_error=float(np.max(delta/np.maximum(widths,2.5e-4*scale)))
        mean_ok=bool(mean_error<1e-8)
        interval_ok=bool(np.all(delta<=np.maximum(.02*widths,5e-6*scale)))
        if (mean_ok and interval_ok) or order>=max_order:break
        lower=high;qlo=q;low_order=order;order=min(max_order,order*2)
    begin=perf_counter();radius=high.ball_radius(mass);radius_low=lower.ball_radius(mass)
    region_error=abs(radius-radius_low);region_seconds=perf_counter()-begin
    region_ok=bool(region_error<1e-5)
    cutoff_mass=high.cut_locus_mass();local_ok=bool(cutoff_mass<=cut_locus_tolerance)
    rank=int(np.count_nonzero(values>0))
    threshold=float(-2*np.log(1-mass)) if rank==2 else float(ndtri((1+mass)/2)**2) if rank==1 else 0.
    mode=np.eye(3)-np.outer(n,n)
    diagnostics=dict(mean_numerical_ok=mean_ok,response_quantiles_ok=interval_ok,
        regions_numerical_ok=region_ok,integration_numerical_ok=bool(mean_ok and interval_ok and region_ok),
        local_approximation_ok=local_ok,numerical_ok=bool(mean_ok and interval_ok and region_ok and local_ok),
        mean_resolution_error=mean_error,quantile_relative_error=quantile_error,
        ball_radius_resolution_error=region_error,angular_order=order,previous_angular_order=low_order,
        refinements=attempts,radial_tail_bound=high.radial_tail,cut_locus_mass=cutoff_mass,
        cut_locus_tolerance=cut_locus_tolerance,mean_seconds=mean_seconds,
        degenerate_covariance=bool(rank<2),local_ellipse_dimension=rank,
        interval_seconds=interval_seconds,region_seconds=region_seconds,
        integration_seconds=perf_counter()-start,
        distribution='Exponential-map tangent Gaussian; periodic projective response integrated without clipping',
        hpd_kind='Local tangent Gaussian ellipsoid; not invariant-density Grassmann HPD',
        unresolved_reason='' if local_ok else 'Nonlocal Gaussian mass crosses the projective cut locus')
    result=dict(mean=M,mode_projector=mode,mode_normal=n,basis=E,covariance=P,
        response_lower=q[:,:,0],response_upper=q[:,:,1],risks=np.einsum('qi,ij,qj->q',W,M-M@M,W),
        ball_radius=radius,hpd_threshold=threshold,numerics=diagnostics)
    if true_normal is not None:
        coordinates,theta=normal_coordinates(n,true_normal,E)
        positive=values>0
        eigenvalues,eigenvectors=np.linalg.eigh(P);local=eigenvectors.T@coordinates
        mahalanobis=(float(np.sum(local[positive]**2/eigenvalues[positive]))
                     if np.linalg.norm(local[~positive])<1e-12 else np.inf)
        result.update(ball_covered=bool(np.sin(theta)<=radius),hpd_covered=bool(mahalanobis<=threshold),
            chordal_error=float(np.sin(theta)),tangent_error=coordinates,mahalanobis_error=mahalanobis,
            directional_standard_deviations=np.sqrt(np.maximum(0,np.diag(P))))
    return result


def laplace_state(A,gap_tolerance=1e-10):
    """Same-parameter normal-line modal covariance; singular cutoffs fail."""
    A=np.asarray(A,dtype=float)
    if A.shape!=(3,3) or not np.isfinite(A).all() or not np.allclose(A,A.T,atol=1e-12):
        raise ValueError('A must be a finite symmetric real 3x3 plane parameter')
    values,U=np.linalg.eigh(-A);values,U=values[::-1],U[:,::-1]
    gaps=values[0]-values[1:]
    if np.min(gaps)<=gap_tolerance:raise ArithmeticError('Singular Laplace precision; no variance floor or false confidence inserted')
    return U[:,0],np.diag(1/(2*gaps)),U[:,1:],gaps


def laplace_summary(A,queries,true_normal=None,mass=.95,**kwargs):
    start=perf_counter()
    try:n,P,E,gaps=laplace_state(A)
    except ArithmeticError as error:
        return _unresolved_summary(queries,error,singular=True,seconds=perf_counter()-start)
    result=gaussian_summary(n,P,queries,true_normal,basis=E,mass=mass,**kwargs)
    result['numerics'].update(singular_precision=False,laplace_min_gap=float(gaps.min()),
        laplace_max_gap=float(gaps.max()),laplace_preparation_seconds=perf_counter()-start-result['numerics']['integration_seconds'])
    return result


class RecursiveTangentGaussian:
    """Cheap manifold EKF with kernel-matched isotropic tangent dispersion.

    Warm-up follows the SAME moment-projected process Bingham law for six
    observations, then initializes from its mode and local curvature. Startup
    cost is counted. Later prediction adds exact-kernel E[theta^2]/2 I; this
    remains a local process approximation. Updates use the stated Jacobian,
    a Joseph covariance correction and geodesic parallel transport. No true
    geometry or current observation enters its own prepared prior.
    """
    def __init__(self,s,kappa,warmup=6):
        if not np.isfinite(s) or s<=0 or warmup<1:raise ValueError('positive variance and warmup required')
        self.s=float(s);self.kappa=float(kappa);self.warmup=int(warmup)
        self.normal=None;self.covariance=None;self.basis=None;self.A=np.zeros((3,3))
        self.t=0;self.initialized=False;self._prepared=False
        self.nonlocal_updates=0
        self.timings=dict(prediction_seconds=0.,update_seconds=0.,warmup_seconds=0.,score_seconds=0.)
        self.last_diagnostics={};self._warmup_prior=None
        self._last_rotation=np.eye(3)
        from .contact_process_prediction import kernel_tangent_variance
        self.process_variance=float(kernel_tangent_variance(kappa))

    def initialize_from_parameter(self,A,t=None,warmup_seconds=0.):
        begin=perf_counter()
        self.normal,self.covariance,self.basis,gaps=laplace_state(A)
        self.A=center(A);self.initialized=True;self._prepared=False
        self.t=self.warmup if t is None else int(t)
        self.timings['warmup_seconds']+=float(warmup_seconds)+perf_counter()-begin
        self.last_diagnostics=dict(initialized=True,warmup_completed=True,minimum_gap=float(gaps.min()))
        return self.normal.copy()

    initialize=initialize_from_parameter

    def predict(self,G=None):
        if self._prepared:
            if G is not None and not np.allclose(G,self._last_rotation,atol=1e-12):
                raise ValueError('A different nominal motion was supplied after preparing this prior')
            return dict(normal=self.normal,covariance=self.covariance,basis=self.basis,initialized=self.initialized)
        begin=perf_counter();G=np.eye(3) if G is None else np.asarray(G,dtype=float)
        if G.shape!=(3,3) or not np.allclose(G.T@G,np.eye(3),atol=1e-10):raise ValueError('nominal rotation must be orthogonal')
        if self.initialized:
            self.normal=G@self.normal;self.basis=G@self.basis
            self.covariance=(self.covariance+self.covariance.T)/2+self.process_variance*np.eye(2)
        else:
            from .contact_process_prediction import predict_contact
            prediction=predict_contact(self.A,self.kappa,G=G)
            self._warmup_prior=prediction.prior
            self.last_diagnostics=dict(initialized=False,warmup=True,prediction_fit=prediction.fit.converged)
        elapsed=perf_counter()-begin;self.timings['prediction_seconds']+=elapsed
        self._last_rotation=G.copy()
        if not self.initialized:self.timings['warmup_seconds']+=elapsed
        self._prepared=True
        return dict(normal=self.normal,covariance=self.covariance,basis=self.basis,initialized=self.initialized)

    def predictive_log_score(self,u,v):
        self.predict();begin=perf_counter();u=np.asarray(u,dtype=float);v=np.asarray(v,dtype=float)
        if self.initialized:
            J=measurement_jacobian(self.normal,v,self.basis)
            S=J@self.covariance@J.T+self.s*np.eye(3);residual=u-measurement(self.normal,v)
            sign,logdet=np.linalg.slogdet(S)
            if sign<=0:raise ArithmeticError('Predictive covariance is not positive definite')
            score=float(-.5*(3*np.log(2*np.pi)+logdet+residual@np.linalg.solve(S,residual)))
            self.last_diagnostics['predictive_score_kind']='Linearized EKF Gaussian observation density, including constants'
        else:
            from .contact_process_prediction import fast_moments
            B=increment(u,self.s,v=v)
            score=float(constant(u,2,self.s,v=v)+fast_moments(self._warmup_prior+B).log_normalizer-
                        fast_moments(self._warmup_prior).log_normalizer)
            self.last_diagnostics['predictive_score_kind']='Process Bingham warmup predictive density, including constants'
        self.timings['score_seconds']+=perf_counter()-begin
        return score

    def update(self,u,v,G=None):
        self.predict(G);begin=perf_counter();u=np.asarray(u,dtype=float);v=np.asarray(v,dtype=float)
        if u.shape!=(3,) or v.shape!=(3,) or not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('finite three-vector observation and input required')
        if not self.initialized:
            warmup_clock=self.timings['warmup_seconds']
            self.A=center(self._warmup_prior+increment(u,self.s,v=v));self.t+=1
            if self.t>=self.warmup:
                try:self.initialize_from_parameter(self.A,t=self.t)
                except ArithmeticError:
                    self.last_diagnostics=dict(initialized=False,initialization_failed=True,
                        unresolved_reason='Warmup Laplace precision singular; accumulate more actual data')
            # initialize_from_parameter counts itself when called externally;
            # here its elapsed time is already inside this complete update.
            self.timings['warmup_seconds']=warmup_clock+perf_counter()-begin
        else:
            n,E,P=self.normal,self.basis,self.covariance
            J=measurement_jacobian(n,v,E);S=J@P@J.T+self.s*np.eye(3)
            K=np.linalg.solve(S,J@P).T;delta=K@(u-measurement(n,v))
            I=np.eye(2);H=I-K@J
            posterior=H@P@H.T+self.s*K@K.T
            tangent=E@delta;rotation=parallel_rotation(n,tangent)
            self.normal=exponential_normal(n,tangent);self.normal/=np.linalg.norm(self.normal)
            transported=rotation@E
            # Reorthogonalize roundoff only, transforming covariance consistently.
            transported-=np.outer(self.normal,self.normal@transported)
            new_basis=np.linalg.qr(transported)[0];T=new_basis.T@transported
            self.basis=new_basis;self.covariance=T@posterior@T.T
            self.covariance=(self.covariance+self.covariance.T)/2;self.t+=1
            self.last_diagnostics=dict(initialized=True,update_angle=float(np.linalg.norm(delta)),
                nonlocal_update=bool(np.linalg.norm(delta)>=np.pi/2),
                covariance_minimum_eigenvalue=float(np.linalg.eigvalsh(self.covariance).min()))
            self.nonlocal_updates+=int(self.last_diagnostics['nonlocal_update'])
            if self.last_diagnostics['covariance_minimum_eigenvalue']<-1e-12:
                raise ArithmeticError('Joseph covariance correction lost positive semidefiniteness')
        self.timings['update_seconds']+=perf_counter()-begin;self._prepared=False
        return None if self.normal is None else self.normal.copy()

    def summary(self,queries,true_normal=None,**kwargs):
        if not self.initialized:raise ArithmeticError('Gaussian warmup has not produced a resolved local state')
        result=gaussian_summary(self.normal,self.covariance,queries,true_normal,basis=self.basis,**kwargs)
        result['numerics']['filter_nonlocal_updates']=self.nonlocal_updates
        return result
