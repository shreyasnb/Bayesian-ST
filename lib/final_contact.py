"""Fixed-scale physical contact experiments for the final simulation protocol.

The plant solves maintained bilateral force balance independently of inference.
Only applied requests, noisy instantaneous velocity and measured wrist rotations
enter the estimators. Hidden geometry is retained solely for scoring and plots.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

from .geometry import center
from .models import increment
from .contact_mechanics import PLANE_NORMAL, force_program, solve_contact_velocity
from .contact_process_prediction import fast_moments, kernel_parameters
from .contact_posterior import normal_quadrature


def _rotation(axis, angle):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    K=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def wrist_frames(T):
    phase=np.linspace(0.,2*np.pi,T)
    R=np.array([_rotation([1.,1.,.4],.7*np.sin(x))@_rotation([.2,1.,-.3],.25*np.sin(2*x)) for x in phase])
    G=np.empty_like(R);G[0]=R[0].T
    G[1:]=np.einsum('tji,tjk->tik',R[1:],R[:-1])
    return R,G


def input_program(T, kind, v0):
    if kind == 'static_balanced':
        return v0*np.eye(3)[np.arange(T)%3]
    if kind != 'static_concentrated':
        raise ValueError('Unknown stationary input design')
    values=[]
    for j in range(T):
        value=np.array([1.,0.,0.]);phase=j%5
        if phase:
            value[(phase+1)//2]=.2 if phase%2 else -.2
        values.append(v0*value/np.linalg.norm(value))
    return np.asarray(values)


def _seed(cfg, trial, split, tag):
    base=cfg['validation_seed'] if split == 'validation' else cfg['test_seed']
    return np.random.SeedSequence([base+cfg.get('stage_seed_offset',0),int(trial),int(tag)])


def _heldout_query_measurements(cfg, world_normals, rotations, sigma2, trial, split):
    """Plant-side independent query observations, never estimator geometry.

    These measurements are generated once, separately from training responses.
    The policy scorer receives only the returned wrist-frame query/response
    pairs; hidden normals are used here only by the physical KKT plant.
    """
    Q=query_directions(cfg['v0'])
    rng=np.random.default_rng(_seed(cfg,trial,split,4103))
    world_clean=np.array([[solve_contact_velocity(cfg['resistance']*q,n,cfg['resistance'])[0]
                           for q in Q] for n in world_normals])
    world_noise=rng.normal(size=world_clean.shape)
    world_observed=world_clean+np.sqrt(sigma2)*world_noise
    queries=np.einsum('qi,tij->tqj',Q,rotations)
    responses=np.einsum('tqi,tij->tqj',world_observed,rotations)
    return dict(heldout_queries=queries,heldout_responses=responses,
        heldout_clean=np.einsum('tqi,tij->tqj',world_clean,rotations),
        heldout_standard_noise=np.einsum('tqi,tij->tqj',world_noise,rotations))


def generate_static_trial(cfg, condition, trial, split='test', normal=None):
    rng=np.random.default_rng(_seed(cfg,trial,split,4101))
    n=rng.normal(size=3) if normal is None else np.asarray(normal,float)
    n=n/np.linalg.norm(n)
    T=cfg['static_T'];V=input_program(T,condition,cfg['v0'])
    clean=[];residuals=[]
    for v in V:
        u,reaction=solve_contact_velocity(cfg['resistance']*v,n,cfg['resistance'])
        clean.append(u);residuals.append(max(abs(n@u),np.linalg.norm(cfg['resistance']*(u-v)-n*reaction)))
    clean=np.asarray(clean);noise=rng.normal(size=(T,3))
    sigma2=float(cfg['static_sigma2'])
    P=np.eye(3)-np.outer(n,n)
    positions=np.cumsum(np.vstack([np.zeros(3),cfg['dt']*clean]),axis=0)
    rotations=np.broadcast_to(np.eye(3),(T,3,3)).copy()
    heldout=_heldout_query_measurements(cfg,np.broadcast_to(n,(T,3)),rotations,sigma2,trial,split)
    return dict(U=clean+np.sqrt(sigma2)*noise,V=V,clean=clean,standard_noise=noise,**heldout,
        P=np.broadcast_to(P,(T,3,3)).copy(),normal=np.broadcast_to(n,(T,3)).copy(),
        G=np.broadcast_to(np.eye(3),(T,3,3)).copy(),R=rotations,
        positions=positions[:-1],path=positions,world_normal=np.broadcast_to(n,(T,3)).copy(),
        world_U=clean+np.sqrt(sigma2)*noise,world_V=V,world_clean=clean,
        forces=cfg['resistance']*V,times=np.arange(T)*cfg['dt'],sigma2=sigma2,
        diagnostics=dict(maximum_kkt_residual=max(residuals),input_energy=float(np.sum(V*V)),
            maximum_projection_error=float(np.max(abs(clean-V@P))),surface='plane',
            fixed_normal=True,normal_supplied_to_estimator=False))


def generate_contact_trial(cfg, condition, trial, split='test', substeps=1):
    """Generate plane/sphere motion; fast_sphere changes force after midpoint.

    Returned normal/projector/position records are scoring information. R is an
    independently prescribed measured orientation, mapping wrist to world.
    The fast_sphere intervention changes both excitation and physical speed.
    """
    if condition.startswith('static_'):
        return generate_static_trial(cfg,condition,trial,split)
    if condition not in ('plane','sphere','fast_sphere'):
        raise ValueError('Unknown physical contact condition')
    T=int(cfg.get('T',cfg.get('horizon',192)));dt=float(cfg['dt'])
    sigma2=float(cfg['sigma2']);d=float(cfg['resistance']);v0=float(cfg['v0'])
    if T<1 or dt<=0 or sigma2<0 or substeps<1:
        raise ValueError('Invalid plant budget')
    rng=np.random.default_rng(_seed(cfg,trial,split,4102))
    initial=rng.normal(size=3);initial/=np.linalg.norm(initial)
    radius=float(cfg['radius'])
    position=np.zeros(3) if condition=='plane' else radius*initial
    positions=[];normals=[];velocities=[];requests=[];forces=[];path=[position.copy()]
    balance=[];constraint=[];projection=[];drift_identity=[]
    def normal(p):
        return PLANE_NORMAL.copy() if condition=='plane' else p/np.linalg.norm(p)
    def project(p):
        return p-PLANE_NORMAL*(PLANE_NORMAL@p) if condition=='plane' else radius*p/np.linalg.norm(p)
    for j in range(T):
        tau=j*dt
        amplitude=3. if condition=='fast_sphere' and j>=T//2 else 1.
        fixed=v0*np.eye(3)[j%3] if condition=='plane' else None
        def request(time):
            return fixed if fixed is not None else amplitude*force_program(time,v0)
        def velocity(p,time):
            return solve_contact_velocity(d*request(time),normal(p),d)[0]
        n=normal(position);v=request(tau);u,reaction=solve_contact_velocity(d*v,n,d)
        positions.append(position.copy());normals.append(n);velocities.append(u);requests.append(v);forces.append(d*v)
        balance.append(np.linalg.norm(d*u-d*v-n*reaction));constraint.append(abs(n@u))
        projection.append(np.linalg.norm(u-(np.eye(3)-np.outer(n,n))@v))
        if condition!='plane':
            ndot=u/radius;Pdot=-(np.outer(ndot,n)+np.outer(n,ndot))
            drift_identity.append(abs(np.linalg.norm(Pdot,2)-np.linalg.norm(u)/radius))
        for sub in range(int(substeps)):
            h=dt/substeps;time=tau+sub*h
            a=velocity(position,time);b=velocity(position+h*a/2,time+h/2)
            c=velocity(position+h*b/2,time+h/2);e=velocity(position+h*c,time+h)
            position=project(position+h*(a+2*b+2*c+e)/6)
        path.append(position.copy())
    positions=np.asarray(positions);normals=np.asarray(normals);clean=np.asarray(velocities)
    Vworld=np.asarray(requests);unit_noise=rng.normal(size=(T,3));Uworld=clean+np.sqrt(sigma2)*unit_noise
    R,G=wrist_frames(T)
    body=lambda value:np.einsum('tji,tj->ti',R,value)
    Pworld=np.eye(3)-np.einsum('ti,tj->tij',normals,normals)
    P=np.einsum('tji,tjk,tkl->til',R,Pworld,R)
    finite_relative=np.nan
    if condition!='plane' and T>=3:
        numerical=np.diff(Pworld,axis=0)/dt
        theoretical=(np.linalg.norm(clean[:-1],axis=1)+np.linalg.norm(clean[1:],axis=1))/(2*radius)
        keep=np.ones(T-1,bool)
        if condition=='fast_sphere':keep[max(0,T//2-2):min(T-1,T//2+1)]=False
        finite_relative=float(np.max(abs(np.linalg.norm(numerical,2,axis=(1,2))[keep]-theoretical[keep])/
                                   np.maximum(theoretical[keep],1e-8)))
    heldout=_heldout_query_measurements(cfg,normals,R,sigma2,trial,split)
    return dict(U=body(Uworld),V=body(Vworld),clean=body(clean),standard_noise=body(unit_noise),**heldout,
        P=P,normal=body(normals),G=G,R=R,positions=positions,path=np.asarray(path),
        world_normal=normals,world_U=Uworld,world_V=Vworld,world_clean=clean,
        forces=np.asarray(forces),times=np.arange(T)*dt,sigma2=sigma2,
        diagnostics=dict(maximum_kkt_residual=float(max(balance)),maximum_constraint_residual=float(max(constraint)),
            maximum_projection_error=float(max(projection)),maximum_sphere_drift_identity_error=max(drift_identity,default=0.),
            finite_difference_drift_relative_error=finite_relative,
            maximum_surface_error=float(np.max(abs(np.asarray(path)@PLANE_NORMAL))) if condition=='plane'
                else float(np.max(abs(np.linalg.norm(path,axis=1)-radius))),
            surface=condition,normal_supplied_to_estimator=False,measured_orientation_supplied=True,
            input_energy=float(np.sum(Vworld*Vworld)),
            event='force amplitude triples at midpoint, also changing excitation' if condition=='fast_sphere' else 'none'))


def mode_projector(A):
    values,E=np.linalg.eigh((np.asarray(A)+np.asarray(A).T)/2)
    normal=E[:,0]
    gap=float(values[1]-values[0])
    return np.eye(3)-np.outer(normal,normal),normal,gap


def mean_projector(A):
    result=fast_moments(A,checked=True)
    return result.mean,dict(numerical_ok=bool(result.numerical_ok),
        mean_resolution_error=float(result.resolution_error),
        log_normalizer_resolution_error=float(result.logz_resolution_error),
        angular_order=int(result.angular_order),log_normalizer=float(result.log_normalizer),
        moment_seconds=float(result.seconds))


def isotropic_mean(M,P):
    alpha=float(np.trace(P@M)/2);xi=float(2-2*alpha)
    D=alpha*P+xi*(np.eye(3)-P)
    if not -1e-8<=xi<=1+1e-8 or not -1e-8<=alpha<=1+1e-8:
        raise ArithmeticError('Matched isotropic mean is not an admissible contraction')
    return (D+D.T)/2,alpha,xi


def query_directions(v0=.02):
    Q=np.linalg.qr(np.array([[1.,.2,.3],[-.4,1.,.1],[.1,-.2,1.]]))[0]
    return float(v0)*Q.T


def response_loss(M,P,queries):
    error=np.asarray(queries)@(M-P).T
    return float(np.mean(np.sum(error*error,axis=1)))


@dataclass
class ConditionalTracker:
    """One correction per pair; no surface geometry enters this interface."""
    sigma2: float
    beta: float=1.
    A0: np.ndarray | None=None

    def __post_init__(self):
        if self.sigma2<=0 or not 0<self.beta<=1:raise ValueError('Positive variance and valid memory required')
        self.A=np.zeros((3,3)) if self.A0 is None else center(np.array(self.A0,float,copy=True))
        self.observations=0

    def update(self,u,v,G=None):
        G=np.eye(3) if G is None else np.asarray(G)
        self.A=center(self.beta*G@self.A@G.T+increment(u,self.sigma2,v=v))
        self.observations+=1
        return self.A

    def mean(self):
        return mean_projector(self.A)


def action_times(cfg,condition):
    return cfg['static_times'] if condition.startswith('static_') else list(range(cfg['action_every'],cfg['T']+1,cfg['action_every']))


def bingham_records(cfg,data,beta,times=None,transport=True,A0=None):
    tracker=ConditionalTracker(float(data['sigma2']),beta,A0)
    times=set(range(cfg['action_every'],len(data['U'])+1,cfg['action_every']) if times is None else times)
    records={};all_A=[]
    for j,(u,v) in enumerate(zip(data['U'],data['V'])):
        tracker.update(u,v,data['G'][j] if transport else np.eye(3));all_A.append(tracker.A.copy())
        if j+1 in times:
            M,diag=tracker.mean();P,normal,gap=mode_projector(tracker.A);iso,alpha,xi=isotropic_mean(M,P)
            records[j+1]=dict(A=tracker.A.copy(),M=M,Pmode=P,normal=normal,mode_gap=gap,
                mode_unique=gap>1e-10,isotropic=iso,alpha=alpha,xi=xi,diagnostics=diag)
    return records,np.asarray(all_A)


def anisotropy_fixture(cfg):
    """An actual, preassigned noisy stationary episode and an actual axial law."""
    local=dict(cfg,static_sigma2=cfg['anisotropy_sigma2'],test_seed=cfg['story_seed'],stage_seed_offset=0)
    data=generate_static_trial(local,cfg['anisotropy_input'],0,normal=cfg['anisotropy_normal'])
    t=cfg['anisotropy_time']
    records,_=bingham_records(local,data,1.,[t]);record=records[t]
    A=record['A'];M=record['M'];mode=record['normal'];Pn=np.outer(mode,mode)
    target=float(np.trace(Pn@(np.eye(3)-M)))
    if target<=1/3+1e-13:kappa=0.
    else:
        upper=max(2.,2/max(1-target,1e-12))
        while kernel_parameters(upper).a<target:upper*=2
        kappa=float(brentq(lambda value:kernel_parameters(value).a-target,0.,upper,xtol=1e-11))
    normal_A=-A;iso_normal_A=center(kappa*Pn)
    full=normal_quadrature(normal_A,64,192);iso=normal_quadrature(iso_normal_A,64,192)
    low=normal_quadrature(iso_normal_A,48,96)
    iso_M=np.eye(3)-iso['mean'];matched,alpha,xi=isotropic_mean(M,record['Pmode'])
    theta=np.linspace(0,2*np.pi,181)
    queries=np.column_stack([np.cos(theta),np.sin(theta),np.zeros_like(theta)])
    additional=np.column_stack([np.zeros_like(theta),np.cos(theta),np.sin(theta)])
    queries=np.vstack([queries,additional])
    risks=np.einsum('qi,ij,qj->q',queries,M-M@M,queries)
    iso_risks=np.einsum('qi,ij,qj->q',queries,iso_M-iso_M@iso_M,queries)
    values=np.linalg.eigvalsh(normal_A)[::-1];curvature=2*(values[0]-values[1:])
    X=np.linalg.svd(np.asarray(cfg['anisotropy_normal'])[None,:],full_matrices=True)[2][1:].T
    from .theory import fisher_information
    F=fisher_information(X,data['V'][:t].T,cfg['anisotropy_sigma2'])
    overlap_error=abs(float(mode@iso['mean']@mode)-target)
    checks=dict(data_generated=True,selected_episode=0,selected_time=t,
        normal_mode_overlap=target,kappa=kappa,isotropic_overlap_error=overlap_error,
        isotropic_mean_match_error=float(np.linalg.norm(iso_M-matched,2)),
        isotropic_resolution_error=float(np.linalg.norm(iso['mean']-low['mean'],2)),
        mode_unique=bool(curvature[0]>1e-10),
        curvature_ratio=float(curvature[-1]/curvature[0]) if curvature[0]>1e-10 else np.nan,
        numerical_ok=bool(record['diagnostics']['numerical_ok'] and overlap_error<1e-8
                          and np.linalg.norm(iso_M-matched,2)<1e-7))
    arrays=dict(plane_A=A,normal_A=normal_A,normal_logz=np.array(full['log_normalizer']),
        normal_mean=np.eye(3)-M,plane_mean=M,iso_normal_A=iso_normal_A,
        iso_normal_logz=np.array(iso['log_normalizer']),iso_normal_mean=iso['mean'],iso_plane_mean=iso_M,
        mode_normal=mode,mode_plane=record['Pmode'],unit_queries=queries,physical_queries=cfg['v0']*queries,
        query_angles=np.tile(theta,2),query_family=np.repeat([0,1],len(theta)),
        query_bases=np.array([[[1.,0.,0.],[0.,1.,0.]],[[0.,1.,0.],[0.,0.,1.]]]),
        risks_full=risks,risks_isotropic=iso_risks,curvature_eigenvalues=curvature,
        fisher_eigenvalues=np.linalg.eigvalsh(F),input_vectors=data['V'][:t],observations=data['U'][:t],
        true_normal=np.asarray(cfg['anisotropy_normal']),kappa=np.array(kappa),report_time=np.array(t))
    return arrays,checks


def test_episode(cfg, condition, trial, selection):
    if condition != 'sphere':
        raise ValueError('The paper moving contact experiment uses the sphere plant')
    data = generate_contact_trial(cfg, condition, trial)
    policy = next(p for p in selection if p['condition'] == condition)
    times = action_times(cfg, condition)
    records, _ = bingham_records(cfg, data, policy['beta'], times)
    identity, _ = bingham_records(cfg, data, policy['beta'], times, transport=False)
    rows = []
    averages = {}
    for time, record in records.items():
        queries = query_directions(cfg['v0']) @ data['R'][time-1]
        methods = dict(mean=record['M'], mode=record['Pmode'],
            isotropic=record['isotropic'], mean_identity=identity[time]['M'])
        for method, estimate in methods.items():
            row = dict(condition=condition, trial=trial, method=method, t=time,
                response_mse=response_loss(estimate, data['P'][time-1], queries),
                response_map_frobenius=float(np.linalg.norm(estimate-data['P'][time-1], 'fro')**2),
                posterior_response_excess=float(np.mean(np.sum((queries@(estimate-record['M']).T)**2, axis=1))),
                moment_resolution_error=record['diagnostics']['mean_resolution_error'],
                numerical_ok=bool(record['diagnostics']['numerical_ok']))
            averages.setdefault(method, []).append(row)
            rows.append(dict(row, endpoint='checkpoint'))
    for method, group in averages.items():
        rows.append(dict(condition=condition, trial=trial, method=method, t=len(data['U']),
            endpoint='episode_average',
            **{name:float(np.mean([r[name] for r in group])) for name in
               ('response_mse', 'response_map_frobenius', 'posterior_response_excess')},
            moment_resolution_error=max(r['moment_resolution_error'] for r in group),
            numerical_ok=all(r['numerical_ok'] for r in group)))
    if not all(row['numerical_ok'] for row in rows):
        raise ArithmeticError('Contact moment resolution failed at the original 5e-8 tolerance')
    return dict(metrics=rows)
