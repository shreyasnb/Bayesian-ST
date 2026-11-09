"""Fixed-scale physical contact experiments for the final simulation protocol.

The plant solves maintained bilateral force balance independently of inference.
Only applied requests, noisy instantaneous velocity and measured wrist rotations
enter the estimators. Hidden geometry is retained solely for scoring and plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from scipy.stats import t as student_t

from .geometry import center
from .models import increment
from .contact_mechanics import PLANE_NORMAL, force_program, solve_contact_velocity
from .contact_process_prediction import fast_moments, kernel_parameters
from .contact_posterior import normal_quadrature


def configuration(stage='verify'):
    if stage not in ('verify', 'pilot', 'paper'):
        raise ValueError('Choose verify, pilot or paper')
    return dict(schema=1, stage=stage, source_commit='a91faabeb10a63b1d1c613d75cc32503cfc0cac6',
        n=3, k=2, T=64 if stage == 'verify' else 192, static_T=32, dt=.05,
        v0=.02, resistance=10., radius=.05, sigma2=.004**2, static_sigma2=.01**2,
        conditions=['static_balanced', 'static_concentrated', 'plane', 'sphere'],
        validation_episodes=2 if stage == 'verify' else 4 if stage == 'pilot' else 64,
        test_episodes=3 if stage == 'verify' else 8 if stage == 'pilot' else 256,
        block_episodes=2 if stage == 'verify' else 3 if stage == 'pilot' else 32,
        beta_grid=[1., .95] if stage == 'verify' else [1., .98, .95, .9],
        great_step_grid=[10.] if stage == 'verify' else [1., 10., 50., 200.],
        great_inner_grid=[1] if stage == 'verify' else [1, 3, 8],
        great_window_grid=[16] if stage == 'verify' else [16, 64],
        great_beta_grid=[.95] if stage == 'verify' else [1., .98, .9],
        static_times=[1, 4, 16, 32], checkpoint_times=[4, 16, 32, 64] if stage == 'verify'
            else [4, 16, 32, 64, 128, 192],
        action_every=4, block_lengths=[1, 4, 16, 32], task_lambda=10.,
        validation_seed=2026111001, test_seed=2026111002, story_seed=2026111003,
        stage_seed_offset=100000 if stage == 'verify' else 200000 if stage == 'pilot' else 0,
        story_episode=0, anisotropy_time=16, anisotropy_normal=[0., 0., 1.],
        anisotropy_input='static_concentrated', anisotropy_sigma2=.01**2,
        moment_tolerance=5e-8, algebra_tolerance=1e-10,
        physical_sampling='Independent instantaneous velocity errors; no differenced positions.',
        variance_convention='sigma2 is variance per real velocity coordinate in (m/s)^2.',
        information='Bingham and regularized point use (u,v,G). GREAT uses response u and measured G only.',
        controls='Mean, mode and isotropic control share exactly the same A, sigma2, beta and G.',
        selection='Independent validation episode-average noisy held-out response MSE from recorded KKT query observations; no hidden normals or test outcomes select settings.',
        validation_measurements='Three common orthonormal world-frame query requests per reporting time, independent direct velocity noise; recorded queries and responses are transformed into the measured wrist frame.',
        primary='Episode-average held-out clean-response MSE at the declared action cadence.',
        query_unit='Fixed common orthonormal query directions, each requested speed v0; queries averaged within episode.',
        statistical_unit='Independent plant/noise episode; query components and reporting times are not repetitions.',
        scope='Ideal maintained real Gr(2,3) bilateral contact with known isotropic resistance; no hard feasibility guarantee.',
        great_reference='GREAT sliding window and exponentially weighted GREAT; original real response-only objective.')


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


def regularized_mode(P,rho):
    return 2*np.eye(3)/3+float(rho)*(P-2*np.eye(3)/3)


def query_directions(v0=.02):
    Q=np.linalg.qr(np.array([[1.,.2,.3],[-.4,1.,.1],[.1,-.2,1.]]))[0]
    return float(v0)*Q.T


def soft_commands(M,queries,lam):
    return np.linalg.solve(np.eye(3)+float(lam)*(np.eye(3)-M),np.asarray(queries).T).T


def response_loss(M,P,queries):
    error=np.asarray(queries)@(M-P).T
    return float(np.mean(np.sum(error*error,axis=1)))


def observed_response_loss(M,queries,responses):
    """Squared prediction error using measured query/response pairs only."""
    error=np.asarray(queries)@np.asarray(M).T-np.asarray(responses)
    return float(np.mean(np.sum(error*error,axis=1)))


def command_cost(commands,P,desired,lam):
    difference=np.asarray(commands)-np.asarray(desired)
    blocked=np.asarray(commands)@(np.eye(3)-P).T
    return float(np.mean(np.sum(difference*difference,axis=1)+float(lam)*np.sum(blocked*blocked,axis=1)))


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


def _great_policies(cfg):
    return [dict(kind='window',window=w,beta=None,stepsize=eta,inner_steps=q)
            for w in cfg['great_window_grid'] for eta in cfg['great_step_grid'] for q in cfg['great_inner_grid']]+[
            dict(kind='ew',window=None,beta=b,stepsize=eta,inner_steps=q)
            for b in cfg['great_beta_grid'] for eta in cfg['great_step_grid'] for q in cfg['great_inner_grid']]


def great_records(data,policy,times,transport=True):
    from benchmarks.common.great import GREAT
    # Same predeclared point start, never the hidden normal. Uniform Bingham
    # initialization itself has no unique modal plane.
    estimator=GREAT(3,2,stepsize=policy['stepsize'],inner_steps=policy['inner_steps'],
        window=policy['window'],beta=policy['beta'],complex_data=False,initial_basis=np.eye(3)[:,1:])
    records={};times=set(times)
    for j,u in enumerate(data['U']):
        X=estimator.update(u,G=data['G'][j] if transport else None)
        if j+1 in times:records[j+1]=X@X.T
    return records,estimator.diagnostics()


def validation_episode(cfg,condition,trial,*,data=None):
    data=generate_contact_trial(cfg,condition,trial,'validation') if data is None else data
    times=action_times(cfg,condition)
    beta_grid=[1.] if condition.startswith('static_') else cfg['beta_grid']
    rows=[]
    for beta in beta_grid:
        records,_=bingham_records(cfg,data,beta,times)
        losses=[];quadratic=linear=0.
        for t,record in records.items():
            P=record['Pmode'];D0=2*np.eye(3)/3;D1=P-D0
            Q=data['heldout_queries'][t-1];Y=data['heldout_responses'][t-1]
            losses.append(observed_response_loss(record['M'],Q,Y))
            slope=Q@D1.T;residual=Q@D0.T-Y
            quadratic+=float(np.mean(np.sum(slope*slope,axis=1)))
            linear+=float(np.mean(np.sum(slope*residual,axis=1)))
        rows.append(dict(condition=condition,trial=trial,family='bingham',beta=beta,
            loss=float(np.mean(losses)),quadratic=quadratic/len(records),linear=linear/len(records),
            policy=None,numerical_ok=all(r['diagnostics']['numerical_ok'] for r in records.values())))
    for policy in _great_policies(cfg):
        records,diagnostic=great_records(data,policy,times)
        losses=[observed_response_loss(P,data['heldout_queries'][t-1],data['heldout_responses'][t-1])
                for t,P in records.items()]
        rows.append(dict(condition=condition,trial=trial,family='great_'+policy['kind'],
            beta=policy['beta'],loss=float(np.mean(losses)),quadratic=np.nan,linear=np.nan,
            policy=policy,numerical_ok=bool(np.isfinite(losses).all()),diagnostics=diagnostic))
    return rows


def select_policies(rows,cfg):
    policies=[]
    for condition in cfg['conditions']:
        selected=[row for row in rows if row['condition']==condition]
        candidate=[]
        for beta in ([1.] if condition.startswith('static_') else cfg['beta_grid']):
            group=[r for r in selected if r['family']=='bingham' and r['beta']==beta]
            if all(r['numerical_ok'] for r in group):
                candidate.append((np.mean([r['loss'] for r in group]),beta,group))
        if not candidate:
            raise ArithmeticError('No contact validation setting resolved its moment budget')
        loss,beta,group=min(candidate,key=lambda value:(value[0],-value[1]))
        rho=float(np.clip(-sum(r['linear'] for r in group)/sum(r['quadratic'] for r in group),0.,1.))
        record=dict(condition=condition,beta=float(beta),regularization_rho=rho,validation_loss=float(loss))
        for kind in ['window','ew']:
            options=[]
            for policy in _great_policies(cfg):
                if policy['kind']!=kind:continue
                g=[r for r in selected if r['family']=='great_'+kind and r['policy']==policy]
                if all(r['numerical_ok'] for r in g):
                    options.append((float(np.mean([r['loss'] for r in g])),policy))
            if not options:raise ArithmeticError('No finite GREAT validation setting')
            best=min(options,key=lambda value:value[0])
            record['great_'+kind]=dict(best[1],validation_loss=best[0])
        policies.append(record)
    return policies


def _metric(D,Ptrue,M,queries,cfg):
    commands=soft_commands(D,queries,cfg['task_lambda'])
    pstar=soft_commands(M,queries,cfg['task_lambda'])
    # For a projector-valued state, the posterior penalty is linear in I-M.
    expected=float(np.mean(np.sum((commands-queries)**2,axis=1)+cfg['task_lambda']*
        np.einsum('qi,ij,qj->q',commands,np.eye(3)-M,commands)))
    star=float(np.mean(np.sum((pstar-queries)**2,axis=1)+cfg['task_lambda']*
        np.einsum('qi,ij,qj->q',pstar,np.eye(3)-M,pstar)))
    H=np.eye(3)+cfg['task_lambda']*(np.eye(3)-M)
    identity=float(np.mean(np.einsum('qi,ij,qj->q',commands-pstar,H,commands-pstar)))
    return dict(response_mse=response_loss(D,Ptrue,queries),
        response_map_frobenius=float(np.linalg.norm(D-Ptrue,'fro')**2),
        command_cost=command_cost(commands,Ptrue,queries,cfg['task_lambda']),
        posterior_response_excess=float(np.mean(np.sum((queries@(D-M).T)**2,axis=1))),
        posterior_command_excess=expected-star,command_excess_identity_error=abs(expected-star-identity))


def test_episode(cfg,condition,trial,selection):
    data=generate_contact_trial(cfg,condition,trial);times=action_times(cfg,condition)
    policy=next(p for p in selection if p['condition']==condition)
    records,Apath=bingham_records(cfg,data,policy['beta'],times)
    great={kind:great_records(data,policy['great_'+kind],times)[0] for kind in ['window','ew']}
    identity,_=bingham_records(cfg,data,policy['beta'],times,transport=False) if not condition.startswith('static_') else ({},None)
    rows=[];averages={}
    keep=set(cfg['static_times'] if condition.startswith('static_') else cfg['checkpoint_times'])
    for t,record in records.items():
        Q=query_directions(cfg['v0'])@data['R'][t-1]
        values=dict(mean=record['M'],mode=record['Pmode'],isotropic=record['isotropic'],
            regularized=regularized_mode(record['Pmode'],policy['regularization_rho']),
            great_window=great['window'][t],great_ew=great['ew'][t])
        if identity:values.update(mean_identity=identity[t]['M'],mode_identity=identity[t]['Pmode'])
        for method,D in values.items():
            metric=_metric(D,data['P'][t-1],record['M'],Q,cfg)
            numerical=record['diagnostics']['numerical_ok']
            row=dict(condition=condition,trial=trial,method=method,t=t,time_s=float(data['times'][t-1]),
                beta=policy['beta'],sigma2=float(data['sigma2']),mode_unique=record['mode_unique'],
                moment_resolution_error=record['diagnostics']['mean_resolution_error'],
                numerical_ok=bool(numerical),**metric)
            averages.setdefault(method,[]).append(row)
            if t in keep:rows.append(dict(row,endpoint='checkpoint'))
    for method,group in averages.items():
        summary={name:float(np.mean([r[name] for r in group])) for name in
            ['response_mse','response_map_frobenius','command_cost','posterior_response_excess',
             'posterior_command_excess','command_excess_identity_error']}
        rows.append(dict(condition=condition,trial=trial,method=method,t=len(data['U']),
            time_s=float(data['times'][-1]),beta=policy['beta'],sigma2=float(data['sigma2']),
            endpoint='episode_average',mode_unique=all(r['mode_unique'] for r in group),
            moment_resolution_error=max(r['moment_resolution_error'] for r in group),
            numerical_ok=all(r['numerical_ok'] for r in group),**summary))
    story={}
    if trial==cfg['story_episode']:
        story={condition+'_'+key:data[key] for key in
            ['U','V','clean','P','normal','G','R','positions','path','world_normal','world_U','world_V','world_clean','forces','times']}
        story[condition+'_A']=Apath
        story[condition+'_report_times']=np.array(list(records))
        for key in ['M','Pmode','isotropic']:
            story[condition+'_'+key]=np.array([r[key] for r in records.values()])
    return dict(metrics=rows,story=story,diagnostics=dict(condition=condition,trial=trial,**data['diagnostics']))


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


def block_episode(cfg,condition,trial,selection):
    """Separate exact endpoint identity, held-output delay and block modeling."""
    from .final_theory import exact_aggregate,align_block
    data=generate_contact_trial(cfg,condition,trial)
    policy=next(p for p in selection if p['condition']==condition)
    beta=policy['beta'];T=len(data['U']);sigma2=float(data['sigma2'])
    A0=center(np.diag([.3,-.1,-.2]))
    singleton,As=bingham_records(cfg,data,beta,list(range(1,T+1)),A0=A0)
    Qworld=query_directions(cfg['v0']);rows=[]
    common_times=set(range(max(cfg['block_lengths']),T+1,max(cfg['block_lengths'])))
    for L in cfg['block_lengths']:
        A=A0.copy();M0,_=mean_projector(A0);world_M=M0;world_commands=soft_commands(M0,Qworld,cfg['task_lambda'])
        elapsed=0.;held_losses=[];held_cost=[];full_losses=[];common_losses=[];common_cost=[];errors=[]
        moments=0;actions=0;last_output=0;constant_losses=[];constant_drift=[]
        aggregate_failures=0;constant_failures=0
        for t in range(1,T+1):
            if t%L==0:
                start=t-L;begin=perf_counter()
                A=exact_aggregate(A,data['U'][start:t],sigma2,V=data['V'][start:t],
                    beta=beta,G=data['G'][start:t])
                M,d=mean_projector(A);Q=Qworld@data['R'][t-1]
                aggregate_failures+=int(not d['numerical_ok'])
                command=soft_commands(M,Q,cfg['task_lambda'])
                elapsed+=perf_counter()-begin;moments+=1;actions+=1;last_output=t
                world_M=data['R'][t-1]@M@data['R'][t-1].T
                world_commands=command@data['R'][t-1].T
                errors.append(float(np.linalg.norm(A-As[t-1],2)))
                if t not in common_times:
                    # All lengths use exactly the same evaluation endpoints.
                    # The primary held-output score below still covers every
                    # observation, including times between data-block outputs.
                    R=data['R'][t-1];P=data['P'][t-1]
                    held=R.T@world_M@R;commands=world_commands@R
                    held_losses.append(response_loss(held,P,Qworld@R))
                    held_cost.append(command_cost(commands,P,Qworld@R,cfg['task_lambda']))
                    full_losses.append(response_loss(singleton[t]['M'],P,Qworld@R))
                    continue
                common_losses.append(response_loss(M,data['P'][t-1],Q))
                common_cost.append(command_cost(command,data['P'][t-1],Q,cfg['task_lambda']))
                # This reset, uniform-prior common-state fit is a DIFFERENT
                # approximation study, never presented as exact aggregation.
                aligned=align_block(np.zeros((3,3)),data['U'][start:t],V=data['V'][start:t],G=data['G'][start:t])
                W,V=aligned['U'],aligned['V']
                B=(W.T@V+V.T@W-V.T@V)/(2*sigma2)
                block_M,bd=mean_projector(center(B))
                constant_failures+=int(not bd['numerical_ok'])
                constant_losses.append(response_loss(block_M,data['P'][t-1],Q))
                endpoint=data['P'][t-1]
                transport=np.eye(3)
                for j in range(t-1,start,-1):transport=transport@data['G'][j]
                constant_drift.append(float(np.linalg.norm(transport@data['P'][start]@transport.T-endpoint,2)))
            R=data['R'][t-1];P=data['P'][t-1];Q=Qworld@R
            held=R.T@world_M@R;commands=world_commands@R
            held_losses.append(response_loss(held,P,Q))
            held_cost.append(command_cost(commands,P,Q,cfg['task_lambda']))
            full_losses.append(response_loss(singleton[t]['M'],P,Q))
            if last_output>t:raise AssertionError('A delayed output used a future observation')
        rows.append(dict(condition=condition,trial=trial,block_length=L,dt_s=cfg['dt'],beta_single=beta,
            beta_block=beta**L,observations=T,action_outputs=actions,moment_evaluations=moments,
            maximum_output_age_observations=L-1,maximum_output_age_s=(L-1)*cfg['dt'],
            first_data_output_observation=L,initial_output='Saved boundary prior only before first completed block',
            exact_parameter_error=max(errors,default=np.nan),
            singleton_response_mse=float(np.mean(full_losses)),delayed_response_mse=float(np.mean(held_losses)),
            delayed_command_cost=float(np.mean(held_cost)),common_endpoint_response_mse=float(np.mean(common_losses)),
            common_endpoint_singleton_response_mse=float(np.mean([
                response_loss(singleton[t]['M'],data['P'][t-1],Qworld@data['R'][t-1]) for t in sorted(common_times)])),
            common_endpoint_count=len(common_times),common_endpoint_observations=sorted(common_times),
            common_endpoint_command_cost=float(np.mean(common_cost)),
            constant_block_response_mse=float(np.mean(constant_losses)),
            mean_within_block_normal_drift=float(np.mean(constant_drift)),
            measured_update_and_action_seconds=elapsed,
            aggregate_numerical_failures=aggregate_failures,constant_block_numerical_failures=constant_failures,
            timing_scope='Concurrent experiment metadata only; isolated cadence benchmarks are separate.',
            numerical_ok=bool(max(errors,default=0.)<1e-9 and not aggregate_failures
                and not constant_failures and all(r['diagnostics']['numerical_ok'] for r in singleton.values())),
            delay_scope='Hold the preceding world-frame response/action until the completed block arrives; known coordinate changes only.',
            constant_state_scope='Separate uniform-prior, unweighted, beta=1 common-state fit on each completed block.'))
    return rows


def paired_table(frame):
    rows=[]
    for keys,g in frame.groupby(['condition','endpoint','t'],sort=False):
        own=g[g.method.eq('mean')].set_index('trial')
        for method,h in g[~g.method.eq('mean')].groupby('method',sort=False):
            h=h.set_index('trial');ids=own.index.intersection(h.index)
            for metric in ['response_mse','command_cost']:
                values=(h.loc[ids,metric]-own.loc[ids,metric]).to_numpy();finite=np.isfinite(values)
                x=values[finite];N=len(x)
                half=float(student_t.ppf(.975,N-1)*np.std(x,ddof=1)/np.sqrt(N)) if N>1 else np.nan
                rows.append(dict(condition=keys[0],endpoint=keys[1],t=keys[2],reference=method,metric=metric,
                    episodes=len(ids),finite_episodes=N,gain=float(np.mean(x)) if N else np.nan,
                    lower=float(np.mean(x)-half) if N else np.nan,upper=float(np.mean(x)+half) if N else np.nan,
                    numerical_ok=bool(finite.all() and own.loc[ids].numerical_ok.all() and h.loc[ids].numerical_ok.all()),
                    interval='Paired pointwise Student95% across independent episodes; no multiple-comparison guarantee',
                    sign='Reference minus full posterior mean; positive favors full mean'))
    return rows
