"""Isolated pipeline costs at declared action, report, and learning cadences.

Timers partition the complete cycle. The native mean timer includes its own
eigensystem; it is never added to a second eigensystem timer. Persistent bytes
are exact NumPy array sizes, whereas tracemalloc excludes native workspaces/RSS.
"""
from dataclasses import dataclass
from time import perf_counter, process_time
import gc
import tracemalloc
import numpy as np
from .geometry import center, gaussian
from .models import increment, constant
from .spectral_series import rank_one_bingham
from .beamforming_numerics import posterior_report
from .contact_posterior import contact_summary, contact_mean, normal_quadrature
from .final_beamforming import checked_moments, modal_weight
from .beamforming_decisions import isotropized_mean
from .array_geometry_study import covariance_shrinkage
from .causal_variance import CausalVarianceTracker
from .block_em import moments as block_moments
from .final_accuracy import complex_reference, complex_response_second_moment
from benchmarks.common.great import GREAT


def configuration(stage='verify'):
    if stage not in ('verify','pilot','paper'):raise ValueError('Unknown stage')
    return dict(schema=1,stage=stage,seed=2026111017,
        dimensions=[3,8,16,32,64,128],observations=64,
        action_cadences=[1,4,8,16,32,64],repetitions=7 if stage=='paper' else 1,
        warmup_repetitions=1,beta=.97,sigma2=.2,tau2=1.,
        moment_tolerance=1e-10,moment_max_order=4096,fallback_max_order=8192,
        real_n3_resolution_tolerance=5e-8,
        spans=[0.,80.,400.,40000.],stress_max_order=4096,
        em_iterations=[2,4],learning_block=32,report_every=32,
        report_dimensions=[3],report_draws=2048 if stage!='paper' else 16384,
        report_relative_mcse=.08 if stage!='paper' else .025,
        report_proposal_cap=1000000,
        numerical_reference_digits=[100,200],
        statistical_scope='Randomized repeated clocks use the same seeded data. Clock repetitions are not independent statistical episodes.',
        timing_scope='Serial warmed single-thread elapsed/cpu costs, not worst-case deadlines. Stream timers exclude data generation and estimator construction and include all corrections, buffer handling, actions, reports and variance fits. Pilot clocks may share the host and are engineering checks only.',
        memory_scope='Exact live persistent NumPy array bytes and separate tracemalloc peak; native BLAS/LAPACK workspaces and process RSS excluded.',
        rank_scope='Native complex rank one and real codimension one only. Physical contact remains n=3; other real dimensions are integration/cost fixtures.',
        precision_scope='Native first-moment/normalizer truncation, independent precision agreement, second moments, and tails have separate flags.')


@dataclass
class CostCase:
    example:str
    n:int
    case:str
    U:np.ndarray
    V:np.ndarray|None
    sigma2:np.ndarray
    tau2:np.ndarray|None
    desired:np.ndarray
    initial:np.ndarray
    beta:float
    truth:np.ndarray
    policy:dict|None=None
    G:np.ndarray|None=None
    adaptive_initial:tuple|None=None


def synthetic_case(example,n,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],n,example=='beamforming']))
    cx=example=='beamforming';T=cfg['observations']
    Q=np.linalg.qr(gaussian(rng,(n,n),cx))[0];h=Q[:,0]
    a=gaussian(rng,n,cx);a/=np.linalg.norm(a)
    P=np.outer(h,h.conj()) if cx else np.eye(n)-np.outer(h,h)
    if cx:
        U=np.sqrt(cfg['tau2'])*gaussian(rng,T,True)[:,None]*h
        U+=np.sqrt(cfg['sigma2'])*gaussian(rng,(T,n),True);V=None
    else:
        V=gaussian(rng,(T,n));U=V@P.T+np.sqrt(cfg['sigma2'])*gaussian(rng,(T,n))
    return CostCase(example,n,'data_T64',U,V,np.full(T,cfg['sigma2']),
        np.full(T,cfg['tau2']) if cx else None,a,Q,cfg['beta'],np.repeat(P[None],T,axis=0))


def native(A,example,cfg):
    if example=='beamforming':
        result,fallback=checked_moments(A,cfg['moment_tolerance'],cfg['moment_max_order'],cfg['fallback_max_order'])
        return result.mean,result.log_normalizer,dict(numerical_ok=result.numerical_ok,
            mean_error_bound=result.moment_error_bound,logz_error_bound=result.log_normalizer_error_bound,
            order=result.order,cap_extension=fallback)
    if len(A)==3:
        from .final_contact import mean_projector
        M,d=mean_projector(A)
        return M,d['log_normalizer'],dict(numerical_ok=d['numerical_ok'],
            mean_error_bound=d.get('mean_resolution_error',np.nan),logz_error_bound=d.get('log_normalizer_resolution_error',np.nan),
            order=np.nan,cap_extension=False)
    result=rank_one_bingham(-A,tol=cfg['moment_tolerance'],max_order=cfg['moment_max_order'])
    return np.eye(len(A))-result.mean,result.log_normalizer+np.trace(A).real,dict(
        numerical_ok=result.accuracy_met,mean_error_bound=result.moment_error_bound,
        logz_error_bound=result.log_normalizer_error_bound,order=result.order,cap_extension=False)


def extract(A,example,method,cfg):
    n=len(A);k=1 if example=='beamforming' else n-1
    if method=='mode':
        _,E=np.linalg.eigh(A);h=E[:,-1] if k==1 else E[:,0]
        P=np.outer(h,h.conj()) if k==1 else np.eye(n)-np.outer(h,h.conj())
        return P,dict(numerical_ok=True),h
    M,psi,d=native(A,example,cfg)
    payload=None
    if method=='isotropic':
        _,E=np.linalg.eigh(A);h=E[:,-1] if k==1 else E[:,0]
        P=np.outer(h,h.conj()) if k==1 else np.eye(n)-np.outer(h,h.conj())
        if k==1:
            M,_=isotropized_mean(M,P)
            parallel=float(np.trace(P@M).real);perpendicular=(1-parallel)/(n-1)
        else:
            alpha=float(np.trace((np.eye(n)-P)@(np.eye(n)-M)).real)
            perpendicular=1-(1-alpha)/(n-1);parallel=1-alpha
            M=perpendicular*P+parallel*(np.eye(n)-P)
        payload=dict(h=h,parallel=parallel,perpendicular=perpendicular)
    return M,d,payload


def action(M,example,a,sigma2,tau2,h=None,rho=1.):
    """Efficient point actions; dense solve only for a full mean/covariance."""
    if example=='beamforming':
        if isinstance(h,dict):
            c=sigma2+tau2*h['perpendicular'];b=tau2*(h['parallel']-h['perpendicular'])
            z=a-b/(c+b)*h['h']*np.vdot(h['h'],a)
            return z/np.vdot(a,z)
        if h is not None:return modal_weight(h,a,sigma2,tau2,rho)
        z=np.linalg.solve(sigma2*np.eye(len(a))+tau2*M,a)
        return z/np.vdot(a,z)
    # Contact mean projection and mode projection are the same task map.
    return M@a


def contact_command(M,a,axial=None,lam=10.,projector=False):
    """Soft contact command; projector/axial cases use inverse identities."""
    if projector:return a-lam/(1+lam)*(a-M@a)
    if isinstance(axial,dict):
        c=1+lam*(1-axial['perpendicular']);b=lam*(axial['perpendicular']-axial['parallel'])
        h=axial['h'];return (a-b/(c+b)*h*np.vdot(h,a))/c
    return np.linalg.solve(np.eye(len(a))+lam*(np.eye(len(a))-M),a)


def array_bytes(obj,seen=None):
    """Count persistent array storage once, including queue/buffer contents."""
    if seen is None:seen=set()
    if id(obj) in seen:return 0
    seen.add(id(obj))
    if isinstance(obj,np.ndarray):
        base=obj
        while isinstance(base.base,np.ndarray):base=base.base
        if base is not obj:
            if id(base) in seen:return 0
            seen.add(id(base))
        return int(base.nbytes)
    if isinstance(obj,dict):return sum(array_bytes(v,seen) for v in obj.values())
    if isinstance(obj,(list,tuple)) or obj.__class__.__name__=='deque':return sum(array_bytes(v,seen) for v in obj)
    if hasattr(obj,'__dict__'):return array_bytes(vars(obj),seen)
    return 0


def run_pipeline(case,method,cadence,cfg,*,report_every=0,em_iterations=0):
    """Time one actual stream. All scoring occurs after the complete timer."""
    cx=case.example=='beamforming';n=case.n;T=len(case.U)
    dtype=complex if cx else float
    A=None if method.startswith('great_') or method=='covariance_loading' else np.zeros((n,n),dtype)
    S=np.zeros((n,n),dtype) if method=='covariance_loading' else None;mass=0.
    beta=case.beta;policy=case.policy or {};gp=policy.get(method,{})
    p=next((p for p in policy.get('policies',[]) if T in p['times']),{})
    if method=='modal_loading':beta=p.get('scalar_beta',beta)
    if method=='covariance_loading':beta=p.get('covariance_beta',beta)
    great=None;adaptive=None
    if method.startswith('great_'):
        great=GREAT(n,1 if cx else n-1,stepsize=gp.get('stepsize',.005 if cx else .0005),
            inner_steps=gp.get('inner_steps',3),window=gp.get('window',32) if method=='great_window' else None,
            beta=gp.get('beta',beta) if method=='great_ew' else None,complex_data=cx,
            initial_basis=case.initial[:,:1 if cx else n-1])
    if method=='adaptive_mean':
        initial_s,initial_q=case.adaptive_initial or (case.sigma2[0],None if not cx else case.tau2[0])
        beta=policy.get('adaptive_policy',{}).get('beta',beta)
        adaptive=CausalVarianceTracker(n,1 if cx else n-1,initial_s,
            initial_q,beta=beta,block_length=cfg['learning_block'],
            em_iterations=em_iterations,floor=1e-8,complex_data=cx,
            moment_tolerance=cfg['moment_tolerance'],max_order=cfg['moment_max_order'])
        if 'adaptive_tracker_kwargs' in policy:
            kwargs=dict(policy['adaptive_tracker_kwargs']);kwargs['em_iterations']=em_iterations
            adaptive=CausalVarianceTracker(**kwargs)
    correction=action_seconds=report_seconds=fitting=0.;actions=reports=fits=0
    mean_ok=fit_ok=report_ok=True;records=[];final_action=None;report_tail_ok=None;report_state_ok=None
    score_ok=True;score_count=0;final_command=None
    start=perf_counter();cpu=process_time()
    for j,u in enumerate(case.U):
        t=j+1;s=float(case.sigma2[j]);q=None if not cx else float(case.tau2[j])
        v=None if cx else case.V[j];g=None if case.G is None else case.G[j]
        desired=case.desired[j] if case.desired.ndim==2 else case.desired
        due=t%cadence==0 or t==T
        if adaptive is not None and due:
            tic=perf_counter();prior=adaptive.predict_parameter(g);s=adaptive.sigma2;q=adaptive.tau2
            M,psi0,d=block_moments(prior,adaptive.k,cx,adaptive.moment_tolerance,adaptive.max_order)
            final_action=action(M,case.example,desired,s,q)
            if not cx:final_command=contact_command(M,desired)
            action_seconds+=perf_counter()-tic;actions+=1;mean_ok &= bool(d['numerical_ok'])
            records.append((t,final_action.copy(),desired.copy()))
            tic=perf_counter()
            _,psi1,d1=block_moments(prior+increment(u,s,q,v,cx),adaptive.k,cx,adaptive.moment_tolerance,adaptive.max_order)
            score=constant(u,1 if cx else n-1,s,q,v,cx)+psi1-psi0
            score_ok &= bool(d1['numerical_ok'] and np.isfinite(score));score_count+=1
            report_seconds+=perf_counter()-tic;reports+=1
        if adaptive is not None:
            record=adaptive.update(u,v,g);A=adaptive.A;s=adaptive.sigma2;q=adaptive.tau2
            correction+=record.correction_seconds;fitting+=record.fitting_seconds
            fits+=record.fit_performed;fit_ok &= record.fit_numerical_ok
        else:
            tic=perf_counter()
            if great is not None:
                value=u
                if method=='great_ew' and cx:
                    eta=q/(s*(s+q));s0=case.sigma2[0];q0=case.tau2[0]
                    value=np.sqrt(eta/(q0/(s0*(s0+q0))))*u
                great.update(value,g)
            elif method=='covariance_loading':
                S=beta*S+np.outer(u,u.conj());mass=beta*mass+1
            else:
                if g is not None:A=g@A@g.conj().T
                A=center(beta*A+increment(u,s,q,v,complex_data=cx))
            correction+=perf_counter()-tic
        if due and adaptive is None:
            tic=perf_counter();h=None
            if great is not None:
                M=great.projector;h=great.basis[:,0] if cx else None;d=dict(numerical_ok=True)
            elif method=='covariance_loading':
                R=covariance_shrinkage(S/mass,p.get('alpha',.1))
                z=np.linalg.solve(R,desired);final_action=z/np.vdot(desired,z)
                M=None;d=dict(numerical_ok=np.isfinite(final_action).all())
            else:M,d,h=extract(A,case.example,'mean' if method=='mean_report' else 'mode' if method in ('modal_loading','regularized') else method,cfg)
            if method=='regularized':
                rho=policy['regularization_rho'];M=rho*M+(1-rho)*(n-1)/n*np.eye(n)
                h=dict(h=h,parallel=(1-rho)*(n-1)/n,perpendicular=rho+(1-rho)*(n-1)/n)
            if M is not None:
                final_action=action(M,case.example,desired,s,q,h,p.get('rho',1.))
                if not cx:final_command=contact_command(M,desired,h,projector=method in ('mode','great_window','great_ew'))
            action_seconds+=perf_counter()-tic;actions+=1;mean_ok &= bool(d['numerical_ok'])
            records.append((t,final_action.copy(),desired.copy()))
        if report_every and (t%report_every==0 or t==T):
            tic=perf_counter()
            if cx:
                rep=posterior_report(A,desired[None],draws=cfg['report_draws'],seed=cfg['seed']+t,
                    max_proposals=cfg['report_proposal_cap'],relative_mcse_tolerance=cfg['report_relative_mcse'])
            else:rep=contact_summary(A,desired[None])
            report_ok &= bool(rep['numerics']['numerical_ok']);report_seconds+=perf_counter()-tic;reports+=1
            info=rep['numerics'];report_tail_ok=bool(info.get('tail_ok',info.get('response_quantiles_ok',False)))
            report_state_ok=bool(info.get('state_ok',info.get('regions_numerical_ok',False)))
    elapsed=perf_counter()-start;cpu_elapsed=process_time()-cpu
    persistent=array_bytes([A,S,great,adaptive])
    loss=[]
    for t,w,desired in records:
        P=case.truth[t-1]
        if cx:loss.append(float(case.sigma2[t-1]*np.vdot(w,w).real+case.tau2[t-1]*np.vdot(w,P@w).real))
        else:loss.append(float(np.linalg.norm(w-P@desired)**2))
    phase=correction+action_seconds+report_seconds+fitting
    return dict(wall_seconds=elapsed,cpu_seconds=cpu_elapsed,correction_seconds=correction,
        action_seconds=action_seconds,report_seconds=report_seconds,fitting_seconds=fitting,
        overhead_seconds=elapsed-phase,actions=actions,reports=reports,fits=fits,
        mean_ok=bool(mean_ok),fit_ok=bool(fit_ok),report_ok=bool(report_ok),
        report_tail_ok=report_tail_ok,report_state_ok=report_state_ok,
        predictive_score_ok=score_ok,predictive_scores=score_count,
        persistent_array_bytes=persistent,
        task_loss=float(np.mean(loss)),final_action=final_action,final_command=final_command,
        parameter=(great.covariance if great is not None else S if A is None else A).copy())


def pipeline_specs(example,cases,cfg):
    specs=[]
    for case in cases:
        methods=['mean','mode','isotropic']
        if example=='beamforming':methods+=['great_window','great_ew','covariance_loading']
        elif case.n==3:methods+=['great_window','great_ew']
        if case.case=='task_adaptive':
            specs.append((case,'adaptive_mean',32,0,case.policy['adaptive_policy']['em_iterations']))
            continue
        if case.case.startswith('task_'):
            if example=='beamforming':methods+=['modal_loading'];cadences=[len(case.U)] if case.case=='task_stationary' else [8]
            else:methods+=['regularized'];cadences=[4]
        else:cadences=cfg['action_cadences']
        for method in methods:
            for cadence in cadences:specs.append((case,method,cadence,0,0))
        if case.n in cfg['report_dimensions'] and case.case=='data_T64':
            specs.append((case,'mean_report',8,cfg['report_every'],0))
        if case.n==(32 if example=='beamforming' else 3) and case.case=='data_T64':
            for em in cfg['em_iterations']:
                for cadence in (8,32):specs.append((case,'adaptive_mean',cadence,0,em))
    return specs


def benchmark(example,cases,cfg,progress=None):
    specs=pipeline_specs(example,cases,cfg);rows=[];memory=[];parameters={}
    rng=np.random.default_rng(cfg['seed']+100)
    for case,method,cadence,report,em in specs:
        for _ in range(cfg['warmup_repetitions']):
            try:run_pipeline(case,method,cadence,cfg,report_every=report,em_iterations=em)
            except (ArithmeticError,ValueError,np.linalg.LinAlgError):pass
    for repetition in range(cfg['repetitions']):
        for index in rng.permutation(len(specs)):
            case,method,cadence,report,em=specs[index]
            prefix=dict(example=example,n=case.n,case=case.case,method=method,
                action_cadence=cadence,report_every=report,learn_every=cfg['learning_block'] if em else 0,
                em_iterations=em,observations=len(case.U),repetition=repetition)
            try:
                result=run_pipeline(case,method,cadence,cfg,report_every=report,em_iterations=em)
                A=result.pop('parameter');result.pop('final_action');result.pop('final_command');failure=''
                parameters[f'{case.case}_n{case.n}_{method}']=A
            except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
                result=dict(mean_ok=False,fit_ok=False,report_ok=False);failure=type(error).__name__+': '+str(error)
            rows.append(dict(prefix,**result,failure=failure))
        if progress:progress(f'{example} timing repetition {repetition+1}/{cfg["repetitions"]}')
    # Allocation tracing is a separate experiment and never surrounds clocks.
    for case,method,cadence,report,em in specs:
        # Cadence changes extraction frequency, not retained state shape.
        # Trace endpoint schedules and each distinct reporting/EM schedule.
        if case.case=='data_T64' and cadence!=len(case.U) and not (report or em):continue
        gc.collect();tracemalloc.start()
        try:
            result=run_pipeline(case,method,cadence,cfg,report_every=report,em_iterations=em)
            _,peak=tracemalloc.get_traced_memory();failure=''
        except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
            _,peak=tracemalloc.get_traced_memory();result={};failure=str(error)
        finally:tracemalloc.stop()
        memory.append(dict(example=example,n=case.n,case=case.case,method=method,action_cadence=cadence,
            report_every=report,em_iterations=em,python_numpy_peak_bytes=int(peak),
            persistent_array_bytes=result.get('persistent_array_bytes'),
            persistent_scope='Live end-of-cycle estimator arrays, including retained EM history; immutable input data and outputs excluded.',failure=failure,
            scope=cfg['memory_scope']))
    return rows,memory,parameters


def numerical_accuracy(example,cases,cfg,saved_parameters=None):
    rows=[];parameters={}
    for case in cases:
        n=case.n;rng=np.random.default_rng(cfg['seed']+n)
        Q=np.linalg.qr(gaussian(rng,(n,n),example=='beamforming'))[0]
        targets=[]
        for span in (cfg['spans'] if case.case=='data_T64' else []):
            lam=np.r_[0.,-np.linspace(.25,1.,n-1)*span]
            B=center((Q*lam)@Q.conj().T);targets.append((f'span_{span:g}',B if example=='beamforming' else -B))
        key=f'{case.case}_n{n}_'+('adaptive_mean' if case.case=='task_adaptive' else 'mean')
        A=None if saved_parameters is None else saved_parameters.get(key)
        if A is None:
            A=np.zeros((n,n),dtype=case.U.dtype)
            for j,u in enumerate(case.U):
                if case.G is not None:A=case.G[j]@A@case.G[j].conj().T
                A=center(case.beta*A+increment(u,case.sigma2[j],None if case.V is not None else case.tau2[j],None if case.V is None else case.V[j],example=='beamforming'))
        targets.append((case.case,A))
        for name,A in targets:
            parameters[f'{name}_n{n}']=A
            row=dict(example=example,n=n,case=name,span=float(np.ptp(np.linalg.eigvalsh(A))),
                mean_requested_tolerance=cfg['real_n3_resolution_tolerance'] if example=='contact' and n==3 else cfg['moment_tolerance'],
                normalizer_requested_tolerance=cfg['real_n3_resolution_tolerance'] if example=='contact' and n==3 else cfg['moment_tolerance'],
                native_max_order=cfg['stress_max_order'] if name=='span_40000' else cfg['moment_max_order'],
                mean_ok=False,normalizer_ok=False,second_moment_evaluated=False,second_moment_ok=None,
                tail_evaluated=False,tail_ok=None,general_rank_supported=False,
                reference_ok=False,reference_scope='',failure='')
            try:
                if name=='span_40000':
                    rank_one_bingham(A if example=='beamforming' else -A,complex_data=example=='beamforming',tol=cfg['moment_tolerance'],max_order=cfg['stress_max_order'])
                M,z,d=native(A,example,cfg)
                row.update(mean_truncation_bound=d['mean_error_bound'],normalizer_truncation_bound=d['logz_error_bound'],native_order=d['order'])
                if example=='beamforming':
                    values,E=np.linalg.eigh(A);ref=complex_reference(values,digits=tuple(cfg['numerical_reference_digits']))
                    diagonal=np.real(np.diag(E.conj().T@M@E));indices=ref['selected_indices']
                    me=float(np.max(abs(diagonal[indices]-ref['selected_mean'][indices])));ze=abs(z-ref['log_normalizer'])
                    row.update(mean_reference_error=me,normalizer_reference_error=ze,reference_ok=ref['numerical_ok'],
                        reference_cross_precision_error=ref['cross_precision_error'],reference_scope=ref['scope'],
                        mean_ok=bool(d['numerical_ok'] and ref['numerical_ok'] and me<cfg['moment_tolerance']),
                        normalizer_ok=bool(d['numerical_ok'] and ref['numerical_ok'] and ze<cfg['moment_tolerance']))
                    if n==3:
                        from .final_theory import native_response_covariance
                        v=case.desired;ind,info=complex_response_second_moment(A,v)
                        covariance,_,meta=native_response_covariance(A,1,v,True)
                        second=covariance+np.outer(M@v,(M@v).conj())
                        error=float(np.linalg.norm(second-ind,2))
                        row.update(second_moment_evaluated=True,second_moment_error=error,
                            second_moment_requested_tolerance=2e-5,second_moment_ok=bool(info['numerical_ok'] and error<2e-5))
                elif n==3:
                    low=normal_quadrature(-A,128,256);high=normal_quadrature(-A,256,512)
                    me=float(np.linalg.norm(M-(np.eye(n)-high['mean']),2));ze=abs(z-(np.trace(A)+high['log_normalizer']))
                    res=max(float(np.linalg.norm(low['mean']-high['mean'],2)),abs(low['log_normalizer']-high['log_normalizer']))
                    row.update(mean_reference_error=me,normalizer_reference_error=ze,
                        reference_ok=res<1e-7,reference_cross_precision_error=res,
                        reference_scope='Concentration-adapted sphere integration resolution comparison; no floating-point enclosure.',
                        mean_ok=bool(d['numerical_ok'] and me<cfg['real_n3_resolution_tolerance'] and res<cfg['real_n3_resolution_tolerance']),
                        normalizer_ok=bool(d['numerical_ok'] and ze<cfg['real_n3_resolution_tolerance'] and res<cfg['real_n3_resolution_tolerance']))
                else:
                    row.update(mean_ok=bool(d['numerical_ok']),normalizer_ok=bool(d['numerical_ok']),
                        reference_scope='Native truncation only; independent real high-dimensional reference not supplied.')
            except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:row['failure']=type(error).__name__+': '+str(error)
            rows.append(row)
    return rows,parameters
