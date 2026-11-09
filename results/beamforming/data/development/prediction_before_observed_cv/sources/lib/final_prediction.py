"""Actual causal-recursion prediction, variance adaptation and mismatch studies."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq
from .causal_variance import CausalVarianceTracker
from .block_em import moments
from .geometry import center
from .models import increment, constant
from .guide_reliability import (gaussian, unit, path, projector, posterior_nodes,
    response_values, real_region_thresholds, weighted_quantile, normal_mixture_quantile,
    gaussian_logscore, logscore, interval, binomial_interval, friction_force)
from .contact_posterior import _ResponseCDF
from .contact_mechanics import solve_contact_velocity


def configuration(stage):
    if stage not in ('verify','pilot','paper'):
        raise ValueError('Use verify, pilot or paper')
    paper=stage=='paper'
    return dict(stage=stage,source_commit='a91faabeb10a63b1d1c613d75cc32503cfc0cac6',
        validation_episodes=64 if paper else 3,test_episodes=256 if paper else 3,
        calibration_validation_episodes=32 if paper else 3,
        calibration_test_episodes=128 if paper else 3,
        mismatch_episodes=64 if paper else 3,
        validation_seed=2026111011,test_seed=2026111012,calibration_seed=2026111013,
        mismatch_seed=2026111014,seed_offset=0 if paper else 100000 if stage=='verify' else 200000,
        primary_beam_n=32,primary_beam_T=256,primary_beam_dt=.01,
        primary_contact_T=192,primary_contact_dt=.05,primary_report_every=32,
        calibration_n=3,calibration_T=64,calibration_times=[32,64],
        calibration_sigma2=.2,calibration_tau2=1.,calibration_drift_angle=.8,
        nominal=[.5,.8,.95],posterior_draws=4096 if paper else 1024,
        real_orders=[[48,96],[64,192]],state_mass_tolerance=.035,
        quantile_resolution_tolerance=.006,
        moment_tolerance=1e-9,max_order=4096,ascent_tolerance=1e-7,
        real_quadrature_resolution_tolerance=1e-7,
        second_moment_tolerance=1e-6,
        numerical_scope='Native mean/log-normalizer flags bound series truncation in exact arithmetic, excluding rounding. The real N3 fallback compares two quadrature resolutions at tolerance1e-7, not a rigorous error enclosure. Tail/CDF and second-moment checks are separate.',
        beam_floor=1e-5,contact_floor=1e-8,calibration_floor=1e-5,
        policy_candidates=[
            dict(beta=1.,block_length=16,em_iterations=2),
            dict(beta=.98,block_length=16,em_iterations=2),
            dict(beta=.9,block_length=16,em_iterations=2),
            dict(beta=.98,block_length=32,em_iterations=2),
            dict(beta=.98,block_length=16,em_iterations=4)],
        beam_conditions=['noise_change','smooth','power_change'],
        contact_conditions=['noise_change','fast_sphere'],
        heldout_draws=128,task_penalty=10.,
        precision='Primary256 independent test episodes; calibration128 independent episodes with exact binomial intervals. Reporting times and queries are averaged within episodes. All precision flags are retained.',
        variance_convention='sigma2 is per-real-coordinate variance, or proper-complex covariance; complex real/imaginary noise components have variance sigma2/2.',
        causal_rule='One singleton correction; completed nonoverlapping past blocks; transported preceding prior fixed during EM; fitted variances used only for future corrections; streaming state is never replaced or augmented by fitted block posterior.',
        timing_scope='Elapsed component clocks are execution metadata. Comparative warmed serial cadence timing is a separate root-owned stage.',
        causal_warm_start='One current-parameter warm start per completed block; if signal variance is zero, also use a declared positive interior fallback. Multiple-start stationary reference experiments are retained separately.',
        validation_scoring_scope='Policy validation uses future true-state task loss; it does not charge or compute optional predictive density reports. Test prediction scores are computed before assimilation.',
        mismatch=dict(contamination_power=.5,secondary_power=.35,noise_correlation=.5,
            friction_coefficient=.35,anisotropic_damping=[1.,2.,4.],
            position_dt=.05,position_sigma=.01))


def seed(cfg, base, example, episode, condition_index=0):
    return np.random.SeedSequence([int(cfg[base])+cfg['seed_offset'],
        0 if example=='beamforming' else 1,int(condition_index),int(episode)])


def tracker_kwargs(example, n, policy, cfg, *, learn, calibration=False):
    if calibration:
        sigma2,tau2,floor=.05,(.2 if example=='beamforming' else None),cfg['calibration_floor']
    elif example=='beamforming':
        sigma2,tau2,floor=.05,.2,cfg['beam_floor']
    else:
        sigma2,tau2,floor=.002**2,None,cfg['contact_floor']
    # Starts are fixed numerical initializations, not supplied true values.
    return dict(n=n,k=1 if example=='beamforming' else n-1,sigma2=sigma2,tau2=tau2,
        beta=policy['beta'],block_length=policy['block_length'],
        em_iterations=policy['em_iterations'],starts=None,floor=floor,
        complex_data=example=='beamforming',learn=learn,
        moment_tolerance=cfg['moment_tolerance'],max_order=cfg['max_order'],
        ascent_tolerance=cfg['ascent_tolerance'])


def same_mode_controls(A,M,k):
    values,Q=np.linalg.eigh(A);X=Q[:,-k:];P=X@X.conj().T;n=len(A)
    alpha=float(np.trace(P@M).real/k)
    xi=float((k-k*alpha)/(n-k))
    Mi=alpha*P+xi*(np.eye(n)-P)
    return P,Mi,float(values[-k]-values[-k-1])


def accuracy_fields(d,cfg):
    quadrature='resolution_error' in d
    return dict(moment_backend=d.get('backend','unknown'),
        mean_accuracy_estimate=float(d.get('resolution_error',d.get('moment_error_bound',np.nan))),
        normalizer_accuracy_estimate=float(d.get('resolution_error',d.get('log_normalizer_error_bound',np.nan))),
        accuracy_tolerance=cfg['real_quadrature_resolution_tolerance'] if quadrature else cfg['moment_tolerance'],
        accuracy_scope='Quadrature resolution difference' if quadrature else 'Exact-arithmetic series truncation')


def mvdr(R,a):
    z=np.linalg.solve(R,a);return z/np.vdot(a,z)


def axial_mvdr(P,a,sigma2,coefficient):
    """Closed-form inverse for sigma2 I + coefficient P."""
    z=a/sigma2-coefficient/(sigma2*(sigma2+coefficient))*(P@a)
    return z/np.vdot(a,z)


def beam_actions(M,P,Mi,a,sigma2,tau2):
    n=len(M);alpha=float(np.trace(P@Mi).real)
    xi=float((np.trace(Mi).real-alpha)/(n-1))
    return dict(mean=mvdr(sigma2*np.eye(n)+tau2*M,a),
        mode=axial_mvdr(P,a,sigma2,tau2),
        iso=axial_mvdr(P,a,sigma2+tau2*xi,tau2*(alpha-xi)))


def physical_task_rows(example, stream, t, tr, cfg, rng, common):
    """Future-only task rules use the pre-correction parameter at time t."""
    G=stream.get('G',None)
    g=None if G is None else G[t]
    A=tr.predict_parameter(g);start=perf_counter()
    M,psi0,diag=moments(A,tr.k,tr.complex_data,tr.moment_tolerance,tr.max_order)
    mean_seconds=perf_counter()-start
    P,Mi,gap=same_mode_controls(A,M,tr.k)
    truth=stream['P'][t]
    sigma2=float(np.asarray(stream['sigma2'])[t])
    tau2=None if example=='contact' else float(np.asarray(stream['tau2'])[t])
    rows=[]
    if example=='beamforming':
        a=stream['a'];n=len(a)
        h=stream['h'][t]
        held=common
        oracle=axial_mvdr(truth,a,sigma2,tau2)
        oracle_loss=float(sigma2*np.vdot(oracle,oracle).real+tau2*abs(np.vdot(oracle,h))**2)
        start=perf_counter();actions=beam_actions(M,P,Mi,a,tr.sigma2,tr.tau2)
        action_seconds=perf_counter()-start
        for control,w in actions.items():
            true=float(sigma2*np.vdot(w,w).real+tau2*abs(np.vdot(w,h))**2)
            heldloss=float(np.mean(abs(held@w.conj())**2))
            rows.append(dict(control=control,future_task_loss=heldloss,true_task_loss=true,
                oracle_loss=oracle_loss,output_oracle_db=float(10*np.log10(true/oracle_loss)),
                heldout_output_oracle_db=float(10*np.log10(heldloss/oracle_loss)),
                chordal_error=float(np.linalg.norm(P-truth,'fro')/np.sqrt(2)),
                distortionless_error=float(abs(np.vdot(w,a)-1)),response_mse=np.nan,
                command_cost=np.nan,action_seconds=action_seconds))
    else:
        V=common;desired=stream['V'][t];lam=cfg['task_penalty']
        for control,D in [('mean',M),('mode',P),('iso',Mi)]:
            start=perf_counter()
            response=np.mean(np.sum(abs(V@(D-truth).T)**2,axis=1))
            command=np.linalg.solve(np.eye(tr.n)+lam*(np.eye(tr.n)-D),desired)
            cost=float(np.linalg.norm(command-desired)**2+lam*np.vdot(command,(np.eye(tr.n)-truth)@command).real)
            rows.append(dict(control=control,future_task_loss=float(response),true_task_loss=float(response),
                oracle_loss=0.,output_oracle_db=np.nan,heldout_output_oracle_db=np.nan,
                chordal_error=float(np.linalg.norm(P-truth,'fro')/np.sqrt(2)),
                distortionless_error=np.nan,response_mse=float(response),command_cost=cost,
                action_seconds=perf_counter()-start))
    for row in rows:row.update(mean_ok=bool(diag['numerical_ok']),mode_gap=gap,mean_seconds=mean_seconds,
                              **accuracy_fields(diag,cfg))
    return rows,dict(A=A,M=M,P=P,Mi=Mi,truth=truth,log_normalizer=psi0,
                     mean_numerical_ok=bool(diag['numerical_ok']))


def adaptation_trial(example, stream, policies, cfg, episode, split, condition, *, story=False, method_subset=None):
    """Run fixed/learned variances on identical measurements and query draws."""
    U=stream['U'];T,n=U.shape
    methods=['known','misspecified','learned'] if method_subset is None else list(method_subset)
    trackers={name:CausalVarianceTracker(**tracker_kwargs(example,n,policies[name],cfg,
               learn=name=='learned')) for name in methods}
    rows,events,arrays=[],[],{}
    rng=np.random.default_rng(seed(cfg,'test_seed' if split=='test' else 'validation_seed',
                                  example,episode,100+cfg['beam_conditions'].index(condition)
                                  if example=='beamforming' else 200+cfg['contact_conditions'].index(condition)))
    report_times=set(range(cfg['primary_report_every']-1,T,cfg['primary_report_every']))
    for t,u in enumerate(U):
        g=None if stream.get('G') is None else stream['G'][t]
        v=None if example=='beamforming' else stream['V'][t]
        common=None
        if t in report_times:
            if example=='beamforming':
                common=stream['h'][t]*gaussian(rng,(cfg['heldout_draws'],1),True)*np.sqrt(stream['tau2'][t])
                common=common+np.sqrt(stream['sigma2'][t])*gaussian(rng,(cfg['heldout_draws'],n),True)
            else:
                scale=float(stream.get('query_scale',.02))
                common=scale*np.eye(n)
        for method,tr in trackers.items():
            if method=='known':
                tr.sigma2=float(stream['sigma2'][t])
                if example=='beamforming':tr.tau2=float(stream['tau2'][t])
            pending=[]
            if t in report_times:
                try:
                    pending,state=physical_task_rows(example,stream,t,tr,cfg,rng,common)
                    started=perf_counter()
                    if split=='validation':
                        score,score_ok=np.nan,True
                    else:
                        try:
                            add=increment(u,tr.sigma2,tr.tau2,v,tr.complex_data)
                            _,psi1,score_diag=moments(state['A']+add,tr.k,tr.complex_data,
                                tr.moment_tolerance,tr.max_order)
                            score=float(constant(u,tr.k,tr.sigma2,tr.tau2,v,tr.complex_data)
                                +psi1-state['log_normalizer'])
                            score_ok=bool(score_diag['numerical_ok'] and state['mean_numerical_ok'])
                        except (ArithmeticError,ValueError,np.linalg.LinAlgError):
                            score,score_ok=np.nan,False
                    score_seconds=perf_counter()-started
                    if example=='beamforming':
                        surrogate=gaussian_logscore(u,np.zeros(n),tr.sigma2*np.eye(n)+tr.tau2*state['M'],True)
                    else:
                        # At physical checkpoints retain the exact mixture score;
                        # full Gaussian predictive covariance is evaluated by
                        # the independent-node calibration cohort below.
                        surrogate=np.nan
                    for row in pending:
                        row.update(example=example,condition=condition,episode=episode,split=split,
                            method=method,t=t+1,time_s=t*stream['dt'],sigma2_true=float(stream['sigma2'][t]),
                            tau2_true=np.nan if example=='contact' else float(stream['tau2'][t]),
                            sigma2_estimate=tr.sigma2,tau2_estimate=tr.tau2,
                            prequential_score=score,gaussian_score=surrogate,
                            score_numerical_ok=score_ok,score_seconds=score_seconds,
                            beta=tr.beta,block_length=tr.block_length,em_iterations=tr.em_iterations,
                            prediction_scope='Before assimilation of the scored observation',
                            variance_information='True scales supplied' if method=='known' else 'Recorded past observations only')
                    if story and method=='learned':
                        prefix=f'{example}_{condition}_t{t+1}_'
                        arrays.update({prefix+key:value for key,value in state.items()})
                except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
                    pending=[dict(example=example,condition=condition,episode=episode,split=split,
                        method=method,control='mean',t=t+1,time_s=t*stream['dt'],
                        sigma2_true=float(stream['sigma2'][t]),sigma2_estimate=tr.sigma2,
                        tau2_true=None if example=='contact' else float(stream['tau2'][t]),
                        tau2_estimate=tr.tau2,mean_ok=False,score_numerical_ok=False,
                        future_task_loss=np.nan,true_task_loss=np.nan,prequential_score=np.nan,
                        error=type(error).__name__+': '+str(error))]
            record=tr.update(u,v,g)
            if pending:
                for row in pending:
                    row.update(sigma2_next=record.sigma2_next,tau2_next=record.tau2_next,
                        fit_performed=record.fit_performed,fit_numerical_ok=record.fit_numerical_ok,
                        fitting_seconds=record.fitting_seconds,
                        correction_seconds=record.correction_seconds)
                rows.extend(pending)
            if record.fit_performed:
                event=dict(example=example,condition=condition,episode=episode,split=split,
                    method=method,t=t+1,time_s=t*stream['dt'],**asdict(record),
                    history=tr.fit_history[-1])
                events.append(event)
        if story:
            prefix=f'{example}_{condition}_'
            arrays[prefix+'U']=U;arrays[prefix+'sigma2_true']=np.asarray(stream['sigma2'])
            if example=='beamforming':arrays[prefix+'tau2_true']=np.asarray(stream['tau2']);arrays[prefix+'h']=stream['h'];arrays[prefix+'a']=stream['a']
            else:arrays[prefix+'V']=stream['V'];arrays[prefix+'G']=stream['G'];arrays[prefix+'positions']=stream.get('positions',np.zeros_like(U))
    if story:
        prefix=f'{example}_{condition}_'
        for method,tr in trackers.items():
            arrays[prefix+method+'_sigma2_used']=np.array([x.sigma2_used for x in tr.records])
            arrays[prefix+method+'_sigma2_next']=np.array([x.sigma2_next for x in tr.records])
            arrays[prefix+method+'_fit_performed']=np.array([x.fit_performed for x in tr.records])
            if example=='beamforming':
                arrays[prefix+method+'_tau2_used']=np.array([x.tau2_used for x in tr.records])
                arrays[prefix+method+'_tau2_next']=np.array([x.tau2_next for x in tr.records])
    return rows,events,arrays


def calibration_stream(example, regime, cfg, episode, split, stress='matched',mismatch=False):
    """Matched or declared-mismatch observations for the actual recursion."""
    rng=np.random.default_rng(seed(cfg,'mismatch_seed' if mismatch else 'calibration_seed' if split=='test' else 'validation_seed',
        example,episode,20+['prior_predictive','fixed_state','drifting'].index(regime)))
    n,T=cfg['calibration_n'],cfg['calibration_T'];complex_data=example=='beamforming'
    h=unit(gaussian(rng,n,complex_data)) if regime=='prior_predictive' else unit(np.array([1.,.5,-.3],dtype=complex if complex_data else float))
    angle=cfg['calibration_drift_angle'] if regime=='drifting' else 0.
    if stress=='fast_drift':angle=3*cfg['calibration_drift_angle']
    H=path(h,T,angle);P=np.array([projector(x,example) for x in H])
    sigma2=np.full(T,cfg['calibration_sigma2']);tau2=np.full(T,cfg['calibration_tau2'])
    Z=gaussian(rng,(T,n),complex_data)
    G=np.broadcast_to(np.eye(n),(T,n,n)).copy();V=None
    if example=='beamforming':
        a=unit(np.ones(n,dtype=complex))
        noise_cov=np.eye(n);extra_vector=None;extra_power=0.
        U=H*gaussian(rng,(T,1),True)*np.sqrt(tau2)[:,None]+np.sqrt(sigma2)[:,None]*Z
        if stress in ('desired_contamination','additional_interference'):
            extra_vector=a if stress=='desired_contamination' else unit(gaussian(rng,n,True))
            power=cfg['mismatch']['contamination_power' if stress=='desired_contamination' else 'secondary_power']
            U+=np.sqrt(power)*gaussian(rng,(T,1),True)*extra_vector
            if stress=='additional_interference':extra_power=power
        if stress=='correlated_noise':
            rho=cfg['mismatch']['noise_correlation']
            C=rho**abs(np.arange(n)[:,None]-np.arange(n)[None,:])
            U+=np.sqrt(sigma2)[:,None]*(Z@np.linalg.cholesky(C).T-Z)
            noise_cov=C
        clean=None
    else:
        V=gaussian(rng,(T,n));V/=np.linalg.norm(V,axis=1)[:,None]
        clean=[]
        known_scaled=stress in ('known_resistance_correct_noise','known_resistance_raw_isotropic')
        Dknown=np.diag(cfg['mismatch']['anisotropic_damping'])
        root=np.diag(np.sqrt(np.diag(Dknown)));invroot=np.linalg.inv(root)
        physical_V=V.copy()
        for j,(v,normal) in enumerate(zip(physical_V,H)):
            D=Dknown if stress=='wrong_resistance' or known_scaled else np.eye(n)
            force=v.copy()
            if stress=='friction':
                force+=friction_force(force,normal,cfg['mismatch']['friction_coefficient'])
            velocity=v if stress=='contact_loss' else solve_contact_velocity(force,normal,D)[0]
            clean.append(velocity)
        clean=np.asarray(clean)
        noise=np.sqrt(sigma2)[:,None]*Z
        if stress=='known_resistance_correct_noise':
            noise=noise@invroot.T
        if stress=='position_differences':
            # Calibrate marginal variance to sigma2 while retaining the
            # negative lag-one correlation of differenced position errors.
            errors=gaussian(rng,(T+1,n))
            noise=np.sqrt(sigma2/2)[:,None]*np.diff(errors,axis=0)
        U=clean+noise
        if known_scaled:
            U=U@root.T;clean=clean@root.T;V=physical_V@invroot.T
            H=np.array([unit(invroot@h) for h in H])
            P=np.array([projector(h,example) for h in H])
    result=dict(U=U,V=V,G=G,P=P,H=H,sigma2=sigma2,tau2=tau2,
                model_correct=stress in ('matched','known_resistance_correct_noise') and regime!='drifting',
                regime=regime,stress=stress,clean=clean,dt=.05)
    if example=='beamforming':
        result.update(a=a,Rtrue=np.array([sigma2[j]*noise_cov+tau2[j]*P[j]+
            (extra_power*np.outer(extra_vector,extra_vector.conj()) if extra_power else 0)
            for j in range(T)]))
    return result


def evaluate_prediction(A,u,v,Ptrue,htrue,example,sigma2,tau2,cfg,rng,query=None):
    """Independent mean/tail/second-moment checks at a saved prior parameter."""
    k=1 if example=='beamforming' else 2
    M,psi0,d=moments(A,k,example=='beamforming',cfg['moment_tolerance'],cfg['max_order'])
    nodes,weights,check,cweights,node_ok=posterior_nodes(A,example,cfg,rng)
    normal_A=A if example=='beamforming' else -A
    values,Q=np.linalg.eigh(normal_A);mode=Q[:,-1]
    distance=1-abs(nodes.conj()@mode)**2;distance2=1-abs(check.conj()@mode)**2
    loss=values[-1]-np.einsum('bi,ij,bj->b',nodes.conj(),normal_A,nodes).real
    loss2=values[-1]-np.einsum('bi,ij,bj->b',check.conj(),normal_A,check).real
    actual_distance=max(0.,1-abs(np.vdot(mode,htrue))**2)
    actual_loss=float(values[-1]-np.vdot(htrue,normal_A@htrue).real)
    tied=bool(np.ptp(values)<1e-12)
    if tied:loss,loss2,actual_loss=distance.copy(),distance2.copy(),actual_distance
    query=unit(gaussian(rng,3,example=='beamforming')) if query is None else np.asarray(query)
    responses=response_values(nodes,query,example)
    true_response=Ptrue@query
    if example=='beamforming':
        clean=abs(nodes.conj()@query)**2;clean2=abs(check.conj()@query)**2
        clean_truth=float(np.vdot(query,Ptrue@query).real)
        predictive_means=np.zeros(len(nodes));predictive_means2=np.zeros(len(check))
        predictive_variances=(sigma2+tau2*abs(nodes[:,0])**2)/2
        predictive_variances2=(sigma2+tau2*abs(check[:,0])**2)/2
        covariance=sigma2*np.eye(3)+tau2*M;surrogate=gaussian_logscore(u,np.zeros(3),covariance,True)
        clean_mean=float(np.vdot(query,M@query).real)
        second_error=0.
    else:
        clean=responses[:,0];clean2=response_values(check,query,example)[:,0]
        clean_truth=float(true_response[0])
        future=response_values(nodes,v,example);future2=response_values(check,v,example)
        predictive_means=future[:,0];predictive_means2=future2[:,0]
        predictive_variances=np.full(len(nodes),sigma2);predictive_variances2=np.full(len(check),sigma2)
        mean=M@v;res=future-mean;res2=future2-mean
        spread=np.einsum('b,bi,bj->ij',weights,res,res)
        spread2=np.einsum('b,bi,bj->ij',cweights,res2,res2)
        second_error=float(np.linalg.norm(spread-spread2,2))
        covariance=sigma2*np.eye(3)+spread;surrogate=gaussian_logscore(u,mean,covariance)
        clean_mean=float((M@query)[0])
        regions=real_region_thresholds(normal_A,cfg['nominal'],192)
        refs=real_region_thresholds(normal_A,cfg['nominal'],384)
        probs=np.ravel([[(1-p)/2,(1+p)/2] for p in cfg['nominal']])
        clean_q=_ResponseCDF(normal_A,128).quantiles(query[None],probs)[0,0].reshape(-1,2)
        clean_ref=_ResponseCDF(normal_A,256).quantiles(query[None],probs)[0,0].reshape(-1,2)
    B=increment(u,sigma2,tau2,v,example=='beamforming')
    _,psi1,pdiag=moments(A+B,k,example=='beamforming',cfg['moment_tolerance'],cfg['max_order'])
    mixture=float(constant(u,k,sigma2,tau2,v,example=='beamforming')+psi1-psi0)
    score_ok=bool(d['numerical_ok'] and pdiag['numerical_ok'])
    result=[]
    for j,nominal in enumerate(cfg['nominal']):
        probs=np.array([(1-nominal)/2,(1+nominal)/2])
        if example=='contact':
            radius=float(regions[0][j]);threshold=float(regions[1][j])
            mass_error=max(abs(refs[2](radius)-nominal),abs(refs[3](threshold)-nominal))
            cq,cq2=clean_q[j],clean_ref[j];clean_error=float(np.max(abs(cq-cq2)))
        else:
            radius=float(weighted_quantile(distance,weights,[nominal])[0])
            threshold=float(weighted_quantile(loss,weights,[nominal])[0])
            mass_error=max(abs(float(cweights@(distance2<=radius))-nominal),
                           abs(float(cweights@(loss2<=threshold))-nominal))
            cq=weighted_quantile(clean,weights,probs);cq2=weighted_quantile(clean2,cweights,probs)
            clean_error=float(np.max(abs(cq-cq2)))
        clean_cdf=float(np.max(abs(np.array([cweights@(clean2<=x) for x in cq])-probs)))
        nq=normal_mixture_quantile(predictive_means,predictive_variances,weights,probs)
        noisy_cdf=float(np.max(abs(np.array([cweights@ndtr((x-predictive_means2)/np.sqrt(predictive_variances2)) for x in nq])-probs)))
        base_ok=bool(d['numerical_ok'] and node_ok and score_ok)
        mass_tol=max(cfg['state_mass_tolerance'],5*np.sqrt(2*nominal*(1-nominal)/len(nodes)))
        clean_tol=cfg['quantile_resolution_tolerance'] if example=='contact' else float(np.max(5*np.sqrt(2*probs*(1-probs)/len(nodes)))+2/len(nodes))
        for kind,truth,lo,hi,ok in [
            ('state_ball',actual_distance,0.,radius,mass_error<=mass_tol),
            ('state_hpd',actual_loss,0.,threshold,mass_error<=mass_tol),
            ('clean_response',clean_truth,*cq,(clean_error if example=='contact' else clean_cdf)<=clean_tol),
            ('noisy_observation',float(np.real(u[0])),*nq,noisy_cdf<=cfg['quantile_resolution_tolerance'])]:
            result.append(dict(kind=kind,nominal=nominal,covered=bool(lo<=truth<=hi),width=float(hi-lo),
                lower=float(lo),upper=float(hi),truth=float(truth),numerical_ok=bool(base_ok and ok),
                mean_numerical_ok=bool(d['numerical_ok']),tail_numerical_ok=bool(ok and node_ok),
                mass_error=mass_error,quantile_error=clean_error,clean_cdf_error=clean_cdf,
                predictive_cdf_error=noisy_cdf,second_moment_error=second_error,
                second_moment_ok=bool(second_error<cfg['second_moment_tolerance']),mode_tied=tied))
            result[-1].update(accuracy_fields(d,cfg))
        sd=np.sqrt(covariance[0,0].real/(2 if example=='beamforming' else 1))
        mean=0. if example=='beamforming' else float((M@v)[0])
        gq=mean+sd*ndtri(probs)
        result.append(dict(kind='gaussian_noisy_observation',nominal=nominal,
            covered=bool(gq[0]<=u[0].real<=gq[1]),width=float(gq[1]-gq[0]),
            lower=float(gq[0]),upper=float(gq[1]),truth=float(u[0].real),
            numerical_ok=bool(base_ok and second_error<cfg['second_moment_tolerance']),
            mean_numerical_ok=bool(d['numerical_ok']),tail_numerical_ok=True,
            mass_error=mass_error,quantile_error=clean_error,clean_cdf_error=clean_cdf,
            predictive_cdf_error=noisy_cdf,second_moment_error=second_error,
            second_moment_ok=bool(second_error<cfg['second_moment_tolerance']),mode_tied=tied))
        result[-1].update(accuracy_fields(d,cfg))
    score=dict(mixture_log_score=mixture,gaussian_log_score=surrogate,
        risk_predicted=float(np.vdot(query,(M-M@M)@query).real),
        risk_realized=float(np.linalg.norm(M@query-true_response)**2),
        clean_mean=clean_mean,clean_truth=clean_truth,numerical_ok=bool(d['numerical_ok'] and node_ok and score_ok),
        second_moment_error=second_error,**accuracy_fields(d,cfg))
    Pmode,Miso,_=same_mode_controls(A,M,k)
    score.update(response_loss_mean=score['risk_realized'],
        response_loss_mode=float(np.linalg.norm(Pmode@query-true_response)**2),
        response_loss_iso=float(np.linalg.norm(Miso@query-true_response)**2))
    return result,score


def score_only_prediction(A,u,v,stream,t,example,sigma2,tau2,cfg,query):
    """Maintained-model scores and true mechanical/task loss under mismatch."""
    k=1 if example=='beamforming' else 2
    M,psi0,diag=moments(A,k,example=='beamforming',cfg['moment_tolerance'],cfg['max_order'])
    B=increment(u,sigma2,tau2,v,example=='beamforming')
    _,psi1,d1=moments(A+B,k,example=='beamforming',cfg['moment_tolerance'],cfg['max_order'])
    mixture=float(constant(u,k,sigma2,tau2,v,example=='beamforming')+psi1-psi0)
    P,Mi,_=same_mode_controls(A,M,k)
    stress=stream['stress']
    true_response=stream['P'][t]@query
    covariance_error=0.
    if example=='beamforming':
        surrogate=gaussian_logscore(u,np.zeros(3),sigma2*np.eye(3)+tau2*M,True)
        R=stream['Rtrue'][t];a=stream['a']
        z=np.linalg.solve(R,a);oracle=z/np.vdot(a,z)
        oracle_power=float(np.vdot(oracle,R@oracle).real)
        actions=beam_actions(M,P,Mi,a,sigma2,tau2)
        task={control:float(np.vdot(w,R@w).real) for control,w in actions.items()}
    else:
        if stress=='contact_loss':
            true_response=query
        elif stress in ('wrong_resistance','friction'):
            force=query.copy()
            if stress=='friction':force+=friction_force(force,stream['H'][t],cfg['mismatch']['friction_coefficient'])
            D=np.diag(cfg['mismatch']['anisotropic_damping']) if stress=='wrong_resistance' else np.eye(3)
            true_response=solve_contact_velocity(force,stream['H'][t],D)[0]
        nodes,weights,check,weights2,node_ok=posterior_nodes(A,example,cfg,None)
        responses=response_values(nodes,v,example);responses2=response_values(check,v,example)
        residual=responses-M@v;residual2=responses2-M@v
        spread=np.einsum('b,bi,bj->ij',weights,residual,residual)
        spread2=np.einsum('b,bi,bj->ij',weights2,residual2,residual2)
        covariance_error=float(np.linalg.norm(spread-spread2,2))
        surrogate=gaussian_logscore(u,M@v,sigma2*np.eye(3)+spread)
        task={control:float(np.linalg.norm(D@query-true_response)**2)
              for control,D in [('mean',M),('mode',P),('iso',Mi)]}
        oracle_power=0.
    risk=float(np.vdot(query,(M-M@M)@query).real)
    return dict(mixture_log_score=mixture,gaussian_log_score=surrogate,risk_predicted=risk,
        risk_realized=float(np.linalg.norm(M@query-true_response)**2),
        response_loss_mean=float(np.linalg.norm(M@query-true_response)**2),
        response_loss_mode=float(np.linalg.norm(P@query-true_response)**2),
        response_loss_iso=float(np.linalg.norm(Mi@query-true_response)**2),
        task_loss_mean=task['mean'],task_loss_mode=task['mode'],task_loss_iso=task['iso'],
        oracle_power=oracle_power,second_moment_error=covariance_error,
        numerical_ok=bool(diag['numerical_ok'] and d1['numerical_ok'] and covariance_error<cfg['second_moment_tolerance']),
        state_target_applicable=stress!='contact_loss',**accuracy_fields(diag,cfg))


def calibration_trial(example,regime,cfg,episode,split,policy,stress='matched',mismatch=False):
    stream=calibration_stream(example,regime,cfg,episode,split,stress,mismatch)
    methods=['exact_known','known_forgetting','misspecified','learned']
    trackers={}
    for name in methods:
        p=dict(policy)
        if name=='exact_known':p['beta']=1.
        trackers[name]=CausalVarianceTracker(**tracker_kwargs(example,3,p,cfg,learn=name=='learned',calibration=True))
    rows,scores,events=[],[],[]
    rng=np.random.default_rng(seed(cfg,'mismatch_seed' if stress!='matched' else 'calibration_seed',
                                  example,episode,700+['prior_predictive','fixed_state','drifting'].index(regime)))
    for t,u in enumerate(stream['U']):
        query_rng=np.random.default_rng(seed(cfg,'calibration_seed',example,episode,900+t))
        common_query=unit(gaussian(query_rng,3,example=='beamforming'))
        for name,tr in trackers.items():
            if name in ('exact_known','known_forgetting'):
                tr.sigma2=float(stream['sigma2'][t])
                if example=='beamforming':tr.tau2=float(stream['tau2'][t])
            v=None if example=='beamforming' else stream['V'][t]
            if t+1 in cfg['calibration_times']:
                try:
                    if mismatch:
                        report=[]
                        score=score_only_prediction(tr.predict_parameter(),u,v,stream,t,example,
                            tr.sigma2,tr.tau2,cfg,common_query)
                    else:
                        report,score=evaluate_prediction(tr.predict_parameter(),u,v,stream['P'][t],
                            stream['H'][t],example,tr.sigma2,tr.tau2,cfg,rng,common_query)
                    common=dict(example=example,regime=regime,stress=stress,episode=episode,split=split,
                        cohort='mismatch' if mismatch else 'calibration',
                        method=name,t=t+1,time_s=t*stream['dt'],beta=tr.beta,
                        sigma2_estimate=tr.sigma2,tau2_estimate=tr.tau2,
                        sigma2_true=float(stream['sigma2'][t]),model_correct=stream['model_correct'],
                        probability_scope='Prior-predictive identity only for exact_known beta=1 matched static likelihood; other recursion/physical drift reports are empirical inclusion diagnostics.')
                    rows.extend(dict(common,**row) for row in report)
                    scores.append(dict(common,**score))
                except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
                    rows.append(dict(example=example,regime=regime,stress=stress,episode=episode,
                        split=split,method=name,t=t+1,kind='unresolved',nominal=np.nan,
                        covered=False,width=np.nan,numerical_ok=False,
                        error=type(error).__name__+': '+str(error)))
            record=tr.update(u,v)
            if record.fit_performed:
                events.append(dict(example=example,regime=regime,stress=stress,episode=episode,
                    split=split,method=name,**asdict(record)))
    return rows,scores,events
