"""Passive array tracking, fixed-time certificates and matched process checks.

All estimators receive row-wise passive snapshots only. Spatial signatures,
angles, covariances and latent coefficients are used by generation or scoring,
never as a known-input observation supplied to the filter.
"""
from time import perf_counter
import numpy as np
import pandas as pd
from .beamforming_protocol import (profile as common_profile,proper_complex,haar_line,
    ula,physical_angles,DESIRED_ANGLE)
from .beamforming_theory import exponential_weights,passive_covariance_certificate
from .beamforming_numerics import (complex_moments,sample_complex_transition,
    predict_complex,posterior_report)
from .geometry import center
from .models import increment,constant
from .theory import regional_bound
from benchmarks.case1 import mvdr,output_power,sample_output_power


def configuration(name):
    cfg=common_profile(name);quick=name=='quick'
    offset=10000 if quick else 0
    cfg.update(validation_seed=6571021+offset,test_seed=6571022+offset,
        process_development_seed=6571031+offset,process_test_seed=6571032+offset,
        physical_T=64 if quick else 128,physical_reports=[8,32,64] if quick else [8,32,64,128],
        betas=[1.,.995,.98,.95,.9],long_n=4,long_T=4096,long_reports=[256,1024,4096],
        long_trials=16 if quick else 256,delta=.05,
        coverage_trials=16 if quick else 256,coverage_snapshots=[8,32],coverage_s=.2,
        coverage_states=['coordinate line','ULA 20 degrees'],
        posterior_draws=2048 if quick else 16384,
        physical_tail_trials=4 if quick else 32,
        physical_tail_times=[32,64] if quick else [64,128],
        process_n=[2,8],process_kappas=[4.,20.],process_T=16 if quick else 24,
        process_reports=[8,16] if quick else [8,16,24],process_s=.2,
        process_trials=16 if quick else 256,process_tail_trials=4 if quick else 32,
        reference_trials=4 if quick else 32,reference_development_trials=2,
        reference_coarse=dict(degree=24,z_order=48,phi_order=96),
        reference_fine=dict(degree=32,z_order=64,phi_order=128),
        scan_degrees=list(np.linspace(-60,60,61)),heldout_snapshots=32 if quick else 128,
        selection='Per-noise beta minimizes terminal physical-ULA modal population beamformer power on independent validation; same beta for mode, mean and isotropization.',
        certificate='Known fixed powers, uniform initialization, deterministic physical path and identity transport; fixed reporting times only. Drift uses scoring truth.',
        tails='Native iid posterior reports only on a predeclared first-trajectory subset; mean accuracy and tail Monte Carlo accuracy have separate flags.',
        process='Known kappa matched invariant complex-line transition; physical ULA is a different deterministic path. No true kappa is assigned to ULA motion.',
        reference='Full-density CP1 Bloch-sphere filter on the first predeclared trajectories. It is not an n8 array reference.',
        statistics='Independent trajectories; query averages within a trajectory, and pointwise intervals at declared times. No optional stopping or success-selected references.')
    return cfg


def signature_projectors(h):
    h=np.asarray(h,complex)
    return np.einsum('...i,...j->...ij',h,h.conj())


def physical_data(cfg,seed,trial,s):
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial,int(round(1000*s)),18021]))
    angles=physical_angles(cfg['physical_T'],moving=True)
    h=np.asarray([ula(cfg['n'],theta) for theta in angles])
    U=h*proper_complex(rng,len(h),cfg['q'])[:,None]+proper_complex(rng,h.shape,s)
    return dict(U=U,h=h,theta=angles,desired=ula(cfg['n'],DESIRED_ANGLE))


def stationary_data(seed,trial,n,N,s,q,h=None,tag=0):
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial,n,tag,12091]))
    h=haar_line(rng,n) if h is None else np.asarray(h,complex)
    U=h[None,:]*proper_complex(rng,N,q)[:,None]+proper_complex(rng,(N,n),s)
    return dict(U=U,h=np.broadcast_to(h,(N,n)),desired=ula(n,DESIRED_ANGLE))


def matched_process_data(cfg,seed,trial,n,kappa):
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial,n,int(kappa),29017]))
    h=haar_line(rng,n);states=[];U=[]
    for j in range(cfg['process_T']):
        if j:h=sample_complex_transition(h,kappa,rng,size=1)[0]
        states.append(h.copy())
        U.append(h*proper_complex(rng,1,cfg['q'])[0]+proper_complex(rng,n,cfg['process_s']))
    return dict(U=np.asarray(U),h=np.asarray(states),desired=ula(n,DESIRED_ANGLE))


def covariance(h,s,q):return s*np.eye(len(h))+q*np.outer(h,h.conj())


def scatter(U,beta=1.):
    alpha,total=exponential_weights(len(U),beta)
    R=(np.asarray(U).T*alpha)@np.asarray(U).conj()
    return (R+R.conj().T)/2,total


def modal_projector(A):
    values,E=np.linalg.eigh(A);h=E[:,-1]
    return np.outer(h,h.conj()),h,float(values[-1]-values[-2])


def matched_isotropization(M,P):
    n=len(M);rho=float((np.trace(P@M).real-1/n)/(1-1/n))
    return rho*P+(1-rho)*np.eye(n)/n,rho


def validation_task(task):
    cfg,s,trial=task;data=physical_data(cfg,cfg['validation_seed'],trial,s);rows=[]
    truth=covariance(data['h'][-1],s,cfg['q'])
    for beta in cfg['betas']:
        R,total=scatter(data['U'],beta);P,_,_=modal_projector(R)
        w=mvdr(s*np.eye(cfg['n'])+cfg['q']*P,data['desired'])
        rows.append(dict(s=s,trial=trial,beta=beta,population_power=output_power(w,truth),weight_sum=total))
    return rows


def choose_memory(rows,cfg):
    frame=pd.DataFrame(rows);selected=[];curves=[]
    for s,g in frame.groupby('s',sort=False):
        options=[]
        for beta,h in g.groupby('beta',sort=False):
            complete=len(h)==cfg['validation_trials'] and np.isfinite(h.population_power).all()
            score=float(h.population_power.mean()) if complete else np.inf
            curves.append(dict(s=s,beta=beta,validation_loss=score,independent_trials=len(h),selected=False,
                finite_boundary=beta==min(cfg['betas'])))
            options.append((score,len(curves)-1))
        _,i=min(options);curves[i]['selected']=True
        selected.append(dict(s=s,beta=float(curves[i]['beta']),validation_loss=curves[i]['validation_loss']))
    return selected,pd.DataFrame(curves)


def fixed_time_record(U,h,s,q,beta,delta):
    begin=perf_counter();Rhat,total=scatter(U,beta);P,mode,gap=modal_projector(Rhat)
    cert=passive_covariance_certificate(h,s,q,beta=beta,delta=delta)
    Ptrue=np.outer(h[-1],h[-1].conj());actual=float(np.sqrt(max(0,1-np.trace(P@Ptrue).real)))
    truth=s*np.eye(len(mode))+q*Ptrue;expected=s*np.eye(len(mode))+q*cert.mean_projector
    noise=float(np.linalg.norm(Rhat-expected,2));error=float(np.linalg.norm(Rhat-truth,2))
    angles=np.deg2rad(np.linspace(-60,60,61));queries=np.asarray([ula(len(mode),a) for a in angles])
    scan_error=float(q*np.max(abs(np.einsum('bi,ij,bj->b',queries.conj(),P-Ptrue,queries).real)))
    return dict(eta=cert.eta,noise=cert.eta,drift=cert.drift,epsilon=cert.epsilon,
        valid=cert.valid,bound=cert.bound,actual_error=actual,neff=cert.neff,weight_sum=cert.weight_sum,
        sample_cov_error=error,realized_centered_noise=noise,
        noise_event=bool(noise<=cert.eta),covariance_event=bool(error<=cert.epsilon),
        certificate_violated=bool(cert.valid and actual>cert.bound),
        scan_error=scan_error,scan_lipschitz_limit=q*actual,modal_eigengap=gap,
        evaluation_seconds=perf_counter()-begin),P,mode


def long_certificate_task(task):
    cfg,trial=task;n=cfg['long_n'];s=.2;q=cfg['q']
    data=stationary_data(cfg['test_seed'],trial,n,cfg['long_T'],s,q,tag=4103);rows=[]
    for t in cfg['long_reports']:
        row,_,_=fixed_time_record(data['U'][:t],data['h'][:t],s,q,1.,cfg['delta'])
        rows.append(dict(condition='Long stationary control',trial=trial,t=t,n=n,s=s,q=q,beta=1.,**row))
    return rows


def posterior_metrics(A,h,queries,s,q,draws,seed):
    report=posterior_report(A,queries=queries,truth=h,draws=draws,seed=seed)
    d=report['numerics'];P=np.outer(h,h.conj());truth=np.einsum('bi,ij,bj->b',queries.conj(),P,queries).real
    qs=np.asarray(report['quadratic_quantiles']);lo=qs[:,0];hi=qs[:,-1];width=hi-lo
    score=width+40*(np.maximum(lo-truth,0)+np.maximum(truth-hi,0))
    M=report['mean'];response_error=np.sum(abs((M-P)@queries.T)**2,axis=0)
    risks=np.einsum('bi,ij,bj->b',queries.conj(),M-M@M,queries).real
    metrics=dict(ball_covered=bool(report['ball_covered']),hpd_covered=bool(report['hpd_covered']),
        ball_radius=float(report['ball_radius']),hpd_threshold=float(report['hpd_threshold']),
        clean_coverage=float(np.mean((truth>=lo)&(truth<=hi))),clean_width=float(q*np.mean(width)),
        interval_score=float(q*np.mean(score)),projection_risk=float(risks.mean()),
        projection_mse=float(response_error.mean()),numerical_ok=bool(d['numerical_ok']),
        mean_ok=bool(d.get('mean_ok',d['numerical_ok'])),state_ok=bool(d.get('state_ok',d['numerical_ok'])),
        tail_ok=bool(d.get('tail_ok',d['numerical_ok'])),report_seconds=float(d.get('total_seconds',np.nan)),
        state_mass_error=float(d.get('state_mass_error',np.nan)),scalar_cdf_error=float(np.max(d.get('scalar_cdf_error',[np.nan]))),
        quantile_mcse=float(np.nanmax(report.get('quantile_mcse',np.nan))))
    return report,metrics


def coverage_task(task):
    cfg,state,trial=task;n=cfg['n'];s=cfg['coverage_s'];q=cfg['q']
    h=np.eye(n,dtype=complex)[0] if state=='coordinate line' else ula(n,np.deg2rad(20.))
    tag=5401 if state=='coordinate line' else 5402
    data=stationary_data(cfg['test_seed'],trial,n,max(cfg['coverage_snapshots']),s,q,h=h,tag=tag)
    rng=np.random.default_rng(6571097);queries=np.asarray([haar_line(rng,n) for _ in range(4)])
    rows=[]
    for t in cfg['coverage_snapshots']:
        A=center(sum((increment(u,s,q,complex_data=True) for u in data['U'][:t]),np.zeros((n,n),complex)))
        _,metrics=posterior_metrics(A,h,queries,s,q,cfg['posterior_draws'],6571098+trial*100+tag+t)
        rows.append(dict(condition='Stationary fixed-state coverage',state=state,trial=trial,t=t,n=n,s=s,q=q,**metrics))
    return rows


def physical_task(task):
    cfg,s,trial,selected=task;data=physical_data(cfg,cfg['test_seed'],trial,s)
    beta=float(next(r['beta'] for r in selected if r['s']==s));policies={'All history':1.,'Selected memory':beta}
    metrics=[];bounds=[];tails=[];scan=[];regional=[];arrays={}
    Q=np.asarray([ula(cfg['n'],np.deg2rad(a)) for a in cfg['scan_degrees']])
    rng=np.random.default_rng(6571091);queries=np.asarray([haar_line(rng,cfg['n']) for _ in range(4)])
    for policy,b in policies.items():
        Sall=np.zeros((cfg['n'],cfg['n']),complex);modes=[];Aall=[]
        update_time=0.
        for j,u in enumerate(data['U']):
            start=perf_counter();Sall=b*Sall+np.outer(u,u.conj());A=center(cfg['q']/(s*(s+cfg['q']))*Sall)
            P,mode,_=modal_projector(A);update_time+=perf_counter()-start
            if trial==0:modes.append(P);Aall.append(A.copy())
            t=j+1
            if t not in cfg['physical_reports']:continue
            budget,_,_=fixed_time_record(data['U'][:t],data['h'][:t],s,cfg['q'],b,cfg['delta'])
            bounds.append(dict(condition='Moving ULA',trial=trial,t=t,n=cfg['n'],s=s,q=cfg['q'],beta=b,policy=policy,**budget))
            begin=perf_counter();evaluation=complex_moments(A);M=evaluation.mean;mean_seconds=perf_counter()-begin
            iso,rho=matched_isotropization(M,P);truth=covariance(data['h'][j],s,cfg['q'])
            held_rng=np.random.default_rng(np.random.SeedSequence([cfg['test_seed'],trial,int(s*1000),t,92017]))
            held=data['h'][j]*proper_complex(held_rng,cfg['heldout_snapshots'],cfg['q'])[:,None]+proper_complex(held_rng,(cfg['heldout_snapshots'],cfg['n']),s)
            for method,D in [('Posterior mean',M),('Mode',P),('Matched isotropization',iso),('Oracle',signature_projectors(data['h'][j]))]:
                C=s*np.eye(cfg['n'])+cfg['q']*D;w=mvdr(C,data['desired'])
                risks=np.einsum('bi,ij,bj->b',queries.conj(),M-M@M,queries).real
                risks+=np.sum(abs((M-D)@queries.T)**2,axis=0)
                error=np.sum(abs((D-signature_projectors(data['h'][j]))@queries.T)**2,axis=0)
                metrics.append(dict(condition='Moving ULA',trial=trial,t=t,n=cfg['n'],s=s,q=cfg['q'],beta=b,policy=policy,
                    method=method,population_power=output_power(w,truth),heldout_power=sample_output_power(w,held),
                    unit_gain_error=float(abs(np.vdot(w,data['desired'])-1)),actual_error=budget['actual_error'],
                    principal_angle=float(np.arcsin(np.clip(budget['actual_error'],0,1))),
                    projection_risk=float(risks.mean()),projection_mse=float(error.mean()),mean_ok=evaluation.numerical_ok,
                    mean_seconds=mean_seconds,update_seconds=update_time,mode_isotropization_rho=rho,
                    anisotropy=float(np.linalg.norm(M-iso,'fro'))))
            if trial<cfg['physical_tail_trials'] and t in cfg['physical_tail_times']:
                report,tail=posterior_metrics(A,data['h'][j],Q,s,cfg['q'],cfg['posterior_draws'],6571057+trial*10000+int(s*1000)+t)
                tails.append(dict(condition='Moving ULA',trial=trial,t=t,s=s,beta=b,policy=policy,**tail))
                if trial==0 and policy=='Selected memory':
                    qs=report['quadratic_quantiles'];ptrue=np.einsum('bi,ij,bj->b',Q.conj(),signature_projectors(data['h'][j]),Q).real
                    for k,angle in enumerate(cfg['scan_degrees']):
                        C=np.outer(Q[k],Q[k].conj());threshold=report['hpd_constraint_threshold'];constraint=report['hpd_constraint_matrix']
                        upper=regional_bound(C,constraint,threshold,1);lower=regional_bound(-C,constraint,threshold,1)
                        regional.append(dict(s=s,t=t,angle=angle,lower=s-cfg['q']*lower.upper,upper=s+cfg['q']*upper.upper,
                            dual_gap=max(lower.gap,upper.gap),feasible=lower.feasible and upper.feasible,
                            hpd_covered=tail['hpd_covered'],truth=s+cfg['q']*ptrue[k],posterior_ok=tail['numerical_ok']))
                        scan.append(dict(s=s,t=t,angle=angle,truth=s+cfg['q']*ptrue[k],mean=s+cfg['q']*np.vdot(Q[k],M@Q[k]).real,
                            mode=s+cfg['q']*np.vdot(Q[k],P@Q[k]).real,q025=s+cfg['q']*qs[k,0],q25=s+cfg['q']*qs[k,1],
                            q50=s+cfg['q']*qs[k,2],q75=s+cfg['q']*qs[k,3],q975=s+cfg['q']*qs[k,4],
                            sampling_ok=tail['numerical_ok']))
        if trial==0:
            key=policy.lower().replace(' ','_');arrays[key+'_mode_projectors']=np.asarray(modes);arrays[key+'_A']=np.asarray(Aall)
    if trial==0:arrays.update(U=data['U'],h=data['h'],theta_rad=data['theta'],times=np.arange(len(data['U'])),s=np.array(s))
    return dict(metrics=metrics,bounds=bounds,tails=tails,scan=scan,regional=regional,arrays=arrays)


def process_task(task):
    cfg,n,kappa,trial=task;s=cfg['process_s'];q=cfg['q']
    data=matched_process_data(cfg,cfg['process_test_seed'],trial,n,kappa)
    from .beamforming_reference import ComplexLineReference,compare_reference
    use_reference=n==2 and trial<cfg['reference_trials']
    coarse=ComplexLineReference(kappa,**cfg['reference_coarse']) if use_reference else None
    fine=ComplexLineReference(kappa,**cfg['reference_fine']) if use_reference else None
    A=np.zeros((n,n),complex);D=A.copy();I=A.copy();rows=[];audits=[];shapes=[];scan=[];scores=[];arrays={}
    queries=np.asarray([ula(n,np.deg2rad(a)) for a in [-45.,-20.,0.,20.,45.]])
    prediction_seconds=update_seconds=0.;maximum_residual=0.;maximum_uncertainty=0.;failure=''
    for j,u in enumerate(data['U']):
        t=j+1;B=increment(u,s,q,complex_data=True)
        before=perf_counter()
        if not failure:
            try:
                prediction=predict_complex(A,kappa);prior=prediction.prior
                prediction_seconds+=perf_counter()-before
                maximum_residual=max(maximum_residual,prediction.fit.residual)
                maximum_uncertainty=max(maximum_uncertainty,prediction.fit.moment_uncertainty)
                psi0=complex_moments(prior).log_normalizer;psi1=complex_moments(prior+B).log_normalizer
                logscore=float(constant(u,1,s,q,complex_data=True)+psi1-psi0)
                start=perf_counter();A=center(prior+B);update_seconds+=perf_counter()-start
            except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:failure=str(error);logscore=np.nan
        else:logscore=np.nan
        scores.append(dict(n=n,kappa=kappa,trial=trial,t=t,method='Process Bingham',log_score=logscore,failure_reason=failure))
        for method,prior in [('Power forgetting',.95*D),('Identity prediction',I)]:
            value=float(constant(u,1,s,q,complex_data=True)+complex_moments(prior+B).log_normalizer-complex_moments(prior).log_normalizer)
            scores.append(dict(n=n,kappa=kappa,trial=trial,t=t,method=method,log_score=value,failure_reason=''))
        D=center(.95*D+B);I=center(I+B)
        if use_reference:
            coarse.update(u,s,q);fine.update(u,s,q)
            scores.append(dict(n=n,kappa=kappa,trial=trial,t=t,method='Full-density CP1 reference',
                log_score=fine.last_log_score,failure_reason=''))
        if t not in cfg['process_reports']:continue
        Ptrue=signature_projectors(data['h'][j]);truth=covariance(data['h'][j],s,q)
        for method,parameter in [('Process Bingham',A),('Power forgetting',D),('Identity prediction',I)]:
            base=dict(n=n,kappa=kappa,trial=trial,t=t,method=method,
                prediction_seconds=prediction_seconds if method=='Process Bingham' else 0.,
                update_seconds=update_seconds if method=='Process Bingham' else np.nan,
                inverse_residual=maximum_residual if method=='Process Bingham' else np.nan,
                moment_uncertainty=maximum_uncertainty if method=='Process Bingham' else np.nan,
                failure_reason=failure if method=='Process Bingham' else '')
            if method=='Process Bingham' and failure:
                rows.append(dict(base,numerical_ok=False,mean_ok=False,state_ok=False,tail_ok=False));continue
            begin=perf_counter();moment=complex_moments(parameter);M=moment.mean;P,_,_=modal_projector(parameter)
            w=mvdr(s*np.eye(n)+q*M,data['desired']);queryerror=np.sum(abs((M-Ptrue)@queries.T)**2,axis=0)
            row=dict(base,mean_ok=moment.numerical_ok,numerical_ok=moment.numerical_ok,
                state_ok=np.nan,tail_ok=np.nan,tail_evaluated=False,
                mean_seconds=perf_counter()-begin,population_power=output_power(w,truth),
                projector_mean_mse=float(np.linalg.norm(M-Ptrue,'fro')**2),
                actual_error=float(np.sqrt(max(0,1-np.trace(P@Ptrue).real))),
                projection_mse=float(queryerror.mean()),projection_risk=float(np.mean(np.einsum('bi,ij,bj->b',queries.conj(),M-M@M,queries).real)))
            if trial<cfg['process_tail_trials']:
                report,tail=posterior_metrics(parameter,data['h'][j],queries,s,q,cfg['posterior_draws'],6571071+trial*10000+n*100+int(kappa)+t)
                row.update(tail,tail_evaluated=True)
                if trial==0 and method=='Process Bingham':
                    for k in range(len(queries)):
                        qs=report['quadratic_quantiles'][k]
                        scan.append(dict(n=n,kappa=kappa,t=t,query=k,truth=s+q*np.vdot(queries[k],Ptrue@queries[k]).real,
                            mean=s+q*np.vdot(queries[k],M@queries[k]).real,q025=s+q*qs[0],q25=s+q*qs[1],q50=s+q*qs[2],q75=s+q*qs[3],q975=s+q*qs[4]))
            rows.append(row)
        if use_reference:
            reference,audit=compare_reference(coarse,fine,queries,data['h'][j]);audits.append(dict(n=n,kappa=kappa,trial=trial,t=t,**audit))
            M=reference['mean'];w=mvdr(s*np.eye(n)+q*M,data['desired']);quantiles=reference['quadratic_quantiles'];trueq=np.einsum('bi,ij,bj->b',queries.conj(),Ptrue,queries).real
            lo=quantiles[:,0];hi=quantiles[:,-1]
            rows.append(dict(n=n,kappa=kappa,trial=trial,t=t,method='Full-density CP1 reference',
                population_power=output_power(w,truth),projector_mean_mse=float(np.linalg.norm(M-Ptrue,'fro')**2),
                projection_mse=float(np.mean(np.sum(abs((M-Ptrue)@queries.T)**2,axis=0))),
                clean_coverage=float(np.mean((trueq>=lo)&(trueq<=hi))),clean_width=float(q*np.mean(hi-lo)),
                interval_score=float(q*np.mean(hi-lo+40*(np.maximum(lo-trueq,0)+np.maximum(trueq-hi,0)))),
                ball_covered=reference['ball_covered'],hpd_covered=reference['hpd_covered'],
                numerical_ok=audit['reference_numerical_ok'],mean_ok=audit['mean_error']<2e-4,
                state_ok=audit['region_mass_error']<.005,tail_ok=audit['scalar_cdf_error']<.005,
                tail_evaluated=True,prediction_seconds=fine.prediction_seconds,update_seconds=fine.update_seconds,
                ball_radius=reference['ball_radius'],hpd_threshold=reference['hpd_threshold']))
            if not failure:
                rho=fine.density();Pnodes=__import__('lib.beamforming_reference',fromlist=['bloch_projectors']).bloch_projectors(fine.nodes)
                psi=complex_moments(A).log_normalizer
                projected=np.exp(np.einsum('ij,bji->b',A,Pnodes).real-psi)
                shapes.append(dict(n=n,kappa=kappa,trial=trial,t=t,
                    density_tv=float(.5*fine.weights@abs(projected-rho)),mean_error=float(np.linalg.norm(complex_moments(A).mean-M,2)),
                    reference_numerical_ok=audit['reference_numerical_ok'],inverse_residual=maximum_residual))
    if trial==0:arrays.update(U=data['U'],h=data['h'])
    return dict(metrics=rows,reference_checks=audits,shape_checks=shapes,scan=scan,scores=scores,arrays=arrays)
