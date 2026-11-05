"""Bounded ordered-power and concentration inference for passive array data."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from dataclasses import asdict
import hashlib,json,platform,sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp
from lib.beamforming_protocol import profile,output_directory,proper_complex,haar_line,ula,physical_angles
from lib.beamforming_numerics import complex_moments,predict_complex,sample_complex_transition
from lib.beamforming_inference import (PowerPrior,sensitivity_prior,stationary_gibbs,full_path_gibbs,
    joint_predictive_score,joint_forecast_covariance,recorded_prior_em,stationary_reference_n2)
from lib.models import increment,constant
from lib.geometry import center
from benchmarks.case1 import mvdr,output_power,sample_output_power


SOURCES=['lib/beamforming_inference.py','scripts/run_beamforming_inference.py',
    'lib/beamforming_numerics.py','lib/beamforming_protocol.py','lib/models.py','lib/em.py',
    'lib/geometry.py','lib/moments.py','lib/spectral_series.py','lib/process_prediction.py',
    'benchmarks/case1/baselines.py','tests/test_beamforming_inference.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def digest(value):return hashlib.sha256(json.dumps(portable(value),sort_keys=True).encode()).hexdigest()
def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


def configuration(name):
    cfg=profile(name)
    cfg.update(component='inference',schema=1,seed_data=6571041,seed_validation=6571040,seed_chains=6571042,
        Ns=[8,32,64],stationary_trials=2 if name=='quick' else 64,
        moving_trials=1 if name=='quick' else 8,moving_T=24,
        chains=2 if name=='quick' else 3,draws=128 if name=='quick' else 512,
        burnin=64 if name=='quick' else 256,moving_draws=64 if name=='quick' else 512,
        moving_burnin=32 if name=='quick' else 256,heldout_snapshots=128,
        known_full_path_trials=2,
        kappa_matched=20.,kappa_physical_control=50.,
        fixed_s=[.1,.2,.5,1.,2.],fixed_q=[.5,1.,2.],validation_N=32,
        priors=[asdict(PowerPrior()),asdict(sensitivity_prior())],
        primary='Parameter-inference pilot; primary decision study has its own 256-trajectory design.',
        replication='64 stationary data sets per noise and 8 moving data sets per law/noise in paper profile; queries/draws are not repetitions.',
        stationary_model='One common complex-line projector, constant ordered a=s+q>b=s. Uniform state prior.',
        moving_model='Full path of 24 states, uniform state 1, 23 transitions, constant powers and concentration; G=I.',
        moving_controls='Known/fixed powers use single-Bingham assumed-density prediction; they are not exact full-path posterior references.',
        known_full_path_control='First two matched-motion data sets per noise have a separate known-power/known-kappa full-path Gibbs control, shared across priors; no outcome selection.',
        physical_scope='Deterministic ULA motion is a process mismatch; no true kappa is assigned.',
        prior_scope='Joint product IG(a)IG(b) restricted to a>b, with one global normalizer. Not independent s,q priors.',
        prospective_scope='Each fit uses observations through N only; the target is observation N+1 before assimilation.',
        decision_scope='Weights use the joint next covariance E[bI+(a-b)P_next], not a product of marginal means.',
        EM_scope='Every pre-correction prior is recorded once and frozen for EM; final correction and future prediction use fitted powers without refitting history.',
        mixing_scope='Three independent starts; ESS, split Rhat and between-chain tails are finite diagnostics, not convergence proofs.',
        display='Preassigned trial 0, s=.2, default prior; stationary N32 plus both moving laws. No outcome-based target selection.',
        excluded_raw='Full Gibbs state arrays are regenerated from seeded scripts; compact preassigned scalar draws and all per-fit diagnostics are committed.')
    return cfg


def dataset(config,law,s,trial,N=None,validation=False):
    n=config['n'];q=config['q'];N=(max(config['Ns']) if law=='stationary' else config['moving_T']) if N is None else N
    law_id={'stationary':0,'matched_motion':1,'ula_motion':2}[law]
    seed=config['seed_validation'] if validation else config['seed_data']
    rng=np.random.default_rng(np.random.SeedSequence([seed,law_id,int(round(s*1000)),trial,717]))
    if law=='stationary':
        h=np.broadcast_to(haar_line(rng,n),(N+1,n)).copy()
    elif law=='matched_motion':
        states=[haar_line(rng,n)]
        for _ in range(N):states.append(sample_complex_transition(states[-1],config['kappa_matched'],rng=rng,size=1)[0])
        h=np.asarray(states)
    else:h=np.array([ula(n,theta) for theta in physical_angles(N+1,moving=True)])
    U=proper_complex(rng,N+1,q)[:,None]*h+proper_complex(rng,(N+1,n),s)
    hold_rng=np.random.default_rng(np.random.SeedSequence([seed,law_id,int(round(s*1000)),trial,9917]))
    return dict(U=U,h=h,holdout_rng=hold_rng)


def fixed_stationary(U,s,q):
    n=U.shape[1];A=center(q/(s*(s+q))*(U.T@U.conj()));moment=complex_moments(A)
    return dict(A=A,mean=moment.mean,covariance=s*np.eye(n)+q*moment.mean,
        numerical_ok=moment.numerical_ok,s=s,q=q,a=s+q,b=s)


def fixed_moving(U,s,q,kappa):
    n=U.shape[1];A=np.zeros((n,n),complex);failures=0;residual=0.
    for j,u in enumerate(U):
        if j:
            prediction=predict_complex(A,kappa);A=prediction.prior
            residual=max(residual,prediction.fit.residual)
            failures+=int(not prediction.fit.converged)
        A=center(A+increment(u,s,tau2=q,complex_data=True))
    prediction=predict_complex(A,kappa);moment=complex_moments(prediction.prior)
    return dict(A=prediction.prior,mean=prediction.target_mean,
        covariance=s*np.eye(n)+q*prediction.target_mean,numerical_ok=bool(moment.numerical_ok and not failures),
        s=s,q=q,a=s+q,b=s,kappa=kappa,maximum_inverse_residual=residual)


def fixed_score(fit,u):
    return float(constant(u,1,fit['s'],tau2=fit['q'],complex_data=True)
        +complex_moments(fit['A']+increment(u,fit['s'],tau2=fit['q'],complex_data=True)).log_normalizer
        -complex_moments(fit['A']).log_normalizer)


def validation_job(task):
    config,s,trial=task;N=config['validation_N'];d=dataset(config,'stationary',s,trial,N=N,validation=True)
    rows=[]
    for estimate_s in config['fixed_s']:
        for estimate_q in config['fixed_q']:
            begin=perf_counter()
            try:
                fit=fixed_stationary(d['U'][:N],estimate_s,estimate_q)
                score=fixed_score(fit,d['U'][N]);ok=fit['numerical_ok'] and np.isfinite(score)
                rows.append(dict(s_true=s,trial=trial,s=estimate_s,q=estimate_q,
                    negative_log_score=-score,numerical_ok=bool(ok),seconds=perf_counter()-begin))
            except ArithmeticError as error:
                rows.append(dict(s_true=s,trial=trial,s=estimate_s,q=estimate_q,
                    negative_log_score=None,numerical_ok=False,reason=str(error),seconds=perf_counter()-begin))
    return rows


def select_fixed(rows,config):
    f=pd.DataFrame(rows);options=[];expected=2*config['validation_trials']
    for (s,q),g in f.groupby(['s','q']):
        valid=len(g)==expected and g.numerical_ok.all() and np.isfinite(g.negative_log_score).all()
        value=float(g.groupby('trial').negative_log_score.mean().mean()) if valid else np.inf
        options.append(dict(s=s,q=q,negative_log_score=value,valid=bool(valid),evaluations=len(g)))
    winner=min(options,key=lambda x:x['negative_log_score'])
    if not winner['valid']:raise ArithmeticError('No resolved fixed-power validation candidate')
    return winner,options


def _decision_metrics(Rbar,Rtrue,a,holdout):
    w=mvdr(Rbar,a);oracle=mvdr(Rtrue,a)
    difference=oracle-w
    excess=float(np.vdot(oracle,Rbar@oracle).real-np.vdot(w,Rbar@w).real)
    quadratic=float(np.vdot(difference,Rbar@difference).real)
    return dict(actual_output_power=output_power(w,Rtrue),heldout_output_power=sample_output_power(w,holdout),
        oracle_output_power=output_power(oracle,Rtrue),posterior_expected_power=output_power(w,Rbar),
        unit_gain_error=float(abs(np.vdot(w,a)-1)),posterior_excess_identity_error=abs(excess-quadratic),
        covariance_error=float(np.linalg.norm(Rbar-Rtrue,'fro')))


def _joint_rows(result,config,law,s,N,trial,prior,Rtrue,desired,holdout,target):
    start=perf_counter();Rbar=joint_forecast_covariance(result)
    prediction=joint_predictive_score(result,target,seed=config['seed_chains']+trial+N)
    row=dict(law=law,s_true=s,q_true=config['q'],a_true=s+config['q'],b_true=s,
        kappa_true=config['kappa_matched'] if law=='matched_motion' else np.nan,
        N=N,trial=trial,method='Joint inference',prior=prior.label,
        numerical_ok=bool(result.diagnostics['completed']),mixing_ok=result.diagnostics['mixing_ok'],
        min_chain_ess=result.diagnostics['min_chain_ess'],max_rhat=result.diagnostics['max_rhat'],
        max_between_chain_tail_fraction=result.diagnostics['max_between_chain_tail_fraction'],
        fit_seconds=result.diagnostics['seconds'],report_seconds=perf_counter()-start,
        prospective_log_score=prediction['log_score'],score_chain_mcse=prediction['chain_log_score_mcse'],
        trained_through=N,scored_observation=N+1,conditioning='Full prefix, no future assimilation',
        status='Approximate Gibbs' if result.diagnostics['mixing_ok'] else 'Unresolved mixing',
        **_decision_metrics(Rbar,Rtrue,desired,holdout))
    for name,values,truth in [('a',result.a,s+config['q']),('b',result.b,s),
        ('q',result.a-result.b,config['q']),('kappa',result.kappa,config['kappa_matched'] if law=='matched_motion' else np.nan)]:
        if np.isfinite(values).all():
            low,median,high=np.quantile(values,[.025,.5,.975])
            row.update({name+'_mean':float(values.mean()),name+'_lower':float(low),
                name+'_median':float(median),name+'_upper':float(high),
                name+'_covered':bool(low<=truth<=high) if np.isfinite(truth) else np.nan})
    terminal=result.states[:,:,-1,:] if result.states.ndim==4 else result.states
    # The product control is only a diagnostic; it never substitutes for Rbar.
    if result.states.ndim==3:M=result.projector_mean
    else:
        from lib.beamforming_numerics import transition_coefficients_complex
        h=terminal.reshape(-1,config['n']);rho=np.array([transition_coefficients_complex(config['n'],k).rho for k in result.kappa.ravel()])
        M=np.mean(1-rho)/config['n']*np.eye(config['n'])+np.einsum('d,di,dj->ij',rho,h,h.conj())/len(rho)
    product=result.b.mean()*np.eye(config['n'])+(result.a-result.b).mean()*M
    row['joint_product_covariance_difference']=float(np.linalg.norm(Rbar-product,'fro'))
    return row


def experiment_job(task):
    config,law,s,trial,fixed=task;Nmax=max(config['Ns']) if law=='stationary' else config['moving_T']
    d=dataset(config,law,s,trial,N=Nmax);Ns=config['Ns'] if law=='stationary' else [Nmax]
    metrics=[];diagnostics=[];draw_rows=[];em_rows=[];n=config['n'];desired=ula(n,config['desired_angle'])
    for N in Ns:
        U=d['U'][:N];target=d['U'][N];truth=d['h'][N];Rtrue=s*np.eye(n)+config['q']*np.outer(truth,truth.conj())
        rng=np.random.default_rng(np.random.SeedSequence([config['seed_data'],trial,N,int(s*1000),9127]))
        holdout=proper_complex(rng,config['heldout_snapshots'],config['q'])[:,None]*truth+proper_complex(rng,(config['heldout_snapshots'],n),s)
        controls=[('Known powers',s,config['q']),('Validation-fixed powers',fixed['s'],fixed['q'])]
        kappa=config['kappa_matched'] if law=='matched_motion' else config['kappa_physical_control']
        for method,ss,qq in controls:
            begin=perf_counter()
            try:
                fit=fixed_stationary(U,ss,qq) if law=='stationary' else fixed_moving(U,ss,qq,kappa)
                metrics.append(dict(law=law,s_true=s,q_true=config['q'],a_true=s+config['q'],b_true=s,
                    kappa_true=config['kappa_matched'] if law=='matched_motion' else np.nan,N=N,trial=trial,
                    method=method,prior='Common control',a_mean=ss+qq,b_mean=ss,q_mean=qq,
                    kappa_mean=kappa if law!='stationary' else np.nan,
                    numerical_ok=fit['numerical_ok'],mixing_ok=np.nan,fit_seconds=perf_counter()-begin,
                    prospective_log_score=fixed_score(fit,target),trained_through=N,scored_observation=N+1,
                    status='Verified fixed-parameter integral' if law=='stationary' else 'Approximate assumed-density prediction',
                    **_decision_metrics(fit['covariance'],Rtrue,desired,holdout)))
            except (ArithmeticError,np.linalg.LinAlgError) as error:
                metrics.append(dict(law=law,s_true=s,N=N,trial=trial,method=method,prior='Common control',numerical_ok=False,status=str(error)))
        begin=perf_counter()
        try:
            em=recorded_prior_em(U,s0=.5,q0=1.,iterations=5,kappa=None if law=='stationary' else kappa)
            fit=dict(em)
            if law!='stationary':
                prediction=predict_complex(em['A'],kappa);fit['A']=prediction.prior
                fit['covariance']=em['s']*np.eye(n)+em['q']*prediction.target_mean
            metrics.append(dict(law=law,s_true=s,q_true=config['q'],a_true=s+config['q'],b_true=s,
                kappa_true=config['kappa_matched'] if law=='matched_motion' else np.nan,N=N,trial=trial,
                method='Recorded-prior EM',prior='Common control',a_mean=em['a'],b_mean=em['b'],q_mean=em['q'],
                kappa_mean=kappa if law!='stationary' else np.nan,numerical_ok=em['ascent_verified'],mixing_ok=np.nan,
                fit_seconds=perf_counter()-begin,prospective_log_score=fixed_score(fit,target),
                trained_through=N,scored_observation=N+1,status='Verified frozen-prior objective only',
                **_decision_metrics(fit['covariance'],Rtrue,desired,holdout)))
            em_rows.extend(dict(law=law,s_true=s,N=N,trial=trial,iteration=j,objective=value,
                accepted=em['accepted'],rejected=em['rejected']) for j,value in enumerate(em['objectives']))
        except (ArithmeticError,np.linalg.LinAlgError) as error:
            metrics.append(dict(law=law,s_true=s,N=N,trial=trial,method='Recorded-prior EM',prior='Common control',numerical_ok=False,status=str(error)))
        for prior_id,prior_values in enumerate(config['priors']):
            prior=PowerPrior(**prior_values);seed=config['seed_chains']+100000*trial+1000*N+100*int(s*10)+10*prior_id+{'stationary':0,'matched_motion':1,'ula_motion':2}[law]
            kwargs=dict(prior=prior,draws=config['draws'] if law=='stationary' else config['moving_draws'],
                burnin=config['burnin'] if law=='stationary' else config['moving_burnin'],chains=config['chains'],seed=seed)
            result=stationary_gibbs(U,**kwargs) if law=='stationary' else full_path_gibbs(U,**kwargs)
            if result.diagnostics['completed']:
                metrics.append(_joint_rows(result,config,law,s,N,trial,prior,Rtrue,desired,holdout,target))
            else:metrics.append(dict(law=law,s_true=s,N=N,trial=trial,method='Joint inference',prior=prior.label,numerical_ok=False,status=str(result.diagnostics['failures'])))
            diagnostics.extend(dict(law=law,s_true=s,N=N,trial=trial,prior=prior.label,**row)
                for row in result.diagnostics['details'])
            if trial==0 and s==.2 and prior_id==0 and (N==32 or law!='stationary'):
                for chain in range(config['chains']):
                    for index in range(result.a.shape[1]):
                        aa=float(result.a[chain,index]);bb=float(result.b[chain,index])
                        draw_rows.append(dict(law=law,s_true=s,N=N,trial=trial,prior=prior.label,chain=chain,draw=index,
                            a=aa,b=bb,s=bb,q=aa-bb,kappa=float(result.kappa[chain,index])))
        if law=='matched_motion' and trial<config['known_full_path_trials']:
            control=full_path_gibbs(U,draws=config['moving_draws'],burnin=config['moving_burnin'],
                chains=config['chains'],seed=config['seed_chains']+900000+100*trial+int(s*10),
                known_variances=(s+config['q'],s),kappa_fixed=config['kappa_matched'])
            if control.diagnostics['completed']:
                row=_joint_rows(control,config,law,s,N,trial,PowerPrior(),Rtrue,desired,holdout,target)
                row.update(method='Known full-path Gibbs',prior='Common control',
                    conditioning='First two predeclared matched data sets per noise, shared across prior sensitivities')
                for name in ['a','b','q','kappa']:
                    for suffix in ['lower','median','upper','covered']:row.pop(name+'_'+suffix,None)
                metrics.append(row)
            else:metrics.append(dict(law=law,s_true=s,N=N,trial=trial,method='Known full-path Gibbs',
                prior='Common control',numerical_ok=False,status=str(control.diagnostics['failures'])))
            diagnostics.extend(dict(law=law,s_true=s,N=N,trial=trial,prior='Known full-path control',**row)
                for row in control.diagnostics['details'])
    return dict(metrics=metrics,mixing=diagnostics,posterior_draws=draw_rows,em_objectives=em_rows)


def run_jobs(function,tasks,out,stage,key,workers):
    folder=out/'checkpoints'/stage;folder.mkdir(parents=True,exist_ok=True)
    completed={};pending=[]
    for index,task in enumerate(tasks):
        path=folder/f'{index:05d}.json';signature=digest(task)
        try:
            previous=json.loads(path.read_text())
            assert previous['source_hash']==key and previous['task_hash']==signature
            assert digest(previous['result'])==previous['result_hash']
            completed[index]=previous['result']
        except (OSError,ValueError,KeyError,AssertionError):pending.append((index,task,path,signature))
    print(stage,'restored',len(completed),'/',len(tasks),flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):(index,path,signature) for index,task,path,signature in pending}
            for future in as_completed(futures):
                index,path,signature=futures[future];result=portable(future.result());completed[index]=result
                save(path,dict(source_hash=key,task_hash=signature,result_hash=digest(result),result=result))
                if len(completed)%4==0 or len(completed)==len(tasks):print(stage,'completed',len(completed),'/',len(tasks),flush=True)
    return [completed[j] for j in range(len(tasks))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick')
    parser.add_argument('--output',type=Path);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();cfg=configuration(args.profile)
    out=(args.output or output_directory(ROOT,args.profile,'inference')).resolve();out.mkdir(parents=True,exist_ok=True)
    gate=None
    if args.profile=='paper':
        from lib.beamforming_validation import require_beamforming_gate
        gate=require_beamforming_gate(ROOT)
    sources={name:sha(ROOT/name) for name in SOURCES}
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=cfg,sources=sources,versions=versions))
    if (out/'manifest.json').exists():
        previous=json.loads((out/'manifest.json').read_text())
        if previous['source_hash']==key and all(sha(out/p)==v for p,v in previous['tables'].items()):
            print('Verified cached inference evidence');return
        raise RuntimeError('Completed inference evidence differs; preserve it and choose a fresh output directory')
    declaration=dict(config=cfg,sources=sources,versions=versions,source_hash=key,
        base_commit=cfg['base_commit'],gate=gate,status='running',workers=args.workers)
    if (out/'preregistered.json').exists() and json.loads((out/'preregistered.json').read_text())['source_hash']!=key:
        raise RuntimeError('Frozen inference sources differ; preserve development run and choose a fresh output')
    save(out/'preregistered.json',declaration);save(out/'status.json',dict(status='running',stage='validation',source_hash=key))
    tasks=[(cfg,s,trial) for s in cfg['noises'] for trial in range(cfg['validation_trials'])]
    rows=[row for group in run_jobs(validation_job,tasks,out,'validation',key,args.workers) for row in group]
    selected,curves=select_fixed(rows,cfg)
    # One symmetric validation-only extension if a finite power boundary wins.
    expansion_s=[];expansion_q=[]
    if selected['s']==min(cfg['fixed_s']):expansion_s=[.05]
    if selected['s']==max(cfg['fixed_s']):expansion_s=[4.]
    if selected['q']==min(cfg['fixed_q']):expansion_q=[.25]
    if selected['q']==max(cfg['fixed_q']):expansion_q=[4.]
    if expansion_s or expansion_q:
        extended=dict(cfg,fixed_s=sorted(set(cfg['fixed_s']+expansion_s)),fixed_q=sorted(set(cfg['fixed_q']+expansion_q)))
        tasks=[(extended,s,trial) for s in cfg['noises'] for trial in range(cfg['validation_trials'])]
        rows=[row for group in run_jobs(validation_job,tasks,out,'validation_expanded',key,args.workers) for row in group]
        selected,curves=select_fixed(rows,extended)
    pd.DataFrame(rows).to_csv(out/'validation.csv',index=False);pd.DataFrame(curves).to_csv(out/'tuning.csv',index=False)
    save(out/'selected.json',dict(selected=selected,source_hash=key,
        expansion=dict(s=expansion_s,q=expansion_q),selected_before_test=True,
        criterion='Mean next-observation negative log predictive density at prefix 32, equally weighted true-noise conditions'))
    tasks=[(cfg,'stationary',s,trial,selected) for s in cfg['noises'] for trial in range(cfg['stationary_trials'])]
    tasks += [(cfg,law,s,trial,selected) for law in ['matched_motion','ula_motion'] for s in cfg['noises'] for trial in range(cfg['moving_trials'])]
    save(out/'test_preregistered.json',dict(**declaration,selected=selected,selection_sha256=sha(out/'selected.json'),heldout_opened_after_selection=True))
    save(out/'status.json',dict(status='running',stage='inference',source_hash=key))
    groups=run_jobs(experiment_job,tasks,out,'inference',key,args.workers)
    frames={name:pd.DataFrame([row for group in groups for row in group[name]]) for name in ['metrics','mixing','posterior_draws','em_objectives']}
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    metric=frames['metrics'];summary=metric.groupby(['law','s_true','N','method','prior'],dropna=False).agg(
        trials=('trial','nunique'),calculation_failures=('numerical_ok',lambda x:int((~x.astype(bool)).sum())),
        unresolved_mixing=('mixing_ok',lambda x:int((x==False).sum())),actual_output_power=('actual_output_power','mean'),
        prospective_log_score=('prospective_log_score','mean'),fit_seconds=('fit_seconds','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    # A separate native CP1 reference validates the hierarchy, not n=8 tails.
    rng=np.random.default_rng(6571059);h=haar_line(rng,2);U=proper_complex(rng,8)[:,None]*h+proper_complex(rng,(8,2),.2)
    reference=stationary_reference_n2(U)
    reference.pop('mean');reference.pop('covariance')
    save(out/'reference.json',reference)
    if any(sha(ROOT/p)!=v for p,v in sources.items()):
        save(out/'status.json',dict(status='source_changed_development',source_hash=key))
        raise RuntimeError('Inference scientific sources changed during execution; preserve this run')
    checks=dict(rows=len(metric),calculation_failures=int((~metric.numerical_ok.astype(bool)).sum()),
        joint_fits=int((metric.method=='Joint inference').sum()),
        unresolved_joint_mixing=int(((metric.method=='Joint inference')&(~metric.mixing_ok.fillna(False).astype(bool))).sum()),
        max_unit_gain_error=float(metric.unit_gain_error.max()),max_posterior_optimality_residual=float(metric.posterior_excess_identity_error.max()),
        stationary_reference_ok=reference['numerical_ok'])
    tables={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.csv','.json'] and p.name not in ['manifest.json','config.json','status.json']}
    save(out/'manifest.json',dict(declaration,status='completed',tables=tables,checks=checks))
    save(out/'config.json',dict(declaration,status='completed',tables=tables,checks=checks))
    save(out/'status.json',dict(status='completed',source_hash=key,checks=checks))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
