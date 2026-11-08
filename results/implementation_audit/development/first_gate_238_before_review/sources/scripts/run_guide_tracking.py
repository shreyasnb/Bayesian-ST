"""Frozen guide tracking/bound cohorts; legacy evidence remains untouched."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib,json,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution,t as student_t
from lib.guide_tracking import (configuration,marginal_certificate,conditional_certificate,
    certificate_observation,marginal_preflight)
from lib.beamforming_protocol import proper_complex,ula,haar_line
from lib.array_geometry_study import (distortionless_weight,covariance_shrinkage,ledoit_wolf,
    scm_statistics,scm_validation_loss,continuous_minimum)
from lib.beamforming_decisions import isotropized_mean
from lib.beamforming_numerics import complex_moments
from lib.geometry import center
from benchmarks.case1.baselines import output_power,sample_output_power
from benchmarks.case2.baselines import GROUSE

SOURCES=['lib/guide_tracking.py','tests/test_guide_tracking.py','scripts/run_guide_tracking.py',
    'lib/models.py','lib/geometry.py','lib/beamforming_protocol.py','lib/array_geometry_study.py',
    'lib/beamforming_decisions.py','lib/beamforming_numerics.py','lib/spectral_series.py',
    'benchmarks/case1/baselines.py','benchmarks/case2/baselines.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(path,x):path.write_text(json.dumps(portable(x),indent=2,allow_nan=False)+'\n')


def path(T,condition='static',rate=0.,complex_data=False):
    angle=np.arange(T)*rate
    if condition=='abrupt':angle[T//2:]=np.pi/4
    h=np.column_stack([np.sin(angle),np.zeros(T),np.cos(angle)])
    if complex_data:h=h.astype(complex);h[:,1]=1j*np.sin(angle)/np.sqrt(2);h[:,0]/=np.sqrt(2)
    return np.einsum('ti,tj->tij',h,h.conj()),h


def scalars(c):
    keys=['n','k','s','tau2','gamma','initialization','drift','noise','epsilon','bound','valid',
        'informative','ceiling','formula_name','manuscript_verified','m','b','h','eta']
    return {name:c[name] for name in keys if name in c}


def exact_binomial(successes,trials):
    if not trials:return np.nan,np.nan
    lower=0. if not successes else float(beta_distribution.ppf(.025,successes,trials-successes+1))
    upper=1. if successes==trials else float(beta_distribution.ppf(.975,successes+1,trials-successes))
    return lower,upper


def bound_study(cfg):
    scenarios=[];phases=[];n=cfg['dimension'];s=cfg['s'];q=cfg['tau2']
    for complex_data in [False,True]:
        kind='complex' if complex_data else 'real'
        P,h=path(cfg['static_T'],complex_data=complex_data)
        for beta in [1.,.995]:
            for t in cfg['static_times']:
                cert=marginal_certificate(P[:t],s,q,beta,cfg['delta'],complex_data=complex_data)
                scenarios.append(dict(observation_type='marginal_'+kind,condition='static',t=t,
                    beta=beta,drift_rate=0.,certificate=cert,h=h[:t],V=None))
        for rate in cfg['dynamic_rates']:
            P,h=path(cfg['dynamic_T'],'drift',rate,complex_data)
            for beta in cfg['bound_betas']:
                cert=marginal_certificate(P,s,q,beta,cfg['delta'],complex_data=complex_data)
                scenarios.append(dict(observation_type='marginal_'+kind,condition='drift',t=len(P),
                    beta=beta,drift_rate=rate,certificate=cert,h=h,V=None))
        for beta in [1.,.9999,.999,.995,.99,.98]:
            for rate in [0.,.00001,.0001,.001]:
                P,_=path(cfg['dynamic_T'],'drift',rate,complex_data)
                cert=marginal_certificate(P,s,q,beta,cfg['delta'],complex_data=complex_data)
                pre=marginal_preflight(n,s,q,len(P),beta,cfg['delta'],complex_data)
                phases.append(dict(observation_type='marginal_'+kind,condition='preflight',t=len(P),
                    beta=beta,drift_rate=rate,**scalars(cert),static_ratio=pre['ratio'],
                    infinite_memory_ratio=pre['limit'],permanently_vacuous=pre['permanently_vacuous']))
        for condition in ['static','concentrated_design','abrupt']:
            T=cfg['abrupt_T'] if condition=='abrupt' else cfg['conditional_T']
            P,h=path(T,condition,complex_data=complex_data)
            V=np.eye(n)[np.arange(T)%n]
            if condition=='concentrated_design':V=np.broadcast_to([1.,0.,0.],(T,n))
            if complex_data:V=V.astype(complex)
            times=sorted(set([T//2,T//2+1,T])) if condition=='abrupt' else [t for t in cfg['conditional_times'] if t<=T]
            for beta in ([1.,.98] if condition=='abrupt' else [1.]):
                for t in times:
                    cert=conditional_certificate(P[:t],V[:t],s,beta,cfg['delta'],complex_data=complex_data)
                    scenarios.append(dict(observation_type='conditional_'+kind,condition=condition,t=t,
                        beta=beta,drift_rate=0.,certificate=cert,h=h[:t],V=V[:t]))
    rows=[]
    for index,scenario in enumerate(scenarios):
        c=scenario['certificate'];T=scenario['t'];complex_data=c['complex_data'];V=scenario['V'];h=scenario['h']
        for trial in range(cfg['episodes']):
            # Fixed path and design across every signal/noise repetition. Within
            # a cell the entire observation stream is an independent episode.
            rng=np.random.default_rng(np.random.SeedSequence([cfg['test_seed'],trial,index,2301]))
            if complex_data:noise=proper_complex(rng,(T,n),s)
            else:noise=rng.normal(size=(T,n))*np.sqrt(s)
            if V is None:
                z=proper_complex(rng,T,q) if complex_data else rng.normal(size=T)*np.sqrt(q)
                U=h*z[:,None]+noise
            else:U=np.einsum('tij,tj->ti',c['Ps'],V)+noise
            result=certificate_observation(c,U,V)
            rows.append(dict(observation_type=scenario['observation_type'],condition=scenario['condition'],
                trial=trial,t=T,beta=scenario['beta'],drift_rate=scenario['drift_rate'],**scalars(c),**result))
        if (index+1)%8==0:print('bound cells',index+1,'/',len(scenarios),flush=True)
    frame=pd.DataFrame(rows);summary=[]
    groups=['observation_type','condition','n','k','t','beta','drift_rate']
    for keys,g in frame.groupby(groups,sort=False):
        valid=g.valid.astype(bool);eligible=int(valid.sum());violations=int(g.loc[valid,'violation'].sum());lo,hi=exact_binomial(violations,eligible)
        summary.append(dict(zip(groups,keys),episodes=len(g),eligible_episodes=eligible,violations=violations if eligible else np.nan,violation_lower=lo,
            violation_upper=hi,valid_fraction=float(valid.mean()),informative_fraction=float(g.informative.mean()),
            violation_upper_one_sided95=(1. if violations==eligible else float(beta_distribution.ppf(.95,violations+1,eligible-violations))) if eligible else np.nan,
            mean_error=float(g.modal_error.mean()),error_se=float(g.modal_error.std(ddof=1)/np.sqrt(len(g))),
            max_decomposition_error=float(g.decomposition_error.max()),
            bound=float(g.bound.iloc[0]),ceiling=float(g.ceiling.iloc[0]),
            theoretical_delta=cfg['delta'],confidence_scope='Pointwise exact binomial95% interval among eligible episodes only; invalid-bound cells have no modal violation test; no simultaneous claim'))
    return frame,pd.DataFrame(summary),pd.DataFrame(phases)


def beam_data(cfg,spec,trial,split):
    seed=cfg['validation_seed'] if split=='validation' else cfg['test_seed']
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial,5021]));n=cfg['beam_n'];T=cfg['beam_T']
    angles=np.full(T,10.)
    if spec['motion']=='drift':angles=np.linspace(3.,30.,T)
    if spec['motion']=='abrupt':angles[T//2:]=30.;angles[:T//2]=3.
    h=np.array([ula(n,np.deg2rad(theta)) for theta in angles]);a=ula(n,0.)
    U=h*proper_complex(rng,T,spec['tau2'])[:,None]+proper_complex(rng,(T,n),spec['s'])
    initial=haar_line(rng,n);held={}
    for t in cfg['beam_times']:
        held[t]=h[t-1]*proper_complex(rng,cfg['heldout'],spec['tau2'])[:,None]+proper_complex(rng,(cfg['heldout'],n),spec['s'])
    return dict(U=U,h=h,a=a,initial=initial,held=held,angles=angles)


def scatters(U,beta,times):
    S=np.zeros((U.shape[1],U.shape[1]),complex);mass=0.;rows={}
    for t,u in enumerate(U,1):
        S=beta*S+np.outer(u,u.conj());mass=beta*mass+1
        if t in times:rows[t]=(S.copy(),mass)
    return rows


def mode(S):
    h=np.linalg.eigh(S)[1][:,-1]
    return np.outer(h,h.conj())


def beam_validation(cfg):
    selected=[];curves=[];n=cfg['beam_n']
    for spec in cfg['beam_conditions']:
        s,q=spec['s'],spec['tau2'];records={b:[] for b in cfg['beam_betas']};grouse=[]
        for trial in range(cfg['validation_episodes']):
            data=beam_data(cfg,spec,trial,'validation');a=data['a']
            for beta in cfg['beam_betas']:
                loss=[];stats=[]
                for t,(S,mass) in scatters(data['U'],beta,cfg['beam_times']).items():
                    P=mode(S);R=s*np.eye(n)+q*np.outer(data['h'][t-1],data['h'][t-1].conj())
                    loss.append(output_power(distortionless_weight(s*np.eye(n)+q*P,a),R))
                    stats.append(scm_statistics(S/mass,a,data['h'][t-1]))
                records[beta].append(dict(trial=trial,loss=float(np.mean(loss)),statistics=stats))
            for step in cfg['beam_grouse_steps']:
                tracker=GROUSE(n,1,step,rng=0);tracker.basis=data['initial'][:,None].copy();loss=[]
                for t,u in enumerate(data['U'],1):
                    X=tracker.update(u)
                    if t in cfg['beam_times']:
                        R=s*np.eye(n)+q*np.outer(data['h'][t-1],data['h'][t-1].conj())
                        loss.append(output_power(distortionless_weight(s*np.eye(n)+q*X@X.conj().T,a),R))
                grouse.append(dict(step=step,trial=trial,loss=float(np.mean(loss))))
        options=[];scm_options=[]
        for beta,rows in records.items():
            score=float(np.mean([r['loss'] for r in rows]));options.append((score,beta))
            statistics=[row for r in rows for row in r['statistics']]
            fit=continuous_minimum(lambda alpha:scm_validation_loss(alpha,statistics,s,q),points=33)
            scm_options.append((fit['loss'],beta,fit['value']))
            curves.append(dict(condition=spec['name'],method='mode',beta=beta,parameter=np.nan,loss=score,
                validation_episodes=cfg['validation_episodes'],within_episode_times=len(cfg['beam_times'])))
            curves.extend(dict(condition=spec['name'],method='scm',beta=beta,parameter=alpha,loss=loss,
                validation_episodes=cfg['validation_episodes'],within_episode_times=len(cfg['beam_times']))
                for alpha,loss in zip(fit['grid'],fit['losses']))
        gsloss={step:float(np.mean([r['loss'] for r in grouse if r['step']==step])) for step in cfg['beam_grouse_steps']}
        for step,loss in gsloss.items():curves.append(dict(condition=spec['name'],method='grouse',beta=np.nan,parameter=step,loss=loss,validation_episodes=cfg['validation_episodes']))
        _,beta=min(options);_,scm_beta,alpha=min(scm_options);step=min(gsloss,key=gsloss.get)
        selected.append(dict(condition=spec['name'],beta=beta,scm_beta=scm_beta,alpha=alpha,grouse_step=step,
            mode_validation_loss=min(options)[0],scm_validation_loss=min(scm_options)[0],
            selection_scope='One mode-selected beta shared by mean/mode/isotropic; own SCM beta/loading and own GROUSE step. All outcomes averaged within episode first.'))
        print('beam validation',spec['name'],'complete',flush=True)
    return selected,pd.DataFrame(curves)


def beam_test(cfg,selected):
    rows=[];stories={};n=cfg['beam_n']
    for spec in cfg['beam_conditions']:
        policy=next(p for p in selected if p['condition']==spec['name']);s,q=spec['s'],spec['tau2']
        for trial in range(cfg['episodes']):
            d=beam_data(cfg,spec,trial,'test');a=d['a'];paths={b:scatters(d['U'],b,cfg['beam_times']) for b in set([1.,policy['beta'],policy['scm_beta']])}
            tracker=GROUSE(n,1,policy['grouse_step'],rng=0);tracker.basis=d['initial'][:,None].copy();Pg={}
            for t,u in enumerate(d['U'],1):
                X=tracker.update(u)
                if t in cfg['beam_times']:Pg[t]=X@X.conj().T
            for t in cfg['beam_times']:
                S,mass=paths[policy['beta']][t];P=mode(S);A=center(q/(s*(s+q))*S)
                moment=complex_moments(A);M=moment.mean;Mi,rho=isotropized_mean(M,P)
                P1=mode(paths[1.][t][0]);Sscm,msc=paths[policy['scm_beta']][t]
                Rlw,alphalw=ledoit_wolf(d['U'][:t]);Ptrue=np.outer(d['h'][t-1],d['h'][t-1].conj());Rtrue=s*np.eye(n)+q*Ptrue
                covariances=dict(mean=s*np.eye(n)+q*M,mode=s*np.eye(n)+q*P,iso=s*np.eye(n)+q*Mi,
                    mode_beta1=s*np.eye(n)+q*P1,scm=covariance_shrinkage(Sscm/msc,policy['alpha']),
                    ew_covariance=S/mass,lw=Rlw,grouse=s*np.eye(n)+q*Pg[t],oracle=Rtrue)
                targets=dict(mean=M,mode=P,iso=Mi,mode_beta1=P1,grouse=Pg[t],oracle=Ptrue)
                Rbar=s*np.eye(n)+q*M
                for method,R in covariances.items():
                    w=distortionless_weight(R,a);D=targets.get(method)
                    rows.append(dict(condition=spec['name'],motion=spec['motion'],n=n,t=t,trial=trial,
                        s=s,tau2=q,method=method,beta=1. if method in ('mode_beta1','lw') else policy['scm_beta'] if method=='scm' else policy['beta'],
                        true_power=output_power(w,Rtrue),heldout_power=sample_output_power(w,d['held'][t]),
                        posterior_power=output_power(w,Rbar),response_frobenius=float(np.linalg.norm(D-Ptrue,'fro')**2) if D is not None else np.nan,
                        mean_ok=moment.numerical_ok,mode_pca_error=float(np.linalg.norm(P-mode(A),'fro')),
                        unit_gain_error=float(abs(np.vdot(w,a)-1)),rho=rho,
                        alpha=policy['alpha'] if method=='scm' else alphalw if method=='lw' else np.nan,
                        transport='Identity',oracle_motion=False))
                if trial==0:
                    pre=spec['name']+f'_t{t}_';stories.update({pre+'A':A,pre+'M':M,pre+'P':P,pre+'truth':Ptrue})
            if trial==0:stories.update({spec['name']+'_'+key:d[key] for key in ['U','h','a','angles']})
        print('beam test',spec['name'],'complete',flush=True)
    frame=pd.DataFrame(rows);pairs=[]
    for (condition,t),g in frame.groupby(['condition','t'],sort=False):
        base=g[g.method.eq('mean')].set_index('trial')
        for method,h in g[~g.method.eq('mean')].groupby('method',sort=False):
            h=h.set_index('trial');indices=base.index.intersection(h.index)
            for endpoint in ['true_power','heldout_power','posterior_power']:
                v=(h.loc[indices,endpoint]-base.loc[indices,endpoint]).to_numpy();half=float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
                pairs.append(dict(condition=condition,t=t,reference=method,endpoint=endpoint,episodes=len(v),
                    gain=float(v.mean()),lower=float(v.mean()-half),upper=float(v.mean()+half),
                    numerical_ok=bool(h.mean_ok.all() and base.mean_ok.all()),
                    interval='Paired pointwise Student95%; independent episodes; no multiplicity correction'))
    return frame,pd.DataFrame(pairs),stories


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',choices=['smoke','paper'],default='smoke');parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg=configuration(args.profile);out=args.output or ROOT/'results/implementation_audit'/args.profile/'tracking';out.mkdir(parents=True,exist_ok=True)
    if args.profile=='paper':
        from lib.implementation_validation import require_gate
        gate=require_gate(ROOT)
    else:gate=None
    sources={p:sha(ROOT/p) for p in SOURCES}
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old['config']!=cfg or old['sources']!=sources:raise RuntimeError('Preserve obsolete outputs and use a fresh directory')
        for name,expected in old['tables'].items():
            if sha(out/name)!=expected:raise RuntimeError('Changed cached artifact: '+name)
        print('Verified existing completed tracking profile');return
    save(out/'preregistered.json',dict(config=cfg,sources=sources,registered_utc=datetime.now(timezone.utc).isoformat()))
    start=perf_counter();bounds,summary,phase=bound_study(cfg)
    selected,validation=beam_validation(cfg);save(out/'selected.json',selected)
    beam,pairs,stories=beam_test(cfg,selected)
    for name,frame in dict(bounds=bounds,summary=summary,phase=phase,validation=validation,beamforming=beam,paired=pairs).items():frame.to_csv(out/(name+'.csv'),index=False)
    np.savez_compressed(out/'stories.npz',**stories)
    checks=dict(bound_reports=len(bounds),bound_cells=len(summary),valid_reports=int(bounds.valid.sum()),
        informative_reports=int(bounds.informative.sum()),violations=int(bounds.violation.sum()),
        maximum_decomposition_error=float(bounds.decomposition_error.max()),
        beam_episodes=cfg['episodes']*len(cfg['beam_conditions']),beam_metric_rows=len(beam),
        native_mean_failures=int((~beam.mean_ok.astype(bool)).sum()),
        maximum_mode_pca_error=float(beam.mode_pca_error.max()),maximum_unit_gain_error=float(beam.unit_gain_error.max()),
        seconds=perf_counter()-start,
        complex_conditional_manuscript_match='not applicable: supplementary.tex absent; separate conservative derivation labeled explicitly')
    save(out/'checks.json',checks)
    if any(sha(ROOT/p)!=expected for p,expected in sources.items()):raise RuntimeError('Scientific producers changed during execution')
    tables={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('manifest.json','status.json')}
    save(out/'manifest.json',dict(status='completed',config=cfg,sources=sources,tables=tables,checks=checks,
        gate_science_hash=gate.get('science_hash') if gate else None))
    save(out/'status.json',dict(status='completed',checks=checks));print(json.dumps(portable(checks),indent=2),flush=True)


if __name__=='__main__':main()
