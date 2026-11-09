"""Frozen, paired conditional-observation tests of memory and causal motion.

Every unknown-motion policy is selected once on the same declared mixture of
validation laws and input frames. Test laws are never used to choose a policy.
All state processes here are deterministic conditional on a Haar initial state;
posterior calibration under those laws is a robustness experiment.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import t as student_t
from .geometry import center, projector, mode, chordal, risk
from .drift_study import generator, _posterior, _summary
from .theory import tracking_bound


BETAS=[1.,.995,.99,.98,.97,.95,.93,.90,.85,.80,.70]
WINDOWS=[8,12,16,24,32,48]
# The initial development check selected lag 16 at its upper boundary.
# Add 32 before any final holdout is evaluated; retain the original lags.
LAGS=[4,8,16,32]
DAMPING=[0.,.5,1.]
METHODS=['identity_1','forgetting_097','tuned_forgetting','rolling_window',
         'causal_velocity','partial_motion','exact_motion']
LABELS=dict(identity_1='Identity, no forgetting',forgetting_097='Identity, beta=0.97',
            tuned_forgetting='Tuned forgetting',rolling_window='Rolling likelihood',
            causal_velocity='Causal velocity',partial_motion='Half motion control',
            exact_motion='Exact motion control')


def default_config():
    return dict(schema=1,n=8,k=3,T=128,s=.03,input_norm=2.,
        validation_seed=3296411,test_seed=3296412,validation_trials=32,test_trials=64,
        validation_laws=['slow','fast','acceleration'],
        test_laws=['slow','fast','acceleration','turn','weak_excitation','low_snr','corruption'],
        input_frames=['sensor','co_rotating'],
        betas=BETAS,windows=WINDOWS,lags=LAGS,damping=DAMPING,
        report_times=[8,16,32,64,96,128],posterior_trials=8,
        draws=1500,burnin=1000,max_draws=6000,max_burnin=4000,
        primary_law='fast',primary_frame='sensor',primary_method='causal_velocity',
        primary_reference='tuned_forgetting',delta=.05,
        selection_loss='Terminal chordal distance, equally weighted validation law/frame mixture',
        primary_endpoint='Terminal chordal distance; paired causal-minus-tuned-memory comparison',
        posterior_laws=['slow','fast'],posterior_frame='sensor',
        posterior_methods=['identity_1','tuned_forgetting','rolling_window','causal_velocity','exact_motion'])


def data(config,seed,trial,law,frame):
    """No inputs, motions, or method settings use the hidden initial subspace."""
    n,k,T=config['n'],config['k'],config['T']
    rng=np.random.default_rng(np.random.SeedSequence([seed,trial,4801]))
    X0=np.linalg.qr(rng.normal(size=(n,k)))[0]
    base_v=config['input_norm']*np.eye(n)[np.arange(T)%(2 if law=='weak_excitation' else n)]
    noise=rng.normal(size=(T,n));outliers=rng.normal(size=(T,n))
    corrupt=rng.random(T)<.03
    s=.2 if law=='low_snr' else config['s']
    K=generator(n);R=np.eye(n)
    X=[];V=[];U=[];rotations=[];Gs=[];partials=[]
    for j in range(T):
        if law=='slow':rate=.003
        elif law=='acceleration':rate=.003+.022*j/max(T-1,1)
        else:rate=.025
        if law=='turn' and j>=T//2:rate=-rate
        G=expm(rate*K);R=G@R
        Y=R@X0;v=R@base_v[j] if frame=='co_rotating' else base_v[j]
        u=projector(Y)@v+np.sqrt(s)*R@noise[j]
        if law=='corruption' and corrupt[j]:u=u+3*outliers[j]
        X.append(Y);V.append(v);U.append(u);rotations.append(R.copy())
        Gs.append(G);partials.append(expm(.5*rate*K))
    return dict(X=np.asarray(X),V=np.asarray(V),U=np.asarray(U),
                rotations=np.asarray(rotations),G=np.asarray(Gs),partial=np.asarray(partials),s=s)


def candidate_specs(config,stage,beta=None):
    if stage=='memory':
        return [dict(candidate=f'beta_{b:g}',family='forgetting',prediction='identity',beta=b) for b in config['betas']]+[
            dict(candidate=f'window_{w}',family='window',prediction='rolling',likelihood_window=w,beta=1.) for w in config['windows']]
    return [dict(candidate=f'lag_{h}_damping_{eta:g}',family='velocity',prediction='velocity',
                 beta=beta,lag=h,damping=eta) for h in config['lags'] for eta in config['damping']]


def specs(selected):
    return [dict(method='identity_1',prediction='identity',beta=1.),
            dict(method='forgetting_097',prediction='identity',beta=.97),
            dict(method='tuned_forgetting',prediction='identity',beta=selected['beta']),
            dict(method='rolling_window',prediction='rolling',beta=1.,likelihood_window=selected['window']),
            dict(method='causal_velocity',prediction='velocity',beta=selected['beta'],
                 lag=selected['lag'],damping=selected['damping']),
            dict(method='partial_motion',prediction='identity',beta=1.),
            dict(method='exact_motion',prediction='identity',beta=1.)]


def _filter(config,d,spec):
    from .prediction import PredictiveBinghamFilter
    keys=['prediction','beta','lag','damping','likelihood_window']
    return PredictiveBinghamFilter(config['n'],config['k'],d['s'],**{k:spec[k] for k in keys if k in spec})


def tune_task(task):
    config,trial,stage,beta=task;rows=[]
    for law in config['validation_laws']:
        for frame in config['input_frames']:
            d=data(config,config['validation_seed'],trial,law,frame)
            for spec in candidate_specs(config,stage,beta):
                f=_filter(config,d,spec);start=perf_counter()
                for u,v in zip(d['U'],d['V']):f.update(u,v=v)
                rows.append(dict(trial=trial,law=law,input_frame=frame,stage=stage,
                    candidate=spec['candidate'],family=spec['family'],beta=spec['beta'],
                    window=spec.get('likelihood_window',np.nan),lag=spec.get('lag',np.nan),
                    damping=spec.get('damping',np.nan),terminal_error=float(chordal(f.basis,d['X'][-1])),
                    seconds=perf_counter()-start))
    return rows


def choose(tuning):
    """One common policy, including deterministic first-candidate tie breaking."""
    memory=tuning[tuning.stage=='memory']
    result={}
    for family,key,column in [('forgetting','beta','beta'),('window','window','window')]:
        g=memory[memory.family==family].groupby('candidate',sort=False)
        scores=g.terminal_error.mean();winner=scores.idxmin()
        result[key]=float(g.get_group(winner)[column].iloc[0])
    result['window']=int(result['window'])
    if (tuning.stage=='velocity').any():
        g=tuning[tuning.stage=='velocity'].groupby('candidate',sort=False)
        winner=g.terminal_error.mean().idxmin();row=g.get_group(winner).iloc[0]
        result['lag']=int(row.lag);result['damping']=float(row.damping)
    return result


def _certificate(config,d,method,spec,error):
    if method in ('causal_velocity','rolling_window'):
        return dict(theorem_applicable=False,theorem_reason='Learned transforms' if method=='causal_velocity' else 'Finite likelihood window',
                    valid=False,bound=np.nan,gamma=np.nan,drift_term=np.nan,noise_term=np.nan,
                    bound_violated=False,ceiling=float(np.sqrt(min(config['k'],config['n']-config['k']))))
    G=d['G'] if method=='exact_motion' else d['partial'] if method=='partial_motion' else None
    if d.get('law')=='corruption':
        return dict(theorem_applicable=False,theorem_reason='Corrupted observation law',valid=False,
                    bound=np.nan,gamma=np.nan,drift_term=np.nan,noise_term=np.nan,
                    bound_violated=False,ceiling=float(np.sqrt(min(config['k'],config['n']-config['k']))))
    b=tracking_bound(d['X'],d['V'],d['s'],spec['beta'],G,delta=config['delta'])
    return dict(theorem_applicable=True,theorem_reason='Real conditional, deterministic transforms, fixed true variance',
        valid=b.valid,bound=b.bound,gamma=b.gamma,drift_term=b.drift,noise_term=b.noise,
        bound_violated=bool(b.valid and error>b.bound),ceiling=b.ceiling)


def test_task(task):
    config,trial,selected=task;rows=[];paths=[];bound_rows=[];cost=[]
    for law in config['test_laws']:
        for frame in config['input_frames']:
            d=data(config,config['test_seed'],trial,law,frame);d['law']=law
            for spec in specs(selected):
                method=spec['method'];f=_filter(config,d,spec)
                error_history=[];forecast_history=[]
                for j,(u,v) in enumerate(zip(d['U'],d['V'])):
                    G=d['G'][j] if method=='exact_motion' else d['partial'][j] if method=='partial_motion' else None
                    prior=f.predict(G=G)
                    forecast=float(chordal(mode(prior,config['k']),d['X'][j]))
                    Y=f.update(u,v=v,G=G);error=float(chordal(Y,d['X'][j]))
                    error_history.append(error);forecast_history.append(forecast)
                    if j+1 in config['report_times'] or trial==0:
                        diag=f.motion_diagnostics
                        row=dict(trial=trial,law=law,input_frame=frame,method=method,t=j+1,
                            forecast_error=forecast,corrected_error=error,beta=spec['beta'],
                            window=spec.get('likelihood_window',0),lag=spec.get('lag',0),
                            damping=spec.get('damping',0.),speed_cap='none',
                            motion_active=diag.get('active',False),motion_reason=diag.get('reason','identity'),
                            maximum_angle=diag.get('max_angle',0.),
                            prediction_seconds=f.prediction_seconds,update_seconds=f.update_seconds)
                        if j+1 in config['report_times']:rows.append(row)
                        if trial==0:paths.append(row)
                error=float(error_history[-1]);certificate=_certificate(config,d,method,spec,error)
                bound_rows.append(dict(trial=trial,law=law,input_frame=frame,method=method,
                    terminal_error=error,**certificate))
                cost.append(dict(trial=trial,law=law,input_frame=frame,method=method,
                    terminal_error=error,terminal_forecast_error=forecast_history[-1],
                    initial_error=float(np.mean(error_history[:16])),
                    mean_error=float(np.mean(error_history)),
                    prediction_seconds=f.prediction_seconds,update_seconds=f.update_seconds,
                    prediction_seconds_per_step=f.prediction_seconds/config['T'],
                    update_seconds_per_step=f.update_seconds/config['T']))
    return dict(metrics=rows,paths=paths,bounds=bound_rows,cost=cost)


def posterior_task(task):
    """Eight trajectories are a pilot; every capped numerical failure is kept."""
    config,trial,selected=task;rows=[]
    allowed=set(config['posterior_methods'])
    for law in config['posterior_laws']:
        frame=config['posterior_frame'];d=data(config,config['test_seed'],trial,law,frame)
        for spec in specs(selected):
            method=spec['method']
            if method not in allowed:continue
            f=_filter(config,d,spec)
            for j,(u,v) in enumerate(zip(d['U'],d['V'])):
                G=d['G'][j] if method=='exact_motion' else None
                prior=f.predict(G=G)
                if j==config['T']-1:terminal_prior=prior.copy()
                f.update(u,v=v,G=G)
            idx=METHODS.index(method);seed=[config['test_seed'],trial,config['test_laws'].index(law),idx,8281]
            kwargs=dict(draws=config['draws'],burnin=config['burnin'],
                        max_draws=config['max_draws'],max_burnin=config['max_burnin'])
            M,chains,diag=_posterior(f.A,config['k'],np.random.SeedSequence(seed),[np.eye(config['n'])],**kwargs)
            values=_summary(f.A,f.basis,d['X'][-1],M,chains,np.eye(config['n']),diag,seed,d['s'])
            # The prior is sampled independently, before the current observation's correction.
            prior_seed=np.random.SeedSequence(seed+[1])
            Mp,prior_chains,prior_diag=_posterior(terminal_prior,config['k'],prior_seed,[np.eye(config['n'])],**kwargs)
            residual=d['U'][-1]-Mp@d['V'][-1]
            prior_expected=config['n']*d['s']+risk(Mp,d['V'][-1])
            bases=np.concatenate([c.bases for c in chains]);Pm=projector(f.basis)
            distances=np.sqrt(np.maximum(0.,config['k']-np.einsum('bni,nm,bmi->b',bases,Pm,bases)))
            losses=np.trace(f.A@Pm)-np.einsum('bni,nm,bmi->b',bases,f.A,bases)
            batches=np.array_split(np.arange(len(bases)),8)
            qmcse=float(np.std([np.quantile(distances[b],.95) for b in batches],ddof=1)/np.sqrt(8))
            hmcse=float(np.std([np.quantile(losses[b],.95) for b in batches],ddof=1)/np.sqrt(8))
            rows.append(dict(trial=trial,law=law,input_frame=frame,method=method,t=config['T'],
                **values,radius_mcse=qmcse,hpd_threshold_mcse=hmcse,
                clean_width=values['q975']-values['q025'],
                predictive_width=values['predictive_q975']-values['predictive_q025'],
                innovation_energy=float(residual@residual),innovation_expected_energy=float(prior_expected),
                innovation_error=float(residual@residual-prior_expected),
                prior_sampling_ok=bool(prior_diag['sampling_ok']),prior_min_ess=prior_diag['min_ess'],
                prior_max_rhat=prior_diag['max_rhat'],prior_component_min_ess=_scalar_ess(prior_chains),
                prior_actual_draws_per_chain=prior_diag['actual_draws_per_chain'],
                prior_actual_burnin=prior_diag['actual_burnin'],prior_sampler_seconds=prior_diag['sampler_seconds']))
    return rows


def _scalar_ess(chains):
    from .moments import effective_sample_size
    return float(min(effective_sample_size(np.sum(c.bases[:,1,:]*c.bases[:,0,:],axis=1)) for c in chains))


def summaries(frames,config):
    """All intervals use independent trajectories, never chain draws as trials."""
    from .reporting import wilson
    cost=frames['cost'];stats=[];paired=[]
    for keys,g in cost.groupby(['law','input_frame','method'],sort=False):
        value=g.terminal_error;se=float(value.std(ddof=1)/np.sqrt(len(value)))
        stats.append(dict(law=keys[0],input_frame=keys[1],method=keys[2],
            mean_error=float(value.mean()),se=se,trials=len(value),
            mean_forecast_error=float(g.terminal_forecast_error.mean()),
            mean_initial_error=float(g.initial_error.mean()),
            prediction_seconds=float(g.prediction_seconds.mean()),update_seconds=float(g.update_seconds.mean())))
    for keys,g in cost.groupby(['law','input_frame'],sort=False):
        pivot=g.pivot(index='trial',columns='method',values='terminal_error')
        for reference in ['identity_1','forgetting_097','tuned_forgetting']:
            for method in METHODS:
                if method==reference:continue
                gain=(pivot[reference]-pivot[method]).to_numpy();se=float(np.std(gain,ddof=1)/np.sqrt(len(gain)))
                radius=float(student_t.ppf(.975,len(gain)-1))*se
                primary=(keys[0]==config['primary_law'] and keys[1]==config['primary_frame'] and
                         method==config['primary_method'] and reference==config['primary_reference'])
                paired.append(dict(law=keys[0],input_frame=keys[1],method=method,reference=reference,
                    mean_gain=float(gain.mean()),se=se,low=float(gain.mean()-radius),high=float(gain.mean()+radius),
                    trials=len(gain),primary=primary,interval_scope='primary' if primary else 'exploratory pointwise'))
    coverage=[]
    if len(frames.get('posterior',[])):
        for keys,g in frames['posterior'].groupby(['law','input_frame','method'],sort=False):
            for field,width in [('ball_covered','posterior_radius'),('hpd_covered','hpd_threshold'),
                                ('clean_response_covered','clean_width'),('predictive_covered','predictive_width')]:
                count=int(g[field].sum());low,high=wilson(count,len(g))
                coverage.append(dict(law=keys[0],input_frame=keys[1],method=keys[2],region=field,
                    successes=count,trials=len(g),coverage=count/len(g),low=low,high=high,
                    mean_width=float(g[width].mean()),width_quantity=width,
                    sampling_passes=int(g.sampling_ok.sum()),prior_sampling_passes=int(g.prior_sampling_ok.sum())))
    return dict(summary=pd.DataFrame(stats),paired=pd.DataFrame(paired),coverage=pd.DataFrame(coverage))
