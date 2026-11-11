"""Read-only replay and optimization diagnostics for the saved EW objective.

New finite-step comparisons use unchanged observations, memory and step size.
Only the number of GREAT inner steps changes. Validation selection uses the
arithmetic mean of measured log output power; hidden covariance never selects
an iteration count. Exact eigenspaces are diagnostic references, not additional
GREAT work. Timing, if explicitly requested after isolation, excludes diagnostics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from benchmarks.common.great import GREAT
from .final_beamforming import (beam_episode, great_likelihood_observations,
    heldout_snapshots, modal_weight, covariance_power, observed_validation_record)
from .final_validation import require_gate, verify_component, HISTORICAL_SOURCE_PROOF
from .final_stationary_evidence import verify_stationary_component, stationary_component_path
from .geometry import center

INNER_STEPS = (1,3,10,30)
METRICS = ('objective_gap','relative_objective_gap','chordal_to_mode',
           'chordal_squared_to_mode','stationarity_norm','measured_power_db','true_ratio_db')
SOURCES = ('lib/beamforming_objective_audit.py','scripts/run_beamforming_objective_audit.py',
    'benchmarks/common/great.py','lib/final_beamforming.py','lib/beamforming_protocol.py',
    'lib/geometry.py','lib/final_validation.py','lib/final_stationary_evidence.py',
    'benchmarks/case1/baselines.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value,message):
    if not value:raise RuntimeError(message)


def _artifact(meta,basename):
    names=[name for name in meta['artifacts'] if Path(name).name==basename]
    _require(len(names)==1,'Expected one declared '+basename)
    return names[0]


def load_context(root,cohort='moving'):
    """Verify original numerical evidence before any new replay."""
    root=Path(root).resolve();require_gate(root)
    if cohort=='moving':
        meta=verify_component(root,'beamforming','paper','performance')
        manifest='results/beamforming/data/paper/performance/manifest.json'
        raw_name='moving_checkpoints.csv.gz';story_name='stories.npz'
    elif cohort=='stationary':
        meta=verify_stationary_component(root,'paper')
        manifest=str(stationary_component_path(root,'paper').relative_to(root))
        raw_name='stationary.csv';story_name='stories.npz'
    else:raise ValueError('Use moving or stationary')
    names={name:_artifact(meta,name) for name in (raw_name,story_name,'selected.json','validation_curves.csv')}
    cfg=meta['config'];raw=pd.read_csv(root/names[raw_name])
    policies=json.loads((root/names['selected.json']).read_text())
    with np.load(root/names[story_name],allow_pickle=False) as z:stories={k:z[k] for k in z.files}
    curves=pd.read_csv(root/names['validation_curves.csv'])
    cases=[]
    for policy in policies:
        if cohort=='moving' and (policy.get('stationary') or policy['n']!=32 or policy['condition'] not in ('smooth','abrupt')):continue
        if cohort=='stationary' and not policy['stationary']:continue
        n=int(policy['n']);condition=policy['condition'];T=256 if cohort=='moving' else 200
        beta=.97 if cohort=='moving' else 1.
        _require(policy['shared_beta']==beta and policy['great_ew']['beta']==beta,
                 'The original common EW memory changed')
        _require(policy['great_ew']['inner_steps']==3,'The original baseline must use three inner steps')
        times=list(range(8,257,8)) if cohort=='moving' else [25,50,100,200]
        _require(policy['times']==times,'Reporting times changed')
        cases.append(dict(case=condition if cohort=='moving' else 'static_n'+str(n),
            condition=condition,n=n,T=T,beta=beta,times=times,stationary=cohort=='stationary',policy=policy))
    _require(len(cases)==(2 if cohort=='moving' else 3),'Incomplete original comparison cases')
    _require(cfg['validation_episodes']==64 and cfg['test_episodes']==256,
             'Expected 64 validation and 256 independent test simulations')
    _require(cfg['sigma2']==.2 and cfg['tau2']==1.,'Generating scale settings changed')
    input_hashes={manifest:sha(root/manifest),**{name:sha(root/name) for name in names.values()}}
    proof=root/HISTORICAL_SOURCE_PROOF
    if proof.exists():input_hashes[HISTORICAL_SOURCE_PROOF]=sha(proof)
    return dict(root=root,cohort=cohort,meta=meta,config=cfg,cases=cases,raw=raw,
        stories=stories,validation_curves=curves,input_hashes=input_hashes,
        source_hashes={name:sha(root/name) for name in SOURCES})


def _measured_power(weight,covariance):
    return float(np.vdot(weight,covariance@weight).real)


def replay_trial(context,case,trial,split,inner_steps):
    """Replay a declared stream and return every report-time objective/action."""
    cfg=context['config'];n=case['n'];T=case['T'];beta=case['beta'];policy=case['policy']
    data=beam_episode(cfg,n,case['condition'],trial,split,case['stationary'],length=T)
    raw_U=data['U'];eta=data['tau2']/(data['sigma2']*(data['sigma2']+data['tau2']))
    W=great_likelihood_observations(raw_U,data['sigma2'],data['tau2'])
    if trial==cfg.get('story',{}).get('episode',0) and split=='test':
        prefix=case['condition']+'_n'+str(n)+'_'
        if prefix+'U' in context['stories']:
            for name in ('U','h','a','angles','sigma2','tau2','time'):
                np.testing.assert_array_equal(data[name],context['stories'][prefix+name])
    if split=='validation':
        observed=observed_validation_record(data,case['times'],cfg['heldout_snapshots'])
        measured=observed['validation_covariances']
    else:
        measured={t:((Z:=heldout_snapshots(data,t,cfg['heldout_snapshots'])).T@Z.conj()/len(Z)) for t in case['times']}
    trackers={q:GREAT(n,1,complex_data=True,initial_basis=data['initial'][:,None],
        window=None,beta=beta,stepsize=policy['great_ew']['stepsize'],inner_steps=int(q)) for q in inner_steps}
    rows=[];A=np.zeros((n,n),complex);max_identity=0.;max_saved=0.;max_mode_saved=0.
    reference=context['raw'];reference=reference[(reference.n.eq(n))&reference.condition.eq(case['condition'])&reference.trial.eq(trial)]
    reference=reference.set_index(['method','t'])
    for t,(u,w) in enumerate(zip(raw_U,W),1):
        states={q:tracker.update(w)[:,0] for q,tracker in trackers.items()}
        A=beta*A+eta[t-1]*np.outer(u,u.conj())
        if t not in case['times']:continue
        C=next(iter(trackers.values())).covariance
        identity=float(np.linalg.norm(C-A/eta[0])/max(np.linalg.norm(C),1e-300))
        max_identity=max(max_identity,identity)
        _require(identity<2e-12,'Likelihood-weighted EW covariance differs from uncentered Bingham parameter')
        for tracker in trackers.values():
            np.testing.assert_array_equal(tracker.covariance,C)
            _require(tracker.gamma==np.sqrt(beta),'GREAT gamma must equal sqrt(beta)')
        values,E=np.linalg.eigh(C);exact=E[:,-1];top=float(values[-1]);bottom=float(values[0]);trace=float(np.trace(C).real)
        gap=float(values[-1]-values[-2]);s=float(data['sigma2'][t-1]);qscale=float(data['tau2'][t-1]);desired=data['a']
        exact_weight=modal_weight(exact,desired,s,qscale)
        exact_measured=_measured_power(exact_weight,measured[t])
        if split=='test':
            h=data['h'][t-1]
            oracle=covariance_power(modal_weight(h,desired,s,qscale),h,s,qscale)
            exact_true=float(10*np.log10(covariance_power(exact_weight,h,s,qscale)/oracle))
            if ('mode',t) in reference.index:
                err=abs(exact_true-float(reference.loc[('mode',t),'true_ratio_db']));max_mode_saved=max(max_mode_saved,err)
                _require(err<3e-9,'Exact common-objective eigenspace fails original modal replay')
        else:oracle=np.nan;exact_true=np.nan
        common=dict(condition=case['condition'],n=n,T=T,split=split,trial=trial,t=t,beta=beta,
            gamma=np.sqrt(beta),eta0=float(eta[0]),lambda_max=top,lambda_min=bottom,
            eigengap=gap,trace_covariance=trace,exact_objective=trace-top,
            statistics_match_relative_error=identity)
        rows.append(dict(common,method='mode_exact',inner_steps=0,stepsize=np.nan,
            objective=trace-top,objective_gap=0.,relative_objective_gap=0.,chordal_to_mode=0.,chordal_squared_to_mode=0.,stationarity_norm=0.,
            measured_power=exact_measured,measured_power_db=10*np.log10(exact_measured),
            true_ratio_db=exact_true,oracle_power=oracle))
        for inner,x in states.items():
            rayleigh=float(np.vdot(x,C@x).real);objective_gap=max(0.,top-rayleigh)
            chordal=float(np.sqrt(max(0.,1-abs(np.vdot(exact,x))**2)))
            _require(gap<=1e-12 or chordal**2<=objective_gap/gap+2e-12,
                     'Rayleigh-gap/chordal inequality failed')
            weight=modal_weight(x,desired,s,qscale);power=_measured_power(weight,measured[t])
            true=float(10*np.log10(covariance_power(weight,data['h'][t-1],s,qscale)/oracle)) if split=='test' else np.nan
            if split=='test' and inner==3:
                saved=reference.loc[('great_ew',t)]
                err=max(abs(true-float(saved.true_ratio_db)),
                    abs(10*np.log10(power/oracle)-float(saved.heldout_ratio_db)))
                max_saved=max(max_saved,err)
                _require(err<3e-9,'Three-step EW GREAT does not replay original saved scores')
            rows.append(dict(common,method='great_ew',inner_steps=int(inner),stepsize=policy['great_ew']['stepsize'],
                objective=trace-rayleigh,objective_gap=objective_gap,
                relative_objective_gap=objective_gap/max(top-bottom,1e-300),
                chordal_to_mode=chordal,chordal_squared_to_mode=chordal**2,stationarity_norm=float(np.linalg.norm(C@x-x*rayleigh)),
                measured_power=power,measured_power_db=10*np.log10(power),
                true_ratio_db=true,oracle_power=oracle))
    checks=dict(case=case['case'],trial=int(trial),split=split,
        maximum_statistics_match_relative_error=max_identity,
        maximum_original_great_ew_db_error=max_saved,maximum_original_mode_db_error=max_mode_saved,
        measurements=T,reporting_times=case['times'],inner_steps=list(map(int,inner_steps)),
        no_covariance_or_truth_used_for_validation_selection=True)
    return pd.DataFrame(rows),checks


def interval(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return None
    mean=float(values.mean());half=float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
    return dict(simulations=len(values),mean=mean,lower=mean-half,upper=mean+half)


def summarize(frame):
    rows=[]
    for keys,g in frame.groupby(['condition','n','t','method','inner_steps'],sort=True):
        for metric in METRICS:
            stats=interval(g.groupby('trial')[metric].mean())
            if stats is not None:rows.append(dict(zip(['condition','n','t','method','inner_steps'],keys),metric=metric,**stats))
    return pd.DataFrame(rows)


def paired_summary(frame,cohort='moving'):
    if cohort=='stationary':frame=frame[frame.t.eq(200)]
    rows=[]
    averages=frame.groupby(['condition','n','trial','method','inner_steps'],sort=True).true_ratio_db.mean().reset_index()
    for (condition,n),g in averages.groupby(['condition','n'],sort=True):
        exact=g[g.method.eq('mode_exact')].set_index('trial').true_ratio_db
        for inner,group in g[g.method.eq('great_ew')].groupby('inner_steps'):
            sample=group.set_index('trial').true_ratio_db-exact
            rows.append(dict(condition=condition,n=n,method='great_ew',inner_steps=int(inner),
                reference='mode_exact',metric=('final_t200_true_ratio_db_difference' if cohort=='stationary'
                    else 'time_average_true_ratio_db_difference'),**interval(sample)))
    return pd.DataFrame(rows)


def validation_scores(frame,case):
    """Measured-only criterion: final t200 stationary, all32 moving reports."""
    selected=frame[frame.method.eq('great_ew')]
    times=[200] if case['stationary'] else case['times']
    selected=selected[selected.t.isin(times)]
    counts=selected.groupby(['inner_steps','trial']).t.nunique()
    _require(counts.eq(len(times)).all(),'Missing validation reporting measurements')
    _require(selected.groupby('inner_steps').trial.nunique().eq(64).all(),
             'Every candidate requires all64 validation simulations')
    return selected.groupby(['inner_steps','trial']).measured_power_db.mean().groupby('inner_steps').mean()


def operation_counts(context,selection):
    rows=[]
    for case in context['cases']:
        for inner in sorted({3,int(selection[case['case']]['selected_inner_steps'])}):
            rows.append(dict(condition=case['condition'],n=case['n'],method='great_ew',inner_steps=inner,
                measurements=case['T'],covariance_updates=case['T'],grassmann_steps=case['T']*inner,
                task_decisions=len(case['times']),diagnostic_eigendecompositions=len(case['times']),
                tracker_eigendecompositions=0,scope='Diagnostic eigenspaces excluded from future timing'))
        rows.append(dict(condition=case['condition'],n=case['n'],method='mode_exact',inner_steps=0,
            measurements=case['T'],covariance_updates=case['T'],grassmann_steps=0,
            task_decisions=len(case['times']),diagnostic_eigendecompositions=0,
            tracker_eigendecompositions=len(case['times']),scope='Exact mode at the common task reporting times'))
    return pd.DataFrame(rows)


def timed_workload(context,case,inner_steps,split='test'):
    """One isolated workload; objective/eigenspace diagnostics are absent."""
    data=beam_episode(context['config'],case['n'],case['condition'],0,split,case['stationary'],length=case['T'])
    W=great_likelihood_observations(data['U'],data['sigma2'],data['tau2'])
    # Observations, scaling and desired signatures are prepared before timing.
    begin=perf_counter();n=case['n'];decision=None;tracking=0.;decisions=0.
    start=perf_counter()
    tracker=(GREAT(n,1,complex_data=True,initial_basis=data['initial'][:,None],window=None,
        beta=case['beta'],stepsize=case['policy']['great_ew']['stepsize'],inner_steps=int(inner_steps))
        if inner_steps else None)
    C=np.zeros((n,n),complex) if inner_steps==0 else None
    tracking+=perf_counter()-start
    for t,w in enumerate(W,1):
        start=perf_counter()
        if tracker is None:
            C=case['beta']*C+np.outer(w,w.conj());C=(C+C.conj().T)/2
            x=np.linalg.eigh(C)[1][:,-1] if t in case['times'] else None
        else:x=tracker.update(w)[:,0]
        tracking+=perf_counter()-start
        if t in case['times']:
            start=perf_counter()
            decision=modal_weight(x,data['a'],data['sigma2'][t-1],data['tau2'][t-1])
            decisions+=perf_counter()-start
    elapsed=perf_counter()-begin
    _require(decision is not None and np.isfinite(decision).all(),'Timed workload failed')
    return dict(total_ms=elapsed*1000,tracking_ms=tracking*1000,decision_ms=decisions*1000,
        overhead_ms=max(0.,elapsed-tracking-decisions)*1000)


def load_objective_audit(root,cohort='moving'):
    """Read completed new audit files; never rerun a tracker or fit a setting."""
    root=Path(root).resolve();folder=root/'results/beamforming_comparison_audit'/cohort/'objective'
    meta=json.loads((folder/'manifest.json').read_text())
    _require(meta['status']=='completed','The objective audit is incomplete')
    for mapping in (meta['source_hashes'],meta['input_hashes'],meta['artifacts']):
        for name,digest in mapping.items():_require(sha(root/name)==digest,'Changed objective evidence: '+name)
    tables={name:pd.read_csv(folder/(name+'.csv')) for name in
        ['validation_checkpoints','validation_summary','test_checkpoints','test_summary','paired_summary','operation_counts']}
    timing=None;timing_folder=folder/'timing'
    if (timing_folder/'manifest.json').exists():
        clock_meta=json.loads((timing_folder/'manifest.json').read_text())
        for name,digest in clock_meta['artifacts'].items():_require(sha(root/name)==digest,'Changed isolated timing artifact')
        timing=dict(raw=pd.read_csv(timing_folder/'raw.csv'),summary=pd.read_csv(timing_folder/'summary.csv'),metadata=clock_meta)
    return dict(config=meta['config'],selection=json.loads((folder/'selected.json').read_text()),
        checks=meta['checks'],provenance=meta,input_hashes=meta['input_hashes'],
        source_hashes=meta['source_hashes'],timing=timing,**tables)
