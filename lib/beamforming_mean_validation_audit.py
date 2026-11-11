"""Independent posterior-mean memory selection on preserved array observations.

This audit never changes the original experiment, settings, or result files.
Selection sees measured training/holdout data only. True covariance is used
after selection for evaluation, on the original independent test simulations.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .final_beamforming import (
    beam_episode, observed_validation_record, weighted_statistics,
    checked_moments, projector_mode, modal_weight, covariance_power,
    oracle_power, heldout_snapshots)
from .beamforming_decisions import isotropized_mean
from .array_geometry_study import distortionless_weight
from benchmarks.case1.baselines import output_power, sample_output_power


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('results/beamforming/data/paper/performance')
DESTINATION = Path('results/beamforming_comparison_audit')
BETAS = (1., .99, .97, .9)
METHODS = ('mean', 'mode', 'isotropic')
AGGREGATIONS = ('all_mean_db', 'burnin_mean_db',
                'all_db_mean_ratio', 'burnin_db_mean_ratio')
SOURCES = (
    'lib/beamforming_mean_validation_audit.py',
    'scripts/run_beamforming_mean_validation_audit.py',
    'lib/final_beamforming.py', 'lib/geometry.py',
    'lib/beamforming_protocol.py', 'lib/beamforming_numerics.py',
    'lib/beamforming_decisions.py', 'lib/array_geometry_study.py',
    'lib/spectral_series.py', 'lib/beamforming_moving_evidence.py',
    'lib/final_stationary_evidence.py', 'lib/final_validation.py',
    'benchmarks/case1/baselines.py')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def _write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False, float_format='%.17g',
                 compression=dict(method='gzip',mtime=0) if path.suffix == '.gz' else None)
    os.replace(temporary, path)


def _case_key(n, condition, scope):
    return condition if scope == 'moving' else 'n'+str(n)


def audit_configuration(root=ROOT, scope='moving'):
    """Freeze a new selection criterion while retaining the original cohort."""
    root = Path(root).resolve()
    if scope not in ('moving', 'stationary'):
        raise ValueError('Use moving or stationary')
    source = ORIGINAL if scope == 'moving' else ORIGINAL/'stationary_T200'
    # Verify producer-bound historical artifacts before binding their current
    # bytes into this new audit; fresh hashes alone cannot certify originals.
    if scope == 'moving':
        from .beamforming_moving_evidence import _verified_original
        manifest,_,_ = _verified_original(root)
    else:
        from .final_stationary_evidence import verify_stationary_component
        manifest = verify_stationary_component(root,'paper')
    if manifest.get('status') != 'completed':
        raise ValueError('The preserved production cohort is not completed')
    cfg = manifest['config']
    expected = dict(validation_episodes=64, test_episodes=256,
        sigma2=.2, tau2=1., overlap=.5, heldout_snapshots=256,
        moment_tolerance=1e-10, moment_max_order=4096,
        moment_fallback_max_order=8192)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected production precision or sampling configuration')
    if tuple(cfg['betas']) != BETAS or cfg['validation_seed'] != 2026111001 or cfg['test_seed'] != 2026111002:
        raise ValueError('The original beta grid and independent split seeds are required')
    if scope == 'moving':
        if cfg['moving_n'] != 32 or cfg['moving_T'] != 256 or cfg['action_cadence'] != 8:
            raise ValueError('Expected n32/T256 with all 32 original report times')
        cases = [dict(n=32, condition=x, key=x) for x in ('smooth', 'abrupt')]
        times = list(range(8, 257, 8))
        raw = source/'moving_checkpoints.csv.gz'
        baseline_beta = .97
        criterion = ('Arithmetic mean across 64 validation simulations of the '
            'mean across all 32 report times t=8,16,...,256 of '
            '10*log10(w^H Rhat_validation w); no oracle normalization, '
            'no burn-in, and no true covariance in selection.')
    else:
        if cfg.get('stationary_T') != 200 or cfg.get('stationary_primary_t') != 200:
            raise ValueError('Expected the preserved fixed-T200 stationary cohort')
        cases = [dict(n=n, condition='static', key='n'+str(n)) for n in (32,64,128)]
        times = [200]
        raw = source/'stationary.csv'
        baseline_beta = 1.
        criterion = ('Arithmetic mean across 64 validation simulations of '
            '10*log10(w_200^H Rhat_validation,200 w_200); final time t=200 '
            'only, no oracle normalization, and no true covariance in selection.')
    return dict(schema=1, scope=scope, cases=cases, times=times,
        betas=list(BETAS), baseline_beta=baseline_beta, original_config=cfg,
        original_manifest=(source/'manifest.json').as_posix(),
        original_raw=raw.as_posix(), original_selected=(source/'selected.json').as_posix(),
        original_story=(source/'stories.npz').as_posix(),
        validation_criterion=criterion, selection_endpoint='measured_power_db',
        selection_aggregation='all_mean_db', validation_simulations=64,
        final_simulations=256, repetition_unit='Independent original simulation',
        selected_family='Posterior-mean action only',
        tie_break='First beta in the original ordered grid, using minimum mean loss',
        evaluation_aggregations=list(AGGREGATIONS) if scope=='moving' else ['all_mean_db'],
        original_selection=('Shared mode-based mean linear measured output power '
            'at report times t>=32; common beta=.97 for both moving cases.'
            if scope=='moving' else 'Controlled stationary beta=1, unchanged.'),
        information='Known generating sigma2=.2 and tau2=1; uniform Bingham '
            'initial parameter A0=0; identity transport; snapshot-only correction. '
            'Independent holdout covariances are measured observations, not true covariances.',
        no_retuning='Only the primary validation criterion selects beta. All '
            'secondary test aggregations are reported after selection, without reselection.',
        scope_note='Replays of the original observations and trajectories, not new cohorts. '
            'Original methods, settings, numerics, notebooks and exports are unchanged. '
            'This is independent posterior-mean selection under a different criterion; '
            'it is separate from the controlled common-parameter comparison.')


def _bindings(root, cfg):
    paths = list(SOURCES)+[cfg[k] for k in ('original_manifest','original_raw',
                                            'original_selected','original_story')]
    return {name:_sha(root/name) for name in paths}


def prepare_audit(root=ROOT, scope='moving'):
    """Write the criterion and source hashes before inspecting new test scores."""
    root = Path(root).resolve()
    cfg = audit_configuration(root, scope)
    destination = root/DESTINATION/scope/'mean_validation'
    prereg = dict(config=cfg, source_hashes=_bindings(root,cfg))
    path = destination/'preregistered.json'
    if path.exists():
        if json.loads(path.read_text()) != prereg:
            raise ValueError('Audit inputs or source code changed; refusing stale restart')
    else:
        _write_json(path, prereg)
    return cfg, destination, _json_hash(prereg)


def _status(destination, phase, **extra):
    _write_json(destination/'status.json', dict(status='running',phase=phase,**extra))


def _weights(state, a, s, q, cfg):
    """Original checked mean and the same-parameter mode/isotropic actions."""
    posterior, extension = checked_moments(state['A'], cfg['moment_tolerance'],
        cfg['moment_max_order'], cfg['moment_fallback_max_order'])
    if not posterior.numerical_ok or not np.isfinite(posterior.mean).all():
        raise ArithmeticError('Unresolved native posterior mean; no fallback estimator is used')
    P,h,gap = projector_mode(state['A'])
    iso,_ = isotropized_mean(posterior.mean,P)
    n = len(a)
    weights = dict(mean=distortionless_weight(s*np.eye(n)+q*posterior.mean,a),
        mode=modal_weight(h,a,s,q),
        isotropic=distortionless_weight(s*np.eye(n)+q*iso,a))
    diagnostics = dict(mean_order=int(posterior.diagnostics['order']),
        mean_error_bound=float(posterior.diagnostics['moment_error_bound']),
        mean_cap_extension=bool(extension), numerical_ok=True,mean_ok=True)
    if diagnostics['mean_error_bound'] > cfg['moment_tolerance']:
        raise ArithmeticError('Native mean bound exceeds the preserved tolerance')
    for w in weights.values():
        if not np.isfinite(w).all() or abs(np.vdot(w,a)-1)>1e-10:
            raise ArithmeticError('Nonfinite or non-distortionless action')
    return weights, diagnostics


def _test_rows(cfg, case, trial, beta):
    original = cfg['original_config']
    stationary = cfg['scope']=='stationary'
    stream = beam_episode(original,case['n'],case['condition'],trial,'test',stationary)
    states = weighted_statistics(stream['U'],stream['sigma2'],stream['tau2'],beta,cfg['times'])
    rows=[]
    for t in cfg['times']:
        s,q,h=stream['sigma2'][t-1],stream['tau2'][t-1],stream['h'][t-1]
        weights,diagnostics=_weights(states[t],stream['a'],s,q,original)
        reference=oracle_power(stream['a'],h,s,q)
        measured=heldout_snapshots(stream,t,original['heldout_snapshots'])
        for method,w in weights.items():
            true=covariance_power(w,h,s,q)
            observed=sample_output_power(w,measured)
            rows.append(dict(n=case['n'],condition=case['condition'],trial=trial,
                t=t,time_s=t*original['dt'],method=method,beta=beta,
                sigma2=float(s),tau2=float(q),true_power=true,heldout_power=observed,
                oracle_power=reference,true_ratio=true/reference,
                heldout_ratio=observed/reference,true_ratio_db=10*np.log10(true/reference),
                heldout_ratio_db=10*np.log10(observed/reference),
                unit_gain_error=float(abs(np.vdot(w,stream['a'])-1)),**diagnostics))
    return pd.DataFrame(rows)


def _preserved(root,cfg):
    frame=pd.read_csv(root/cfg['original_raw'])
    cases={(x['n'],x['condition']) for x in cfg['cases']}
    selected=frame[[ (int(n),str(c)) in cases for n,c in zip(frame.n,frame.condition) ]]
    selected=selected[selected.t.isin(cfg['times'])].copy()
    expected=256*len(cfg['times'])*10*len(cases)
    if len(selected)!=expected or selected.duplicated(['n','condition','trial','t','method']).any():
        raise ValueError('Incomplete preserved cohort')
    for _,g in selected.groupby(['n','condition','t','method']):
        if set(g.trial)!=set(range(256)) or not g.numerical_ok.all() or not np.isfinite(g.true_ratio_db).all():
            raise ValueError('Unresolved or incomplete preserved simulations')
    common=selected[selected.method.isin(METHODS)]
    if not np.allclose(common.beta,cfg['baseline_beta'],rtol=0,atol=0):
        raise ValueError('The preserved common memory differs from the predeclared baseline')
    return selected


def verify_preserved_replay(root=ROOT,scope='moving'):
    """Prove trial-0 observation/trajectory equality and baseline score replay."""
    root=Path(root).resolve()
    cfg,destination,binding=prepare_audit(root,scope)
    path=destination/'replay_checks.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result.get('binding')!=binding:
            raise ValueError('Stale replay certificate')
        return result
    preserved=_preserved(root,cfg)
    story=np.load(root/cfg['original_story'])
    checks=[]
    started=perf_counter()
    for case in cfg['cases']:
        stream=beam_episode(cfg['original_config'],case['n'],case['condition'],0,
                            'test',scope=='stationary')
        prefix=f"{case['condition']}_n{case['n']}_"
        fields=[]
        for name in ('U','h','a','angles','sigma2','tau2','time'):
            key=prefix+name
            if key in story:
                if not np.array_equal(stream[name],story[key]):
                    raise AssertionError('Original story/seed mismatch: '+key)
                fields.append(name)
        if scope=='moving' and len(fields)!=7:
            raise AssertionError('Missing original moving story evidence')
        replay=_test_rows(cfg,case,0,cfg['baseline_beta'])
        old=preserved[(preserved.n==case['n'])&(preserved.condition==case['condition'])
            &(preserved.trial==0)&preserved.method.isin(METHODS)]
        joined=replay.merge(old,on=['n','condition','trial','t','method'],suffixes=('_new','_old'),validate='one_to_one')
        if len(joined)!=3*len(cfg['times']):
            raise AssertionError('Incomplete baseline replay')
        errors={}
        for metric in ('true_power','heldout_power','oracle_power','true_ratio_db','heldout_ratio_db'):
            a,b=joined[metric+'_new'],joined[metric+'_old']
            if not np.allclose(a,b,rtol=3e-12,atol=3e-12):
                raise AssertionError('Preserved common-beta replay mismatch: '+metric)
            errors[metric]=float(np.max(np.abs(a-b)))
        checks.append(dict(n=case['n'],condition=case['condition'],trial=0,
            beta=cfg['baseline_beta'],story_fields_identical=fields,
            checkpoint_method_rows=len(joined),maximum_absolute_errors=errors))
    result=dict(status='passed',binding=binding,cases=checks,
        seconds=perf_counter()-started,
        scope='Original trial-0 saved observations/trajectory and same-beta actions; '
            'all preserved trial IDs/checkpoint grids also checked. Numerical comparison '
            'uses 3e-12 absolute/relative tolerance to allow CSV round trips.')
    _write_json(path,result)
    return result


def _resume_frame(path,binding,expected_rows):
    meta=path.with_name(path.name+'.json')
    if not path.exists() and not meta.exists():
        return None
    if not path.exists() or not meta.exists():
        # An interrupted atomic pair can be recomputed without accepting stale data.
        return None
    record=json.loads(meta.read_text())
    if record.get('binding')!=binding or record.get('sha256')!=_sha(path):
        raise ValueError('Stale or modified audit trial artifact: '+str(path))
    frame=pd.read_csv(path,float_precision='round_trip')
    if len(frame)!=expected_rows:
        raise ValueError('Incomplete saved trial artifact')
    return frame


def _save_frame(path,frame,binding):
    _write_csv(path,frame)
    _write_json(path.with_name(path.name+'.json'),
                dict(binding=binding,sha256=_sha(path),rows=len(frame)))


def _measured_record(root,cfg,case,trial,path,binding):
    """Save only U, a, known scales, and measured Rhat; no hidden truth."""
    meta=path.with_name(path.name+'.json')
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if record.get('binding')!=binding or record.get('sha256')!=_sha(path):
            raise ValueError('Stale measured validation artifact')
        with np.load(path) as z:
            required={'U','a','sigma2','tau2','times','Rhat'}
            if set(z.files)!=required or not np.array_equal(z['times'],cfg['times']):
                raise ValueError('Unexpected measured selection inputs')
            return dict(U=z['U'],a=z['a'],sigma2=z['sigma2'],tau2=z['tau2'],
                validation_covariances={t:R for t,R in zip(z['times'],z['Rhat'])})
    stream=beam_episode(cfg['original_config'],case['n'],case['condition'],trial,
                        'validation',cfg['scope']=='stationary')
    observed=observed_validation_record(stream,cfg['times'],256)
    del stream
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,U=observed['U'],a=observed['a'],sigma2=observed['sigma2'],
            tau2=observed['tau2'],times=np.array(cfg['times']),
            Rhat=np.array([observed['validation_covariances'][t] for t in cfg['times']]))
    os.replace(temporary,path)
    _write_json(meta,dict(binding=binding,sha256=_sha(path),
        fields=['U','a','sigma2','tau2','times','Rhat'],
        information='Measured finite holdout covariance; no hidden signatures, true '
            'covariance, oracle power, angles or test records supplied to selection.'))
    return {k:observed[k] for k in ('U','a','sigma2','tau2','validation_covariances')}


def validate_mean_memory(root=ROOT,scope='moving',progress=None):
    root=Path(root).resolve()
    cfg,destination,binding=prepare_audit(root,scope)
    verify_preserved_replay(root,scope)
    rows=[]
    for case in cfg['cases']:
        for trial in range(64):
            trial_path=destination/'validation'/case['key']/f'trial_{trial:03d}.csv.gz'
            frame=_resume_frame(trial_path,binding,4*len(cfg['times']))
            observed=_measured_record(root,cfg,case,trial,
                destination/'measured_validation'/case['key']/f'trial_{trial:03d}.npz',binding)
            if frame is None:
                records=[]
                for beta in BETAS:
                    states=weighted_statistics(observed['U'],observed['sigma2'],observed['tau2'],beta,cfg['times'])
                    for t in cfg['times']:
                        weights,diag=_weights(states[t],observed['a'],observed['sigma2'][t-1],
                            observed['tau2'][t-1],cfg['original_config'])
                        power=output_power(weights['mean'],observed['validation_covariances'][t])
                        if not np.isfinite(power) or power<=0:
                            raise ArithmeticError('Unresolved measured validation power')
                        records.append(dict(n=case['n'],condition=case['condition'],
                            trial=trial,t=t,beta=beta,measured_power=float(power),
                            measured_power_db=float(10*np.log10(power)),**diag))
                frame=pd.DataFrame(records)
                _save_frame(trial_path,frame,binding)
            rows.append(frame)
            if trial%8==0:
                _status(destination,'validation',case=case['key'],completed_trial=trial)
                if progress: progress(scope,'validation',case['key'],trial)
    scores=pd.concat(rows,ignore_index=True)
    _write_csv(destination/'validation_scores.csv.gz',scores)
    selection={}
    for case in cfg['cases']:
        part=scores[(scores.n==case['n'])&(scores.condition==case['condition'])]
        per_trial=part.groupby(['beta','trial'],sort=False).measured_power_db.mean()
        means=per_trial.groupby('beta',sort=False).mean()
        selected=min(BETAS,key=lambda x:float(means.loc[x]))
        selection[case['key']]=dict(n=case['n'],condition=case['condition'],
            selected_beta=selected,baseline_beta=cfg['baseline_beta'],
            criterion=cfg['validation_criterion'],scored_times=cfg['times'],
            simulations=64,candidate_mean_loss_db={str(b):float(means.loc[b]) for b in BETAS},
            selected_using='Independent posterior-mean validation, primary criterion only')
    result=dict(binding=binding,selection=selection,scope=scope,
        no_test_outcomes_used=True,validation_scores_sha256=_sha(destination/'validation_scores.csv.gz'))
    path=destination/'selection.json'
    if path.exists() and json.loads(path.read_text())!=result:
        raise ValueError('Restart would change an already frozen selection')
    _write_json(path,result)
    return result


def evaluate_selected_memory(root=ROOT,scope='moving',progress=None):
    root=Path(root).resolve()
    cfg,destination,binding=prepare_audit(root,scope)
    selection_path=destination/'selection.json'
    if not selection_path.exists():
        raise ValueError('Freeze independent measured validation before evaluation')
    selected=json.loads(selection_path.read_text())
    if selected['binding']!=binding:
        raise ValueError('Stale selected policy')
    preserved=_preserved(root,cfg)
    outputs=[]
    for case in cfg['cases']:
        beta=selected['selection'][case['key']]['selected_beta']
        for trial in range(256):
            path=destination/'test'/case['key']/f'trial_{trial:03d}.csv.gz'
            frame=_resume_frame(path,binding,3*len(cfg['times']))
            if frame is None:
                if beta==cfg['baseline_beta']:
                    frame=preserved[(preserved.n==case['n'])&(preserved.condition==case['condition'])
                        &(preserved.trial==trial)&preserved.method.isin(METHODS)].copy()
                    frame['evaluation_source']='Reused preserved baseline-beta checkpoint records'
                else:
                    frame=_test_rows(cfg,case,trial,beta)
                    frame['evaluation_source']='Replayed original observations at independently selected beta'
                _save_frame(path,frame,binding)
            outputs.append(frame)
            if trial%16==0:
                _status(destination,'evaluation',case=case['key'],completed_trial=trial)
                if progress: progress(scope,'evaluation',case['key'],trial)
    result=pd.concat(outputs,ignore_index=True)
    _write_csv(destination/'test_checkpoints.csv.gz',result)
    _write_csv(destination/'preserved_checkpoints.csv.gz',preserved)
    return result


def _interval(values):
    values=np.asarray(values,float)
    if len(values)!=256 or not np.isfinite(values).all():
        raise ValueError('Intervals require all 256 independent finite simulations')
    mean=float(values.mean())
    se=float(values.std(ddof=1)/np.sqrt(len(values)))
    radius=float(student_t.ppf(.975,len(values)-1)*se)
    return dict(mean=mean,lower=mean-radius,upper=mean+radius,
                standard_error=se,simulations=len(values),degrees_of_freedom=len(values)-1)


def _episode_scores(frame,policy,cfg):
    rows=[]
    for (n,condition,trial,method),g in frame.groupby(['n','condition','trial','method'],sort=False):
        if set(g.t)!=set(cfg['times']) or len(g)!=len(cfg['times']):
            raise ValueError('Incomplete report-time grid; no silent checkpoint omission')
        for aggregation in cfg['evaluation_aggregations']:
            part=g[g.t>=32] if aggregation.startswith('burnin') else g
            row=dict(n=int(n),condition=condition,policy=policy,beta=float(g.beta.iloc[0])
                if np.isfinite(g.beta.iloc[0]) else None,trial=int(trial),method=method,
                aggregation=aggregation,checkpoints=len(part))
            for endpoint in ('true','heldout'):
                row[endpoint+'_ratio_db']=float(part[endpoint+'_ratio_db'].mean()
                    if aggregation.endswith('mean_db') else 10*np.log10(part[endpoint+'_ratio'].mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_mean_validation(root=ROOT,scope='moving'):
    root=Path(root).resolve()
    cfg,destination,binding=prepare_audit(root,scope)
    own=pd.read_csv(destination/'test_checkpoints.csv.gz')
    old=_preserved(root,cfg)
    episodes=pd.concat([_episode_scores(old,'preserved_common',cfg),
        _episode_scores(own,'independently_selected',cfg)],ignore_index=True)
    summaries=[]
    for keys,g in episodes.groupby(['n','condition','policy','method','aggregation'],sort=False):
        n,condition,policy,method,aggregation=keys
        if set(g.trial)!=set(range(256)):
            raise ValueError('Missing independent test simulations')
        for endpoint in ('true_ratio_db','heldout_ratio_db'):
            summaries.append(dict(n=int(n),condition=condition,policy=policy,method=method,
                aggregation=aggregation,beta=float(g.beta.iloc[0]) if np.isfinite(g.beta.iloc[0]) else None,
                endpoint=endpoint,**_interval(g[endpoint])))
    paired=[]
    for (n,condition,aggregation),g in episodes.groupby(['n','condition','aggregation'],sort=False):
        for policy in ('preserved_common','independently_selected'):
            mean=g[(g.policy==policy)&(g.method=='mean')].set_index('trial')
            comparisons=[(policy,x,'same_policy') for x in g[g.policy==policy].method.unique() if x!='mean']
            if policy=='independently_selected':
                comparisons += [('preserved_common',x,'selected_mean_vs_preserved')
                    for x in g[g.policy=='preserved_common'].method.unique()]
            for reference_policy,reference,comparison in comparisons:
                other=g[(g.policy==reference_policy)&(g.method==reference)].set_index('trial')
                if set(other.index)!=set(mean.index) or len(other)!=256:
                    raise ValueError('Unpaired simulations')
                for endpoint in ('true_ratio_db','heldout_ratio_db'):
                    interval=_interval(other.loc[mean.index,endpoint]-mean[endpoint])
                    paired.append(dict(n=int(n),condition=condition,aggregation=aggregation,
                        policy=policy,reference_policy=reference_policy,reference=reference,
                        comparison=comparison,endpoint=endpoint,
                        mean_difference=interval.pop('mean'),**interval))
    _write_csv(destination/'episode_scores.csv.gz',episodes)
    _write_csv(destination/'summary.csv',pd.DataFrame(summaries))
    _write_csv(destination/'paired_summary.csv',pd.DataFrame(paired))
    # The production manifests remain immutable; this new audit has its own seal.
    artifacts={}
    for p in sorted(destination.rglob('*')):
        if p.is_file() and p.name not in ('manifest.json','status.json') and not p.name.endswith('.tmp'):
            artifacts[p.relative_to(destination).as_posix()]=dict(sha256=_sha(p),bytes=p.stat().st_size)
    manifest=dict(status='completed',scope=scope,component='independent_mean_validation',
        binding=binding,config=cfg,source_hashes=_bindings(root,cfg),artifacts=artifacts,
        checks=dict(selection_uses_measured_only=True,no_test_retuning=True,
            original_data_unchanged=True,all_test_simulations=256,
            candidate_grid=list(BETAS),numerical_tolerance=1e-10,
            primary_aggregation='all_mean_db',secondary_aggregations_do_not_select_beta=True),
        difference_convention='Comparator minus posterior mean, paired by original trial ID; '
            'positive values favor the mean. Confidence intervals are pointwise Student '
            '95% intervals across 256 independent original simulations, not simultaneous intervals.',
        scope_note=cfg['scope_note'])
    _write_json(destination/'manifest.json',manifest)
    _write_json(destination/'status.json',dict(status='completed',scope=scope))
    return manifest


def load_mean_validation_audit(root=ROOT,scope='moving',verify=True):
    """Data-only API for the new audit notebooks; never executes an experiment."""
    root=Path(root).resolve()
    destination=root/DESTINATION/scope/'mean_validation'
    manifest=json.loads((destination/'manifest.json').read_text())
    if manifest.get('status')!='completed':
        raise ValueError('The independent mean validation audit is incomplete')
    if verify:
        for name,record in manifest['artifacts'].items():
            if _sha(destination/name)!=record['sha256']:
                raise ValueError('Modified audit artifact: '+name)
        for name,digest in manifest['source_hashes'].items():
            if _sha(root/name)!=digest:
                raise ValueError('Audit source changed: '+name)
    return dict(config=manifest['config'],manifest=manifest,
        selection=json.loads((destination/'selection.json').read_text())['selection'],
        validation_scores=pd.read_csv(destination/'validation_scores.csv.gz'),
        test_checkpoints=pd.read_csv(destination/'test_checkpoints.csv.gz'),
        preserved_checkpoints=pd.read_csv(destination/'preserved_checkpoints.csv.gz'),
        episode_scores=pd.read_csv(destination/'episode_scores.csv.gz'),
        summary=pd.read_csv(destination/'summary.csv'),
        paired=pd.read_csv(destination/'paired_summary.csv'))
