"""Read-only independent checks of the separate beamforming comparison audits.

The checker recomputes sample means and Student intervals from per-simulation
records, and checks selected objective diagnostics by replaying saved trial-zero
observations. It creates no observations, selects no settings, times no workload,
and changes no scientific result. Every aggregate contributes one value per trial.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = Path('results/beamforming_comparison_audit')
METHODS = ('mean', 'mode', 'isotropic', 'great_window', 'great_ew')
ALL_METHODS = ('mean', 'mode', 'isotropic', 'modal_loading', 'covariance_loading',
               'great_window', 'great_ew', 'grouse', 'ledoit_wolf', 'oracle')
SPECIFICATIONS = (
    ('mean_db', 'all', 8, 32, True),
    ('mean_db', 't_ge_32', 32, 29, False),
    ('log_mean_ratio', 'all', 8, 32, False),
    ('log_mean_ratio', 't_ge_32', 32, 29, False),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_mapping(root, mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise AssertionError('Missing ' + label + ' mapping')
    for name, record in mapping.items():
        expected = record['sha256'] if isinstance(record, dict) else record
        path = Path(root)/name
        assert path.is_file() and sha(path) == expected, label + ': ' + name
        if isinstance(record, dict) and 'bytes' in record:
            assert path.stat().st_size == record['bytes'], label + ' size: ' + name


def interval(values, simulations=256):
    import numpy as np
    from scipy.stats import t as student_t
    values = np.asarray(values, float)
    assert values.shape == (simulations,) and np.isfinite(values).all()
    mean = float(np.sum(values)/simulations)
    centered = values-mean
    variance = float(centered@centered/(simulations-1))
    half = float(student_t.ppf(.975, simulations-1)*(variance/simulations)**.5)
    return dict(simulations=simulations, mean=mean, lower=mean-half, upper=mean+half)


def summary(frame, keys):
    import numpy as np
    import pandas as pd
    rows = []
    for attributes, group in frame.groupby(keys, sort=True, dropna=False):
        attributes = attributes if isinstance(attributes, tuple) else (attributes,)
        group = group.sort_values('trial')
        np.testing.assert_array_equal(group.trial, np.arange(256))
        rows.append(dict(zip(keys, attributes), **interval(group.value_db.to_numpy())))
    return pd.DataFrame(rows)


def paired(frame, keys, methods):
    import numpy as np
    import pandas as pd
    statistics, differences = [], []
    for attributes, group in frame.groupby(keys, sort=True, dropna=False):
        attributes = attributes if isinstance(attributes, tuple) else (attributes,)
        common = dict(zip(keys, attributes))
        samples = group.pivot(index='trial', columns='method', values='value_db')
        np.testing.assert_array_equal(samples.index, np.arange(256))
        for reference in methods:
            if reference == 'mean':
                continue
            delta = samples[reference].to_numpy()-samples['mean'].to_numpy()
            row = interval(delta)
            statistics.append(dict(common, reference=reference, gain=row['mean'], **row))
            differences.extend(dict(common, reference=reference, trial=trial,
                                    difference_db=float(value)) for trial, value in enumerate(delta))
    return pd.DataFrame(statistics), pd.DataFrame(differences)


def compare_frame(actual, expected, keys, label):
    import pandas as pd
    assert set(actual.columns) == set(expected.columns), label + ': columns differ'
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected[actual.columns], check_dtype=False,
                                  check_exact=False, rtol=2e-11, atol=2e-12,
                                  obj=label)


def _read(folder, name):
    import pandas as pd
    path = folder/(name+'.csv')
    if not path.is_file():
        path = folder/(name+'.csv.gz')
    return pd.read_csv(path, float_precision='round_trip')


def verify_manifest(root, path):
    """Bind the actual producer bytes, preserved inputs and all output bytes."""
    root, path = Path(root), Path(path)
    manifest = json.loads(path.read_text())
    assert manifest.get('status') in ('completed', 'passed', 'verified'), str(path)
    mappings = 0
    for key in ('sources', 'source_hashes', 'inputs', 'input_hashes'):
        if manifest.get(key):
            verify_mapping(root, manifest[key], key)
            mappings += len(manifest[key])
    artifacts = manifest.get('artifacts', manifest.get('files'))
    assert isinstance(artifacts, dict) and artifacts, 'Missing output hashes: ' + str(path)
    for name, record in artifacts.items():
        base = root if Path(name).parts[0] in ('results', 'notebooks', 'lib', 'scripts', 'benchmarks', 'archive') else path.parent
        verify_mapping(base, {name: record}, 'audit artifact')
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path),
                outputs=len(artifacts), source_and_input_hashes=mappings)


def check_existing(root):
    """Recalculate all four aggregation definitions directly from frozen rows."""
    import numpy as np
    import pandas as pd
    from lib.beamforming_moving_evidence import _verified_original
    from lib.final_stationary_evidence import verify_stationary_component
    root = Path(root)
    _verified_original(root)
    verify_stationary_component(root, 'paper')
    folder = root/OUTPUT/'moving/existing'
    meta = verify_manifest(root, folder/'manifest.json')
    raw = pd.read_csv(root/'results/beamforming/data/paper/performance/moving_checkpoints.csv.gz')
    raw = raw[raw.condition.isin(['smooth', 'abrupt']) & raw.n.eq(32)
              & raw.method.isin(METHODS)].copy()
    assert len(raw) == 2*5*32*256
    assert not raw.duplicated(['condition', 'n', 'trial', 't', 'method']).any()
    assert raw.numerical_ok.astype(bool).all() and raw.mean_ok.astype(bool).all()
    np.testing.assert_allclose(raw.true_ratio, raw.true_power/raw.oracle_power,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(raw.true_ratio_db, 10*np.log10(raw.true_ratio),
                               rtol=1e-12, atol=1e-12)
    checkpoint = raw[['condition','n','trial','t','method','true_ratio_db']].rename(columns={'true_ratio_db':'value_db'})
    checkpoint_summary = summary(checkpoint, ['condition','n','t','method'])
    cp, cd = paired(checkpoint, ['condition','n','t'], METHODS)
    compare_frame(_read(folder,'checkpoint_summary'), checkpoint_summary,
                  ['condition','n','t','method'], 'Moving checkpoint means/CIs')
    compare_frame(_read(folder,'paired_checkpoint_summary'), cp,
                  ['condition','n','t','reference'], 'Moving checkpoint paired CIs')
    compare_frame(_read(folder,'paired_checkpoint_differences'), cd,
                  ['condition','n','t','reference','trial'], 'Moving checkpoint pairs')
    aggregates = []
    for aggregate, window, start, count, primary in SPECIFICATIONS:
        for keys, group in raw[raw.t.ge(start)].groupby(['condition','trial','method'],sort=True):
            np.testing.assert_array_equal(np.sort(group.t), np.arange(start,257,8))
            value = (np.sum(group.true_ratio_db.to_numpy())/count if aggregate=='mean_db'
                     else 10*np.log10(np.sum((group.true_power/group.oracle_power).to_numpy())/count))
            aggregates.append(dict(condition=keys[0],trial=int(keys[1]),method=keys[2],
                value_db=float(value),aggregation=aggregate,window=window,start_t=start,
                reporting_points=count,primary=primary))
    aggregates = pd.DataFrame(aggregates)
    keys = ['condition','aggregation','window','start_t','reporting_points','primary']
    means = summary(aggregates, [*keys,'method'])
    pairs, differences = paired(aggregates, keys, METHODS)
    for name, expected, sorting in [
        ('simulation_aggregates',aggregates,['condition','trial','method','aggregation','window']),
        ('aggregate_summary',means,[*keys,'method']),
        ('paired_aggregate_summary',pairs,[*keys,'reference']),
        ('paired_aggregate_differences',differences,[*keys,'reference','trial']),
        ('primary_summary',means[means.primary],[*keys,'method']),
        ('paired_primary_summary',pairs[pairs.primary],[*keys,'reference'])]:
        compare_frame(_read(folder,name), expected, sorting, name)
    # Jensen's inequality is checked per simulation, not just for a grand mean.
    wide = aggregates.pivot(index=['condition','trial','method','window'],
                            columns='aggregation',values='value_db')
    assert (wide.log_mean_ratio-wide.mean_db >= -2e-12).all()
    # The legacy quantity remains separately recoverable, without relabeling it.
    legacy = pd.read_csv(root/'results/beamforming/tables/paper/performance/moving_episodes.csv')
    legacy = legacy[legacy.condition.isin(['smooth','abrupt']) & legacy.n.eq(32)
                    & legacy.method.isin(METHODS)]
    joined = aggregates[aggregates.aggregation.eq('log_mean_ratio') & aggregates.window.eq('t_ge_32')].merge(
        legacy, on=['condition','trial','method'], validate='one_to_one')
    np.testing.assert_allclose(joined.value_db, joined.true_ratio_db, rtol=2e-12, atol=2e-12)
    # Primary old four-method summaries must remain identical with EW added.
    original = pd.read_csv(root/'results/beamforming_moving/tables/time_average_summary.csv')
    original.method = original.method.replace({'iso':'isotropic'})
    current = means[means.primary & means.method.isin(original.method.unique())]
    merged = current.merge(original,on=['condition','method'],validate='one_to_one',suffixes=('_audit','_old'))
    np.testing.assert_allclose(merged[['mean_audit','lower_audit','upper_audit']],
                               merged[['mean_old','lower_old','upper_old']], rtol=2e-12, atol=2e-12)

    stationary_folder = root/OUTPUT/'stationary/existing'
    stationary_meta = verify_manifest(root, stationary_folder/'manifest.json')
    stationary = pd.read_csv(root/'results/beamforming/data/paper/performance/stationary_T200/stationary.csv')
    assert len(stationary) == 3*10*4*256
    assert stationary['T'].eq(200).all() and stationary.condition.eq('static').all()
    assert set(stationary.method) == set(ALL_METHODS)
    np.testing.assert_allclose(stationary.true_ratio_db,
        10*np.log10(stationary.true_power/stationary.oracle_power),rtol=1e-12,atol=1e-12)
    static = stationary[['n','trial','t','method','true_ratio_db']].rename(columns={'true_ratio_db':'value_db'})
    static_means = summary(static,['n','t','method'])
    static_pairs, static_differences = paired(static,['n','t'],ALL_METHODS)
    for name, expected, sorting in [
        ('checkpoint_summary',static_means,['n','t','method']),
        ('paired_checkpoint_summary',static_pairs,['n','t','reference']),
        ('paired_checkpoint_differences',static_differences,['n','t','reference','trial']),
        ('primary_summary',static_means[static_means.t.eq(200)],['n','t','method']),
        ('paired_primary_summary',static_pairs[static_pairs.t.eq(200)],['n','t','reference'])]:
        compare_frame(_read(stationary_folder,name), expected, sorting, 'Stationary '+name)
    return dict(status='passed',moving_manifest=meta,stationary_manifest=stationary_meta,
        moving_rows=len(raw),moving_checkpoint_intervals=len(checkpoint_summary),
        moving_aggregate_intervals=len(means),moving_paired_aggregate_intervals=len(pairs),
        moving_simulation_aggregates=len(aggregates),moving_paired_simulation_differences=len(differences),
        stationary_rows=len(stationary),stationary_checkpoint_intervals=len(static_means),
        stationary_primary_intervals=int(static_means.t.eq(200).sum()),
        stationary_paired_primary_intervals=int(static_pairs.t.eq(200).sum()),
        original_primary_summaries_preserved=True,legacy_secondary_reproduced=True,
        jensen_per_simulation_verified=True,degrees_of_freedom=255,
        primary='Arithmetic mean of individual true_ratio_db over all32 reports within each trial; then trial mean and Student95% CI.')


def preservation(root):
    root = Path(root)
    path = root/OUTPUT/'preserved_files.json'
    proof = json.loads(path.read_text())
    # Presentation-only changes explicitly requested after the audit began are
    # permitted by exact before/after hashes. Scientific inputs remain frozen.
    authorization = root/'results/application_plotting_update/authorized_updates.json'
    if authorization.is_file():
        from lib.presentation_update_provenance import verify_presentation_preservation
        tracked = verify_presentation_preservation(root, proof['tracked_files'])
    else:
        verify_mapping(root, proof['tracked_files'], 'preserved original')
        tracked = dict(status='passed', files=len(proof['tracked_files']))
    local_path = Path('/tmp/beamforming-comparison-preservation-baseline.json')
    local = dict(status='not_present',scope='Local regenerable caches need not exist in a fresh checkout.')
    if local_path.is_file():
        records = json.loads(local_path.read_text())
        if authorization.is_file():
            local = verify_presentation_preservation(root, records)
            local.update(files=len(records),sha256=sha(local_path))
        else:
            verify_mapping(root,records,'preserved local original')
            local = dict(status='passed',files=len(records),sha256=sha(local_path))
    return dict(status='passed',tracked_files=len(proof['tracked_files']),
                proof_sha256=sha(path),tracked=tracked,local=local)


def check_mean_validation(root, scope):
    """Verify independent memory selection and all aggregate-level comparisons."""
    import numpy as np
    import pandas as pd
    root = Path(root)
    folder = root/OUTPUT/scope/'mean_validation'
    manifest_record = verify_manifest(root,folder/'manifest.json')
    manifest = json.loads((folder/'manifest.json').read_text())
    prereg = json.loads((folder/'preregistered.json').read_text())
    binding = hashlib.sha256(json.dumps(prereg,sort_keys=True,separators=(',',':'),
                                       allow_nan=False).encode()).hexdigest()
    assert manifest['binding'] == binding and manifest['config'] == prereg['config']
    config = manifest['config']
    times = list(range(8,257,8)) if scope=='moving' else [200]
    betas = [1.,.99,.97,.9]
    baseline = .97 if scope=='moving' else 1.
    assert config['times']==times and config['betas']==betas
    assert config['baseline_beta']==baseline
    assert config['selection_aggregation']=='all_mean_db'
    assert config['validation_simulations']==64 and config['final_simulations']==256
    frozen = json.loads((folder/'selection.json').read_text())
    assert frozen['binding']==binding and frozen['no_test_outcomes_used'] is True
    validation = _read(folder,'validation_scores')
    assert sha(folder/'validation_scores.csv.gz')==frozen['validation_scores_sha256']
    assert not validation.duplicated(['n','condition','trial','t','beta']).any()
    np.testing.assert_allclose(validation.measured_power_db,10*np.log10(validation.measured_power),
                               rtol=2e-12,atol=2e-12)
    assert validation.numerical_ok.astype(bool).all() and validation.mean_ok.astype(bool).all()
    assert np.isfinite(validation.mean_error_bound).all() and validation.mean_error_bound.le(1e-10).all()
    choices = []
    for case in config['cases']:
        rows=validation[validation.n.eq(case['n'])&validation.condition.eq(case['condition'])]
        assert len(rows)==64*len(times)*4
        losses={}
        for beta in betas:
            candidate=rows[rows.beta.eq(beta)]
            by_trial=[]
            for trial,group in candidate.groupby('trial',sort=True):
                np.testing.assert_array_equal(np.sort(group.t),times)
                by_trial.append(float(group.measured_power_db.sum()/len(times)))
            assert len(by_trial)==64 and set(candidate.trial)==set(range(64))
            stats=interval(by_trial,64)
            losses[beta]=stats['mean']
            choices.append(dict(case=case['key'],beta=beta,**stats))
        chosen=min(betas,key=losses.get)
        recorded=frozen['selection'][case['key']]
        assert recorded['selected_beta']==chosen and recorded['scored_times']==times
        assert recorded['simulations']==64 and recorded['baseline_beta']==baseline
        for beta in betas:
            np.testing.assert_allclose(recorded['candidate_mean_loss_db'][str(beta)],losses[beta],
                                       rtol=2e-12,atol=2e-12)

    selected = _read(folder,'test_checkpoints')
    preserved = _read(folder,'preserved_checkpoints')
    assert len(selected)==256*len(times)*3*len(config['cases'])
    assert set(selected.method)=={'mean','mode','isotropic'}
    for frame in (selected,preserved):
        assert not frame.duplicated(['n','condition','trial','t','method']).any()
        assert frame.numerical_ok.astype(bool).all() and frame.mean_ok.astype(bool).all()
        for endpoint,power in [('true','true_power'),('heldout','heldout_power')]:
            np.testing.assert_allclose(frame[endpoint+'_ratio'],frame[power]/frame.oracle_power,
                                       rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(frame[endpoint+'_ratio_db'],10*np.log10(frame[endpoint+'_ratio']),
                                       rtol=2e-12,atol=2e-12)
    original=pd.read_csv(root/config['original_raw'],float_precision='round_trip')
    selected_cases={(case['n'],case['condition']) for case in config['cases']}
    original=original[[ (n,c) in selected_cases for n,c in zip(original.n,original.condition) ]]
    original=original[original.t.isin(times)]
    join=preserved.merge(original,on=['n','condition','trial','t','method'],validate='one_to_one',suffixes=('_audit','_old'))
    assert len(join)==len(preserved)==len(original)
    for metric in ['true_power','heldout_power','oracle_power','true_ratio_db','heldout_ratio_db']:
        np.testing.assert_allclose(join[metric+'_audit'],join[metric+'_old'],rtol=2e-12,atol=2e-12)
    reuse=[]
    for case in config['cases']:
        beta=frozen['selection'][case['key']]['selected_beta']
        part=selected[selected.n.eq(case['n'])&selected.condition.eq(case['condition'])]
        assert part.beta.eq(beta).all()
        if beta==baseline:
            old=preserved[preserved.n.eq(case['n'])&preserved.condition.eq(case['condition'])&
                          preserved.method.isin(['mean','mode','isotropic'])]
            joined=part.merge(old,on=['n','condition','trial','t','method'],validate='one_to_one',suffixes=('_selected','_old'))
            assert len(joined)==len(part)
            errors=[]
            for metric in ['true_power','heldout_power','oracle_power','true_ratio_db','heldout_ratio_db']:
                error=float(np.max(np.abs(joined[metric+'_selected']-joined[metric+'_old'])))
                assert error<3e-12
                errors.append(error)
            reuse.append(dict(case=case['key'],beta=beta,numerical_tie=True,
                              maximum_copy_error=max(errors)))
    episodes=[]
    for policy,frame in [('preserved_common',preserved),('independently_selected',selected)]:
        for (n,condition,trial,method),group in frame.groupby(['n','condition','trial','method'],sort=False):
            np.testing.assert_array_equal(np.sort(group.t),times)
            for aggregate in config['evaluation_aggregations']:
                part=group[group.t.ge(32)] if aggregate.startswith('burnin') else group
                beta=group.beta.iloc[0]
                record=dict(n=int(n),condition=condition,policy=policy,
                    beta=float(beta) if np.isfinite(beta) else None,trial=int(trial),
                    method=method,aggregation=aggregate,checkpoints=len(part))
                for endpoint in ['true','heldout']:
                    record[endpoint+'_ratio_db']=float(part[endpoint+'_ratio_db'].mean()
                        if aggregate.endswith('mean_db') else 10*np.log10(part[endpoint+'_ratio'].mean()))
                episodes.append(record)
    episodes=pd.DataFrame(episodes)
    compare_frame(_read(folder,'episode_scores'),episodes,
                  ['n','condition','policy','trial','method','aggregation'],scope+' memory-selection trial aggregates')
    results=[];pairs=[]
    for attributes,group in episodes.groupby(['n','condition','policy','method','aggregation'],sort=False):
        n,condition,policy,method,aggregation=attributes
        np.testing.assert_array_equal(np.sort(group.trial),np.arange(256))
        beta=group.beta.iloc[0]
        for endpoint in ['true_ratio_db','heldout_ratio_db']:
            stats=interval(group[endpoint])
            stats.update(standard_error=float(group[endpoint].std(ddof=1)/16),degrees_of_freedom=255)
            results.append(dict(n=int(n),condition=condition,policy=policy,method=method,
                aggregation=aggregation,beta=float(beta) if np.isfinite(beta) else None,endpoint=endpoint,**stats))
    for (n,condition,aggregation),group in episodes.groupby(['n','condition','aggregation'],sort=False):
        for policy in ['preserved_common','independently_selected']:
            own=group[group.policy.eq(policy)&group.method.eq('mean')].set_index('trial')
            references=[(policy,m,'same_policy') for m in group[group.policy.eq(policy)].method.unique() if m!='mean']
            if policy=='independently_selected':
                references += [('preserved_common',m,'selected_mean_vs_preserved') for m in group[group.policy.eq('preserved_common')].method.unique()]
            for reference_policy,reference,comparison in references:
                other=group[group.policy.eq(reference_policy)&group.method.eq(reference)].set_index('trial')
                np.testing.assert_array_equal(np.sort(other.index),np.arange(256))
                for endpoint in ['true_ratio_db','heldout_ratio_db']:
                    difference=other.loc[own.index,endpoint]-own[endpoint]
                    stats=interval(difference)
                    value=stats.pop('mean')
                    stats.update(standard_error=float(difference.std(ddof=1)/16),degrees_of_freedom=255)
                    pairs.append(dict(n=int(n),condition=condition,aggregation=aggregation,
                        policy=policy,reference_policy=reference_policy,reference=reference,
                        comparison=comparison,endpoint=endpoint,mean_difference=value,**stats))
    compare_frame(_read(folder,'summary'),pd.DataFrame(results),
        ['n','condition','policy','method','aggregation','endpoint'],scope+' independent-memory means/CIs')
    compare_frame(_read(folder,'paired_summary'),pd.DataFrame(pairs),
        ['n','condition','aggregation','policy','reference_policy','reference','comparison','endpoint'],scope+' independent-memory paired CIs')
    return dict(status='passed',manifest=manifest_record,validation_rows=len(validation),
        validation_candidate_intervals=choices,test_checkpoint_rows=len(selected),
        aggregate_intervals=len(results),paired_intervals=len(pairs),
        same_beta_reuse=reuse,selection_uses_one_primary_aggregate_per_validation_trial=True,
        no_oracle_used_for_selection=True,secondary_criteria_do_not_select_settings=True,
        replay_scope='Production sources and all immutable original rows verified; baseline numerical replay uses saved simulation0, and selected same-beta records reuse every original simulation.')


def check_stationary_selection(root):
    """Check final-time measured selection, full history, and paired test scores."""
    import numpy as np
    import pandas as pd
    root=Path(root);folder=root/OUTPUT/'stationary/selection'
    manifest_record=verify_manifest(root,folder/'manifest.json')
    meta=json.loads((folder/'manifest.json').read_text());config=meta['config']
    assert config['T']==config['t']==200 and config['n']==[32,64,128]
    assert config['beta']==1. and config['ew_beta']==1.
    assert config['validation_simulations']==64 and config['test_simulations']==256
    assert config['step_multipliers']==[.02,.08,.24] and config['inner_steps']==[1,3]
    assert config['covariance_loading_grid_points']==33
    freeze=json.loads((folder/'selection_freeze.json').read_text())
    assert freeze['status']=='selected_before_test_scoring'
    assert freeze['source_signature']==meta['source_signature']
    for key,name in [('selected_settings_sha256','selected_settings.json'),
                     ('validation_trials_sha256','validation_trials.csv'),
                     ('validation_candidates_sha256','validation_candidates.csv')]:
        assert freeze[key]==sha(folder/name)
    choices=json.loads((folder/'selected_settings.json').read_text())
    trials=_read(folder,'validation_trials');candidates=_read(folder,'validation_candidates')
    assert trials.t.eq(200).all() and trials['T'].eq(200).all()
    assert trials.numerical_ok.astype(bool).all()
    assert not trials.duplicated(['n','candidate_id','trial']).any()
    np.testing.assert_allclose(trials.measured_log_power_db,10*np.log10(trials.measured_power),
                               rtol=2e-12,atol=2e-12)
    candidate_intervals=[]
    for (n,identifier),group in trials.groupby(['n','candidate_id'],sort=True):
        np.testing.assert_array_equal(np.sort(group.trial),np.arange(64))
        stats=interval(group.measured_log_power_db,64)
        row=candidates[candidates.n.eq(n)&candidates.candidate_id.eq(identifier)]
        assert len(row)==1
        np.testing.assert_allclose(row[['mean','lower','upper']].iloc[0],
                                   [stats['mean'],stats['lower'],stats['upper']],
                                   rtol=2e-12,atol=2e-12)
        assert row.simulations.iloc[0]==64
        candidate_intervals.append(dict(n=int(n),candidate_id=identifier,**stats))
    covariance_checks=0;selected=[]
    for n in [32,64,128]:
        rows=candidates[candidates.n.eq(n)]
        window=rows[rows.family.eq('great_window')]
        ew=rows[rows.family.eq('great_ew')]
        assert set(window.window.astype(int))=={n//2,n,2*n,200}
        assert len(window)==24 and len(ew)==6
        for record in window.itertuples():
            assert record.inner_steps in (1,3) and record.step_multiplier in (.02,.08,.24)
            np.testing.assert_allclose(record.stepsize,record.step_multiplier/(record.window*1.2),rtol=2e-12)
        for record in ew.itertuples():
            assert record.beta==1. and pd.isna(record.window)
            np.testing.assert_allclose(record.stepsize,record.step_multiplier/(2*n*1.2),rtol=2e-12)
        settings=choices[str(n)]['settings']
        for method,family,restriction in [('great_window_final_db','great_window',None),
                ('great_ew_final_db','great_ew',None),
                ('great_full_history_final_db','great_window',200)]:
            candidate=settings[method]
            eligible=rows[rows.family.eq(family)]
            if restriction is not None:
                eligible=eligible[eligible.window.eq(restriction)]
            chosen=eligible[eligible.candidate_id.eq(candidate['candidate_id'])]
            assert len(chosen)==1
            assert candidate['selection_t']==200 and candidate['validation_simulations']==64
            assert chosen['mean'].iloc[0]<=eligible['mean'].min()+2e-12
            np.testing.assert_allclose(candidate['validation_mean_db'],chosen['mean'].iloc[0],rtol=2e-12,atol=2e-12)
            for key in ['stepsize','step_multiplier','inner_steps']:
                np.testing.assert_allclose(candidate[key],chosen[key].iloc[0],rtol=2e-12,atol=2e-12)
            if restriction is not None:
                assert candidate['window']==200
            selected.append(dict(n=n,method=method,candidate_id=candidate['candidate_id'],
                                 validation_mean_db=candidate['validation_mean_db']))
        # Independently evaluate the scalar covariance-loading formula from
        # the saved measured spectral statistics, without a true covariance.
        loading=rows[rows.family.eq('covariance_loading')]
        assert len(loading)==34
        alpha=settings['covariance_loading_final_db']['alpha']
        assert 0<=alpha<=1
        chosen=loading[loading.candidate_id.eq('covariance_loading_selected')]
        assert len(chosen)==1
        assert chosen['mean'].iloc[0]<=loading['mean'].min()+2e-12
        for trial in range(64):
            with np.load(folder/'checkpoints/validation'/f'n{n}_{trial:03d}.npz',allow_pickle=False) as data:
                eigenvalues=data['values'];p=data['p'];quadratic=data['quadratic']
            for record in loading.itertuples():
                diagonal=(1-record.alpha)*eigenvalues+record.alpha*eigenvalues.mean()
                inverse=1/diagonal
                power=float(inverse@quadratic@inverse/(p@inverse)**2)
                saved=trials[trials.n.eq(n)&trials.trial.eq(trial)&trials.candidate_id.eq(record.candidate_id)]
                assert len(saved)==1
                np.testing.assert_allclose(saved.measured_power.iloc[0],power,rtol=3e-12,atol=3e-12)
                covariance_checks+=1
        np.testing.assert_allclose(alpha,chosen.alpha.iloc[0],rtol=0,atol=2e-15)
    frame=_read(folder,'test_metrics')
    assert len(frame)==3*14*256 and frame.t.eq(200).all() and frame['T'].eq(200).all()
    assert not frame.duplicated(['n','trial','method']).any() and frame.numerical_ok.astype(bool).all()
    assert frame.unit_gain_error.le(1e-10).all()
    assert set(frame.method)==set(ALL_METHODS)|{'great_window_final_db','great_ew_final_db',
        'great_full_history_final_db','covariance_loading_final_db'}
    np.testing.assert_allclose(frame.true_ratio_db,10*np.log10(frame.true_power/frame.oracle_power),
                               rtol=2e-12,atol=2e-12)
    old=pd.read_csv(root/'results/beamforming/data/paper/performance/stationary_T200/stationary.csv')
    old=old[old.t.eq(200)]
    preserved=frame[frame.source.eq('saved_original')]
    joined=preserved.merge(old,on=['n','trial','method'],validate='one_to_one',suffixes=('_audit','_old'))
    assert len(joined)==len(old)==len(preserved)
    for metric in ['true_power','oracle_power','true_ratio_db']:
        np.testing.assert_allclose(joined[metric+'_audit'],joined[metric+'_old'],rtol=2e-12,atol=2e-12)
    means=[];pairs=[]
    for n,group in frame.groupby('n',sort=True):
        wide=group.pivot(index='trial',columns='method',values='true_ratio_db')
        np.testing.assert_array_equal(wide.index,np.arange(256))
        for method in wide:
            means.append(dict(n=int(n),t=200,method=method,**interval(wide[method])))
            if method!='mean':
                pairs.append(dict(n=int(n),t=200,reference=method,**interval(wide[method]-wide['mean']),
                    sign='Reference minus original posterior mean; positive favors posterior mean'))
    compare_frame(_read(folder,'test_summary'),pd.DataFrame(means),['n','t','method'],'Final-time selection means/CIs')
    compare_frame(_read(folder,'paired_vs_mean'),pd.DataFrame(pairs),['n','t','reference'],'Final-time selection paired CIs')
    fingerprints=_read(folder,'replay_fingerprints')
    validation=fingerprints[fingerprints.split.eq('validation')]
    final=fingerprints[fingerprints.split.eq('test')]
    assert len(validation)==192 and len(final)==768
    assert not set(validation.observation_sha256)&set(final.observation_sha256)
    for field in ['mode_reference_db_error','great_window_reference_db_error',
                  'great_ew_reference_db_error','covariance_loading_reference_db_error']:
        assert final[field].max()<1e-10
    return dict(status='passed',manifest=manifest_record,validation_records=len(trials),
        validation_candidate_intervals=len(candidate_intervals),selected_GREAT=selected,
        measured_covariance_loading_powers_reconstructed=covariance_checks,
        test_records=len(frame),test_intervals=len(means),paired_test_intervals=len(pairs),
        full_history_window=200,selection_uses_only_final_measured_db=True,
        original_methods_unchanged=10,disjoint_validation_test_observation_hashes=True)


def _objective_spectral_replay(root, scope, frame, config, settings):
    """Recheck saved trial-zero bases on actual saved observations, not new data."""
    import numpy as np
    from benchmarks.common.great import GREAT
    from lib.beamforming_protocol import haar_line
    from lib.final_beamforming import CONDITIONS
    root=Path(root)
    story_path=(root/'results/beamforming/data/paper/performance/stories.npz' if scope=='moving'
                else root/'results/beamforming/data/paper/performance/stationary_T200/stories.npz')
    cases=config['cases'] if scope=='moving' else [case for case in config['cases'] if case['n']==64]
    checks=[]
    with np.load(story_path,allow_pickle=False) as story:
        for case in cases:
            condition,n=case['condition'],int(case['n'])
            prefix=f'{condition}_n{n}_'
            U=story[prefix+'U'];sigma=story[prefix+'sigma2'];tau=story[prefix+'tau2']
            eta=tau/(sigma*(sigma+tau));W=U*np.sqrt(eta/eta[0])[:,None]
            words=np.random.SeedSequence([config['test_seed'],n,CONDITIONS.index(condition),0]).generate_state(4).tolist()
            child=np.random.SeedSequence(words).spawn(3)[1]
            initial=haar_line(np.random.default_rng(child),n)
            counts=sorted({3,int(settings[case['case']]['selected_inner_steps'])})
            step=config['original_fixed_stepsizes'][case['case']]
            trackers={count:GREAT(n,1,complex_data=True,initial_basis=initial[:,None],
                beta=case['beta'],window=None,stepsize=step,inner_steps=count) for count in counts}
            A=np.zeros((n,n),complex);maximum_gap_error=maximum_distance_error=maximum_covariance_error=0.
            rows=frame[frame.condition.eq(condition)&frame.n.eq(n)&frame.trial.eq(0)]
            for t,(u,w) in enumerate(zip(U,W),1):
                bases={count:tracker.update(w)[:,0] for count,tracker in trackers.items()}
                A=case['beta']*A+eta[t-1]*np.outer(u,u.conj())
                if t not in case['times']:
                    continue
                C=next(iter(trackers.values())).covariance
                maximum_covariance_error=max(maximum_covariance_error,float(np.linalg.norm(C-A/eta[0])/np.linalg.norm(C)))
                values,E=np.linalg.eigh(C);mode=E[:,-1]
                for count,x in bases.items():
                    saved=rows[rows.t.eq(t)&rows.method.eq('great_ew')&rows.inner_steps.eq(count)]
                    assert len(saved)==1
                    gap=float(max(0.,values[-1]-np.vdot(x,C@x).real))
                    squared=float(max(0.,1-abs(np.vdot(mode,x))**2))
                    distance=squared**.5
                    np.testing.assert_allclose(saved.objective_gap.iloc[0],gap,rtol=2e-10,atol=3e-11)
                    np.testing.assert_allclose(saved.chordal_to_mode.iloc[0],distance,rtol=2e-10,atol=3e-8)
                    np.testing.assert_allclose(saved.chordal_squared_to_mode.iloc[0],squared,rtol=2e-10,atol=3e-12)
                    maximum_gap_error=max(maximum_gap_error,abs(float(saved.objective_gap.iloc[0])-gap))
                    maximum_distance_error=max(maximum_distance_error,abs(float(saved.chordal_to_mode.iloc[0])-distance))
            checks.append(dict(condition=condition,n=n,simulation=0,reporting_times=case['times'],
                independent_saved_observation_replay=True,maximum_gap_absolute_error=maximum_gap_error,
                maximum_chordal_absolute_error=maximum_distance_error,
                maximum_covariance_match_relative_error=maximum_covariance_error))
    return checks


def check_objective(root, scope, require_timing=True):
    """Check matching covariance, objective gaps, validation and convergence cost."""
    import numpy as np
    import pandas as pd
    root=Path(root);folder=root/OUTPUT/scope/'objective'
    manifest_record=verify_manifest(root,folder/'manifest.json')
    meta=json.loads((folder/'manifest.json').read_text());config=meta['config']
    prereg=json.loads((folder/'preregistered.json').read_text())
    assert prereg==config
    declaration=hashlib.sha256(json.dumps(prereg,sort_keys=True,allow_nan=False).encode()).hexdigest()
    assert declaration==meta['declaration_sha256']
    assert config['inner_steps']==[1,3,10,30] and config['original_inner_steps']==3
    assert config['validation_simulations']==64 and config['test_simulations']==256
    selected=json.loads((folder/'selected.json').read_text())
    validation=_read(folder,'validation_checkpoints');test=_read(folder,'test_checkpoints')
    for split,frame,count in [('validation',validation,64),('test',test,256)]:
        assert frame.split.eq(split).all()
        assert not frame.duplicated(['condition','n','trial','t','method','inner_steps']).any()
        required=['objective','objective_gap','relative_objective_gap','chordal_to_mode',
            'chordal_squared_to_mode','stationarity_norm','measured_power','measured_power_db',
            'lambda_max','lambda_min','eigengap','trace_covariance','exact_objective',
            'statistics_match_relative_error']
        assert np.isfinite(frame[required].to_numpy()).all()
        assert frame.measured_power.gt(0).all()
        np.testing.assert_allclose(frame.measured_power_db,10*np.log10(frame.measured_power),rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(frame.gamma,np.sqrt(frame.beta),rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(frame.eta0,1/(.2*1.2),rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(frame.exact_objective,frame.trace_covariance-frame.lambda_max,rtol=2e-12,atol=2e-11)
        np.testing.assert_allclose(frame.objective-frame.exact_objective,frame.objective_gap,rtol=2e-10,atol=3e-10)
        np.testing.assert_allclose(frame.relative_objective_gap,
            frame.objective_gap/(frame.lambda_max-frame.lambda_min),rtol=2e-10,atol=3e-12)
        np.testing.assert_allclose(frame.chordal_squared_to_mode,frame.chordal_to_mode**2,rtol=2e-12,atol=2e-12)
        assert frame.chordal_to_mode.between(0,1+1e-12).all()
        assert (frame.objective_gap-frame.eigengap*frame.chordal_squared_to_mode >=
                -3e-10*np.maximum(frame.lambda_max,1.)).all()
        assert frame.statistics_match_relative_error.max()<2e-12
        if split=='validation':
            assert frame.true_ratio_db.isna().all() and frame.oracle_power.isna().all()
        else:
            assert np.isfinite(frame.true_ratio_db).all() and frame.oracle_power.gt(0).all()
        for case in config['cases']:
            part=frame[frame.n.eq(case['n'])&frame.condition.eq(case['condition'])]
            counts=([1,3,10,30] if split=='validation' else sorted({3,selected[case['case']]['selected_inner_steps']}))
            assert set(part.inner_steps)=={0,*counts}
            assert len(part)==count*len(case['times'])*(len(counts)+1)
            assert part.beta.eq(case['beta']).all()
            for (method,inner,trial),rows in part.groupby(['method','inner_steps','trial']):
                np.testing.assert_array_equal(np.sort(rows.t),case['times'])
                assert method==('mode_exact' if inner==0 else 'great_ew')
            for (method,inner,t),rows in part.groupby(['method','inner_steps','t']):
                np.testing.assert_array_equal(np.sort(rows.trial),np.arange(count))
            ew=part[part.method.eq('great_ew')]
            np.testing.assert_allclose(ew.stepsize,config['original_fixed_stepsizes'][case['case']],rtol=2e-14,atol=2e-14)
    for case in config['cases']:
        rows=validation[validation.n.eq(case['n'])&validation.condition.eq(case['condition'])&validation.method.eq('great_ew')]
        times=[200] if scope=='stationary' else case['times']
        part=rows[rows.t.isin(times)]
        losses=part.groupby(['inner_steps','trial']).measured_power_db.mean().groupby('inner_steps').mean()
        policy=selected[case['case']]
        assert policy['validation_selection_times']==times
        assert policy['beta']==case['beta'] and policy['original_inner_steps']==3
        assert losses.loc[policy['selected_inner_steps']]<=losses.min()+2e-12
        for inner in [1,3,10,30]:
            np.testing.assert_allclose(policy['measured_validation_db_by_inner_steps'][str(inner)],
                                       losses.loc[inner],rtol=2e-12,atol=2e-12)
    raw_path=(root/'results/beamforming/data/paper/performance/moving_checkpoints.csv.gz' if scope=='moving'
              else root/'results/beamforming/data/paper/performance/stationary_T200/stationary.csv')
    old=pd.read_csv(raw_path)
    for method,old_method,inner in [('mode_exact','mode',0),('great_ew','great_ew',3)]:
        rows=test[test.method.eq(method)&test.inner_steps.eq(inner)]
        reference=old[old.method.eq(old_method)]
        joined=rows.merge(reference,on=['condition','n','trial','t'],validate='one_to_one',suffixes=('_audit','_old'))
        assert len(joined)==len(rows)
        np.testing.assert_allclose(joined.true_ratio_db_audit,joined.true_ratio_db_old,rtol=3e-10,atol=3e-9)
    metrics=['objective_gap','relative_objective_gap','chordal_to_mode','chordal_squared_to_mode',
             'stationarity_norm','measured_power_db','true_ratio_db']
    summary_count={}
    for split,frame,count in [('validation',validation,64),('test',test,256)]:
        records=[]
        for attributes,rows in frame.groupby(['condition','n','t','method','inner_steps'],sort=True):
            common=dict(zip(['condition','n','t','method','inner_steps'],attributes))
            for metric in metrics:
                if split=='validation' and metric=='true_ratio_db':
                    continue
                stats=interval(rows.sort_values('trial')[metric],count)
                records.append(dict(common,metric=metric,**stats))
        compare_frame(_read(folder,split+'_summary'),pd.DataFrame(records),
            ['condition','n','t','method','inner_steps','metric'],scope+' EW '+split+' diagnostic intervals')
        summary_count[split]=len(records)
    pair_rows=[]
    primary=test[test.t.eq(200)] if scope=='stationary' else test
    means=primary.groupby(['condition','n','trial','method','inner_steps']).true_ratio_db.mean().reset_index()
    for (condition,n),rows in means.groupby(['condition','n'],sort=True):
        mode=rows[rows.method.eq('mode_exact')].set_index('trial').true_ratio_db
        for inner,group in rows[rows.method.eq('great_ew')].groupby('inner_steps',sort=True):
            own=group.set_index('trial').true_ratio_db
            pair_rows.append(dict(condition=condition,n=n,method='great_ew',inner_steps=int(inner),
                reference='mode_exact',metric='final_t200_true_ratio_db_difference' if scope=='stationary'
                    else 'time_average_true_ratio_db_difference',**interval(own.loc[mode.index]-mode)))
    compare_frame(_read(folder,'paired_summary'),pd.DataFrame(pair_rows),
        ['condition','n','method','inner_steps','reference','metric'],scope+' EW power differences to exact mode')
    operations=_read(folder,'operation_counts')
    workloads=0
    for case in config['cases']:
        part=operations[operations.n.eq(case['n'])&operations.condition.eq(case['condition'])]
        counts={0,3,int(selected[case['case']]['selected_inner_steps'])}
        assert set(part.inner_steps)==counts and len(part)==len(counts)
        workloads+=len(counts)
        for _,row in part.iterrows():
            assert row.measurements==row.covariance_updates==case['T']
            assert row.task_decisions==len(case['times'])
            assert row.grassmann_steps==case['T']*row.inner_steps
            if row.inner_steps:
                assert row.method=='great_ew' and row.tracker_eigendecompositions==0
                assert row.diagnostic_eigendecompositions==len(case['times'])
            else:
                assert row.method=='mode_exact' and row.diagnostic_eigendecompositions==0
                assert row.tracker_eigendecompositions==len(case['times'])
    spectral=_objective_spectral_replay(root,scope,test,config,selected)
    report=dict(status='passed',manifest=manifest_record,validation_rows=len(validation),test_rows=len(test),
        diagnostic_intervals=summary_count,paired_power_intervals=len(pair_rows),
        selection=selected,gamma_equals_sqrt_beta=True,covariance_matches_Bingham=True,
        chordal_distance_and_squared_distance_distinguished=True,
        rayleigh_gap_chordal_bound_verified=True,independent_saved_observation_replay=spectral,
        operation_count_workloads=workloads,diagnostic_eigensolves_excluded_from_timings=True,
        sign='EW GREAT minus exact mode; positive power difference favors the exact mode.')
    if require_timing:
        report['timing']=check_timing(root,scope,config,selected)
    return report


def check_timing(root,scope,config,selected):
    """Check isolated computational repetitions without treating them as trials."""
    import numpy as np
    import pandas as pd
    root=Path(root);folder=root/OUTPUT/scope/'objective/timing'
    meta_record=verify_manifest(root,folder/'manifest.json')
    meta=json.loads((folder/'manifest.json').read_text());protocol=meta['protocol']
    assert protocol['isolation_confirmed'] is True and protocol['blas_threads']==1
    assert protocol['warmups']==1 and protocol['repetitions']==7
    assert protocol['objective_manifest_sha256']==sha(folder.parent/'manifest.json')
    raw=_read(folder,'raw');summary=_read(folder,'summary')
    assert raw.total_ms.gt(0).all()
    phases=[name for name in ['tracking_ms','decision_ms','overhead_ms'] if name in raw]
    assert {'tracking_ms','decision_ms'}<=set(phases)
    assert np.isfinite(raw[phases].to_numpy()).all() and raw[phases].ge(0).all().all()
    np.testing.assert_allclose(raw[phases].sum(axis=1),raw.total_ms,rtol=2e-10,atol=2e-9)
    for case in config['cases']:
        rows=raw[raw.condition.eq(case['condition'])&raw.n.eq(case['n'])]
        counts={0,3,int(selected[case['case']]['selected_inner_steps'])}
        assert set(rows.inner_steps)==counts and len(rows)==7*len(counts)
        assert rows['T'].eq(case['T']).all()
        for count,group in rows.groupby('inner_steps'):
            np.testing.assert_array_equal(np.sort(group.repetition),np.arange(7))
            assert set(group.method)=={'mode_exact' if count==0 else 'great_ew'}
    expected=raw.groupby(['condition','n','T','method','inner_steps']).total_ms.agg(
        repetitions='size',median_ms='median',p90_ms=lambda x:x.quantile(.9)).reset_index()
    compare_frame(summary,expected,['condition','n','T','method','inner_steps'],scope+' isolated timings')
    return dict(status='passed',manifest=meta_record,repetitions_per_workload=7,
        raw_repetitions=len(raw),workloads=len(expected),scope=protocol,
        medians=expected.to_dict('records'))


def check_notebook_pdfs(root):
    """Read PDF text/fonts/bounds; this function never executes a notebook."""
    from scripts.check_manuscript_plot_exports import audit_pdf,figure_declarations
    from PIL import Image
    root=Path(root);records={}
    for scope in ('moving','stationary'):
        folder=root/OUTPUT/scope
        manifest=json.loads((folder/'figure_manifest.json').read_text())
        notebook=json.loads((root/manifest['notebook']).read_text())
        declarations={}
        for cell in notebook['cells']:
            if cell['cell_type']!='code':
                continue
            source=cell['source'];source=''.join(source) if isinstance(source,list) else source
            for item in figure_declarations(source):
                assert item['filename'] not in declarations
                declarations[item['filename']]=item
        assert set(declarations)==set(manifest['figures'])
        figures={}
        for filename,item in manifest['figures'].items():
            source=declarations[filename]
            assert source['figure_size_in']==[15,5] and source['dpi']==300.
            assert source['primary_font_sizes_pt']==dict(LABEL_FONT_SIZE=25.,TICK_FONT_SIZE=23.,LEGEND_FONT_SIZE=22.)
            record=audit_pdf(root/item['pdf'],[[15,5]],expected_dpi=300.,
                primary_fonts=source['primary_font_sizes_pt'],main_panel_policy=[],require_primary_sizes=True)
            assert not record['errors'], item['pdf']+': '+str(record['errors'])
            with Image.open(root/item['png']) as image:
                assert image.size==(4500,1500) and image.format=='PNG'
                resolution=image.info.get('dpi')
                assert resolution and all(abs(value-300.)<.02 for value in resolution)
                preview=dict(pixels=list(image.size),dpi=list(resolution))
            record['preview']=preview
            figures[filename]=record
        records[scope]=dict(status='passed',figures=figures,count=len(figures))
    return dict(status='passed',scopes=records,
        scope='Actual PDF dimensions, embedded vector LaTeX fonts, primary type size, page bounds, '
              'panel letters and preview pixels/dpi; no simulation, estimator or plotting is executed.')


def _pdf_audit_with_available_python(root):
    """The system Python may provide PyMuPDF while the science venv does not."""
    import importlib.util
    import shutil
    import subprocess
    if importlib.util.find_spec('pymupdf') is not None or importlib.util.find_spec('fitz') is not None:
        return check_notebook_pdfs(root)
    candidates=[shutil.which('python'),'/usr/local/bin/python','/usr/bin/python3']
    errors=[];seen={sys.executable}
    for candidate in candidates:
        # Do not resolve symlinks: the venv and system interpreter use different
        # package paths even when they ultimately share an executable binary.
        if not candidate or candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        process=subprocess.run([candidate,str(Path(__file__).absolute()),'--root',str(root),'--pdf-only'],
            cwd=root,text=True,capture_output=True,check=False)
        if process.returncode==0:
            return json.loads(process.stdout)
        errors.append(dict(python=candidate,error=process.stderr[-3000:]))
    raise AssertionError('No available interpreter completed the read-only PDF audit: '+str(errors))


def check_notebooks(root):
    """Check visible plotting code, complete execution, hashes and untiled exports."""
    from lib.beamforming_comparison_summaries import load_existing_beamforming_comparisons
    root=Path(root);records={}
    # The moving raw cohort predates fixed-T work. Its recorded scientific
    # producer can legitimately be an archived source, verified by this loader;
    # its hash must not be compared to a later file at the old producer path.
    original_evidence=load_existing_beamforming_comparisons(root)
    expected_figures={
        'moving':{'smooth_history','abrupt_history','paired_primary','posterior_mean_beta',
                  'ew_objective_gap','ew_task_power','computation_time'},
        'stationary':{'original_settings','final_time_selection','paired_final_time','posterior_mean_beta',
                      'ew_objective_gap','ew_task_power','computation_time'},
    }
    for scope in ('moving','stationary'):
        folder=root/OUTPUT/scope
        manifest_path=folder/'figure_manifest.json'
        manifest_record=verify_manifest(root,manifest_path)
        manifest=json.loads(manifest_path.read_text())
        assert manifest['scope']==scope
        notebook_path=root/f'notebooks/beamforming_{scope}_comparison_audit.ipynb'
        assert manifest['notebook']==notebook_path.relative_to(root).as_posix()
        assert manifest['notebook_sha256']==sha(notebook_path)
        assert manifest['producer']=='scripts/make_beamforming_comparison_audit_notebooks.py'
        assert manifest['producer_sha256']==sha(root/manifest['producer'])
        notebook=json.loads(notebook_path.read_text());meta=notebook['metadata']
        audit=meta['beamforming_comparison_audit'];editable=meta['editable_plotting']
        assert audit['scope']==scope and audit['config']==manifest['config']
        assert audit['producer']==manifest['producer'] and audit['producer_sha256']==manifest['producer_sha256']
        for name in ['input_hashes','loader_sources']:
            assert audit[name]==manifest[name]
            verify_mapping(root,audit[name],scope+' notebook '+name)
        assert audit['historical_producer_sources']==manifest['historical_producer_sources']
        assert audit['historical_producer_sources']==original_evidence[scope]['source_hashes']
        assert audit['original_settings_unchanged'] is True
        assert audit['new_parameters_selected_on_validation_only'] is True
        assert editable['mode']=='inline_matplotlib'
        assert editable['figure_count']==editable['rendered_figure_count']==7
        assert editable['output_directory']==str(OUTPUT/scope/'figures')
        assert editable['input_hashes']==audit['input_hashes'] and editable['loader_sources']==audit['loader_sources']
        plots=[];executions=[]
        for index,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code':
                continue
            source=cell['source'];source=''.join(source) if isinstance(source,list) else source
            tree=ast.parse(source)
            assert isinstance(cell.get('execution_count'),int) and cell['execution_count']>0
            executions.append(cell['execution_count'])
            assert not any(output.get('output_type')=='error' for output in cell.get('outputs',[]))
            if cell.get('metadata',{}).get('editable_plotting_role')!='plot':
                continue
            calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
            def name(node):
                return (node.id if isinstance(node,ast.Name) else
                        name(node.value)+'.'+node.attr if isinstance(node,ast.Attribute) else '')
            constructors=[node for node in calls if name(node.func) in ('plt.figure','plt.subplots')]
            assert len(constructors)==1, 'Each candidate figure must be untiled and editable'
            assert not constructors[0].args and {item.arg for item in constructors[0].keywords}>={'figsize','dpi'}
            keywords={item.arg:item.value for item in constructors[0].keywords}
            assert isinstance(keywords['figsize'],ast.Name) and keywords['figsize'].id=='FIGSIZE'
            assert isinstance(keywords['dpi'],ast.Name) and keywords['dpi'].id=='DPI'
            controls={}
            for node in tree.body:
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Name):
                            try:controls[target.id]=ast.literal_eval(node.value)
                            except (ValueError,TypeError):pass
            assert controls['DPI']==300 and tuple(controls['FIGSIZE'])==(15,5)
            assert controls['SAVE_PDF'] is True
            assert [name(node.func) for node in calls].count('plt.show')==1
            assert sum(name(node.func)=='fig.savefig' for node in calls)==2
            assert not any(name(node.func) in ('exec','eval') for node in calls)
            assert "'text.usetex': True" in source
            assert any(name(node.func).startswith(('ax.plot','ax.errorbar','ax.barh','ax.scatter','ax.boxplot')) for node in calls)
            figure=cell['metadata']['figure_id'];assert figure in expected_figures[scope]
            assert (figure+'.pdf') in manifest['figures']
            assert any(output.get('output_type') in ('display_data','execute_result') and
                       'image/png' in output.get('data',{}) for output in cell.get('outputs',[]))
            plots.append(dict(cell=index,figure=figure))
        assert [item['figure'] for item in plots] and {item['figure'] for item in plots}==expected_figures[scope]
        # A style-only rerun preserves the untouched cells' output objects and
        # counters. Every cell must be executed, but counters need not increase
        # from top to bottom after selective figure edits.
        assert len(plots)==7
        for filename,figure in manifest['figures'].items():
            assert Path(filename).stem in expected_figures[scope]
            assert figure['figure_size_in']==[15,5] and figure['dpi']==300 and figure['text_usetex'] is True
            for suffix in ['pdf','png']:
                expected=str(OUTPUT/scope/'figures'/(Path(filename).stem+'.'+suffix))
                assert figure[suffix]==expected and sha(root/expected)==figure[suffix+'_sha256']
        records[scope]=dict(status='passed',notebook=manifest['notebook'],manifest=manifest_record,
            executed_code_cells=len(executions),editable_untiled_figures=plots)
    from scripts.check_solid_trajectories import check_source_styles
    trajectory_style=check_source_styles(root,notebooks=tuple(
        f'notebooks/beamforming_{scope}_comparison_audit.ipynb' for scope in ('moving','stationary')))
    assert trajectory_style['status']=='passed',trajectory_style['errors']
    return dict(status='passed',notebooks=records,pdf_and_preview_checks=_pdf_audit_with_available_python(root),
                method_trajectory_styles=trajectory_style)


def fullcheck(root=ROOT, existing_only=False):
    from lib.final_validation import require_current_gate
    root = Path(root).resolve()
    gate = require_current_gate(root)
    result = dict(status='passed',scope='Existing records only' if existing_only else 'Complete comparison audit',
                  scientific_sha256=gate['final_science_hash'],existing=check_existing(root))
    if not existing_only:
        result['mean_validation']={scope:check_mean_validation(root,scope) for scope in ['moving','stationary']}
        result['stationary_selection']=check_stationary_selection(root)
        result['objective']={scope:check_objective(root,scope) for scope in ['moving','stationary']}
        result['notebooks']=check_notebooks(root)
    result['preservation'] = preservation(root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--existing-only',action='store_true')
    parser.add_argument('--pdf-only',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    report = check_notebook_pdfs(args.root) if args.pdf_only else fullcheck(args.root,args.existing_only)
    report.update(checker='scripts/check_beamforming_comparison_audit.py',checker_sha256=sha(__file__))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
