"""Paper tables from the retained independent simulation records."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

STATIONARY_METHODS = ('mean', 'mode', 'great_window', 'covariance_loading')
MOVING_METHODS = ('mean', 'mode', 'isotropic', 'great_window')


def mean_interval(values):
    values = np.asarray(values, float)
    if values.shape != (256,) or not np.isfinite(values).all():
        raise ValueError('A paper interval requires 256 finite independent simulations')
    mean = float(values.mean())
    se = float(values.std(ddof=1)/16)
    half = float(student_t.ppf(.975, 255)*se)
    return dict(mean=mean, lower=mean-half, upper=mean+half,
                standard_error=se, simulations=256, degrees_of_freedom=255)


def _results(root, results=None):
    return Path(results).resolve() if results is not None else Path(root).resolve()/'results'


def _read(root, name, results=None):
    return pd.read_csv(_results(root,results)/'beamforming/data'/f'{name}.csv.gz',
                       float_precision='round_trip')


def _complete(frame, group_columns, id_column):
    if frame.duplicated(group_columns+[id_column]).any():
        raise ValueError('Duplicate simulation records')
    for _, group in frame.groupby(group_columns, sort=True):
        if len(group) != 256 or set(group[id_column]) != set(range(256)):
            raise ValueError('Incomplete paper simulation cohort')


def _resolved(frame, flags):
    for flag in flags:
        if frame[flag].isna().any() or not frame[flag].astype(str).str.lower().isin(['true','1','1.0']).all():
            raise ValueError('Unresolved numerical record: '+flag)


def _power_identity(frame):
    if not np.isfinite(frame[['true_power','oracle_power','true_ratio_db']]).all().all():
        raise ValueError('Nonfinite exact-covariance power')
    if not (frame.true_power.gt(0).all() and frame.oracle_power.gt(0).all()):
        raise ValueError('Exact-covariance powers must be positive')
    np.testing.assert_allclose(frame.true_ratio_db,
        10*np.log10(frame.true_power/frame.oracle_power),rtol=1e-11,atol=1e-11)


def stationary_tables(root, results=None):
    frame = _read(root, 'stationary',results)
    if set(frame.n) != {32,64,128} or set(frame.t) != {200} or set(frame.condition) != {'static'}:
        raise ValueError('Expected the fixed T=200 stationary cohort')
    if set(frame.method) != set(STATIONARY_METHODS):
        raise ValueError('The stationary table requires all four paper methods')
    if len(frame) != 3*4*256:
        raise ValueError('Incorrect stationary cohort size')
    _complete(frame,['n','t','method'],'trial')
    _resolved(frame,['numerical_ok','mean_ok'])
    _power_identity(frame)
    rows=[];paired=[]
    for (n,method), group in frame.groupby(['n','method'],sort=True):
        rows.append(dict(n=int(n),t=200,method=method,**mean_interval(group.true_ratio_db)))
    for n, group in frame.groupby('n',sort=True):
        pivot=group.pivot(index='trial',columns='method',values='true_ratio_db')
        for method in STATIONARY_METHODS[1:]:
            paired.append(dict(n=int(n),t=200,reference=method,
                baseline='mean',**mean_interval(pivot[method]-pivot['mean'])))
    policies=json.loads((_results(root,results)/'beamforming/data/policies_stationary.json').read_text())
    settings=[]
    for p in policies:
        selected=next(item for item in p['policies'] if 200 in item['times'])
        great=p['great_window']
        settings.append(dict(n=p['n'],T=200,great_window=great['window'],
            great_steps=great['inner_steps'],great_stepsize=great['stepsize'],
            covariance_alpha=selected['alpha'],beta=p['shared_beta']))
    return pd.DataFrame(settings),pd.DataFrame(rows),pd.DataFrame(paired)


def moving_tables(root, results=None):
    frame=_read(root,'moving',results)
    if set(frame.condition) != {'smooth','abrupt'} or set(frame.n) != {32}:
        raise ValueError('Expected the independent smooth and abrupt n=32 cohorts')
    if set(frame.t) != set(range(8,257,8)) or set(frame.method) != set(MOVING_METHODS):
        raise ValueError('The moving cohort reporting grid changed')
    if len(frame) != 2*4*32*256:
        raise ValueError('Incorrect moving cohort size')
    _complete(frame,['condition','t','method'],'trial')
    _resolved(frame,['numerical_ok','mean_ok'])
    _power_identity(frame)
    per_trial=frame.groupby(['condition','trial','method'],sort=True).true_ratio_db.mean().rename('time_average_db').reset_index()
    rows=[];paired=[]
    for (condition,method),group in per_trial.groupby(['condition','method'],sort=True):
        rows.append(dict(condition=condition,method=method,**mean_interval(group.time_average_db)))
    for condition,group in per_trial.groupby('condition',sort=True):
        pivot=group.pivot(index='trial',columns='method',values='time_average_db')
        for method in MOVING_METHODS[1:]:
            paired.append(dict(condition=condition,reference=method,
                baseline='mean',**mean_interval(pivot[method]-pivot['mean'])))
    return pd.DataFrame(rows),pd.DataFrame(paired),per_trial


def noise_tables(root, results=None):
    frame=_read(root,'noise',results)
    if set(frame.condition) != {'noise_change'} or set(frame.t) != set(range(32,257,32)):
        raise ValueError('Expected the fixed-direction noise-change cohort')
    frame=frame[frame.control.eq('mean')]
    if set(frame.method) != {'known','misspecified','learned'} or len(frame) != 3*8*256:
        raise ValueError('Incomplete noise-change mean-control cohort')
    _complete(frame,['t','method'],'episode')
    _resolved(frame,['mean_ok','score_numerical_ok','fit_numerical_ok'])
    if not np.isfinite(frame[['output_oracle_db','prequential_score']]).all().all():
        raise ValueError('Unresolved power or predictive score')
    per_trial=frame.groupby(['method','episode'],sort=True).output_oracle_db.mean().rename('time_average_db').reset_index()
    rows=[dict(method=method,**mean_interval(group.time_average_db))
          for method,group in per_trial.groupby('method',sort=True)]
    return pd.DataFrame(rows)


def _latex(path, columns, rows):
    lines=[r'\begin{tabular}{'+'l'*len(columns)+'}',r'\hline',
           ' & '.join(columns)+r' \\',r'\hline']
    lines.extend(' & '.join(row)+r' \\' for row in rows)
    lines.extend([r'\hline',r'\end{tabular}',''])
    path.write_text('\n'.join(lines))


def export_tables(root, results=None):
    root=Path(root).resolve();target=_results(root,results)/'beamforming/tables';target.mkdir(parents=True,exist_ok=True)
    settings,stationary,stationary_pairs=stationary_tables(root,results)
    moving,moving_pairs,per_trial=moving_tables(root,results)
    noise=noise_tables(root,results)
    frames={'stationary_settings':settings,'stationary_results':stationary,
        'stationary_paired':stationary_pairs,'moving_averages':moving,
        'moving_paired':moving_pairs,'noise_averages':noise}
    for name,frame in frames.items():frame.to_csv(target/(name+'.csv'),index=False)
    def cell(row):return fr"${row['mean']:.3f}\ [{row['lower']:.3f},{row['upper']:.3f}]$"
    rows=[[str(int(r.n)),str(int(r.great_window)),str(int(r.great_steps)),
        f'{r.great_stepsize:.9g}',f'{r.covariance_alpha:.3f}'] for r in settings.itertuples()]
    _latex(target/'stationary_settings.tex',[r'$n$',r'$L_{\mathrm G}$','Steps','Step size',r'$\alpha$'],rows)
    rows=[]
    for n in (32,64,128):
        data=stationary[stationary.n.eq(n)].set_index('method')
        rows.append([str(n)]+[cell(data.loc[m]) for m in STATIONARY_METHODS])
    _latex(target/'stationary_results.tex',[r'$n$','Mean','Mode','GREAT','Regularized covariance'],rows)
    rows=[]
    for condition in ('smooth','abrupt'):
        data=moving[moving.condition.eq(condition)].set_index('method')
        rows.append([condition.capitalize()]+[cell(data.loc[m]) for m in MOVING_METHODS])
    _latex(target/'moving_averages.tex',['Motion','Mean','Mode','Isotropic','GREAT'],rows)
    return {name:str(target/(name+'.csv')) for name in frames}


def validate_tables(root, results=None):
    _,stationary,_=stationary_tables(root,results)
    moving,pairs,_=moving_tables(root,results)
    noise=noise_tables(root,results)
    expected_stationary={32:[.119,.114,.363,.379],64:[.237,.223,.350,.576],128:[.482,.440,.677,.856]}
    for n,expected in expected_stationary.items():
        frame=stationary[stationary.n.eq(n)].set_index('method')
        assert [round(float(frame.loc[m,'mean']),3) for m in STATIONARY_METHODS]==expected
    expected_moving={'smooth':[.330,.331,.332,.296],'abrupt':[.680,.665,.695,.694]}
    for condition,expected in expected_moving.items():
        frame=moving[moving.condition.eq(condition)].set_index('method')
        assert [round(float(frame.loc[m,'mean']),3) for m in MOVING_METHODS]==expected
    expected_noise={'learned':[.435,.425,.445],'misspecified':[1.056,1.044,1.069],'known':[.378,.371,.385]}
    for method,expected in expected_noise.items():
        row=noise.set_index('method').loc[method]
        assert [round(float(row[key]),3) for key in ('mean','lower','upper')]==expected
    abrupt=_read(root,'moving',results);abrupt=abrupt[abrupt.condition.eq('abrupt')&abrupt.t.eq(144)]
    powers=abrupt.groupby('method').true_ratio_db.mean()
    assert [round(float(powers[m]),2) for m in ('mean','mode','isotropic')]==[2.63,3.17,2.82]
    paired=pairs[pairs.condition.eq('abrupt')&pairs.reference.eq('great_window')].iloc[0]
    assert [round(float(paired[key]),3) for key in ('mean','lower','upper')]==[.013,.006,.021]
    return dict(status='passed',stationary_simulations=256,moving_simulations=256,
                noise_simulations=256,all_manuscript_power_targets_match=True)
