"""Independently verify notebook uncertainty summaries from saved trial data.

This presentation audit does not simulate data, fit a model, modify numerical
results or render figures. It checks all task/adaptation/calibration interval
payloads, including confidence limits supplied for curves whose bands are not
shown, and the beamforming task boxplot samples. Run from any working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/bayesian-st-cache')

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.notebook_beamforming_data import load_beamforming_plot_data
from lib.notebook_contact_data import load_contact_plot_data
from lib.final_stationary_evidence import (stationary_component_path,
                                           verify_stationary_component)


def audit(root=ROOT):
    root = Path(root).resolve()
    records=[];summaries={};input_hashes={};reporting_timing={};cohorts={}
    def record_hash(path):
        p=root/path;input_hashes[path]=hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    def read(path):
        return pd.read_csv(record_hash(path))
    def interval(values):
        a=np.asarray(values,float);a=a[np.isfinite(a)]
        m=float(np.mean(a));half=float(student_t.ppf(.975,len(a)-1)*np.std(a,ddof=1)/np.sqrt(len(a)))
        return [m,m-half,m+half]
    def compare(example,figure,metric,method,time,expected,actual,count,kind):
        a,b=np.asarray(expected,float),np.asarray(actual,float)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-11)
        records.append(dict(example=example,figure=figure,metric=metric,method=method,time=float(time),independent_trials=int(count),interval=kind,maximum_absolute_difference=float(np.max(abs(a-b))),center=float(a[0]),lower=float(a[1]),upper=float(a[2]),width=float(a[2]-a[1])))

    def normalize(frame):
        f=frame.rename(columns={'trial':'episode'}).copy()
        if 'method' in f:f['method']=f.method.replace({'isotropic':'iso','modal_loading':'mode_loaded','covariance_loading':'scm_loaded','ledoit_wolf':'lw'})
        return f
    for example,loader in [('beamforming',load_beamforming_plot_data),('contact',load_contact_plot_data)]:
        frame=normalize(read(f'results/{example}/tables/paper/prediction/adaptation.csv'))
        frame=frame[(frame.split=='test')&(frame.control=='mean')&(frame.condition=='noise_change')]
        time_column = 't' if example == 'beamforming' else 'time_s'
        # The raw table records the variance BEFORE each reported update.
        # A fit after that measurement supplies sigma2_next, used for future
        # measurements. Sparse post-step connections are not a dense trajectory.
        config_name=f'results/{example}/data/paper/prediction/manifest.json'
        config_path=root/config_name
        input_hashes[config_name]=hashlib.sha256(config_path.read_bytes()).hexdigest()
        config=json.loads(config_path.read_text())['config']
        prefix='primary_beam_' if example=='beamforming' else 'primary_contact_'
        dt=float(config[prefix+'dt']);length=int(config[prefix+'T'])
        learned=frame[frame.method.eq('learned')]
        assert learned.episode.nunique()==256
        expected_reporting=np.arange(int(config['primary_report_every']),length+1,
                                     int(config['primary_report_every']))
        assert not learned.duplicated(['episode','t']).any()
        variance_keys=['sigma2','tau2'] if example=='beamforming' else ['sigma2']
        maximum_transition_error=0.
        for _,simulation in learned.groupby('episode',sort=True):
            simulation=simulation.sort_values('t')
            np.testing.assert_array_equal(simulation.t,expected_reporting)
            assert simulation.fit_performed.astype(bool).all()
            for key in variance_keys:
                # A fit performed after u_t must first appear in the estimate
                # used at a later report; the post-fit value is not used at t.
                before=simulation[key+'_estimate'].to_numpy()[1:]
                previous_fit=simulation[key+'_next'].to_numpy()[:-1]
                np.testing.assert_allclose(before,previous_fit,rtol=1e-13,atol=1e-15)
                maximum_transition_error=max(maximum_transition_error,
                    float(np.max(np.abs(before-previous_fit))))
        assert frame.prediction_scope.eq('Before assimilation of the scored observation').all()
        assert frame.score_numerical_ok.astype(bool).all()
        episode=int(learned.episode.min())
        trajectory=learned[learned.episode.eq(episode)].sort_values('t')
        first=trajectory.iloc[0]
        changed=trajectory[trajectory.sigma2_true.ne(first.sigma2_true)].iloc[0]
        assert bool(first.fit_performed)
        assert not np.isclose(first.sigma2_estimate,first.sigma2_next,rtol=0.,atol=1e-14)
        assert np.isclose(trajectory.iloc[1].sigma2_estimate,first.sigma2_next)
        change_observation=length//2+1
        reporting_timing[example]=dict(
            clock=('The displayed clock is the stored integer measurement index t; time_s=(t-1)*dt is retained only for checking the original records.' if example=='beamforming' else 'The adaptation table uses time_s=(t-1)*dt, with the first observation at zero.'),
            independently_selected_episode=episode,
            reporting_interval_observations=int(config['primary_report_every']),
            first_reporting_observation=int(first.t),first_reporting_time_s=float(first.time_s),
            first_pre_update_sigma2=float(first.sigma2_estimate),
            first_post_fit_sigma2_next=float(first.sigma2_next),
            first_fit_effective_from_next_observation_time_s=float(first.time_s+dt),
            physical_noise_change_observation=change_observation,
            physical_noise_change_time_s=float((change_observation-1)*dt),
            first_report_with_changed_truth_observation=int(changed.t),
            first_report_with_changed_truth_time_s=float(changed.time_s),
            displayed_event_marker=('Measurement 129, the first measurement using the changed generating variance; the true step uses where=post.' if example=='beamforming' else 'First reporting checkpoint with changed generating variance; not the exact physical change time.'),
            independently_checked_prequential_simulations=int(learned.episode.nunique()),
            maximum_prequential_variance_transition_error=maximum_transition_error,
            source='Noise variance is multiplied by four at zero-based index T//2 in the saved-study generator; no stream is regenerated here.',
            interpretation='Displayed estimates are pre-correction checkpoint values. Connecting segments are not additional measurements or a reconstruction of scale updates between checkpoints.')
        payload=loader(root,'noise_adaptation')
        if example=='beamforming':
            assert payload['event']==129 and payload['T']==256
            assert payload['task_endpoint']=='output_oracle_db'
            assert payload['score_endpoint']=='prequential_score'
            reporting_timing[example]['displayed_event_measurement']=129
            for key,panel in payload['variance'].items():
                np.testing.assert_array_equal(panel['x'],np.arange(1,257))
                expected_truth=(np.where(np.arange(1,257)<=128,.2,.8)
                                if key=='sigma2' else np.ones(256))
                np.testing.assert_array_equal(panel['truth'],expected_truth)
                np.testing.assert_array_equal(panel['events'],[129])
                assert panel['truth_step_where']=='post'
                for trace in panel['traces']:
                    for idx,tm in enumerate(trace['x']):
                        g=frame[(frame.method==trace['method'])&(frame[time_column]==tm)]
                        assert g.episode.nunique()==len(g)==256
                        sample=g[key+'_estimate'].to_numpy();lo,m,hi=np.quantile(sample,[.25,.5,.75])
                        compare(example,'noise_adaptation',key,trace['method'],tm,[m,lo,hi],[trace['y'][idx],trace['low'][idx],trace['high'][idx]],len(sample),'Empirical 25–75% interquartile range')
            trace_sets=[('task',payload['task'],'output_oracle_db',1.),('score',payload['score'],'prequential_score',1.)]
        else:
            for trace in payload['variance']:
                for idx,tm in enumerate(trace['t']):
                    g=frame[(frame.method==trace['method'])&(frame[time_column]==tm)]
                    sample=g.sigma2_estimate.to_numpy()*1e6;lo,m,hi=np.quantile(sample,[.25,.5,.75])
                    compare(example,'noise_adaptation','sigma2',trace['method'],tm,[m,lo,hi],[trace['median'][idx],trace['lower'][idx],trace['upper'][idx]],len(sample),'Empirical 25–75% interquartile range')
            trace_sets=[('task',payload['future'],'future_task_loss',1e6),('score',payload['scores'],'prequential_score',1.)]
        for metric,traces,column,scale in trace_sets:
            for trace in traces:
                for row in trace['points']:
                    tm=row[0];g=frame[(frame.method==trace['method'])&(frame[time_column]==tm)]
                    sample=g.groupby('episode')[column].mean().dropna()*scale
                    compare(example,'noise_adaptation',metric,trace['method'],tm,interval(sample),row[1:],len(sample),'Pointwise Student 95% confidence interval for mean')
        # Every displayed coverage error bar, independently from binary trial records.
        raw=read(f'results/{example}/tables/paper/prediction/calibration.csv')
        for page,regime in [(2,'prior_predictive'),(6,'drifting')]:
            data=loader(root,'predictive_calibration',page)
            for panel in data['panels']:
                target=panel.get('kind',panel.get('target'))
                for trace in panel.get('traces',panel.get('curves')):
                    points=trace.get('points')
                    if points is None:points=np.column_stack([trace['x'],trace['y'],trace['low'],trace['high']])
                    for row in points:
                        nominal=row[0];g=raw[(raw.regime==regime)&(raw.t==64)&(raw.kind==target)&(raw.method==trace['method'])&(raw.nominal==nominal)&(raw.split=='test')]
                        assert g.episode.nunique()==len(g)
                        count=len(g);success=int(g.covered.sum())
                        lo=0. if success==0 else beta.ppf(.025,success,count-success+1)
                        hi=1. if success==count else beta.ppf(.975,success+1,count-success)
                        compare(example,'predictive_calibration',regime+':'+target,trace['method'],nominal,[success/count,lo,hi],row[1:],count,'Pointwise exact binomial 95% confidence interval for coverage')
        task=loader(root,'task_performance')
        if example=='beamforming':
            stationary_meta=verify_stationary_component(root,stage='paper')
            sidecar=stationary_component_path(root,stage='paper')
            record_hash(sidecar.relative_to(root).as_posix())
            raw=[name for name in stationary_meta['artifacts']
                 if Path(name).name=='stationary.csv']
            assert len(raw)==1
            perf=normalize(read(raw[0]))
            assert perf.condition.eq('static').all()
            assert perf.stationary.astype(bool).all()
            assert perf['T'].eq(200).all()
            assert perf.sigma2.eq(.2).all() and perf.tau2.eq(1.).all()
            assert perf[perf.method.isin(['mean','mode','iso'])].beta.eq(1.).all()
            np.testing.assert_array_equal(np.sort(perf.t.unique()),[25,50,100,200])
            np.testing.assert_allclose(perf.true_ratio_db,
                10*np.log10(perf.true_power/perf.oracle_power),rtol=1e-12,atol=1e-12)
            assert task['stationary_T']==task['stationary_primary_t']==200
            perf=perf[perf.t.eq(200)]
            assert not perf.duplicated(['n','method','episode']).any()
            for trace in task['curves']:
                for row in trace['points']:
                    n=row[0];g=perf[(perf.n==n)&(perf.method==trace['method'])]
                    np.testing.assert_array_equal(np.sort(g.episode),np.arange(256))
                    sample=g.true_ratio_db.to_numpy()
                    compare(example,'task_performance','output_power',trace['method'],n,interval(sample),row[1:],len(sample),'Pointwise Student 95% confidence interval for mean')
            cohorts['stationary']=dict(raw_input=raw[0],T=200,primary_t=200,
                dimensions=[32,64,128],reporting_t=[25,50,100,200],beta=1.,
                sigma2=.2,tau2=1.,simulations_per_dimension=256,
                endpoint='true_ratio_db',role='Supplementary fixed-dimension comparison')

            abrupt=loader(root,'abrupt_tracking')
            moving_name='results/beamforming/data/paper/performance/moving_checkpoints.csv.gz'
            moving=normalize(read(moving_name))
            moving=moving[moving.condition.eq('abrupt')&moving.n.eq(32)
                          &moving.method.isin(['mean','mode','iso','great_window'])]
            assert not moving.duplicated(['method','episode','t']).any()
            assert np.isfinite(moving.t).all() and np.equal(moving.t,np.floor(moving.t)).all()
            np.testing.assert_array_equal(np.sort(moving.t.unique()),np.arange(8,257,8))
            assert moving.sigma2.eq(.2).all() and moving.tau2.eq(1.).all()
            assert moving[moving.method.isin(['mean','mode','iso'])].beta.eq(.97).all()
            assert moving.numerical_ok.astype(bool).all()
            np.testing.assert_allclose(moving.true_ratio_db,
                10*np.log10(moving.true_power/moving.oracle_power),rtol=1e-12,atol=1e-12)
            assert abrupt['n']==32 and abrupt['T']==256 and abrupt['event']==129
            assert abrupt['beta']==.97 and abrupt['endpoint']=='true_ratio_db'
            np.testing.assert_array_equal(abrupt['time'],np.arange(8,257,8))
            assert [trace['method'] for trace in abrupt['curves']]==['mean','mode','iso','great_window']
            for trace in abrupt['curves']:
                assert len(trace['points'])==32
                for row in trace['points']:
                    tm=row[0];g=moving[moving.method.eq(trace['method'])&moving.t.eq(tm)]
                    np.testing.assert_array_equal(np.sort(g.episode),np.arange(256))
                    sample=g.true_ratio_db.to_numpy()
                    compare(example,'abrupt_tracking','output_power',trace['method'],tm,
                        interval(sample),row[1:],len(sample),
                        'Pointwise Student 95% confidence interval for mean')
            cohorts['abrupt']=dict(raw_input=moving_name,n=32,T=256,event=129,
                reporting_t=list(range(8,257,8)),beta=.97,sigma2=.2,tau2=1.,
                methods=['mean','mode','iso','great_window'],simulations_per_checkpoint=256,
                endpoint='true_ratio_db',role='Main abrupt-interference comparison')
        else:
            perf=normalize(read('results/contact/tables/paper/performance/metrics.csv'))
            checkpoints=perf[(perf.endpoint=='checkpoint')&(perf.condition=='static_concentrated')]
            for trace in task['curves']:
                for row in trace['points']:
                    tm=row[0];g=checkpoints[(checkpoints.t==tm)&(checkpoints.method==trace['method'])]
                    sample=g.groupby('episode').response_mse.mean()*1e6
                    compare(example,'task_performance','response_MSE',trace['method'],tm,interval(sample),row[1:],len(sample),'Pointwise Student 95% confidence interval for mean')
            moving=perf[(perf.endpoint=='episode_average')&(perf.condition=='sphere')]
            for point in task['absolute']:
                sample=moving[moving.method==point['method']].groupby('episode').response_mse.mean()*1e6
                compare(example,'task_performance','moving_response_MSE',point['method'],192,interval(sample),[point['mean'],point['lower'],point['upper']],len(sample),'Pointwise Student 95% confidence interval for mean')

    for example in ['beamforming','contact']:
        rs=[r for r in records if r['example']==example]
        summary={'recomputed_points':len(rs),'maximum_absolute_difference':max(r['maximum_absolute_difference'] for r in rs),'trial_counts':sorted(set(r['independent_trials'] for r in rs)),'adaptation_widths':{}}
        for metric in ['sigma2','tau2','task','score']:
            rr=[r for r in rs if r['figure']=='noise_adaptation' and r['metric']==metric]
            for method in sorted(set(r['method'] for r in rr)):
                points=[r for r in rr if r['method']==method]
                summary['adaptation_widths'][metric+':'+method]={'minimum':min(r['width'] for r in points),'maximum':max(r['width'] for r in points),'centers':[r['center'] for r in points],'widths':[r['width'] for r in points]}
        summaries[example]=summary
    report={'status':'passed','scope':'Independent recalculation of every interval used by displayed task/adaptation/calibration plots in the two notebooks; plotting/numerical arrays unchanged.','summaries':summaries,'records':records,'raw_input_sha256':input_hashes,'artificial_widths':False,'postprocessing':'Only the declared sample quantiles, Student-t intervals, or exact binomial intervals; no smoothing, rescaling, minimum display width, or added jitter in shaded bands.','important_distinction':'Repeated simulation trials produce empirical spread and uncertainty in aggregate means/coverage; these bands are not posterior credible tubes or Monte Carlo integration error. Finite integration error is checked separately.'}

    file='results/beamforming/tables/paper/performance/moving_episodes.csv'
    f=normalize(read(file))
    f=f[(f.condition=='smooth')&(f.n==32)]
    base=f[f.method=='mean'].set_index('episode').true_ratio_db
    box_rows=[]
    for point in load_beamforming_plot_data(root,'task_performance')['paired']:
        other=f[f.method==point['method']].set_index('episode').true_ratio_db
        index=base.index.intersection(other.index)
        expected=(other.loc[index]-base.loc[index]).to_numpy()
        np.testing.assert_array_equal(expected,point['values'])
        box_rows.append(dict(method=point['method'],independent_trials=len(expected),
            saved_values_exact=True,median=float(np.median(expected)),
            interquartile_range=np.quantile(expected,[.25,.75]).tolist(),
            description='Matplotlib computes boxes, whiskers and outliers directly from these paired trial differences.'))
    report['boxplots']=dict(example='beamforming',figure='task_performance',
        raw_input=file,raw_input_sha256=input_hashes[file],rows=box_rows)
    report['producer']='scripts/check_notebook_bands.py'
    report['producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['counts']=dict(interval_summaries=len(records),boxplot_sample_sets=len(box_rows))
    report['reporting_scope']='Adaptation values are pre-update estimates at sparse reporting times, not a dense trajectory between reports.'
    report['reporting_timing']=reporting_timing
    report['beamforming_cohorts']=cohorts
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='Optional JSON report path; saved numerical files are never modified.')
    args = parser.parse_args()
    report = audit()
    serialized = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
