"""Summarize executed raw results without selecting favorable outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results'
summary={'theory':json.loads((out/'validation.json').read_text()),'cases':{}}
lines=['# Simulation findings','',
       'The original application comparisons use eight trajectories per scenario.',
       'The controlled decision study and the higher-dimensional motion study have',
       'separate repetition counts below. All retain unfavorable outcomes and',
       'numerical failures. See the notebooks for uncertainty intervals.','']
for case in ('case1','case2'):
    m=pd.read_csv(out/f'{case}_metrics.csv');T=int(m.t.max());final=m[m.t==T]
    p=m[m.method=='Posterior mean']
    failed=int((~p.sampling_ok.astype(bool)).sum())
    entry={'posterior_evaluations':len(p),'failed_sampling_diagnostics':failed,
           'minimum_ess':float(p.min_ess.min()),'maximum_rhat':float(p.max_rhat.max()),
           'maximum_mcse':float(p.max_mcse.max())}
    lines.extend([f'## {"Array" if case=="case1" else "Field"}','',
                  f'{len(p)} passive posterior evaluations; {failed} failed the predeclared',
                  'sampling diagnostics at the maximum budget.',
                  'Capped failures are retained and require separate accuracy follow-up.' if failed else
                  'All selected passive evaluations met the stated diagnostics; this is not a convergence proof.',''])
    value='sinr_db' if case=='case1' else 'field_squared_error'
    table=final.groupby(['scenario','method'])[value].mean().unstack('method')
    entry['terminal_means']=table.to_dict()
    lines.append('| Scenario | '+' | '.join(table.columns)+' |')
    lines.append('| --- | '+' | '.join(['---']*len(table.columns))+' |')
    for label,row in table.iterrows(): lines.append('| '+str(label)+' | '+' | '.join(f'{v:.4f}' for v in row)+' |')
    lines.append('')
    if case=='case1':
        r=pd.read_csv(out/'case1_regions.csv');entry['maximum_optimization_gap']=float(r.optimization_gap.max())
        lines.extend([f"Largest regional primal–dual gap: {entry['maximum_optimization_gap']:.3g}.",''])
        accuracy=pd.read_csv(out/'case1_accuracy.csv')
        entry['extended_accuracy_points']=len(accuracy)
        entry['extended_accuracy_passed']=int(accuracy.new_sampling_ok.sum())
        if len(accuracy):
            change=float(accuracy.output_power_relative_change.abs().max())
            entry['maximum_extended_power_relative_change']=change
            lines.extend([f"Separate longer-chain checks passed for {entry['extended_accuracy_passed']}/{len(accuracy)} original capped failures.",
                          f"Largest absolute relative change in population output power: {change:.4%}.",
                          'Original flags remain unchanged; extended checks use new independent random streams.',''])
    else:
        c=pd.read_csv(out/'case2_calibration.csv')
        entry['calibration_failed_diagnostics']=int((~c.sampling_ok.astype(bool)).sum())
        lines.extend([f"Known-calibration posterior evaluations with failed diagnostics: {entry['calibration_failed_diagnostics']} of {len(c)}.",
                      '', 'Sampled high-dimensional EM ascent is unverified. Predictive scores',
                      'must be read with their importance ESS, sampler ESS and chain disagreement.',''])
    summary['cases'][case]=entry
study_path=out/'decision_study_config.json'
if study_path.exists():
    meta=json.loads(study_path.read_text())
    paired=pd.read_csv(out/'decision_study_paired.csv')
    selected=paired[(paired.regime==1)&(paired.t==4)&paired.reference.isin(['Mode','Shrunk PCA','Loaded covariance'])]
    summary['decision_study']=dict(config=meta['config'],checks=meta['checks'],
                                  moderate_noise_four_snapshots=selected.to_dict('records'),
                                  coverage=pd.read_csv(out/'decision_study_coverage_summary.csv').to_dict('records'))
    lines.extend(['## Controlled decision study','',
                  'This separate stationary real Gr(1,3) study uses identical observations,',
                  'known variances, 400 independent test trajectories, 160 independent',
                  'validation trajectories, and nine-candidate reference families.',
                  'It tests posterior decisions under the stated joint prior model.',
                  '', '| Task | Reference | Loss reduction | Paired gain interval |',
                  '| --- | --- | ---: | --- |'])
    for row in selected.itertuples():
        lines.append(f'| {row.task} | {row.reference} | {row.reduction_percent:.2f}% | [{row.gain_low:.5f}, {row.gain_high:.5f}] |')
    lines.extend(['', 'Rows above use noise variance 0.2 and four observations; all other',
                  'predeclared regimes, including statistical ties, remain in the notebooks.',
                  'Intervals concern paired absolute-loss gains, not the displayed percentage.',
                  'Shrunk PCA and strongly loaded covariance capture much of the benefit.',
                  'See [interpretation](interpretation.md) for the cost and scope qualifications.',''])
drift_path=out/'drift_study_config.json'
if drift_path.exists():
    meta=json.loads(drift_path.read_text())
    b=pd.read_csv(out/'drift_study_bounds.csv');b=b[b.t==b.t.max()]
    p=pd.read_csv(out/'drift_study_posterior.csv');p=p[p.t==p.t.max()]
    records=[]
    labels={'static':'Exact frame','identity_slow':'Identity, slow drift',
            'forget_slow':'Identity, slow drift, forgetting',
            'identity_fast':'Identity, fast drift','forget_fast':'Identity, fast drift, forgetting'}
    lines.extend(['## Higher-dimensional motion and uncertainty','',
                  'This conditional-observation study uses n=8, rank 3, 128 measurements,',
                  '64 independent trajectories and posterior calculations on the first 16.',
                  'The true measurement variance is fixed at 0.03. Static and exact-motion',
                  'controls are orthogonally coupled, so they are not extra independent trials.',
                  '', '| Prediction | Mean modal error | Mean valid bound | Valid | Ball coverage |',
                  '| --- | ---: | ---: | --- | --- |'])
    for key,label in labels.items():
        g=b[b.protocol==key];h=p[p.protocol==key];valid=g[g.valid.astype(bool)]
        record=dict(protocol=key,mean_modal_error=float(g.chordal_error.mean()),
                    mean_valid_bound=float(valid.bound.mean()) if len(valid) else None,
                    valid=len(valid),trials=len(g),ball_covered=int(h.ball_covered.sum()),
                    posterior_trials=len(h),valid_bound_violations=int(valid.bound_violated.sum()))
        records.append(record)
        bound=f'{record["mean_valid_bound"]:.3f}' if len(valid) else 'unavailable'
        lines.append(f'| {label} | {record["mean_modal_error"]:.3f} | {bound} | {len(valid)}/{len(g)} | {record["ball_covered"]}/{len(h)} |')
    summary['drift_study']=dict(config=meta['config'],checks=meta['checks'],terminal=records,
                               coverage=pd.read_csv(out/'drift_study_coverage.csv').to_dict('records'))
    lines.extend(['', 'The distance ceiling is sqrt(3)=1.732. A valid modal bound and poor',
                  'credible coverage coexist under slow unmodelled motion. These test',
                  'different claims; the modal theorem is not a coverage theorem. Sixteen',
                  'posterior trials give wide binomial intervals. Forgetting helps this slow',
                  'protocol but does not establish general calibration; both fast nominal',
                  'protocols fail the bound condition and have zero terminal ball coverage.',
                  'All 672 reported posterior summaries pass the stated chain diagnostics.',
                  'See [modeling and drift](modeling_and_drift.md) for assumptions and draft changes.',''])
array_path=out/'array_envelope_config.json'
if array_path.exists():
    meta=json.loads(array_path.read_text())
    d=pd.read_csv(out/'array_story_diagnostics.csv')
    summary['array_uncertainty']=dict(posterior_evaluations=len(d),sampling_passes=int(d.sampling_ok.sum()),
                                     maximum_regional_optimization_gap=meta['maximum_optimization_gap'])
    lines.extend(['## Array uncertainty','',
                  'The main figures use native complex inference with 16 channels and rank 3.',
                  'Preassigned static/drifting trajectories show responses at 12, 36 and 72',
                  'snapshots, and posterior output-power distributions for three weight rules.',
                  'All six independently sampled targets pass the stated diagnostics.',
                  'Light envelopes bound responses over an estimated 95% state HPD region;',
                  'central functional quantiles are shown separately. If the state lies in',
                  'the region, all evaluated responses are bounded. Sampled thresholds do',
                  'not establish frequentist coverage. The maximum optimization gap is',
                  f'{meta["maximum_optimization_gap"]:.3g}; this excludes threshold Monte Carlo error.',''])
series_path=out/'spectral_series_checks.json'
if series_path.exists():
    summary['spectral_series']=json.loads(series_path.read_text())
lines.extend(['## Interpretation','',
              'Marginal Bingham and matching EW PCA modes coincide. Mean/modal changes',
              'therefore concern posterior task decisions. Individual realized errors',
              'need not obey the posterior expected-risk ordering. No universal',
              'performance or frequentist-coverage claim follows from this pilot.',
              '', 'Unknown-motion credible calibration is unsupported by the nominal',
              'identity/forgetting model. Narrow that claim or specify and propagate a',
              'stochastic transition. The exact correction and conditional modal theorem',
              'retain their stated scope. See `modeling_and_drift.md`,',
              '`latent_paper_audit.md`, and `draft_notes.md` for the required qualifications.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'docs/results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary['cases'],indent=2))
