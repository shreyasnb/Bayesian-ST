"""Summarize executed raw results without selecting favorable outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results'
summary={'theory':json.loads((out/'validation.json').read_text()),'cases':{}}
lines=['# Simulation findings','',
       'These are the recorded eight-trajectory pilot runs. Comparisons below use',
       'all scenarios and retain unfavorable outcomes and numerical failures.',
       'See the notebooks for individual scenarios and uncertainty intervals.','']
for case in ('case1','case2'):
    m=pd.read_csv(out/f'{case}_metrics.csv');T=int(m.t.max());final=m[m.t==T]
    p=m[m.method=='Posterior mean']
    failed=int((~p.sampling_ok.astype(bool)).sum())
    entry={'posterior_evaluations':len(p),'failed_sampling_diagnostics':failed,
           'minimum_ess':float(p.min_ess.min()),'maximum_rhat':float(p.max_rhat.max()),
           'maximum_mcse':float(p.max_mcse.max())}
    lines.extend([f'## {"Array" if case=="case1" else "Field"}','',
                  f'{len(p)} passive posterior evaluations; {failed} failed the predeclared',
                  'sampling diagnostics at the maximum budget. Such evaluations cannot',
                  'support precise posterior uncertainty claims.',''])
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
lines.extend(['## Interpretation','',
              'Marginal Bingham and matching EW PCA modes coincide. Mean/modal changes',
              'therefore concern posterior task decisions. Individual realized errors',
              'need not obey the posterior expected-risk ordering. No universal',
              'performance or frequentist-coverage claim follows from this pilot.',
              '', 'No inconsistency in the stated mathematical assumptions was found.',
              'The required numerical qualifications and Algorithm 1 clarifications',
              'are recorded in `draft_notes.md`.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(ROOT/'docs/results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary['cases'],indent=2))
