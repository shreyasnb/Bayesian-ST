"""Build the two chronological, render-only final paper notebooks.

The command refuses missing/stale paper components or an incomplete figure set.
It never generates observations, integrates a posterior, fits a parameter,
selects a trajectory or calls the plotting/simulation runners.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.final_validation import require_gate,verify_component,verify_mapping,verify_history
from lib.final_paper_visuals import CATALOGUE,BUNDLES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_figures(root,example):
    root=Path(root).resolve()
    gate=require_gate(root)
    for component in ('theory','performance','prediction','computation'):
        verify_component(root,example,'paper',component)
    folder=root/'results'/example
    manifest=folder/'figure_manifest.json'
    meta=json.loads(manifest.read_text())
    expected={name+'.pdf' for name in [*CATALOGUE,*BUNDLES]}
    if (meta.get('status')!='completed' or meta.get('stage')!='paper'
            or meta.get('example')!=example or set(meta.get('figures',{}))!=expected):
        raise RuntimeError('A complete, current nineteen-PDF paper figure set is required for '+example)
    if (meta.get('titles') is not False or meta.get('text_usetex') is not True
            or meta.get('preview_dpi')!=300 or meta.get('paper_width_in')!=7.):
        raise RuntimeError('Paper presentation requirements are unresolved')
    from lib.final_validation import gate_path
    if meta.get('validation_sha256')!=sha(gate_path(root)):
        raise RuntimeError('Final figures refer to another general-validation run')
    verify_mapping(root,meta['sources']);verify_mapping(root,meta['input_manifests'])
    for name,digest in meta['files'].items():
        if sha(folder/name)!=digest:raise RuntimeError('Changed final figure/preview: '+name)
    required=('configurations','seed_sets','repetition_unit','repetition_counts',
              'numerical_tolerances','numerical_status','source_hashes',
              'data_manifest_hashes','proposed_latex_caption','placement',
              'formula_or_claim','producer','pages','output_hashes','omitted_conditions')
    for name,entry in meta['figures'].items():
        if entry.get('status')!='completed' or any(key not in entry for key in required):
            raise RuntimeError('Incomplete figure provenance: '+name)
        verify_mapping(root,entry['source_hashes']);verify_mapping(root,entry['data_manifest_hashes'])
        for artifact,digest in entry['output_hashes'].items():
            if sha(folder/artifact)!=digest:raise RuntimeError('Changed figure output: '+artifact)
        if any(page.get('titles') is not False or abs(page['figure_size_in'][0]-7.)>1e-12 for page in entry['pages']):
            raise RuntimeError('A paper page violates width/title requirements: '+name)
    return meta


SECTIONS=(
    ('1. Model, geometry and theoretical checks',('setup','information','posterior_curvature','tracking_bound','block_updates')),
    ('2. Held-out task performance and controls',('task_performance','additional_comparisons')),
    ('3. Directional posterior uncertainty',('posterior_anisotropy',)),
    ('4. Variance learning and future predictions',('variance_estimation','noise_adaptation','predictive_calibration','predictive_scores','model_mismatch')),
    ('5. Computation and numerical precision',('computation_time','numerical_accuracy')),
    ('6. Main-paper composition',('paper_summary',)),
)


INTRODUCTION={
'beamforming':r'''# Final beamforming study

The latent interference line is tracked from fully observed, interference-only snapshots
\[
u_t=h_tc_t+e_t,\qquad c_t\sim\mathcal{CN}(0,\tau_t^2),\quad e_t\sim\mathcal{CN}(0,\sigma_t^2I_n).
\]
The coefficient and interference direction are never estimator inputs. The known desired steering vector enters the subsequent distortionless beamforming decision. The principal cohort adds rank-one task studies at $n=32,64$ and the declared $n=128$ feasibility condition.

For a block of $L$ snapshots generated at a common line with fixed variances, the increment is $\tau^2 WW^H/[\sigma^2(\sigma^2+\tau^2)]$, without division by $L$. Proper-complex noise has real and imaginary coordinate variances $\sigma^2/2$. Discounted/transported aggregation additionally retains every within-block weight and the recorded frame transformations.

The common $\overline P_t^+$, modal projector and $\overline P_t^{\mathrm{iso}}$ controls share the same parameter, scales, data and reporting times. The isotropic mean matches the mode and mean squared chordal distance while discarding directional differences. With uniform initialization and fixed marginal scales the mode equals EW PCA; these are not independent algorithms. GREAT is explicitly a complex extension of the real published algorithm. Loaded modal and EW covariance references have independently selected validation settings.

This notebook reads frozen, content-verified paper evidence only. Its order is theory, task performance, anisotropy, variance learning/prediction, and computation. Notebook markdown supplies explanations; the PDFs contain only panel letters, axes, legends and event markers. The published catalogue uses seven-inch PDF pages. Editable notebook exports use the per-cell FIGSIZE values below, with actual LaTeX text, vector text/lines and 300-dpi raster layers.
''',
'contact':r'''# Final contact study

The physical example is an ideal maintained bilateral contact in $\mathbb R^3$. Known resistance, the applied input and measured wrist-frame changes define recorded transformed input/output pairs
\[
u_t=P_{\mathcal X_t}v_t+e_t,\qquad \mathcal X_t\in\mathrm{Gr}(2,3),\quad e_t\sim\mathcal N(0,\sigma_t^2I_3).
\]
For matched isotropic resistance $D=dI$, the recorded requested velocity is $v_t=f_{\mathrm{eff},t}/d$. Let $n_t$ be the generating unit normal, so $P_{\mathcal X_t}=I_3-n_tn_t^T$; $n_t$ is withheld from the estimator. A noisy velocity is directly measured rather than inferred from differenced positions in the matched model. Surface positions and normals generate/score the experiment; the estimator receives only the recorded pairs and measured coordinate changes. Friction, contact loss, resistance errors and differenced-position measurements are separate mismatch experiments.

For a common-state block of $L$ observations with fixed variance, the increment is $(WV^T+VW^T-VV^T)/(2\sigma^2)$, without division by $L$. Known frame changes are aligned before a fixed-boundary block fit. Holding an output between completed blocks, exact aggregation and fitting one constant state over a moving block are different comparisons.

An admissible plane is represented by an unoriented normal line: $n$ and $-n$ describe the same plane. If the plane has parameter $A$, the normal has parameter $-A$. The actual and matched-isotropic normal laws share a mode and mean squared chordal distance. Their directional response risks can still differ. The anisotropy risk panel uses unit query directions; physical task panels use the declared requested speed and millimetre-per-second units. The constructed swapped-eigenvalue illustration is separately identified as an exact identity, not data-generated task performance.

The common $\overline P_t^+$, modal and $\overline P_t^{\mathrm{iso}}$ controls share the same posterior parameter and inputs. The regularized projector is $\rho P_{\widehat{\mathcal X}_t}+(1-\rho)(2/3)I_3$, with $\rho$ selected on independent validation episodes. This control uses the recorded inputs. GREAT is response-only and receives no recorded $v_t$, so its comparison cannot isolate Bayesian uncertainty. The plant trajectory is driven by a preassigned force program. Estimated responses and soft commands are scored counterfactually on this common stream; they are not fed back to regenerate a closed-loop trajectory. Soft command costs are expected penalties, not hard feasibility guarantees.

This notebook displays saved, source-verified evidence only, in theoretical-to-application order. All paper PDFs are title-free, seven inches wide, use actual LaTeX text and keep vector text/lines. Preview and raster layers use 300 dpi. Figures do not promise universal superiority, calibrated coverage under unmodelled motion, or exact EM ascent for approximate E-steps.
'''}


SECTION_NOTES={
1:r'''These checks distinguish the known-input and marginalized observation laws, finite-data curvature and expected information, exact aggregation and delayed actions. The sharper tracking envelope is evaluated only where its stated sufficient condition holds. Bound gaps, initialization and drift use generating quantities available for analysis, not online credible radii. Nonuniform priors, zero-signal data and concentrated inputs remain separate cohorts.''',
2:r'''Losses are paired on Monte Carlo simulations and common held-out data. Observation/query averages within one episode are not extra repetitions. Main panels use predeclared representative conditions; the complete grids and strong controls remain in the additional comparisons and machine-readable records. A gain over GREAT may reflect input information or optimization/tracking differences; a gain attributable to posterior anisotropy requires the same-mode and matched-isotropic comparisons.''',
3:r'''A fixed-radius 3D sphere encodes density by color under one shared logarithmic normalization. The continuous blue--purple--pink--yellow gradient changes only the color assigned to a density value; saved coordinates, camera, density values and angular widths are retained. Colors do not denote probability-mass quantiles. Gray curves are visible coordinate great circles that help locate directions on the sphere; they are not density contours or credible regions. Contact normals use true antipodal real-line densities. The complex-array sphere is a real-coefficient density section at the $n=64$, $t=T=200$ stationary checkpoint: it is not a physical far-field radiation pattern, a full posterior, or a normalized sphere marginal. The setup figure shows the physical array response. No estimator or benchmark parameters have changed. Mode, radius and mean overlap do not determine directional response risk. Visible anisotropy does not itself prove a repeated-experiment task advantage. PDFs retain vector LaTeX text and curves; all current preview and raster layers are 300 dpi. The font sizes, line widths and figure layout follow the implementation-notebook protocol. Inline images are scaled for reading; the full-resolution PNG and PDF links preserve the export resolution.''',
4:r'''Causal scale fits use completed disjoint past blocks and change future variances only. Each observation is assimilated once. The saved pre-observation predictive score excludes the scored observation from fitting. State, clean-response and noisy-next-observation intervals are distinct targets. Inclusion, widths and proper scores must be read together; integration/quantile precision and frequentist calibration are different checks.''',
5:r'''The cost plots separate measurement updates, task decisions, uncertainty reports and variance fitting. Each timing on the first PDF page processes exactly 64 observations. Its upper panel includes an update for every observation and computes a task decision at the stated interval. The lower panel separates the nonoverlapping operations for the same 64 observations: optional uncertainty reports and EM fits occur every 32 observations, independently of how often a task decision is computed. Less frequent task decisions still assimilate every observation. Summed medians of individual operations need not equal the median total elapsed time. The second PDF page reports memory, with stored arrays distinguished from tracked allocations. Timings use one predeclared data set and repeated warmed single-thread executions; repetitions of the clock are not additional Monte Carlo simulations or worst-case deadlines. Native rank/codimension-one first-moment accuracy does not certify arbitrary-rank moments, floating-point error, second moments or tails.''',
6:r'''The summary is a composition of the same verified arrays. It contains no extra experiment and no favorable-trajectory selection. Proposed captions, exact page selectors, configuration/seed sets, omitted-condition locations, numerical status and hashes are in the figure manifest.''',
}


def _table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def _number(value):
    return f'{float(value):.3g}'


def _stationary_beamforming_findings(root):
    """Summarize the fixed-horizon records without reusing old t/n tables."""
    import numpy as np
    import pandas as pd
    from scipy.stats import t as student_t
    from lib.final_stationary_evidence import (
        verify_stationary_component, stationary_component_path,
    )

    manifest=verify_stationary_component(root,stage='paper')
    folder=stationary_component_path(root,stage='paper').parent
    config=manifest['config']
    horizon=int(config['stationary_T'])
    primary=int(config['stationary_primary_t'])
    if horizon!=200 or primary!=horizon:
        raise RuntimeError('The stationary dimension comparison requires T=t=200')
    dimensions=[int(value) for value in config['dimensions']]
    scheduled=int(config['test_episodes'])
    frame=pd.read_csv(folder/'stationary.csv').rename(columns={'trial':'episode'})
    if 'split' in frame:
        frame=frame[frame.split.eq('test')]
    frame=frame[frame.condition.eq('static')&frame.t.eq(primary)&frame.n.isin(dimensions)]
    if frame.duplicated(['n','method','episode']).any():
        raise RuntimeError('Duplicated fixed-horizon stationary measurements')

    def interval(values):
        values=np.asarray(values,float)
        values=values[np.isfinite(values)]
        if len(values)<2:
            raise RuntimeError('Insufficient resolved fixed-horizon simulations')
        mean=float(values.mean())
        half=float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
        return mean,mean-half,mean+half,len(values)

    methods=('mean','mode','covariance_loading','great_window')
    pairs=[];means=[]
    for n in dimensions:
        group=frame[frame.n.eq(n)]
        for method in (*methods,'isotropic'):
            if group[group.method.eq(method)].episode.nunique()!=scheduled:
                raise RuntimeError(f'Missing stationary T={horizon} records for n={n}, method={method}')
        values=group.pivot(index='episode',columns='method',values='true_ratio_db')
        for method in methods:
            mean,lower,upper,count=interval(values[method])
            means.append(dict(n=n,t=primary,method=method,mean=mean,lower=lower,upper=upper,
                              resolved_pairs=count,episodes=scheduled))
        for reference in ('mode','isotropic','covariance_loading','great_window'):
            gain,lower,upper,count=interval(values[reference]-values['mean'])
            pairs.append(dict(n=n,t=primary,reference=reference,gain=gain,lower=lower,upper=upper,
                              resolved_pairs=count,episodes=scheduled))
    return config,pd.DataFrame(pairs),pd.DataFrame(means)


def paper_findings(root,example):
    """Describe fixed paper cohorts from their published paired summaries."""
    import pandas as pd
    folder=Path(root)/'results'/example/'tables/paper'
    labels={'mode':r'$P_{\widehat{\mathcal X}_t}$',
        'isotropic':r'$\overline P_t^{\mathrm{iso}}$',
        'regularized':'Regularized projector',
        'covariance_loading':'Regularized sample covariance','great_window':'GREAT'}
    rows=[];signs=[]
    mean_power_table=''
    if example=='beamforming':
        config,chosen,means=_stationary_beamforming_findings(root)
        dimensions=[int(value) for value in config['dimensions']]
        unit='dB';selectors=(r'the stationary $t=T=200$ measurements at $n='
            +','.join(map(str,dimensions))+r'$, using $\ell_T$ from the exact covariance')
        for row in chosen.itertuples():
            rows.append([rf'$n={row.n},\ t={row.t}$',labels[row.reference],
                _number(row.gain)+' ['+_number(row.lower)+', '+_number(row.upper)+']',str(row.resolved_pairs)+'/'+str(row.episodes)])
            signs.append((row.lower,row.upper))
        if len(rows)!=4*len(dimensions):raise RuntimeError('Missing fixed-horizon beamforming paired cells')
        mean_rows=[]
        for n in dimensions:
            group=means[means.n.eq(n)].set_index('method')
            mean_rows.append([f'$n={n}$',*[
                _number(group.loc[method,'mean'])+' ['+_number(group.loc[method,'lower'])+', '
                +_number(group.loc[method,'upper'])+']'
                for method in ('mean','mode','covariance_loading','great_window')]])
        mean_power_table=('At the same $T=200$ observations for every sensor count, the mean exact-covariance '
            r'losses $\ell_T$ and pointwise Student 95% intervals in dB are:'+'\n\n'
            +_table(['Sensors','Posterior mean','Mode','Regularized sample covariance','GREAT'],mean_rows)+'\n\n')
    else:
        frame=pd.read_csv(folder/'performance/paired.csv')
        chosen=frame[frame.metric.eq('response_mse')&frame.reference.isin(['mode','isotropic','regularized','great_window'])
            &((frame.condition.eq('static_concentrated')&frame.endpoint.eq('checkpoint')&frame.t.eq(16))
              |(frame.condition.eq('sphere')&frame.endpoint.eq('episode_average'))) ]
        unit='$(\\mathrm{mm}/\\mathrm{s})^2$'
        selectors='the concentrated-input stationary checkpoint $t=16$ and the full sphere simulation'
        for row in chosen.itertuples():
            label='Concentrated, $t=16$' if row.condition=='static_concentrated' else 'Sphere simulation'
            rows.append([label,labels[row.reference],
                _number(1e6*row.gain)+' ['+_number(1e6*row.lower)+', '+_number(1e6*row.upper)+']',str(row.finite_episodes)+'/'+str(row.episodes)])
            signs.append((row.lower,row.upper))
        if len(rows)!=8:raise RuntimeError('Expected all eight predeclared contact paired cells')
    positive=sum(lower>0 for lower,upper in signs);negative=sum(upper<0 for lower,upper in signs)
    comparisons=len(rows)
    task=(mean_power_table+'The paired table reports reference loss minus posterior-mean loss on '+selectors+'. Positive values favor the posterior mean; brackets are paired pointwise 95% Student intervals in '+unit+'. '
        +f'Across these {comparisons} fixed comparisons, {positive} intervals lie above zero, {negative} lie below zero, and {comparisons-positive-negative} include zero. '
        +'Resolved pairs are shown relative to scheduled Monte Carlo simulations. These cells are displayed together, including losses and uncertain differences; this count is descriptive and has no simultaneous-coverage guarantee.\n\n'
        +_table(['Condition','Reference','Paired gain [95% CI]','Resolved pairs'],rows)
        +'\n\nA benefit over the mode can arise from shrinkage. A benefit over the matched isotropic control supports the value of directional posterior moments for that task and cohort. The latter comparison is the direct anisotropy test. A GREAT comparison also changes the estimator and, in contact, the available input information.')
    if example=='contact':
        checks=frame[frame.metric.eq('response_mse')&frame.reference.eq('isotropic')
            &((frame.condition.eq('plane')&frame.endpoint.eq('episode_average'))
              |(frame.condition.eq('sphere')&frame.endpoint.eq('checkpoint')&frame.t.eq(192)))]
        if len(checks)!=2:raise RuntimeError('Missing fixed plane/final-sphere uncertainty controls')
        checks=checks.set_index('condition')
        def paired_text(row):
            return _number(1e6*row.gain)+' ['+_number(1e6*row.lower)+', '+_number(1e6*row.upper)+']'
        task+='\n\nThe full grid also retains the plane time-averaged isotropic-minus-full gain, '+paired_text(checks.loc['plane'])+', and the final sphere checkpoint gain, '+paired_text(checks.loc['sphere'])+', in $(\\mathrm{mm}/\\mathrm{s})^2$. These fixed checks show why an time-averaged gain should not be interpreted as uniform superiority at every time and input condition.'
    paired=pd.read_csv(folder/'prediction/paired.csv')
    chosen=paired[paired.reference.eq('misspecified')&paired.reference_control.eq('mean')&paired.endpoint.eq('future_task_loss')]
    scale=1e6 if example=='contact' else 1.
    noise_rows=[[row.condition.replace('_',' '),_number(scale*row.gain)+' ['+_number(scale*row.lower)+', '+_number(scale*row.upper)+']',row.episodes]
        for row in chosen.itertuples()]
    if len(noise_rows)!=(2 if example=='contact' else 3):raise RuntimeError('Missing declared noise-learning comparison')
    noise=('Across all declared interventions, fixed-minus-learned observed future-task gains are '+
        '; '.join(row[0]+': '+row[1] for row in noise_rows)+'. These are paired pointwise 95% intervals over '
        +str(int(chosen.episodes.iloc[0]))+' Monte Carlo simulations; positive gains favor learning. '
        +('Contact values are in $(\\mathrm{mm}/\\mathrm{s})^2$.' if example=='contact' else 'Beamforming values are held-out disturbance power, not decibels.')
        +' Observation/squared response errores are averaged inside each episode. The [paired summaries](../results/'+example+'/tables/paper/prediction/paired.csv) retain every control and endpoint.\n\n'
        +'A fit changes only future observations and decisions. The shaded variance trajectories show Monte Carlo spread, not a posterior distribution of the variance. '
        +('The supplementary sphere page separately shows an individual pre-observation subspace belief. ' if example=='contact' else '')
        +'These displays do not establish calibration of empirical-Bayes scales.')
    classifications=[]
    for selected,label in [(chosen.lower.gt(0),'lie above zero'),
                           (chosen.upper.lt(0),'lie below zero'),
                           (chosen.lower.le(0)&chosen.upper.ge(0),'include zero')]:
        names=chosen.loc[selected,'condition'].str.replace('_',' ').tolist()
        if names:classifications.append('the intervals '+label+' for '+', '.join(names))
    noise+='\n\nIn these fixed intervention comparisons, '+'; '.join(classifications)+'.'
    calibration=pd.read_csv(folder/'prediction/calibration_summary.csv')
    chosen=calibration[calibration.regime.eq('drifting')&calibration.stress.eq('matched')&calibration.split.eq('test')
        &calibration.method.eq('learned')&calibration.t.eq(64)&calibration.nominal.eq(.95)
        &calibration.kind.isin(['state_hpd','clean_response','noisy_observation'])]
    cal_labels={'state_hpd':'State HPD',
        'clean_response':r'$q^H Pq$' if example=='beamforming' else r'$e_1^T Pq$',
        'noisy_observation':r'$\operatorname{Re}(u_{t,1})$'}
    if len(chosen)!=3:raise RuntimeError('Missing fixed drifting calibration checkpoint')
    matched=calibration[calibration.regime.eq('prior_predictive')&calibration.stress.eq('matched')&calibration.split.eq('test')
        &calibration.method.eq('exact_known')&calibration.t.eq(64)&calibration.nominal.eq(.95)
        &calibration.kind.isin(['state_hpd','clean_response','noisy_observation'])]
    if len(matched)!=3:raise RuntimeError('Missing matched prior-predictive checkpoint')
    def coverage_sentence(frame):
        return '; '.join(cal_labels[row.kind]+': '+_number(row.coverage)+' ['+_number(row.lower)+', '+_number(row.upper)+']'
            for row in frame.itertuples())
    misses=[cal_labels[row.kind] for row in chosen.itertuples() if row.upper<.95]
    calibration_note=('For the matched joint prior-predictive test with true variances, the predeclared $t=64$ 95% reports have inclusion '+coverage_sentence(matched)+'. '
        +'The corresponding learned-variance reports under drift have inclusion '+coverage_sentence(chosen)+'. Brackets are exact binomial 95% intervals over '
        +str(int(chosen.episodes.iloc[0]))+' Monte Carlo simulations. '
        +('The upper interval is below 0.95 for '+', '.join(misses)+', supporting undercoverage in those drifting targets. ' if misses else 'These drifting intervals do not establish a general coverage guarantee. ')
        +'There are '+str(int(matched.unresolved.sum()))+' unresolved matched reports and '+str(int(chosen.unresolved.sum()))+' unresolved drifting reports in these cells.\n\n'
        +'The [calibration summaries](../results/'+example+'/tables/paper/prediction/calibration_summary.csv) also report widths in each target\'s native units. '
        +'Numerical tolerance controls how accurately a report is computed; it does not establish repeated-experiment coverage under motion or an incorrect likelihood.')
    timing=pd.read_csv(folder/'computation/summary.csv')
    case,n,decision_interval=('task_contact',3,4) if example=='contact' else ('task_stationary',64,64)
    chosen=timing[timing.case.eq(case)&timing.n.eq(n)&timing.action_cadence.eq(decision_interval)&timing.report_every.eq(0)
        &timing.method.isin(['mean','mode','isotropic','great_window'])]
    if len(chosen)!=4:raise RuntimeError('Missing matched task-decision-interval task timing cohort')
    mean_time=float(chosen[chosen.method.eq('mean')].median_ms.iloc[0])
    time_parts=[]
    for row in chosen.itertuples():
        label='Posterior mean' if row.method=='mean' else labels[row.method]
        time_parts.append(label+': '+_number(row.median_ms)+' ms (90th percentile '+_number(row.p90_ms)+' ms)')
    cost=('With task decisions every '+str(decision_interval)+' observations and no uncertainty reports, the warmed single-thread times to process '+str(int(chosen.observations.iloc[0]))+' observations are '
        +'; '.join(time_parts)+'. Each clock processes '+str(int(chosen.observations.iloc[0]))+' observations from the same predeclared task episode; '
        +str(int(chosen.repetitions.iloc[0]))+' clock repetitions measure execution variability, not independent task-performance episodes. '
        +'The posterior mean takes '+', '.join(_number(mean_time/row.median_ms)+' times the '+labels[row.method]+' time' for row in chosen.itertuples() if row.method!='mean')+'. '
        +'The [cost summaries](../results/'+example+'/tables/paper/computation/summary.csv) include every schedule and phase.\n\n'
        +'The framework is useful when posterior means improve a consequential decision, directional risk or predictive uncertainty reporting is required, or future variance learning is useful enough to justify its measured cost. '
        +'A modal or optimized tracking rule remains a sensible choice when only a subspace point is required and that cost dominates. '
        +'The present results do not support universal superiority, arbitrary-rank scalability, or automatic coverage under mismatch.')
    return dict(task=task,noise=noise,calibration=calibration_note,cost=cost)


def _preview_selection(meta,name,**selector):
    pages=meta['figures'][name+'.pdf']['pages']
    matches=[page for page in pages if all((page.get('condition')==value if key=='condition'
        else page.get('selection',{}).get(key)==value) for key,value in selector.items())]
    if len(matches)!=1:raise RuntimeError(f'Expected one predeclared {name} page: {selector}')
    return int(matches[0]['page'])


DETERMINISTIC_CHECKS={
'beamforming':r"""import numpy as np
from lib.notebook_inputs import load_final_array, load_final_config
from lib.final_prediction import mvdr
record=load_final_array(ROOT,EXAMPLE,'performance','stories')
M,a=record['setup_M'],record['setup_a']
R=float(record['setup_sigma2'])*np.eye(len(M))+float(record['setup_tau2'])*M
w=mvdr(R,a)
assert np.allclose(np.vdot(w,a),1.,rtol=1e-11,atol=1e-11)
assert np.allclose(R@w,np.vdot(w,R@w).real*a,rtol=1e-10,atol=1e-11)
index=list(record['setup_methods']).index('mean')
assert np.allclose(w,record['setup_weights'][index],rtol=1e-10,atol=1e-11)
print('Saved mean: distortionless action, first-order condition and saved weight agree.')
""",
'contact':r"""import numpy as np
from lib.notebook_inputs import load_final_array, load_final_config
from lib.final_contact import soft_commands
from lib.geometry import risk
record=load_final_array(ROOT,EXAMPLE,'performance','anisotropy')
M=record['plane_mean']
lam=float(load_final_config(ROOT,EXAMPLE,'performance')['task_lambda'])
desired=record['physical_queries']
commands=soft_commands(M,desired,lam)
assert np.allclose(commands@(np.eye(3)+lam*(np.eye(3)-M)).T,desired,rtol=1e-11,atol=1e-12)
computed=np.asarray([risk(M,v) for v in record['unit_queries']])
assert np.allclose(computed,record['risks_full'],rtol=1e-10,atol=1e-11)
Q,Mi=record['unit_queries'],record['iso_plane_mean']
expected_iso=np.sum((Q@M.T)*Q,axis=1)-2*np.sum((Q@M.T)*(Q@Mi.T),axis=1)+np.sum((Q@Mi.T)**2,axis=1)
excess=np.sum((Q@(M-Mi).T)**2,axis=1)
assert np.allclose(expected_iso,computed+excess,rtol=1e-10,atol=1e-11)
print('Saved mean: command first-order condition, directional risk and isotropic-action excess agree.')
"""}



MAIN_PLOT_NOTES = {
'beamforming_stationary': r'''
The stationary curves use the same $t=T=200$ observations at every sensor count
$n\in\{32,64,128\}$. The settings were selected using separate validation measurements.
The plotted `true_ratio_db` values use the exact covariance; shading is a pointwise
Student 95% interval for the posterior-mean average over 256 Monte Carlo simulations.
The separate supplementary dimension plot retains the same fixed-horizon results.
''',
'beamforming_task': r'''
The array comparison uses the exact covariance
$R_t=\sigma_t^2I_n+\tau_t^2P_{\mathcal X_t}$ and
$p_t^\star=(a^H R_t^{-1}a)^{-1}$. Its ordinate is
$\ell_t=10\log_{10}(w_t^H R_tw_t/p_t^\star)$ in dB.
The stationary curves use the same $t=T=200$ observations at every sensor count
$n\in\{32,64,128\}$. The settings were selected using separate validation measurements.
Shading is a pointwise Student 95% interval for the posterior-mean average over 256 Monte Carlo simulations.
The separate supplementary dimension plot retains the same fixed-horizon results.
''',
'beamforming_noise': r'''
Here $n=32$, $T=256$, and EM uses $L=32$ measurements per block and two iterations.
The true noise variance is $0.2$ for $t\le128$ and $0.8$ for $t\ge129$;
$\tau_t^2=1$. A fit completed after measurement $t$ is used from $t+1$.
The displayed $\widehat\sigma_t^2,\widehat\tau_t^2$ are the estimates available
before receiving $u_t$, at the stored integer reporting indices.
The full dashed truth changes between measurements 128 and 129.
Forgetting factors remain $\beta=0.98,0.90,0.98$ for true, fixed and EM variances.
The lower-left plot uses the same exact-covariance $\ell_t$, with the beamformer formed
from the prior mean $\overline P_t^-$.
The lower-right plot is $\log\pi_t(u_t)$, where
$\pi_t(u)=\int p(u\mid\mathcal X,\widehat\theta_t)q_{A_t^-}(\mathcal X)\,d\mu(\mathcal X)$.
Variance bands show the middle 50% of pre-correction estimates across simulations;
power and log-density bands are Student 95% confidence intervals for the Monte Carlo means.
Averaging the eight reporting values per simulation gives exact output-power
comparisons of $0.434633$, $1.056295$ and $0.378372$ dB for EM, fixed and true
variances. Corresponding mean log densities are $-53.3$, $-237.4$ and $-40.9$.
''',
'contact_task': r'''
For $v_t^{(j)}=v_0R_t^\top e_j$, $v_0=0.02\,\mathrm{m/s}$, the velocity error is
$\mathcal E_t(D_t)=\frac13\sum_{j=1}^3\|D_tv_t^{(j)}-P_{\mathcal X_t}v_t^{(j)}\|^2$.
The squared norm sums the three velocity coordinates; there is no further division by three.
Multiplying the saved values by $10^6$ converts $(\mathrm{m/s})^2$ to $(\mathrm{mm/s})^2$.
The stationary panel uses $t=1,4,16,32$. On the sphere of radius $0.05$ m,
errors are averaged over all $t=4,8,\ldots,192$ within each simulation before
forming the mean and Student 95% confidence interval across the 256 simulations.
The left plot shows the posterior-mean interval; the right shows an interval for every method.
''',
'contact_risk': r'''
The saved concentrated-input example has $t=16$ and generating normal $e_3$.
The spheres show density relative to normalized invariant measure, with their
unchanged camera, coordinates, common mode marker and logarithmic color limits.
Antipodal normals describe the same plane. Define
$v(\theta)=v_0(\cos\theta e_2+\sin\theta e_3)$ and
$r_t(v)=v^\top[\overline P_t^+-(\overline P_t^+)^2]v$.
Panel (c) plots $r_t(v(\theta))/v_0^2$ with $\theta$ in radians.
The saved unit-input curve equals this normalized quantity exactly; no numerical
rescaling is applied. Each curve uses its own distribution and mean matrix.
''',
}


def editable_setup(example):
    from importlib import import_module
    data_import=import_module('scripts.notebook_plot_cells_'+example).DATA_IMPORT
    stem='01_array' if example=='beamforming' else '03_contact'
    return rf'''from pathlib import Path
import os
import sys
ROOT = Path.cwd() if (Path.cwd()/'lib').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
EXAMPLE = {example!r}

# Use the optional local TeX installation when running in the cloud environment.
# On a machine with a system TeX installation this block has no effect.
LOCAL_TEX = Path('/workspace/.local/tex')
if (LOCAL_TEX / 'usr/bin/dvipng').exists():
    os.environ['PATH'] = str(LOCAL_TEX / 'usr/bin') + os.pathsep + os.environ['PATH']
    os.environ.setdefault('TEXMFHOME', str(LOCAL_TEX / 'usr/share/texmf'))

# Import data readers before setting the visible plotting defaults.
{data_import}
from lib.notebook_inputs import verify_saved_inputs
from IPython import get_ipython
from IPython.display import display, Markdown
import numpy as np
import matplotlib.pyplot as plt

INPUT_HASHES = verify_saved_inputs(ROOT, EXAMPLE)
EDITABLE_OUTPUT = ROOT / 'results/notebook_exports/{stem}'
EDITABLE_OUTPUT.mkdir(parents=True, exist_ok=True)

# Shared defaults are editable here. Each figure cell also owns DPI and FIGSIZE.
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
    get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {{'bbox_inches': None}}")
plt.rcParams.update({{
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{{amsmath}}\usepackage{{amssymb}}',
    'font.family': 'serif', 'font.size': 25,
    'axes.labelsize': 25, 'axes.titlesize': 25,
    'legend.fontsize': 22, 'xtick.labelsize': 23, 'ytick.labelsize': 23,
    'lines.linewidth': 1.6, 'lines.markersize': 4,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': None,
    'axes.spines.top': False, 'axes.spines.right': False,
}})
print('Verified', len(INPUT_HASHES), 'saved numerical inputs.')
print('Editable PDF output:', EDITABLE_OUTPUT)
'''

def build(root,example):
    root=Path(root);meta=verify_figures(root,example);findings=paper_findings(root,example)
    displayed=[('setup',1),('information',1),('task_performance',1),('posterior_anisotropy',1),
        ('noise_adaptation',_preview_selection(meta,'noise_adaptation',condition='noise_change')),
        ('predictive_calibration',_preview_selection(meta,'predictive_calibration',condition='prior_predictive',t=64,diagnostic='inclusion')),
        ('predictive_calibration',_preview_selection(meta,'predictive_calibration',condition='drifting',t=64,diagnostic='inclusion')),
        ('computation_time',1)]
    previews={}
    for name,page in displayed:
        suffix='' if page==1 else '_page'+str(page)
        path=Path('results')/example/'preview'/(name+suffix+'.png')
        previews[path.as_posix()]=sha(root/path)
    cells=[nbformat.v4.new_markdown_cell(INTRODUCTION[example])]
    manifest_digest=sha(root/'results'/example/'figure_manifest.json')
    producer_digest=sha(root/'scripts/make_final_notebooks.py')
    cells.append(nbformat.v4.new_code_cell(editable_setup(example)))
    def markdown(text):cells.append(nbformat.v4.new_markdown_cell(text))
    def image(index):
        name,page=displayed[index]
        module_name='scripts.notebook_plot_cells_'+example
        from importlib import import_module
        source=import_module(module_name).get_plot_cell(name,page)
        info={'editable_plotting_role':'plot','figure_id':name,'source_page':page}
        if example=='beamforming' and name=='task_performance':
            info.update(figure_id='task_performance',source_catalogue_page=True)
        cells.append(nbformat.v4.new_code_cell(source,metadata=info))
    markdown('## 1. Measurements and geometry\n\n'+SECTION_NOTES[1]);image(0)
    if example=='contact':
        information_note=r'''## 2. Which normal rotations the observations distinguish

The contact likelihood conditions on the applied inputs. Its Fisher matrix $\mathcal F_{\mathrm{cond}}$ measures how strongly the distribution of observed velocities changes when the contact plane rotates. This is a property of the observation law; it is different from the curvature of a finite-data posterior.

Use the same 16 recorded inputs from the fixed concentrated-input example. At its generating plane, $n_0=e_3$ and $X_0=(e_1,e_2)$. The manuscript's local graph coordinates are $z=\operatorname{vec}(Z)=(z_1,z_2)^\top$, with
\[
n(Z)=\frac{n_0-X_0z}{\sqrt{1+\|z\|^2}}.
\]
Let $p_z$ be the joint Gaussian velocity law at this rotated plane, with means $P_{\mathcal X(Z)}v_j$ and the same true measurement variance. For this fixture the exact divergence is
\[
D_{\mathrm{KL}}(p_0\|p_z)
=\frac{z^\top\mathcal F_{\mathrm{cond}}z}{2(1+\|z\|^2)}.
\]
The left panel shows equal values of this divergence in the two native graph directions. The right panel converts each direction into the physical normal-rotation angle $\delta$: a $5^\circ$ rotation in the first direction and a $30^\circ$ rotation in the second both give a divergence of about $0.24$. The same input program therefore distinguishes these rotations very differently. Near the origin the expression reduces to the paper's $D_{\mathrm{KL}}=\tfrac12z^\top\mathcal F_{\mathrm{cond}}z+o(\|z\|^2)$.

These contours describe differences between likelihoods, rather than posterior credible regions. The generating plane and true variance are used for this theoretical analysis; neither the true normal nor extra observations are provided to the estimator. No new simulation, fitting or posterior calculation is introduced.'''
    else:
        information_note=r'''## 2. What the information calculation checks

The expected Fisher information describes the observation law and excitation; finite-data posterior curvature is a different object. The first panel compares analytical and empirical eigenvalues in aligned tangent coordinates. The second separates recorded-input and marginalized-input information.'''
    markdown(information_note+''' [Curvature](../results/'''+example+'''/posterior_curvature.pdf), [sharp tracking bound](../results/'''+example+'''/tracking_bound.pdf), and [block updates](../results/'''+example+'''/block_updates.pdf) retain the locality flags, sufficient conditions, initialization/noise/drift contributions and approximation checks.''');image(1)
    task_equation=(r'''With $\overline P_t^+=\mathbb E[P_{\mathcal X_t}\mid u_{1:t}]$, the Bayes disturbance covariance is $\overline R_t=\sigma^2I_n+\tau^2\overline P_t^+$. For known desired steering vector $a$, the distortionless decision is
\[
w_t^{\mathrm B}=\frac{\overline R_t^{-1}a}{a^H\overline R_t^{-1}a}.
\]
The mode and matched-isotropic controls substitute their own projectors/means into the same decision.''' if example=='beamforming' else r'''For a recorded query $v=f_{\rm eff}/d$, the squared-error Bayes response is $\widehat u_t^{\mathrm B}=\overline P_t^+v$. For desired velocity $v_{\rm des}$ and the soft penalty $\lambda$, the command is
\[
v_t^{\mathrm B}=[I_3+\lambda(I_3-\overline P_t^+)]^{-1}v_{\rm des}.
\]
The mode, matched-isotropic and regularized-projector controls substitute their own matrices into the same task rule. $\overline P_t^+$ is a plane-projector mean, so $I_3-\overline P_t^+$ is the corresponding normal-line mean. The penalty is an expected cost, not a hard contact guarantee.''')
    markdown('## 3. Task decisions and outcomes\n\n'+task_equation+'\n\n'+SECTION_NOTES[2]+'\n\n'+MAIN_PLOT_NOTES[example+'_task'])
    markdown(r'''Define the matched isotropic mean by:
\[
\zeta_t=k-\operatorname{tr}(\overline P_t^+P_{\widehat{\mathcal X}_t}),\qquad
\overline P_t^{\mathrm{iso}}=
\left(1-\frac{\zeta_t}{k}\right)P_{\widehat{\mathcal X}_t}
+\frac{\zeta_t}{n-k}(I_n-P_{\widehat{\mathcal X}_t}).
\]
It retains the mode and mean overlap and makes the eigenvalues equal within the modal subspace and within its orthogonal complement. In beamforming this is the saved matched **mean**; no isotropic density or concentration is fitted for the plot.''')
    markdown('The following deterministic library calls reuse the displayed episode’s stored mean; they do not rerun inference or generate observations.')
    cells.append(nbformat.v4.new_code_cell(DETERMINISTIC_CHECKS[example]));image(2)
    if example=='contact':
        markdown(r'''The left plot uses stationary concentrated inputs. The right plot reports absolute time-averaged $\mathcal E_t(D_t)$ for the moving sphere condition: every method, including $\overline P_t^+$, has a pointwise 95% interval across the same 256 Monte Carlo simulations. MSE is the squared error relative to the clean response $P_{\mathcal X_t}v_t$, averaged within each episode, and is displayed in $(\mathrm{mm}/\mathrm{s})^2$. The supplementary second page shows $\mathrm{MSE}_{\mathrm{ref}}-\mathrm{MSE}_{\mathrm B}$ on paired episodes. This is an absolute difference in squared velocity units, rather than a ratio or relative MSE. The mean's self-difference is exactly zero, which explains its absence from the old difference panel. Intervals of absolute means do not replace paired comparisons.''')
    markdown(findings['task'])
    if example=='beamforming':
        from scripts.notebook_plot_cells_beamforming import get_plot_cell
        cells.append(nbformat.v4.new_code_cell(get_plot_cell('stationary_dimensions',1),
            id='stationary-dimensions-T200',metadata={'editable_plotting_role':'plot',
            'figure_id':'task_performance','source_page':1,'source_catalogue_page':False}))
    markdown(r'''## 4. Which uncertainty matters for the decision

For a fixed query, the posterior squared response risk is
\[
\mathbb E[\|P_{\mathcal X_t}v-\overline P_t^+v\|^2\mid u_{1:t}]
=v^H[\overline P_t^+-(\overline P_t^+)^2]v.
\]
Under the full belief, using another action $\overline P_t^{\mathrm{iso}}v$ adds exactly $\|(\overline P_t^+-\overline P_t^{\mathrm{iso}})v\|^2$ to this expected response error. This is model-based optimality; paired test Monte Carlo simulations determine whether it yields a practical advantage under the data-generating process. The matched isotropic control has the same mode and mean squared chordal distance, but can change directional risk and the task action. '''+SECTION_NOTES[3]+(r''' In the contact risk panel, $v(\theta)=\cos\theta\,e_2+\sin\theta\,e_3$ is a unit query and $\theta$ is measured in radians. The vertical axis is $v(\theta)^\top[\overline P-(\overline P)^2]v(\theta)$, where $\overline P$ denotes the full $\overline P_t^+$ or the matched $\overline P_t^{\mathrm{iso}}$ indicated in the legend. Each curve uses its own belief and its own mean; they are not two algorithms' errors against a common truth. The visible orange marker identifies the common modal normal; its antipode is the same contact plane. The constructed equal-distance-law counterexample is a separately identified supplementary page.''' if example=='contact' else r''' The lower panel is a paired Monte Carlo difference against the matched isotropic action. The sphere above is a real-coefficient restriction of the complex density; it has no normalized real-sphere probability interpretation.''')+('\n\n'+MAIN_PLOT_NOTES['contact_risk'] if example=='contact' else ''));image(3)
    markdown('## 5. Learning variances for future observations\n\n'+SECTION_NOTES[4]+('\n\n'+MAIN_PLOT_NOTES['beamforming_noise'] if example=='beamforming' else ''));image(4);markdown(findings['noise'])
    calibration_targets=(r'For a known unit complex query $q$, the clean scalar target is $q^H Pq$.' if example=='beamforming' else r'For a known unit real query $q$, the clean scalar target is $e_1^T Pq$, the first component of its projected response.')
    markdown('## 6. Coverage of credible regions\n\nThese calibration experiments use $n=3$ reliability fixtures with unit queries. '+calibration_targets+r' The noisy target is $\operatorname{Re}(u_{t,1})$; contact observations are real. State ball/HPD reports target the latent subspace and remain separate from these scalar reports. '+('The sensor counts $n=32,64,128$ belong to the task-performance cohorts; these calibration fixtures do not establish coverage at those dimensions. ' if example=='beamforming' else 'The scalar widths use these fixture units rather than the physical trajectory velocity scale. ')+'The first page checks the matched joint prior-predictive regime at $t=64$; the second shows the predeclared drift diagnostic at the same checkpoint. The appended PDF pages show 95% state-ball chordal radii and scalar interval widths at t=64 across all three regimes; each value is computed per independent episode. Fixed-state inclusion checks and all numerical records remain in the full PDF and summary tables.');image(5);image(6);markdown(findings['calibration'])
    markdown('## 7. Computation and the decision to use the framework\n\n'+SECTION_NOTES[5]);image(7);markdown(findings['cost'])
    catalogue=['## Figure catalogue and provenance','',
        '[Main-draft PDFs](../results/main/'+example+'/) and [supplementary PDFs](../results/supplementary/'+example+'/) provide the manuscript page selection. The complete original catalogue is linked below. The paper summary reuses the same evidence and is linked rather than displayed a second time.','']
    for name in [*CATALOGUE,*BUNDLES]:
        catalogue.append('- ['+name.replace('_',' ').capitalize()+'](../results/'+example+'/'+name+'.pdf)')
    catalogue.extend(['','The [figure manifest](../results/'+example+'/figure_manifest.json) and [caption notes](../results/'+example+'/caption_notes.md) supply proposed captions, exact selectors, source/data/configuration hashes, seeds, episode and clock repetition counts, numerical budgets, failure status and insertion paths. Historic evidence remains under `archive/a91faab/` and its preserved result paths.'])
    markdown('\n'.join(catalogue))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},
        'final_study':{'example':example,'stage':'paper','figure_manifest_sha256':manifest_digest,
            'reference_previews':previews,'rendering':'inline_matplotlib','producer':'scripts/make_final_notebooks.py',
            'producer_sha256':producer_digest,
            'calculations':'Deterministic task actions and identities from stored moments; no inference, integration, sampling or fitting'}})
    from lib.notebook_inputs import saved_input_hashes
    loader='lib/notebook_'+example+'_data.py'
    nb.metadata['editable_plotting']={
        'schema':1,'mode':'inline_matplotlib','figure_count':sum(
            cell.metadata.get('editable_plotting_role')=='plot' for cell in cells),
        'rendered_figure_count':sum(
            isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='plt'
            and node.func.attr in ('figure','subplots')
            for cell in cells if cell.cell_type=='code'
            for node in ast.walk(ast.parse(cell.source))),
        'input_hashes':saved_input_hashes(root,example),
        'loader_sources':{loader:sha(root/loader),'lib/notebook_inputs.py':sha(root/'lib/notebook_inputs.py')},
        'output_directory':'results/notebook_exports/'+('01_array' if example=='beamforming' else '03_contact'),
        'controls':'DPI, FIGSIZE, styles, labels and PDF saves are ordinary visible code in every figure cell',
        'source_arrays':'Frozen published numerical data; no inference or benchmark rerun',
    }
    nb.cells[0].source+='\n\nEvery figure is now drawn by the full Matplotlib code in its own cell. Edit `DPI`, `FIGSIZE`, labels, styles, color scales or the PDF save command there. The setup cell exposes the shared `rcParams`. The one-line data loaders read the saved numerical arrays only. Notebook exports are separate from the published PDFs, so plotting experiments leave the scientific results intact.'
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            cell.source=cell.source.replace('episode-average','time-averaged').replace('within-episode','within-simulation')
            cell.source=re.sub(r'\bepisodes\b','Monte Carlo simulations',cell.source)
            cell.source=re.sub(r'\bepisode\b','simulation',cell.source)
    nbformat.validate(nb)
    return nb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['paper'],default='paper')
    parser.add_argument('--example',choices=['beamforming','contact','both'],default='both')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();examples=['beamforming','contact'] if args.example=='both' else [args.example]
    require_gate(ROOT);verify_history(ROOT)
    for example in examples:verify_figures(ROOT,example)
    if args.check_only:
        print('Final notebook prerequisites verified; no files written.');return
    for example in examples:
        name='01_array.ipynb' if example=='beamforming' else '03_contact.ipynb'
        nbformat.write(build(ROOT,example),ROOT/'notebooks'/name)
        print('Wrote',name,'from completed verified paper evidence')


if __name__=='__main__':main()
