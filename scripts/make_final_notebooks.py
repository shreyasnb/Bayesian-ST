"""Build the two chronological, render-only final paper notebooks.

The command refuses missing/stale paper components or an incomplete figure set.
It never generates observations, integrates a posterior, fits a parameter,
selects a trajectory or calls the plotting/simulation runners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

The common $\overline P_t^+$, modal projector and $M_{\rm iso}$ controls share the same parameter, scales, data and reporting times. The isotropic mean matches the mode and mean squared chordal distance while discarding directional differences. With uniform initialization and fixed marginal scales the mode equals EW PCA; these are not independent algorithms. GREAT is explicitly a complex extension of the real published algorithm. Loaded modal and EW covariance references have independently selected validation settings.

This notebook reads frozen, content-verified paper evidence only. Its order is theory, task performance, anisotropy, variance learning/prediction, and computation. Notebook markdown supplies explanations; the PDFs contain only panel letters, axes, legends and event markers. Every PDF page is seven inches wide with actual LaTeX text, vector text/lines and 300-dpi raster layers/previews.
''',
'contact':r'''# Final contact study

The physical example is an ideal maintained bilateral contact in $\mathbb R^3$. Known resistance, the applied input and measured wrist-frame changes define recorded transformed input/output pairs
\[
u_t=P_{\mathcal X_t}v_t+e_t,\qquad \mathcal X_t\in\mathrm{Gr}(2,3),\quad e_t\sim\mathcal N(0,\sigma_t^2I_3).
\]
For matched isotropic resistance $D=dI$, the recorded requested velocity is $v_t=f_{\mathrm{eff},t}/d$. Let $n_t$ be the generating unit normal, so $P_{\mathcal X_t}=I_3-n_tn_t^T$; $n_t$ is withheld from the estimator. A noisy velocity is directly measured rather than inferred from differenced positions in the matched model. Surface positions and normals generate/score the experiment; the estimator receives only the recorded pairs and measured coordinate changes. Friction, contact loss, resistance errors and differenced-position measurements are separate mismatch experiments.

For a common-state block of $L$ observations with fixed variance, the increment is $(WV^T+VW^T-VV^T)/(2\sigma^2)$, without division by $L$. Known frame changes are aligned before a fixed-boundary block fit. Holding an output between completed blocks, exact aggregation and fitting one constant state over a moving block are different comparisons.

An admissible plane is represented by an unoriented normal line: $n$ and $-n$ describe the same plane. If the plane has parameter $A$, the normal has parameter $-A$. The actual and matched-isotropic normal laws share a mode and mean squared chordal distance. Their directional response risks can still differ. The anisotropy risk panel uses unit query directions; physical task panels use the declared requested speed and millimetre-per-second units. The constructed swapped-eigenvalue illustration is separately identified as an exact identity, not data-generated task performance.

The common $\overline P_t^+$, modal and $M_{\rm iso}$ controls share the same posterior parameter and inputs. $\widetilde M_t=\rho P_{\widehat{\mathcal X}_t}+(1-\rho)(2/3)I_3$ denotes the validation-regularized input-aware spectral point rule. GREAT is response-only and receives no recorded $v_t$, so its comparison cannot isolate Bayesian uncertainty. The plant trajectory is driven by a preassigned force program. Estimated responses and soft commands are scored counterfactually on this common stream; they are not fed back to regenerate a closed-loop trajectory. Soft command costs are expected penalties, not hard feasibility guarantees.

This notebook displays saved, source-verified evidence only, in theoretical-to-application order. All paper PDFs are title-free, seven inches wide, use actual LaTeX text and keep vector text/lines. Preview and raster layers use 300 dpi. Figures do not promise universal superiority, calibrated coverage under unmodelled motion, or exact EM ascent for approximate E-steps.
'''}


SECTION_NOTES={
1:r'''These checks distinguish the known-input and marginalized observation laws, finite-data curvature and expected information, exact aggregation and delayed actions. The sharper tracking envelope is evaluated only where its stated sufficient condition holds. Bound gaps, initialization and drift use generating quantities available for analysis, not online credible radii. Nonuniform priors, zero-signal data and concentrated inputs remain separate cohorts.''',
2:r'''Losses are paired on independent episodes and common held-out data. Observation/query averages within one episode are not extra repetitions. Main panels use predeclared representative conditions; the complete grids and strong controls remain in the additional comparisons and machine-readable records. A gain over GREAT may reflect input information or optimization/tracking differences; a gain attributable to posterior anisotropy requires the same-mode and matched-isotropic comparisons.''',
3:r'''A fixed-radius 3D sphere encodes density by color under one shared normalization. The original saved coordinates, oblique camera and blue-to-pink scale are restored. Gray curves are visible coordinate great circles that help locate directions on the sphere; they are not density contours or credible regions. Contact normals use true antipodal real-line densities. The complex-array sphere is a real-coefficient density section at the original $n=t=64$ checkpoint: it is not a physical far-field radiation pattern, a full posterior, or a normalized sphere marginal. The setup figure shows the physical array response. No estimator or benchmark parameters have changed. Mode, radius and mean overlap do not determine directional response risk. Visible anisotropy does not itself prove a repeated-experiment task advantage. PDFs retain vector LaTeX text and curves; all current preview and raster layers are 300 dpi. Inline images are scaled for reading; the full-resolution PNG and PDF links preserve the export resolution.''',
4:r'''Causal scale fits use completed disjoint past blocks and change future variances only. Each observation is assimilated once. The saved pre-observation predictive score excludes the scored observation from fitting. State, clean-response and noisy-next-observation intervals are distinct targets. Inclusion, widths and proper scores must be read together; integration/quantile precision and frequentist calibration are different checks.''',
5:r'''The cost plots use warmed single-thread schedules and separate additive updates, actions, uncertainty reports and scale fitting. The phase panel uses a 64-observation synthetic cycle: EM fits and optional uncertainty reports occur every 32 observations, independently of the action cadence printed on each row. A cadence experiment still assimilates every singleton; it does not claim a raw-block matrix-product speedup. Memory records distinguish persistent arrays from tracked allocations. Task-state timing points use a predeclared episode; repeated clocks are not hundreds of independent task-performance episodes. Native rank/codimension-one first-moment accuracy does not certify arbitrary-rank moments, floating-point error, second moments or tails.''',
6:r'''The summary is a composition of the same verified arrays. It contains no extra experiment and no favorable-trajectory selection. Proposed captions, exact page selectors, configuration/seed sets, omitted-condition locations, numerical status and hashes are in the figure manifest.''',
}


def _table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def _number(value):
    return f'{float(value):.3g}'


def paper_findings(root,example):
    """Describe fixed paper cohorts from their published paired summaries."""
    import pandas as pd
    folder=Path(root)/'results'/example/'tables/paper'
    labels={'mode':'Mode','isotropic':'Matched isotropic','regularized':'Regularized spectral rule',
        'covariance_loading':'Loaded EW covariance','great_window':
        'GREAT (complex extension)' if example=='beamforming' else 'GREAT (response only)'}
    rows=[];signs=[]
    if example=='beamforming':
        frame=pd.read_csv(folder/'performance/paired_stationary.csv')
        chosen=frame[frame.endpoint.eq('heldout_ratio_db')&frame.n.isin([32,64])&frame.t.eq(frame.n)
            &frame.reference.isin(['mode','isotropic','covariance_loading','great_window'])]
        unit='dB';selectors='the predeclared stationary $L=n$ cells at $n=32,64$'
        for row in chosen.itertuples():
            rows.append([f'$n=L={row.n}$',labels[row.reference],
                _number(row.gain)+' ['+_number(row.lower)+', '+_number(row.upper)+']',str(row.resolved_pairs)+'/'+str(row.episodes)])
            signs.append((row.lower,row.upper))
        if len(rows)!=8:raise RuntimeError('Expected all eight predeclared beamforming paired cells')
    else:
        frame=pd.read_csv(folder/'performance/paired.csv')
        chosen=frame[frame.metric.eq('response_mse')&frame.reference.isin(['mode','isotropic','regularized','great_window'])
            &((frame.condition.eq('static_concentrated')&frame.endpoint.eq('checkpoint')&frame.t.eq(16))
              |(frame.condition.eq('sphere')&frame.endpoint.eq('episode_average'))) ]
        unit='$(\\mathrm{mm}/\\mathrm{s})^2$'
        selectors='the concentrated-input stationary checkpoint $t=16$ and the full sphere episode'
        for row in chosen.itertuples():
            label='Concentrated, $t=16$' if row.condition=='static_concentrated' else 'Sphere episode'
            rows.append([label,labels[row.reference],
                _number(1e6*row.gain)+' ['+_number(1e6*row.lower)+', '+_number(1e6*row.upper)+']',str(row.finite_episodes)+'/'+str(row.episodes)])
            signs.append((row.lower,row.upper))
        if len(rows)!=8:raise RuntimeError('Expected all eight predeclared contact paired cells')
    positive=sum(lower>0 for lower,upper in signs);negative=sum(upper<0 for lower,upper in signs)
    task=('The table reports reference loss minus posterior-mean loss on '+selectors+'. Positive values favor the posterior mean; brackets are paired pointwise 95% Student intervals in '+unit+'. '
        +f'Across these eight fixed comparisons, {positive} intervals lie above zero, {negative} lie below zero, and {8-positive-negative} include zero. '
        +'Resolved pairs are shown relative to scheduled independent episodes. These cells are displayed together, including losses and uncertain differences; this count is descriptive and has no simultaneous-coverage guarantee.\n\n'
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
        task+='\n\nThe full grid also retains the plane episode-average isotropic-minus-full gain, '+paired_text(checks.loc['plane'])+', and the final sphere checkpoint gain, '+paired_text(checks.loc['sphere'])+', in $(\\mathrm{mm}/\\mathrm{s})^2$. These fixed checks show why an episode-average gain should not be interpreted as uniform superiority at every time and input condition.'
    paired=pd.read_csv(folder/'prediction/paired.csv')
    chosen=paired[paired.reference.eq('misspecified')&paired.reference_control.eq('mean')&paired.endpoint.eq('future_task_loss')]
    scale=1e6 if example=='contact' else 1.
    noise_rows=[[row.condition.replace('_',' '),_number(scale*row.gain)+' ['+_number(scale*row.lower)+', '+_number(scale*row.upper)+']',row.episodes]
        for row in chosen.itertuples()]
    if len(noise_rows)!=(2 if example=='contact' else 3):raise RuntimeError('Missing declared noise-learning comparison')
    noise=('Across all declared interventions, fixed-minus-learned observed future-task gains are '+
        '; '.join(row[0]+': '+row[1] for row in noise_rows)+'. These are paired pointwise 95% intervals over '
        +str(int(chosen.episodes.iloc[0]))+' independent episodes; positive gains favor learning. '
        +('Contact values are in $(\\mathrm{mm}/\\mathrm{s})^2$.' if example=='contact' else 'Beamforming values are held-out disturbance power, not decibels.')
        +' Observation/query losses are averaged inside each episode. The [paired summaries](../results/'+example+'/tables/paper/prediction/paired.csv) retain every control and endpoint.\n\n'
        +'A fit changes only future observations and decisions. The shaded variance trajectories show repeated-episode spread, not a posterior distribution of the variance. '
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
        +str(int(chosen.episodes.iloc[0]))+' independent episodes. '
        +('The upper interval is below 0.95 for '+', '.join(misses)+', supporting undercoverage in those drifting targets. ' if misses else 'These drifting intervals do not establish a general coverage guarantee. ')
        +'There are '+str(int(matched.unresolved.sum()))+' unresolved matched reports and '+str(int(chosen.unresolved.sum()))+' unresolved drifting reports in these cells.\n\n'
        +'The [calibration summaries](../results/'+example+'/tables/paper/prediction/calibration_summary.csv) also report widths in each target\'s native units. '
        +'Numerical tolerance controls how accurately a report is computed; it does not establish repeated-experiment coverage under motion or an incorrect likelihood.')
    timing=pd.read_csv(folder/'computation/summary.csv')
    case,n,cadence=('task_contact',3,4) if example=='contact' else ('task_stationary',64,64)
    chosen=timing[timing.case.eq(case)&timing.n.eq(n)&timing.action_cadence.eq(cadence)&timing.report_every.eq(0)
        &timing.method.isin(['mean','mode','isotropic','great_window'])]
    if len(chosen)!=4:raise RuntimeError('Missing matched-cadence task timing cohort')
    mean_time=float(chosen[chosen.method.eq('mean')].median_ms.iloc[0])
    time_parts=[]
    for row in chosen.itertuples():
        label='Posterior mean' if row.method=='mean' else labels[row.method]
        time_parts.append(label+': '+_number(row.median_ms)+' ms (90th percentile '+_number(row.p90_ms)+' ms)')
    cost=('With actions every '+str(cadence)+' observations and no uncertainty reports, the warmed single-thread full-cycle times are '
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
from lib.final_paper_visuals import load_evidence
from lib.final_prediction import mvdr
record=load_evidence(ROOT,EXAMPLE,'paper',components=['performance']).array('performance','stories')
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
from lib.final_paper_visuals import load_evidence
from lib.final_contact import soft_commands
from lib.geometry import risk
evidence=load_evidence(ROOT,EXAMPLE,'paper',components=['performance'])
record=evidence.array('performance','anisotropy')
M=record['plane_mean']
lam=float(evidence.config('performance')['task_lambda'])
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
    cells.append(nbformat.v4.new_code_cell(f'''from pathlib import Path
import sys,json
ROOT=Path.cwd() if (Path.cwd()/'lib').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
from scripts.make_final_notebooks import verify_figures,sha
from IPython.display import display,Image,Markdown
EXAMPLE={example!r}
meta=verify_figures(ROOT,EXAMPLE)
assert sha(ROOT/'results'/EXAMPLE/'figure_manifest.json')=={manifest_digest!r}, 'Figures changed: regenerate this notebook'
assert sha(ROOT/'scripts/make_final_notebooks.py')=={producer_digest!r}, 'Notebook maker changed: regenerate this notebook'
FOLDER=ROOT/'results'/EXAMPLE
DISPLAYED_PREVIEWS={previews!r}
print('Verified',EXAMPLE,'paper evidence:',len(meta['figures']),'PDFs')

def show_saved(name,page=1):
    entry=meta['figures'][name+'.pdf']
    assert any(record['page']==page for record in entry['pages'])
    suffix='' if page==1 else '_page'+str(page)
    path=FOLDER/'preview'/(name+suffix+'.png')
    relative=path.relative_to(ROOT).as_posix()
    assert sha(path)==DISPLAYED_PREVIEWS[relative], 'Preview changed: regenerate notebook'
    display(Image(data=path.read_bytes(),width=1400))
    display(Markdown('[Vector PDF and supplementary pages](../results/'+EXAMPLE+'/'+name+'.pdf) | [Full-resolution PNG](../results/'+EXAMPLE+'/preview/'+name+suffix+'.png)'))
'''))
    def markdown(text):cells.append(nbformat.v4.new_markdown_cell(text))
    def image(index):
        name,page=displayed[index];cells.append(nbformat.v4.new_code_cell(f'show_saved({name!r},page={page})'))
    markdown('## 1. Measurements and geometry\n\n'+SECTION_NOTES[1]);image(0)
    information_note=('The second panel separates recorded-input and marginalized-input information.'
        if example=='beamforming' else 'The second panel shows the unequal tangent Fisher eigenvalues produced by the declared anisotropic contact inputs.')
    markdown(r'''## 2. What the information calculation checks

The expected Fisher information describes the observation law and excitation; finite-data posterior curvature is a different object. The diagonal compares analytical and empirical eigenvalues in aligned tangent coordinates. '''+information_note+''' [Curvature](../results/'''+example+'''/posterior_curvature.pdf), [sharp tracking bound](../results/'''+example+'''/tracking_bound.pdf), and [block updates](../results/'''+example+'''/block_updates.pdf) retain the locality flags, sufficient conditions, initialization/noise/drift contributions and approximation checks.''');image(1)
    task_equation=(r'''With $M=\mathbb E[P_{\mathcal X}\mid u_{1:t}]$, the Bayes disturbance covariance is $\overline R=\sigma^2I+\tau^2M$. For known desired steering vector $a$, the distortionless decision is
\[
w_B=\frac{\overline R^{-1}a}{a^H\overline R^{-1}a}.
\]
The mode and matched-isotropic controls substitute their own projectors/means into the same decision.''' if example=='beamforming' else r'''For a recorded query $v=f_{\rm eff}/d$, the squared-error Bayes response is $\widehat u_B=Mv$. For desired velocity $v_{\rm des}$ and the soft penalty $\lambda$, the command is
\[
v_B=[I+\lambda(I-M)]^{-1}v_{\rm des}.
\]
The mode, matched-isotropic and regularized spectral controls substitute their own matrices into the same task rule. $M$ is a plane-projector mean, so $I-M$ is the corresponding normal-line mean. The penalty is an expected cost, not a hard contact guarantee.''')
    markdown('## 3. Task decisions and paired outcomes\n\n'+task_equation+'\n\n'+SECTION_NOTES[2]+'\n\nThe following deterministic library calls reuse the displayed episode’s stored mean; they do not rerun inference or generate observations.');cells.append(nbformat.v4.new_code_cell(DETERMINISTIC_CHECKS[example]));image(2);markdown(findings['task'])
    markdown(r'''## 4. Which uncertainty matters for the decision

For a fixed query, the posterior squared response risk is
\[
\mathbb E[\|Pv-Mv\|^2\mid u_{1:t}]=v^H(M-M^2)v.
\]
Under the full belief, using another action $M_{\mathrm{iso}}v$ adds exactly $\|(M-M_{\mathrm{iso}})v\|^2$ to this expected response error. This is model-based optimality; paired test episodes determine whether it yields a practical advantage under the data-generating process. The matched isotropic control has the same mode and mean squared chordal distance, but can change directional risk and the task action. '''+SECTION_NOTES[3]+(r''' In the contact risk panel, $v(\theta)=\cos\theta\,e_2+\sin\theta\,e_3$ is a unit query. The two curves are the two beliefs' own predictive risks; they are not two algorithms' errors against a common truth. The visible orange marker identifies the common modal normal; its antipode is the same contact plane. The constructed equal-distance-law counterexample is a separately identified supplementary page.''' if example=='contact' else r''' The lower panel is a paired repeated-episode difference against the matched isotropic action. The sphere above is a real-coefficient restriction of the complex density; it has no normalized real-sphere probability interpretation.'''));image(3)
    markdown('## 5. Learning variances for future observations\n\n'+SECTION_NOTES[4]);image(4);markdown(findings['noise'])
    calibration_targets=(r'For a known unit complex query $q$, the clean scalar target is $q^H Pq$.' if example=='beamforming' else r'For a known unit real query $q$, the clean scalar target is $e_1^T Pq$, the first component of its projected response.')
    markdown('## 6. Inclusion is a separate test\n\nThese calibration experiments use $n=3$ reliability fixtures with unit queries. '+calibration_targets+r' The noisy target is $\operatorname{Re}(u_{t,1})$; contact observations are real. State ball/HPD reports target the latent subspace and remain separate from these scalar reports. '+('The sensor counts $n=32,64,128$ belong to the task-performance cohorts; these calibration fixtures do not establish coverage at those dimensions. ' if example=='beamforming' else 'The scalar widths use these fixture units rather than the physical trajectory velocity scale. ')+'The first page checks the matched joint prior-predictive regime at $t=64$; the second shows the predeclared drift diagnostic at the same checkpoint. The appended PDF pages show 95% state-ball chordal radii and scalar interval widths at t=64 across all three regimes; each value is computed per independent episode. Fixed-state inclusion checks and all numerical records remain in the full PDF and summary tables.');image(5);image(6);markdown(findings['calibration'])
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
            'displayed_previews':previews,'producer':'scripts/make_final_notebooks.py',
            'producer_sha256':producer_digest,
            'calculations':'Deterministic task actions and identities from stored moments; no inference, integration, sampling or fitting'}})
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
