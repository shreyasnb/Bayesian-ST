"""Build question-oriented case notebooks after the numerical-core gate passes."""
from pathlib import Path
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.validation import require_gate
require_gate(ROOT)


def md(text):return nbf.v4.new_markdown_cell(text)
def code(text):return nbf.v4.new_code_cell(text)
def write(name,cells):
    if len(sys.argv)>1 and sys.argv[1]!=name:return
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,ROOT/'notebooks'/name)

common='''from pathlib import Path
import sys, json, hashlib, subprocess
ROOT=Path.cwd() if (Path.cwd()/'lib').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
from lib.validation import require_gate
print('Numerical core:',require_gate(ROOT)['status'])
from lib.plotting import plt
from lib.reporting import run_or_load
from lib.visuals import (array_overview,field_overview,paired_field,coverage_regions,
    cost_tradeoff,field_maps,calibration_overview,noise_overview,sampler_overview,predictive_scores)
from lib.decision_visuals import load_study,gain_maps,decision_calibration
import numpy as np
import pandas as pd
from IPython.display import display
OUT=ROOT/'results'
assert plt.rcParams['text.usetex'] and plt.rcParams['figure.dpi']==200
assert plt.rcParams['savefig.dpi']==200

def show(fig):
    assert fig.dpi==200
    plt.show()
    plt.close(fig)

# Unchanged raw application simulations are checksum-verified and reused.
config=dict(repeats=8,T=72,draws=6000,burnin=2500,max_draws=24000,max_burnin=8000)
'''
controlled='''# This runs only if the controlled-study source or parameters changed.
subprocess.run([sys.executable,str(ROOT/'scripts/run_decision_study.py')],cwd=ROOT,check=True)
study=load_study(ROOT)
display(study['config'])
assert study['accuracy'].reference_difference.max()<=study['config']['tolerance']
'''
write('01_array.ipynb',[
md(r'''# Array interference suppression
The main question is whether posterior decisions improve beamforming enough to justify their additional cost. These figures distinguish **tracking**, **posterior averaging**, and **model mismatch**. They retain every scenario, including cases where a cheaper reference wins.

The original complex study has 16 channels, rank three, interference-only training, independent evaluation snapshots, and supplied motion equal to half the physical motion. Desired orientation, snapshot count, drift, unequal powers, source correlation, and desired-signal contamination are varied separately. Mean and modal beamformers share parameters and variances. The marginal Bingham mode equals matching EW PCA.

Every figure uses TeX, 200 dpi, and a larger canvas. Blue in a signed-gain map favors the posterior mean; red favors the reference. A confidence interval crossing zero does not establish an improvement. The eight-trajectory application experiment is a pilot; intervals are paired Student intervals, not simultaneous bands.'''),
code(common+"\nresult=run_or_load(ROOT,'case1',dict(seed=1729,workers=4,checkpoint_dir='results/checkpoints/array',**config))\nm=result['metrics'];regions=result['regions']\ndisplay(result['config'])"),
md(r'''## Separate the two comparisons
The left map isolates posterior averaging by comparing the mean and mode from the **same posterior**. The right compares against validation-tuned loaded covariance. Gains versus loaded covariance can include structural assumptions and different estimators; they are not all attributable to posterior uncertainty.

Each cell gives the paired mean SINR gain and its 95% interval in dB. Color scales are separate so small mean–mode effects remain visible. The source tables preserve independent held-out power and lower-tail SINR as well.'''),
code("show(array_overview(result,path=OUT/'array_results'))\nassert m.ew_modal_distance.max()<1e-6\nprint('Bingham and matching EW PCA modal distance:',m.ew_modal_distance.max())"),
md(r'''## Read accuracy together with computation
Horizontal position is measured cumulative work through 72 observations, including the three moment evaluations for the posterior mean. It is not per-observation latency or an intrinsic lower bound for Bayesian methods. The implementation uses a reference MCMC evaluator. Modal updates can run without it, and posterior summaries can be requested less frequently.

The oracle line uses the true covariance and is not an available competitor. The contamination panel is a useful failure case: posterior averaging helps relative to the mode, while loaded covariance is substantially better than both.'''),
code("show(cost_tradeoff(result,path=OUT/'array_tradeoff'))\ndisplay(m[m.t==m.t.max()].groupby('method')[['sinr_db','heldout_power','decision_seconds']].mean())"),
md(r'''## Regions answer a directional question
The left panel shows empirical terminal coverage with Wilson intervals. “Matched” covariance assumptions do not make the approximate motion model exact. Coverage can fail under forgetting, unmodelled drift, or misspecification.

The right panel compares optimized leakage bounds at equal posterior mass. A ratio below one means the HPD set gives a smaller bound for this beamformer direction; above one means the ball gives the smaller bound. There is no uniformly better region shape. Optimization gaps certify the specified region, not its sampled probability threshold.'''),
code("show(coverage_regions(result,array=True,path=OUT/'array_regions'))\nprint('Largest regional optimization gap:',regions.optimization_gap.max())\nassert regions.feasible.all() and regions.optimization_gap.max()<1e-5"),
md(r'''## A controlled setting where posterior decisions matter
The next experiment isolates low-data decision quality in a stationary **real** rank-one problem in three dimensions. It is a diagnostic counterpart of beamforming, not a claim of superiority in the 16-channel complex study.

All methods receive identical passive observations, the same known noise levels, and independent desired-steering probes. Latent subspaces are sampled from the stated uniform prior. GROUSE, scalar-shrunk PCA, and loaded covariance have independent validation tuning with matched candidate counts. The posterior mean uses deterministic quadrature with separate order-doubling checks.

This stronger control matters: an advantage over a point estimate can come largely from ordinary shrinkage. The scalar-shrunk PCA and loaded-covariance panels show how much remains beyond those cheaper choices. The objective is expected interference power; a lower average power does not imply a win in every realization or for every alternative SINR objective.'''),
code(controlled+"\nshow(gain_maps(study,task='Beamforming',path=OUT/'array_low_data'))\ndisplay(study['paired'][(study['paired'].task=='Beamforming')&study['paired'].t.isin([1,16])][['regime','t','reference','baseline_loss','posterior_loss','reduction_percent','gain_low','gain_high']])"),
md(r'''## Numerical accuracy is part of the result
Orange outlines identify where the original budget missed the ESS threshold. All original failures remain visible. The separate longer-chain follow-up reconstructs the same posterior targets with independent random streams; it does not replace inconvenient original observations.

The historical follow-up is reused only after checking its raw-table checksum, original numerical dependencies, and core certificate. Display-only changes are excluded from that numerical dependency comparison. Diagnostics are evidence about the stated budgets, not a convergence proof.'''),
code('''show(sampler_overview(result,array=True,path=OUT/'array_accuracy'))
a_meta=json.loads((OUT/'case1_accuracy.json').read_text())
assert a_meta['core_hash']==require_gate(ROOT)['core_hash']
assert hashlib.sha256((OUT/'case1_accuracy.csv').read_bytes()).hexdigest()==a_meta['result_sha256']
for name,digest in a_meta['source']['files'].items():
    if name=='lib/plotting.py':continue  # only its presentation settings changed
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for name,digest in a_meta['input_sha256'].items():
    if name=='results/validation.json':continue  # the core hash above is the invariant
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
accuracy=pd.read_csv(OUT/'case1_accuracy.csv')
assert accuracy.new_sampling_ok.all()
display(accuracy[['scenario','repeat','t','old_min_ess','new_min_ess','new_max_rhat','output_power_relative_change']])
print('Maximum population-power sensitivity:',accuracy.output_power_relative_change.abs().max())
''')])

write('02_field.ipynb',[
md(r'''# Field reconstruction and calibration
This notebook separates three questions: which tracker reconstructs the field better, whether posterior averaging improves on the identical modal estimate, and what useful uncertainty information the distribution provides.

The application uses eight orthonormal DCT spatial modes on an 8-by-8 grid, rank-three changing subspaces, and supplied motion equal to half the physical motion. Passive data and known calibration patterns are separate protocols. Validation trajectories choose benchmark settings independently of evaluation data. Mean and mode share the same Bingham parameters; the mode equals matching EW PCA.

Figures use larger canvases, TeX, and 200 dpi. Signed gain maps, paired trials, spatial errors, coverage forests, and cost–accuracy plots replace repeated trajectory curves. The application has only eight independent repetitions per scenario; intervals should be read as exploratory pilot evidence.'''),
code(common+"\nresult=run_or_load(ROOT,'case2',dict(seed=2718,**config))\nm=result['metrics'];c=result['calibration'];noise=result['noise']\ndisplay(result['config'])"),
md(r'''## Attribute the gains correctly
The three maps compare the posterior mean against GROUSE, its **own mode**, and rolling PCA. Blue means lower field MSE; red means higher MSE. Cells show paired absolute-loss differences scaled by the observed mean reference loss, with paired 95% intervals on that scale—not a confidence interval for a random loss ratio.

Large gains versus GROUSE under mismatch should not be attributed entirely to Bayesian averaging: the mode/EW-PCA control already captures most of them. The rolling-PCA panel prevents selecting only a weak reference. Intervals crossing zero do not establish a difference.'''),
code("show(field_overview(result,path=OUT/'field_results'))\nshow(paired_field(result,path=OUT/'field_pairs'))\ndisplay(m[m.t==m.t.max()].groupby(['scenario','method']).field_squared_error.agg(['mean','std','count']))"),
md(r'''## Inspect the spatial errors
The four field images share one amplitude scale. The two residual maps use a separate, common error scale, so spatial bias and smoothing are visible. This is one preassigned matched trajectory at the final observation, not a selected best reconstruction. Aggregate paired results above carry the comparison.'''),
code("show(field_maps(result,path=OUT/'field_maps'))"),
md(r'''## Known-input prediction is a different use of the posterior
Calibration inputs are preassigned independently of the hidden state and have transported ranks 6, 7, or 8. The left map measures realized response-error reduction against the mode. The right shows its nonnegative **posterior** excess risk from Eq. (30). The identity is conditional on a posterior; it does not require a realized true-state error ordering in every run.

These drifting paths do not test common-subspace global identifiability. The general theory notebook supplies the controlled ambiguity and excitation tests.'''),
code("show(calibration_overview(result,path=OUT/'field_calibration'))"),
md(r'''## Controlled prediction with identical information
A separate stationary real Gr(1,3) experiment tests the decision benefit under the actual Bayesian model. Every method receives the same passive training observations and fresh independent unit query vectors at decision time. No method is given hidden training inputs. Noise levels, rank, and prior-generating law are stated and shared.

The scalar-shrunk PCA control asks whether generic regularization is sufficient. GROUSE and shrinkage each receive the same number of independent validation candidates, separately per task and horizon. Posterior means use deterministic quadrature; all reported targets are checked against a higher order. All predeclared noise levels and horizons are shown, including ties or losses.'''),
code(controlled+"\nshow(gain_maps(study,task='Projection',path=OUT/'field_low_data_prediction'))\nshow(gain_maps(study,task='Reconstruction',path=OUT/'field_low_data_reconstruction'))\ndisplay(study['paired'][(study['paired'].task=='Projection')&study['paired'].t.isin([1,4,16])][['regime','t','reference','baseline_loss','posterior_loss','reduction_percent','gain_low','gain_high']])"),
md(r'''## Calibrated under the model does not mean robust to drift
The controlled experiment's coverage panel uses the first predeclared trajectories at the terminal time and continuous-CDF region thresholds. Its risk panel bins independent terminal trajectories by predicted risk and compares them with observed loss.

The application coverage below is a different evaluation law: it contains unmodelled motion and approximate forgetting. In the original matched-covariance field run, both nominal 95% regions have 0/8 terminal coverage. This contrast identifies the state-model limitation; it is not a claim that point trackers intrinsically provide competing credible regions.'''),
code("show(decision_calibration(study,path=OUT/'field_model_calibration'))\nshow(coverage_regions(result,path=OUT/'field_regions'))"),
md(r'''## Noise estimation needs the uncertainty term
Separate fits use eight stored priors and two EM iterations. The stacked bars decompose the same window residual statistic into the residual of the mean response and the nonnegative subspace-uncertainty term. Treating the mean residual alone as the variance statistic omits that term.

Sampled high-dimensional EM ascent remains unverified. Predictive scores are recorded before fitting and assimilation, with Gaussian constants retained. Different cells correspond to different incoming observations; an increasing log score is not evidence of ascent of one fixed objective. Importance ESS and sampler ESS answer different questions.'''),
code("show(noise_overview(result,path=OUT/'field_noise'))\nshow(predictive_scores(result,path=OUT/'field_scores'))\ncal=noise[noise.protocol=='Calibration']\nassert np.allclose(cal.expected_residual_variance,cal.naive_residual_variance+cal.uncertainty_variance)\ndisplay(noise[['protocol','t','score_chain_difference','minimum_ess','numerical_ascent_verified']])"),
md(r'''## Decide whether the extra computation is worthwhile
The original implementation spends far more time sampling moments than updating a point estimate. The horizontal axis counts cumulative tracking and moment work through 72 measurements, excluding the final field matrix-vector multiplication. Oracle timing is omitted. The mode also represents the matching EW-PCA control; its duplicate point is not plotted.

If the requirement is only a cheap subspace basis or matched passive reconstruction, these results do not justify the full sampling overhead. Posterior computation is motivated by uncertainty-aware response decisions, directional risk and bounds, or noise diagnostics. It can be requested only when those quantities are needed; faster accurate moment backends remain an implementation opportunity.'''),
code("show(cost_tradeoff(result,field=True,path=OUT/'field_tradeoff'))\nshow(sampler_overview(result,path=OUT/'field_accuracy'))\np=m[m.method=='Posterior mean']\nassert p.sampling_ok.astype(bool).all() and c.sampling_ok.astype(bool).all()\ndisplay(m[m.t==m.t.max()].groupby('method')[['decision_seconds','update_seconds']].median())")])
print('Created the requested example notebooks with 200 dpi, question-oriented figures.')
