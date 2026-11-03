"""Create the two examples only after the general notebook has passed."""
from pathlib import Path
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.validation import require_gate
require_gate(ROOT)


def write(name,cells):
    if len(sys.argv)>1 and sys.argv[1]!=name: return
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,ROOT/'notebooks'/name)


def md(s):return nbf.v4.new_markdown_cell(s)
def code(s):return nbf.v4.new_code_cell(s)
common='''from pathlib import Path
import sys, os, json
ROOT=Path.cwd() if (Path.cwd()/'lib').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
from lib.validation import require_gate
print('Theory gate:',require_gate(ROOT)['status'])
from lib.plotting import plt,save
from lib.reporting import run_or_load,coverage_table,error_curve
import numpy as np
import pandas as pd
from IPython.display import display
OUT=ROOT/'results'
# Set refresh=True to resimulate; cached tables are verified against source/config hashes.
config=dict(repeats=8,T=72,draws=6000,burnin=2500,max_draws=24000,max_burnin=8000)
'''
write('01_array.ipynb',[
md(r'''# Array interference suppression
Complex $n=16,k=3$ interference-only training, independent held-out evaluation, and proper complex Gaussian noise. The orthonormal interference modes start from three ULA steering vectors; known diagonal phase motion accounts for half the physical motion. Desired orientation, training count, drift, unequal powers, correlation, and desired-signal contamination are varied separately.

The mean and modal decisions use exactly the same Bingham parameter and noise values. The EW PCA mode is identical to the Bingham mode; this is a control, not an omitted performance gain. Loaded covariance and Bingham forgetting are selected using three candidates each on separate seeded validation streams. Oracle covariance is an evaluation reference. See `benchmarks/case1/README.md` for references and omissions.

Four worker processes run independent trajectories with checksum-verified resumable checkpoints. Eight trajectories per scenario provide a reproducible pilot with explicit uncertainty intervals. Increase repeats and chain budgets before using lower-tail or coverage results as precise empirical claims. Sampling diagnostics are part of the result; a capped failed diagnostic is retained and reported.'''),
code(common+"\nresult=run_or_load(ROOT,'case1',dict(seed=1729,workers=4,checkpoint_dir='results/checkpoints/array',**config))\nm=result['metrics'];r=result['regions'];display(result['config'])"),
md('''## Matched decisions and drift
Held-out power and population SINR use unit desired-signal power. Error bars show one standard error across independent trajectories. Posterior-mean benefits need not hold for every realized true covariance.'''),
code('''fig,axs=plt.subplots(2,2,figsize=(6.5,4.8))
for col,scenario in enumerate(['Static','Drift']):
    subset=m[m.scenario==scenario]
    error_curve(axs[0,col],subset,'sinr_db',['Posterior mean','Mode','Loaded covariance','Oracle'])
    error_curve(axs[1,col],subset,'heldout_power',['Posterior mean','Mode','Loaded covariance','Oracle'])
    axs[0,col].set_title(scenario);axs[0,col].set_ylabel('SINR (dB)');axs[1,col].set_ylabel('Held-out power')
axs[0,0].legend();fig.tight_layout();save(fig,OUT/'array_tracking');plt.show()
print('Largest Bingham/EW modal distance:',m.ew_modal_distance.max())
assert m.ew_modal_distance.max()<1e-6
'''),
md('''## Orientation and mismatch
The tenth-percentile SINR is computed across trajectories at the terminal time, not across correlated training snapshots. The same-rank isotropic model deliberately remains unchanged under mismatch. The numerical tables preserve unfavorable results.'''),
code('''final=m[m.t==config['T']]
summary=final.groupby(['scenario','method']).agg(mean_sinr=('sinr_db','mean'),tail_sinr=('sinr_db',lambda x:x.quantile(.1)),power=('output_power','mean'),heldout=('heldout_power','mean'))
display(summary)
scenarios=list(final.scenario.unique());methods=['Posterior mean','Mode','Loaded covariance','Oracle']
fig,axs=plt.subplots(1,2,figsize=(9,2.8));x=np.arange(len(scenarios));width=.19
for j,method in enumerate(methods):
    g=summary.xs(method,level='method').reindex(scenarios)
    axs[0].bar(x+(j-1.5)*width,g.mean_sinr,width,label=method)
    axs[1].bar(x+(j-1.5)*width,g.tail_sinr,width,label=method)
for ax in axs:ax.set_xticks(x,scenarios,rotation=30,ha='right')
axs[0].set_ylabel('Mean SINR (dB)');axs[1].set_ylabel('10th percentile SINR (dB)');axs[0].legend()
save(fig,OUT/'array_mismatch');plt.show()
'''),
md(r'''## Credible regions and leakage
Coverage is measured under repeated observations of deterministic paths with the stated forgetting rule. It is not guaranteed at 95%. Wilson intervals reflect only the finite trajectory count; sampler threshold errors are reported separately. Both regions use posterior mass 0.95. The HPD region can be wider in one task direction and narrower in another.

The optimization gap certifies Eq. (33) at the estimated region threshold, not the threshold itself. Sampling checks do not certify credible-tail accuracy. Failed diagnostics remain in the tables and are counted below.'''),
code('''posterior=final[final.method=='Posterior mean']
coverage=coverage_table(posterior);display(coverage)
fig,axs=plt.subplots(1,2,figsize=(8,2.8))
for region,offset in [('Ball',-.08),('HPD',.08)]:
    g=coverage[coverage.region==region].set_index('scenario').reindex(scenarios)
    axs[0].errorbar(np.arange(len(g))+offset,g.coverage,yerr=np.maximum(0,[g.coverage-g.low,g.high-g.coverage]),fmt='o',capsize=2,label=region)
axs[0].axhline(.95,color='k',ls=':');axs[0].set_xticks(x,scenarios,rotation=30,ha='right');axs[0].set_ylim(0,1.05);axs[0].set_ylabel('Coverage');axs[0].legend()
paired=r[r.t==config['T']].pivot(index=['scenario','repeat'],columns='region',values='upper')
for scenario,g in paired.groupby(level='scenario'):axs[1].scatter(g.Ball,g.HPD,s=16,label=scenario)
lim=max(paired.max())*1.05;axs[1].plot([0,lim],[0,lim],'k:',lw=.8);axs[1].set_xlabel('Ball leakage bound');axs[1].set_ylabel('HPD leakage bound');axs[1].legend(fontsize=6,loc='upper left')
save(fig,OUT/'array_regions');plt.show()
print('Largest optimization gap:',r.optimization_gap.max())
assert r.feasible.all() and r.optimization_gap.max()<1e-5
'''),
md('''## Numerical accuracy and cost
Modal updates, posterior moments, and regional optimization are timed separately. `decision_seconds` includes cumulative modal updates and cumulative moment evaluations for the posterior-mean decision. The loaded covariance reference includes its own updates. Oracle timings include only the decision solve. Predictive covariance, fitted scalar noise, and modal basis are shared in the mean/modal comparison.'''),
code('''diagnostics=m[m.method=='Posterior mean'][['scenario','repeat','t','sampling_ok','min_ess','max_rhat','max_mcse','actual_draws_per_chain','actual_burnin','moment_seconds']]
display(diagnostics.groupby('scenario').agg(pass_fraction=('sampling_ok','mean'),minimum_ess=('min_ess','min'),largest_rhat=('max_rhat','max'),largest_mcse=('max_mcse','max'),mean_moment_seconds=('moment_seconds','mean')))
display(final.groupby('method')[['decision_seconds','update_seconds']].mean())
fig,axs=plt.subplots(1,2,figsize=(6.5,2.4))
for scenario,g in diagnostics.groupby('scenario'):axs[0].scatter(g.t,g.min_ess,s=12,label=scenario)
axs[0].axhline(50,color='k',ls=':');axs[0].set_xlabel('Observations');axs[0].set_ylabel('Minimum ESS')
final.groupby('method').decision_seconds.mean().plot.bar(ax=axs[1],logy=True,rot=30);axs[1].set_xlabel('');axs[1].set_ylabel('Decision time (s)')
save(fig,OUT/'array_cost');plt.show()
print('Failed sampling diagnostics:',int((~diagnostics.sampling_ok.astype(bool)).sum()),'of',len(diagnostics))
display(result['tuning'])
'''),
md('''## Accuracy follow-up
Original capped failures remain in the tables. A separate longer-chain check reconstructs the identical posterior targets and uses new reproducible random streams. It measures changes in beamforming power and credible thresholds without selecting for a preferred performance ranking. Cached follow-up results are verified against source and input checksums.'''),
code('''import subprocess
subprocess.run([sys.executable,str(ROOT/'scripts/check_array_accuracy.py'),'--workers','4'],cwd=ROOT,check=True)
accuracy=pd.read_csv(OUT/'case1_accuracy.csv')
if len(accuracy):
    display(accuracy[['scenario','repeat','t','old_min_ess','new_min_ess','new_max_rhat','new_max_mcse','output_power_relative_change','radius_change','hpd_threshold_change']])
    assert accuracy.new_sampling_ok.all(), 'Extended-budget accuracy remains unresolved'
else:
    print('No original capped failures required a follow-up.')
''')])
write('02_field.ipynb',[
md(r'''# Field reconstruction and calibration
An orthonormal $64\times8$ spatial DCT basis and a changing rank-three coefficient subspace generate noisy $8\times8$ fields. Passive measurements use unknown Gaussian coefficients; known calibration patterns use a separate conditional protocol. Supplied frame motion accounts for half the physical motion.

Mean and modal estimates share Bingham parameters. Passive benchmarks are full-observation GROUSE, EW covariance PCA, and rolling PCA with equal candidate counts and independent validation trajectories. Reconstruction uses the current corrected posterior as in Eq. (45). Oracle projection is a reference. See `benchmarks/case2/README.md` for references. Eight trajectories are a reproducible pilot, not a precise calibration study.'''),
code(common+"\nresult=run_or_load(ROOT,'case2',dict(seed=2718,**config))\nm=result['metrics'];c=result['calibration'];noise=result['noise'];display(result['config'])"),
md('''## Reconstruction and mismatch
Errors are squared clean-field errors in all 64 coordinates. The DCT lift is isometric, so coefficient and field errors agree. Unequal powers and sparse corruptions violate the fitted isotropic Gaussian model.'''),
code('''fig,axs=plt.subplots(1,3,figsize=(9,2.7))
for ax,scenario in zip(axs,['Matched','Unequal powers','Sparse corruption']):
    error_curve(ax,m[m.scenario==scenario],'field_squared_error',['Posterior mean','Mode','EW PCA','Rolling PCA','GROUSE','Oracle'])
    ax.set_title(scenario);ax.set_ylabel('Squared field error')
axs[0].legend(ncol=2);save(fig,OUT/'field_reconstruction');plt.show()
final=m[m.t==config['T']]
display(final.groupby(['scenario','method']).field_squared_error.agg(['mean','std','count']))
f=result['fields'];images=[f.clean,f.observed,f.posterior_mean,f['mode']]
vmax=max(max(abs(v)) for v in images)
fig,axs=plt.subplots(1,4,figsize=(8,2.2))
for ax,data,title in zip(axs,images,['Clean','Observed','Posterior mean','Mode']):
    im=ax.imshow(np.asarray(data).reshape(8,8),cmap='RdBu_r',vmin=-vmax,vmax=vmax);ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
fig.colorbar(im,ax=axs,shrink=.75);save(fig,OUT/'field_maps');plt.show()
'''),
md(r'''## Known calibration patterns
Preassigned patterns are transported by the supplied frame so their transported rank stays at 6, 7, or 8. They never use the hidden subspace. Physical drift remains, so these runs are not a common-subspace proof of Proposition 4; the general notebook supplies the controlled identifiability counterexamples.

Eq. (30) predicts posterior response risk and the nonnegative modal excess risk. Realized errors against one true trajectory need not respect this ordering.'''),
code('''display(c.groupby(['input_rank','t'])[['mean_squared_error','modal_squared_error','posterior_risk','modal_excess_risk']].mean())
fig,axs=plt.subplots(1,2,figsize=(6.5,2.6))
for rank,g in c.groupby('input_rank'):
    mean=g.groupby('t')[['mean_squared_error','modal_squared_error','posterior_risk','modal_excess_risk']].mean()
    axs[0].plot(mean.index,mean.mean_squared_error,'o-',label=rf'Mean, rank {rank}')
    axs[0].plot(mean.index,mean.modal_squared_error,':',label=rf'Mode, rank {rank}')
    axs[1].plot(mean.index,mean.modal_excess_risk,'o-',label=rf'Rank {rank}')
for ax in axs:ax.set_xlabel('Observations')
axs[0].set_ylabel('Response squared error');axs[1].set_ylabel('Posterior modal excess risk');axs[0].legend(ncol=2);axs[1].legend()
save(fig,OUT/'field_calibration');plt.show()
assert c.modal_excess_risk.min()>=0
'''),
md('''## Noise estimation and predictive scores
These are separate runs with eight stored priors, fits at predeclared times, and two EM iterations. Predictions are scored before fitting or correcting the incoming measurement. Historical priors stay fixed within each fit, and no historical observation is corrected again.

High-dimensional sampled E-steps do not have a verified ascent guarantee here; the general notebook verifies ascent using deterministic references. Window residual and uncertainty diagnostics below use a common final-candidate E-step target. Their sum is the expected residual statistic, rather than a claim to equal the preceding finite EM update. Only the eight observed coefficient coordinates are used for fitting; the 56 complementary pure-noise coordinates are excluded.'''),
code('''display(noise)
fig,axs=plt.subplots(1,3,figsize=(9,2.5))
for protocol,g in noise.groupby('protocol'):
    axs[0].plot(g.t,g.s,'o-',label=protocol)
    axs[1].plot(g.t,g.predictive_log_score,'o-',label=protocol)
    axs[2].plot(g.t,g.score_importance_ess,'o-',label=protocol)
axs[0].axhline(.2,color='k',ls=':');axs[0].set_ylabel('Noise variance');axs[1].set_ylabel('Predictive log score');axs[2].set_ylabel('Importance ESS')
for ax in axs:ax.set_xlabel('Observations')
axs[0].legend();save(fig,OUT/'field_noise');plt.show()
cal=noise[noise.protocol=='Calibration']
fig,ax=plt.subplots(figsize=(3.25,2.4));ax.plot(cal.t,cal.naive_residual_variance,'o-',label='Mean residual');ax.plot(cal.t,cal.expected_residual_variance,'o-',label='Expected residual');ax.set_xlabel('Observations');ax.set_ylabel('Variance statistic');ax.legend();save(fig,OUT/'field_noise_uncertainty');plt.show()
assert np.allclose(cal.expected_residual_variance,cal.naive_residual_variance+cal.uncertainty_variance)
# Finite-sample log-mean-exp scores have bias; importance and chain ESS differ.
display(noise[['protocol','t','score_importance_ess','score_sampler_ess','score_chain_difference','minimum_ess','numerical_ascent_verified']])
'''),
md('''## Calibration, numerical accuracy, and runtime
Coverage below uses terminal repeated-trajectory observations. Wilson intervals reflect finite repetition count. Raw sampler failures are retained; they do not become evidence for calibrated posterior coverage. Field timings count cumulative tracking and posterior moment work, excluding the final matrix-vector reconstruction. Oracle timing is omitted.'''),
code('''p=final[final.method=='Posterior mean'];coverage=coverage_table(p);display(coverage)
diagnostics=m[m.method=='Posterior mean']
display(diagnostics.groupby('scenario').agg(pass_fraction=('sampling_ok','mean'),minimum_ess=('min_ess','min'),largest_rhat=('max_rhat','max'),largest_mcse=('max_mcse','max'),moment_seconds=('moment_seconds','mean')))
display(final.groupby('method')[['decision_seconds','update_seconds']].mean())
fig,axs=plt.subplots(1,2,figsize=(6.5,2.6))
scenarios=list(p.scenario.unique())
for i,(region,g) in enumerate(coverage.groupby('region')):
    g=g.set_index('scenario').reindex(scenarios)
    x=np.arange(len(g))+.12*i
    axs[0].errorbar(x,g.coverage,yerr=np.maximum(0,[g.coverage-g.low,g.high-g.coverage]),fmt='o',capsize=2,label=region)
axs[0].set_xticks(np.arange(3),scenarios,rotation=25);axs[0].axhline(.95,color='k',ls=':');axs[0].set_ylim(0,1.05);axs[0].set_ylabel('Coverage');axs[0].legend()
final[final.method!='Oracle'].groupby('method').decision_seconds.mean().plot.bar(ax=axs[1],logy=True,rot=30);axs[1].set_xlabel('');axs[1].set_ylabel('Update and moment time (s)')
save(fig,OUT/'field_coverage_cost');plt.show()
print('Failed passive sampling diagnostics:',int((~diagnostics.sampling_ok.astype(bool)).sum()),'of',len(diagnostics))
print('Failed calibration sampling diagnostics:',int((~c.sampling_ok.astype(bool)).sum()),'of',len(c))
display(result['tuning'])
''')])
print('Created requested example notebooks after a passing theory gate.')
