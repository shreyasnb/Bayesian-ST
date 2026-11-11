"""Create separate moving and stationary array-comparison audit notebooks.

Every figure is ordinary visible Matplotlib source. Numerical readers verify
new audit exports and their original observations; this script never executes
an estimator, chooses parameters or changes an existing notebook/result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from textwrap import dedent

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUTPUT=Path('results/beamforming_comparison_audit')
NOTEBOOKS={scope:Path('notebooks')/f'beamforming_{scope}_comparison_audit.ipynb'
           for scope in ('moving','stationary')}
FIGURE_NAMES={
    'moving':['smooth_history','abrupt_history','paired_primary','posterior_mean_beta',
              'ew_objective_gap','ew_task_power','computation_time'],
    'stationary':['original_settings','final_time_selection','paired_final_time','posterior_mean_beta',
                  'ew_objective_gap','ew_task_power','computation_time'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_files(root):
    own={root/path for path in NOTEBOOKS.values()}
    paths=set((root/'notebooks').glob('*.ipynb'))-own
    for path in (root/'results').rglob('*'):
        if path.is_file() and OUTPUT not in path.relative_to(root).parents:
            paths.add(path)
    return {str(path.relative_to(root)):sha(path) for path in sorted(paths)}


def check_preservation(root,mapping):
    changed=[name for name,value in mapping.items() if not (root/name).is_file() or sha(root/name)!=value]
    if changed:raise RuntimeError('An existing notebook/result changed: '+', '.join(changed))
    return dict(status='passed',files_checked=len(mapping),changed=[],
        existing_notebooks=sum(Path(name).parent==Path('notebooks') and name.endswith('.ipynb') for name in mapping),
        baseline_sha256=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest())


SETUP=r'''from pathlib import Path
import os
import sys
ROOT = Path.cwd() if (Path.cwd() / 'lib').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
LOCAL_TEX = Path('/workspace/.local/tex')
if (LOCAL_TEX / 'usr/bin/dvipng').exists():
    os.environ['PATH'] = str(LOCAL_TEX / 'usr/bin') + os.pathsep + os.environ['PATH']
    os.environ.setdefault('TEXMFHOME', str(LOCAL_TEX / 'usr/share/texmf'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as student_t
from IPython import get_ipython
from IPython.display import Markdown, HTML, display
from lib.beamforming_comparison_summaries import load_existing_beamforming_comparisons
from lib.beamforming_mean_validation_audit import load_mean_validation_audit
from lib.beamforming_objective_audit import load_objective_audit

SCOPE = __SCOPE__
# These controls identify already completed audits; changing them does not fit a model.
CONFIG = __CONFIG__
existing = load_existing_beamforming_comparisons(ROOT)[SCOPE]
mean_audit = load_mean_validation_audit(ROOT, scope=SCOPE)
optimization = load_objective_audit(ROOT, cohort=SCOPE)
assert existing['config']['T'] == CONFIG['T']
assert existing['config']['beta'] == CONFIG['common_beta']
if optimization['timing'] is None:
    raise RuntimeError('Complete the isolated timing audit before executing this notebook')
if SCOPE == 'stationary':
    from lib.beamforming_stationary_selection_audit import load_stationary_selection_audit
    selection = load_stationary_selection_audit(ROOT)
EDITABLE_OUTPUT = ROOT / 'results/beamforming_comparison_audit' / SCOPE / 'figures'
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
    get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': None}")
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
    'font.family': 'serif', 'lines.linewidth': 1.8, 'lines.markersize': 5.,
    'savefig.bbox': None, 'figure.max_open_warning': 0,
})
print('Verified completed comparison, validation, objective and isolated-timing records:', SCOPE)
'''

STYLE=r'''LABELS = {
    'mean': r'$\overline P_t^+$', 'mode': r'$P_{\widehat{\mathcal X}_t}$',
    'isotropic': r'$\overline P_t^{\mathrm{iso}}$',
    'great_window': 'GREAT (sliding window)', 'great_ew': 'GREAT (exponential weighting)',
    'covariance_loading': 'Regularized sample covariance',
    'modal_loading':'Regularized mode', 'grouse':'GROUSE',
    'ledoit_wolf':'Ledoit--Wolf', 'oracle':'True covariance',
    'great_window_final_db': r'GREAT, $W=\widehat W_n$',
    'great_full_history_final_db': r'GREAT, $W=T$',
    'great_ew_final_db': 'GREAT (exponential weighting)',
    'covariance_loading_final_db': 'Regularized sample covariance',
}
COLORS = {'mean':'#0072B2', 'mode':'#D55E00', 'isotropic':'#CC79A7',
    'great_window':'#009E73', 'great_ew':'#56B4E9', 'covariance_loading':'#484848',
    'great_window_final_db':'#009E73', 'great_full_history_final_db':'#006747',
    'great_ew_final_db':'#56B4E9', 'covariance_loading_final_db':'#484848'}
MARKERS = {'mean':'o', 'mode':'s', 'isotropic':'^', 'great_window':'D', 'great_ew':'v',
    'covariance_loading':'X', 'great_window_final_db':'D', 'great_full_history_final_db':'P',
    'great_ew_final_db':'v', 'covariance_loading_final_db':'X'}
LINESTYLES = {'mean':'-', 'mode':'--', 'isotropic':'-.', 'great_window':':',
    'great_ew':(0,(5,1,1,1)), 'covariance_loading':'-',
    'great_window_final_db':':', 'great_full_history_final_db':':',
    'great_ew_final_db':(0,(5,1,1,1)), 'covariance_loading_final_db':'-'}
MARK_OFFSETS = {'mean':0, 'mode':1, 'isotropic':2, 'great_window':0, 'great_ew':2,
    'covariance_loading':1, 'great_window_final_db':0,'great_full_history_final_db':0,
    'great_ew_final_db':2,'covariance_loading_final_db':1}
'''

CONTROLS=r'''# Edit the canvas, labels, styles and save commands directly in this cell.
DPI = 300
FIGSIZE = (15, 5)
LABEL_FONT_SIZE = 25.
TICK_FONT_SIZE = 23.
LEGEND_FONT_SIZE = 22.
plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': LABEL_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE, 'axes.titlesize': LABEL_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE, 'ytick.labelsize': TICK_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE, 'axes.labelpad': 8.,
    'figure.dpi': DPI, 'savefig.dpi': DPI,
})
SAVE_PDF = True
'''
SAVE=r'''# Complete rectangular frames; ticks on all sides, labels only left/bottom.
for _axes in fig.axes:
    for _spine in _axes.spines.values():
        _spine.set_visible(True)
        _spine.set_linewidth(.95)
    _axes.tick_params(direction='in', top=True, right=True, labeltop=False, labelright=False)
fig.get_layout_engine().set(h_pad=.18, w_pad=.16)
if SAVE_PDF:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, dpi=DPI, bbox_inches=None)
    PNG_PATH = PDF_PATH.with_suffix('.png')
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches=None)
plt.show()
'''


def plot_source(stem,body):
    return CONTROLS+f"PDF_PATH = EDITABLE_OUTPUT / '{stem}.pdf'\n\n"+STYLE+'\n'+dedent(body).strip()+'\n\n'+SAVE


MOVING_HISTORY=r'''
condition = CONDITION
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for index, curve in enumerate(existing['curves'][condition]):
    method = curve['method']
    points = np.asarray(curve['points'])
    ax.plot(points[:,0], points[:,1], color=COLORS[method], marker=MARKERS[method],
        linestyle=LINESTYLES[method], markevery=(index%4,4), label=LABELS[method])
    if method == 'mean':
        ax.fill_between(points[:,0], points[:,2], points[:,3],
                        color=COLORS[method], alpha=.14, linewidth=0)
if condition == 'abrupt':
    ax.axvline(129, color='#7B3294', linewidth=1.3, linestyle='--', zorder=20,
               label='Direction change')
ax.set(xlabel=r'$t$', ylabel=r'$\ell_t$ (dB)', xlim=(1,256))
ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
           ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE)
'''

MOVING_PAIRS=r'''
frame = existing['paired_primary_summary']
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
positions, labels = [], []
for condition in ['smooth','abrupt']:
    for method in ['mode','isotropic','great_window','great_ew']:
        row = frame[frame.condition.eq(condition) & frame.reference.eq(method)].iloc[0]
        position = len(labels)
        labels.append(('Smooth' if condition == 'smooth' else 'Abrupt')+': '+LABELS[method])
        positions.append(position)
        ax.errorbar(row['mean'], position,
            xerr=[[row['mean']-row['lower']],[row['upper']-row['mean']]],
            fmt=MARKERS[method], color=COLORS[method], capsize=4., markersize=7.)
ax.axvline(0., color='#222222', linestyle='--', linewidth=1.2)
ax.set(yticks=positions, yticklabels=labels,
       xlabel=r'$L(D)-L(\overline P^+)$ (dB)')
ax.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)
ax.invert_yaxis()
ax.grid(axis='x', color='#BFC4C8', alpha=.30, linewidth=.45)
'''

AGGREGATION_TABLES=r'''# Every declared aggregation is retained, including reversals and unresolved differences.
for aggregation, window in [('mean_db','all'),('mean_db','t_ge_32'),
                            ('log_mean_ratio','all'),('log_mean_ratio','t_ge_32')]:
    frame = existing['paired_aggregate_summary']
    frame = frame[frame.aggregation.eq(aggregation) & frame.window.eq(window)]
    frame = frame[['condition','reference','mean','lower','upper','simulations']].copy()
    frame['reference'] = frame.reference.map(LABELS)
    frame = frame.rename(columns={'mean':r'$\Delta$ (dB)','lower':'95% lower','upper':'95% upper'})
    label = ('Mean individual dB' if aggregation == 'mean_db' else 'dB of mean linear ratio')
    label += ', all 32 reports' if window == 'all' else ', reports at t >= 32 (29 reports)'
    table = frame.to_html(index=False, escape=False, float_format=lambda value:f'{value:.6g}')
    display(HTML('<details '+('open' if (aggregation,window)==('mean_db','all') else '')+
        '><summary>'+label+'</summary>'+table+'</details>'))
'''


ORIGINAL_SETTINGS=r'''rows=[]
records=(list(existing['selected_settings'].items()) if SCOPE == 'moving' else
         [(str(record['n']),record) for record in existing['selected_settings']])
for case, record in records:
    for method in ['great_window','great_ew']:
        value=record[method]
        rows.append({'Case':case,'Method':LABELS[method],
            'Common beta':record['shared_beta'],'EW beta':value.get('beta'),
            'Window W':value.get('window'),'Gradient steps s':value['inner_steps'],
            'Step size':value['stepsize']})
display(pd.DataFrame(rows))
'''

BETA_VALIDATION=r'''
frame = mean_audit['validation_scores']
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
cases = list(frame[['condition','n']].drop_duplicates().itertuples(index=False,name=None))
palette = ['#0072B2','#D55E00','#CC79A7']
for index, (condition,n) in enumerate(cases):
    subset = frame[frame.condition.eq(condition) & frame.n.eq(n)]
    points = []
    for beta, group in subset.groupby('beta',sort=True):
        # One validation loss per independent simulation; reporting times average inside it.
        values = group.groupby('trial').measured_power_db.mean().to_numpy()
        mean = values.mean()
        half = student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
        points.append([beta,mean,mean-half,mean+half])
    points = np.asarray(points)
    key = condition if SCOPE == 'moving' else 'n'+str(int(n))
    selected_beta = float(mean_audit['selection'][key]['selected_beta'])
    case_label = condition.capitalize() if SCOPE == 'moving' else r'$n='+str(int(n))+r'$'
    label = case_label+', '+r'$\widehat\beta='+f'{selected_beta:g}'+r'$'
    ax.plot(points[:,0],points[:,1], 'o-', color=palette[index],label=label)
    ax.fill_between(points[:,0],points[:,2],points[:,3],color=palette[index],alpha=.10)
    selected = points[np.isclose(points[:,0],selected_beta)][0]
    ax.scatter(selected[0],selected[1],s=100,facecolors='none',edgecolors='#111111',linewidths=1.4,zorder=5)
ax.set(xlabel=r'$\beta$', ylabel=r'$\widehat V(\beta)$ (dB)')
ax.set_xticks(sorted(frame.beta.unique()))
ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
           ncol=3 if SCOPE == 'stationary' else 2, frameon=False, fontsize=LEGEND_FONT_SIZE)
'''

BETA_RESULTS=r'''baseline = float(mean_audit['config']['baseline_beta'])
choices = {case:float(value['selected_beta']) for case,value in mean_audit['selection'].items()}
if all(np.isclose(value,baseline) for value in choices.values()):
    display(Markdown('The independent posterior-mean searches select the existing '+
        r'$\beta='+f'{baseline:g}'+r'$ in every case. The corresponding test records are reused exactly; '
        'there is no additional posterior-mean curve or improved method to report.'))
    display(Markdown('Differences below $10^{-10}$ dB in repeated CSV summaries are round-trip '
        'precision, not an advantage or a separately resolved confidence-interval result.'))
else:
    display(Markdown('The independently selected memory is a separate comparison. It changes the '
        'posterior parameter, so its difference from a common-memory mode is not solely an uncertainty effect.'))
summary = mean_audit['summary']
summary = summary[summary.endpoint.eq('true_ratio_db')].copy()
paired = mean_audit['paired'].copy()
paired = paired[paired.endpoint.eq('true_ratio_db')]
for table in [summary,paired]:
    for field in ['policy','reference_policy']:
        if field in table:
            table[field] = table[field].replace({'preserved_common':'Original common memory',
                'independently_selected':'Independent mean validation'})
    for field in ['method','reference']:
        if field in table:table[field] = table[field].map(LABELS)
summary = summary.rename(columns={'policy':'Memory choice'})
paired = paired.rename(columns={'policy':'Memory choice','reference_policy':'Reference memory',
                               'mean_difference':'Difference (dB)'})
display(pd.DataFrame([{'case':case,'original beta':baseline,'selected beta':beta,
    'same setting':bool(np.isclose(beta,baseline))} for case,beta in choices.items()]))
display(HTML('<details><summary>All mean-selection test summaries and secondary aggregations</summary>'+
    summary.to_html(index=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
display(HTML('<details><summary>All paired mean-selection comparisons and secondary aggregations</summary>'+
    paired.to_html(index=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
'''

OBJECTIVE_GAP=r'''
frame = optimization['test_checkpoints']
if SCOPE == 'stationary':
    frame = frame[frame.t.eq(200)]
# These are per-simulation optimization errors, not posterior credible widths.
values = frame.groupby(['condition','n','method','inner_steps','trial'],sort=True).relative_objective_gap.mean().reset_index()
samples, labels, shades = [], [], []
for (condition,n,method,steps), group in values.groupby(['condition','n','method','inner_steps'],sort=True):
    case = condition.capitalize() if SCOPE == 'moving' else r'$n='+str(int(n))+r'$'
    label = 'Exact mode' if method == 'mode_exact' else r'GREAT (EW), $s='+str(int(steps))+r'$'
    samples.append(group.relative_objective_gap.to_numpy())
    labels.append(case+': '+label)
    shades.append(COLORS['mode'] if method == 'mode_exact' else COLORS['great_ew'])
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
boxes = ax.boxplot(samples, vert=False, patch_artist=True,
    medianprops={'color':'#111111','linewidth':1.3},
    flierprops={'markersize':1.8,'alpha':.35}, manage_ticks=False)
ax.set_yticks(np.arange(1,len(labels)+1), labels=labels)
for box, color in zip(boxes['boxes'], shades):box.set_facecolor(color)
ax.set_xlabel(r'$\overline\delta$' if SCOPE == 'moving' else r'$\delta_T$')
ax.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)
ax.axvline(0.,color='#222222',linestyle='--',linewidth=1.)
ax.grid(axis='x',color='#BFC4C8',alpha=.30,linewidth=.45)
'''

OBJECTIVE_TASK=r'''
frame = optimization['paired_summary']
primary_pairs = existing['paired_primary_summary']
cases = ([('smooth',32),('abrupt',32)] if SCOPE == 'moving' else
         [(str(frame.condition.iloc[0]),n) for n in CONFIG['dimensions']])
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
labels = []
for condition,n in cases:
    case = condition.capitalize() if SCOPE == 'moving' else r'$n='+str(int(n))+r'$'
    for row in frame[frame.condition.eq(condition)&frame.n.eq(n)].sort_values('inner_steps').itertuples():
        label = r'GREAT (EW), $s='+str(int(row.inner_steps))+r'$'
        position = len(labels);labels.append(case+': '+label)
        ax.errorbar(row.mean,position,
            xerr=[[row.mean-row.lower],[row.upper-row.mean]],
            fmt='o',color=COLORS['great_ew'],capsize=4.,markersize=7.)
    # The original mode is the same exact common-EW eigenspace (checked in the audit).
    row = primary_pairs[primary_pairs.reference.eq('mode')]
    row = row[row.condition.eq(condition)] if SCOPE == 'moving' else row[row.n.eq(n)]
    row = row.iloc[0]
    value, lower, upper = -row['mean'], -row['upper'], -row['lower']
    position = len(labels);labels.append(case+': '+LABELS['mean'])
    ax.errorbar(value,position,xerr=[[value-lower],[upper-value]],
                fmt='o',color=COLORS['mean'],capsize=4.,markersize=7.)
ax.axvline(0.,color=COLORS['mode'],linestyle='--',linewidth=1.2,label='Exact mode')
ax.set(yticks=np.arange(len(labels)),yticklabels=labels,
       xlabel=r'$L(D)-L(P_{\widehat{\mathcal X}})$ (dB)' if SCOPE == 'moving'
       else r'$\ell_T(D)-\ell_T(P_{\widehat{\mathcal X}_T})$ (dB)')
ax.tick_params(axis='y',labelsize=LEGEND_FONT_SIZE)
ax.invert_yaxis()
ax.grid(axis='x',color='#BFC4C8',alpha=.30,linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',frameon=False,fontsize=LEGEND_FONT_SIZE)
'''

OBJECTIVE_TABLE=r'''display(Markdown('The following are pooled checkpoint diagnostics over '+
    ('all 32 moving reports' if SCOPE == 'moving' else 'all four stationary reports')+
    '. Their mean and maximum are optimization diagnostics, not the primary task loss.'))
rows = []
for (condition,n,method,steps), group in optimization['test_checkpoints'].groupby(
        ['condition','n','method','inner_steps'],sort=True):
    rows.append({'condition':condition,'n':n,'method':'Exact mode' if method == 'mode_exact' else 'GREAT (EW)',
        'steps':np.nan if method == 'mode_exact' else steps,
        'mean checkpoint gap':group.relative_objective_gap.mean(),
        'maximum relative gap':group.relative_objective_gap.max(),
        'maximum chordal distance to the mode':group.chordal_to_mode.max()})
display(pd.DataFrame(rows))
display(optimization['paired_summary'])
'''


OPTIMIZATION_PAIRS_TO_MEAN=r'''# Pair every optimizer with the original common-parameter posterior mean.
if SCOPE == 'moving':
    baseline=existing['simulation_aggregates']
    baseline=baseline[baseline.primary & baseline.method.eq('mean')].copy()
    baseline['n']=32
    baseline=baseline[['condition','n','trial','value_db']]
else:
    baseline=existing['raw_stats']
    baseline=baseline[baseline.t.eq(200)&baseline.method.eq('mean')]
    baseline=baseline[['condition','n','trial','true_ratio_db']].rename(columns={'true_ratio_db':'value_db'})
frame=optimization['test_checkpoints']
if SCOPE == 'stationary':frame=frame[frame.t.eq(200)]
scores=frame.groupby(['condition','n','method','inner_steps','trial'],sort=True).true_ratio_db.mean().reset_index()
paired=scores.merge(baseline,on=['condition','n','trial'],validate='many_to_one')
paired['difference_db']=paired.true_ratio_db-paired.value_db
rows=[]
for (condition,n,method,steps), group in paired.groupby(['condition','n','method','inner_steps'],sort=True):
    assert len(group)==256 and group.trial.nunique()==256
    values=group.difference_db.to_numpy()
    mean=values.mean()
    half=student_t.ppf(.975,255)*values.std(ddof=1)/16.
    rows.append({'condition':condition,'n':n,
        'Reference':'Exact mode' if method=='mode_exact' else 'GREAT (EW)',
        'Gradient steps':np.nan if method=='mode_exact' else int(steps),
        'Reference minus posterior mean (dB)':mean,'95% lower':mean-half,'95% upper':mean+half,
        'simulations':256})
optimizer_pairs=pd.DataFrame(rows)
display(optimizer_pairs)
EDITABLE_OUTPUT.mkdir(parents=True,exist_ok=True)
optimizer_pairs.to_csv(EDITABLE_OUTPUT/'optimizer_paired_vs_mean.csv',index=False)
display(Markdown('These differences use the same original simulation in each pair. Positive '
    'values favor the posterior mean; negative values favor the reference. The selected finite-step '
    'GREAT result is not assumed to be fully converged.'))
'''

TIMING=r'''
clock = optimization['timing']['raw']
samples, labels, shades = [], [], []
for (condition,n,method,steps), group in clock.groupby(['condition','n','method','inner_steps'],sort=True):
    case = condition.capitalize() if SCOPE == 'moving' else r'$n='+str(int(n))+r'$'
    label = 'Exact mode' if method == 'mode_exact' else r'GREAT (EW), $s='+str(int(steps))+r'$'
    samples.append(group.total_ms.to_numpy())
    labels.append(case+': '+label)
    shades.append(COLORS['mode'] if method == 'mode_exact' else COLORS['great_ew'])
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
boxes = ax.boxplot(samples,vert=False,patch_artist=True,
    medianprops={'color':'#111111','linewidth':1.3},flierprops={'markersize':1.8,'alpha':.35},manage_ticks=False)
ax.set_yticks(np.arange(1,len(labels)+1),labels=labels)
for box,color in zip(boxes['boxes'],shades):box.set_facecolor(color)
ax.set_xlabel('Time for '+str(CONFIG['T'])+' measurements (ms)')
ax.tick_params(axis='y',labelsize=LEGEND_FONT_SIZE)
ax.grid(axis='x',color='#BFC4C8',alpha=.30,linewidth=.45)
'''

STATIONARY_ORIGINAL=r'''
frame = existing['primary_summary']
methods = ['mean','mode','isotropic','great_window','great_ew','covariance_loading']
fig, ax = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout='constrained')
for method in methods:
    group = frame[frame.method.eq(method)].sort_values('n')
    ax.plot(group.n,group['mean'],marker=MARKERS[method],color=COLORS[method],
            linestyle=LINESTYLES[method],markevery=(MARK_OFFSETS[method],3),label=LABELS[method])
    if method == 'mean':ax.fill_between(group.n,group.lower,group.upper,color=COLORS[method],alpha=.14,linewidth=0)
ax.set(xlabel=r'$n$',ylabel=r'$\ell_T$ (dB)')
ax.set_xticks(CONFIG['dimensions'])
ax.grid(color='#BFC4C8',alpha=.30,linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=LEGEND_FONT_SIZE)
'''

STATIONARY_NEW=r'''
frame = selection['test_summary']
methods = ['mean','mode','isotropic','great_window_final_db','great_full_history_final_db',
           'great_ew_final_db','covariance_loading_final_db']
selected_window = frame[frame.method.eq('great_window_final_db')].sort_values('n')
full_history = frame[frame.method.eq('great_full_history_final_db')].sort_values('n')
same_history = all(record['settings']['great_window_final_db']['window'] == 200
                   for record in selection['selected_settings'].values())
if same_history and np.allclose(selected_window[['mean','lower','upper']],full_history[['mean','lower','upper']],rtol=0,atol=1e-10):
    methods.remove('great_window_final_db')  # Same settings; tables retain both records.
fig, ax = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout='constrained')
for method in methods:
    group = frame[frame.method.eq(method)].sort_values('n')
    ax.plot(group.n,group['mean'],marker=MARKERS[method],color=COLORS[method],
            linestyle=LINESTYLES[method],markevery=(MARK_OFFSETS[method],3),label=LABELS[method])
    if method == 'mean':ax.fill_between(group.n,group.lower,group.upper,color=COLORS[method],alpha=.14,linewidth=0)
ax.set(xlabel=r'$n$',ylabel=r'$\ell_T$ (dB)')
ax.set_xticks(CONFIG['dimensions'])
ax.grid(color='#BFC4C8',alpha=.30,linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=LEGEND_FONT_SIZE)
'''

STATIONARY_PAIRS=r'''
frame = selection['paired_vs_mean']
frame = frame[frame.n.eq(64)]
methods = ['mode','isotropic','great_window_final_db','great_full_history_final_db',
           'great_ew_final_db','covariance_loading_final_db']
selected_window = frame[frame.reference.eq('great_window_final_db')].iloc[0]
full_history = frame[frame.reference.eq('great_full_history_final_db')].iloc[0]
same_history = selection['selected_settings']['64']['settings']['great_window_final_db']['window'] == 200
if same_history and np.allclose(selected_window[['mean','lower','upper']].to_numpy(dtype=float),
               full_history[['mean','lower','upper']].to_numpy(dtype=float),rtol=0,atol=1e-10):
    methods.remove('great_window_final_db')
fig, ax = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout='constrained')
labels=[]
for method in methods:
    row = frame[frame.reference.eq(method)].iloc[0]
    position=len(labels);labels.append(LABELS[method])
    ax.errorbar(row['mean'],position,
        xerr=[[row['mean']-row['lower']],[row['upper']-row['mean']]],
        fmt=MARKERS[method],color=COLORS[method],capsize=4.,markersize=7.)
ax.axvline(0.,color='#222222',linestyle='--',linewidth=1.2)
ax.set(yticks=np.arange(len(labels)),yticklabels=labels,
       xlabel=r'$\ell_T(D)-\ell_T(\overline P^+)$ (dB)')
ax.tick_params(axis='y',labelsize=LEGEND_FONT_SIZE)
ax.invert_yaxis()
ax.grid(axis='x',color='#BFC4C8',alpha=.30,linewidth=.45)
'''

STATIONARY_TABLES=r'''# Original settings and the new final-time criterion remain separate.
original = existing['primary_summary'].copy()
original['method'] = original.method.map(LABELS)
original_pairs = existing['paired_primary_summary'].copy()
original_pairs['reference'] = original_pairs.reference.map(LABELS)
display(HTML('<details><summary>All original final-measurement means</summary>'+
    original.to_html(index=False,escape=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
display(HTML('<details><summary>All original paired differences</summary>'+
    original_pairs.to_html(index=False,escape=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
rows=[]
for n, record in selection['selected_settings'].items():
    for method, parameters in record['settings'].items():
        rows.append({'n':int(n),'method':LABELS[method],
            **{name:parameters.get(name,np.nan) for name in
               ['window','inner_steps','beta','stepsize','step_multiplier','alpha','validation_mean_db']}})
display(HTML('<details><summary>Actual settings selected by the final-measurement criterion</summary>'+
    pd.DataFrame(rows).to_html(index=False,escape=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
new_summary = selection['test_summary'].copy()
new_summary['method'] = new_summary.method.map(LABELS)
new_pairs = selection['paired_vs_mean'].copy()
new_pairs['reference'] = new_pairs.reference.map(LABELS)
display(HTML('<details><summary>All final-time-selection means, including original references</summary>'+
    new_summary.to_html(index=False,escape=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
display(HTML('<details><summary>All paired final-time-selection differences</summary>'+
    new_pairs.to_html(index=False,escape=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
if all(record['settings']['great_window_final_db']['window'] == 200
       for record in selection['selected_settings'].values()):
    display(Markdown('The unrestricted selected window is $W=200$ at every dimension. It coincides '
        'with the explicit full-history comparison, so only one of these identical curves is drawn. '
        'Both audit records remain in the tables.'))
means=selection['test_summary'].pivot(index='n',columns='method',values='mean')
comparison=means[['mean','mode','great_window','great_full_history_final_db','great_ew_final_db']].copy()
comparison.columns=['Posterior mean','Mode','Original sliding window','Full history W=200','GREAT EW (final-time criterion)']
display(comparison)
if all(means['great_full_history_final_db'] < means['mean']):
    display(Markdown('With matched full observation history, GREAT has lower mean final-measurement '
        'dB loss than the posterior mean at every dimension in this fixed generating cohort. '
        'The substantial original sliding-window gap largely disappears. This identifies observation '
        'history as an important confound in that comparison; it does not contradict the posterior '
        'expected-linear-power decision rule.'))
linear_rows=[]
for (n,method), group in selection['test_metrics'].groupby(['n','method'],sort=True):
    values=(group.true_power/group.oracle_power).to_numpy()
    mean=values.mean()
    half=student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
    linear_rows.append({'n':n,'method':method,'mean linear ratio':mean,
                       'lower':mean-half,'upper':mean+half,'simulations':len(values)})
linear_summary=pd.DataFrame(linear_rows)
display(HTML('<details><summary>Mean linear power ratios: a separate test risk</summary>'+
    linear_summary.to_html(index=False,float_format=lambda value:f'{value:.6g}')+'</details>'))
EDITABLE_OUTPUT.mkdir(parents=True,exist_ok=True)
linear_summary.to_csv(EDITABLE_OUTPUT/'linear_power_summary.csv',index=False)
'''


def _load(root,scope):
    from lib.beamforming_comparison_summaries import load_existing_beamforming_comparisons
    from lib.beamforming_mean_validation_audit import load_mean_validation_audit
    from lib.beamforming_objective_audit import load_objective_audit
    result={'existing':load_existing_beamforming_comparisons(root)[scope],
            'mean':load_mean_validation_audit(root,scope=scope),
            'objective':load_objective_audit(root,cohort=scope)}
    if result['objective']['timing'] is None:
        raise RuntimeError('The isolated timing audit is not completed: '+scope)
    if scope=='stationary':
        from lib.beamforming_stationary_selection_audit import load_stationary_selection_audit
        result['selection']=load_stationary_selection_audit(root)
    return result


def _metadata_inputs(root,scope,evidence):
    inputs=dict(evidence['existing']['input_hashes'])
    phases=['existing','mean_validation','objective','objective/timing']
    if scope=='stationary':phases.append('selection')
    for phase in phases:
        path=root/OUTPUT/scope/phase/'manifest.json'
        if not path.is_file():raise RuntimeError('Missing completed audit manifest: '+str(path))
        inputs[str(path.relative_to(root))]=sha(path)
    modules=['lib/beamforming_comparison_summaries.py','lib/beamforming_mean_validation_audit.py',
             'lib/beamforming_objective_audit.py']
    if scope=='stationary':modules.append('lib/beamforming_stationary_selection_audit.py')
    sources={name:sha(root/name) for name in modules}
    return inputs,sources


def build(root,scope):
    import nbformat
    evidence=_load(root,scope);inputs,sources=_metadata_inputs(root,scope,evidence)
    config=({'n':32,'T':256,'conditions':['smooth','abrupt'],'common_beta':.97,
             'primary_aggregation':'mean_db','primary_window':'all','test_simulations':256,
             'sigma2':.2,'tau2':1.,'validation_simulations':64}
            if scope=='moving' else {'T':200,'dimensions':[32,64,128],'common_beta':1.,
                                    'primary_t':200,'test_simulations':256,
                                    'sigma2':.2,'tau2':1.,'validation_simulations':64})
    cells=[]
    def md(text,identifier):cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip(),id=identifier))
    def code(text,identifier,figure=None):
        metadata={'editable_plotting_role':'plot','figure_id':figure,'source_page':1} if figure else {}
        cells.append(nbformat.v4.new_code_cell(text,id=identifier,metadata=metadata))
    md('# '+('Moving' if scope=='moving' else 'Stationary')+' beamforming comparisons\n\n'
       'This notebook separates posterior uncertainty, finite optimization and validation choices. '
       'It reads completed audit records from the original observations and independent validation/test '
       'simulation identifiers. This comparison audit preserves original raw records and estimator settings; '
       'it does not rewrite existing application notebooks or their result files. '
       'Every figure has visible plotting code and exports only to this audit directory.','audit-introduction')
    code(SETUP.replace('__SCOPE__',repr(scope)).replace('__CONFIG__',repr(config)),'audit-setup')
    md(r'''## Loss and evaluation procedure

For $R_t=\sigma^2I_n+\tau^2P_{\mathcal X_t}$, $p_t^\star=(a^HR_t^{-1}a)^{-1}$ and a distortionless beamformer $w_t$, the reported loss is
\[
\ell_t=10\log_{10}\frac{w_t^HR_tw_t}{p_t^\star}.
\]
The posterior-mean MVDR rule minimizes posterior **expected linear output power**, $\mathbb E_q[w^H R_{\mathcal X}w]$, under the distortionless constraint. It does not generally minimize $\mathbb E_q[10\log_{10}(w^H R_{\mathcal X}w)]$. The primary benchmark below averages individual dB losses over a fixed generating cohort, which is a different risk. A reversed ranking therefore does not contradict the linear-power Bayes decision.

Evaluation uses this saved exact-covariance loss. Parameter selection uses independent measured output power on 64 validation simulations; true covariance and oracle power do not select a setting. Reported test confidence intervals are pointwise Student 95% intervals across 256 paired test simulations. They are not simultaneous confidence statements across algorithms and cases. Validation bands use the separate 64 validation simulations.

The posterior mean, mode and matched isotropic mean use the same posterior parameter in the controlled comparison. Their differences preserve memory and scale settings. Changing a separately selected memory, observation window or number of gradient steps is another comparison.''','loss-definition')
    if scope=='moving':
        md(r'''## Moving data: define averaging before comparing methods

There are 32 saved reporting measurements $\mathcal T=\{8,16,\ldots,256\}$. The primary outcome is the arithmetic mean of the individual dB losses inside each simulation,
\[
L(D)=\frac1{32}\sum_{t\in\mathcal T}\ell_t(D_t).
\]
The posterior mean, mode, isotropic mean and exponentially weighted GREAT retain their original common $\beta=0.97$. EW denotes exponential weighting. Sliding-window GREAT retains its independently selected original window and step settings. Both GREAT references are shown. Their original common-memory and GREAT settings were selected by mean measured **linear** output power at $t\ge32$; the primary evaluation here uses all 32 individual dB reports. The independent mean-validation audit below explicitly checks the latter criterion.

Three secondary outcomes are retained after selection: restrict the arithmetic mean to $t\ge32$ (29 reports), use $10\log_{10}$ of the mean linear power ratio over all 32 reports, or use that linear-ratio aggregation at $t\ge32$. The restriction omits only $t=8,16,24$. The dB of a mean linear ratio is a different outcome from a mean of individual dB values; the logarithm does not commute with averaging. No ranking chooses which outcome is primary.''','moving-averaging')
        code(STYLE+'\n'+ORIGINAL_SETTINGS,'original-moving-settings')
        for condition in ('smooth','abrupt'):
            md(('### Smooth direction changes' if condition=='smooth' else '### Abrupt direction change')+'\n\n'
               'These are the original selected algorithms evaluated on the same reporting measurements. '
               'Blue shading is the confidence interval for the posterior-mean Monte Carlo mean.'
               +(' The direction changes at $t=129$.' if condition=='abrupt' else ''),'history-note-'+condition)
            code(plot_source(condition+'_history',MOVING_HISTORY.replace('CONDITION',repr(condition))),
                 'history-'+condition,condition+'_history')
        md(r'''### Paired primary differences

Each point below is a same-simulation difference $L(D)-L(\overline P^+)$. Positive values favor the posterior mean; negative values favor the reference. A confidence interval containing zero does not resolve its sign. This figure retains all four references in both moving cases.''','moving-paired-note')
        code(plot_source('paired_primary',MOVING_PAIRS),'moving-paired-primary','paired_primary')
        code(STYLE+'\n'+AGGREGATION_TABLES,'all-moving-aggregations')
        md(r'''The primary comparisons do not have a uniform winner. Under smooth motion, sliding-window GREAT has lower mean loss than the posterior mean, while the saved three-step GREAT EW has higher loss; the mean–mode interval contains zero. Under abrupt change, the posterior mean improves on both GREAT references, while the common mode improves on the posterior mean. The full posterior mean improves on its matched isotropic mean in both cases, with small paired differences.

The mode–mean ordering changes under the secondary averaging conventions. The original primary statistic remains the arithmetic mean of all 32 individual dB losses. These sign changes are reported rather than selecting an averaging rule that favors a method.''','moving-existing-findings')
    else:
        md(r'''## Stationary data: exactly $T=200$ measurements at every dimension

The primary endpoint is $\ell_T$ at $T=200$, for $n=32,64,128$. The controlled mean/mode/isotropic comparison retains $\beta=1$. The first figure uses the original settings, including both GREAT references and the regularized sample covariance. Those GREAT/window/loading settings were chosen using the original four-checkpoint mean **linear** measured-power procedure. EW denotes exponential weighting. Their outcomes must not be confused with the new final-measurement validation criterion below.''','stationary-original-note')
        code(STYLE+'\n'+ORIGINAL_SETTINGS,'original-stationary-settings')
        code(plot_source('original_settings',STATIONARY_ORIGINAL),'stationary-original','original_settings')
        md(r'''### Final-measurement validation and observation history

The separate audit selects settings using the mean individual measured dB output power at $t=200$ on the same 64 validation simulations. The test simulations remain independent. Sliding-window GREAT additionally admits the full observation history $W=T=200$. The unrestricted selected window is denoted $\widehat W_n$; a separately selected $W=T$ comparison is retained even when it is not the unrestricted choice. GREAT EW and covariance shrinkage are also selected under this final-measurement criterion.

The posterior mean, mode and isotropic controls below retain their original parameters. Thus an improvement in this section may reflect a more appropriate history or a different measured validation criterion, rather than posterior uncertainty. Exact selected windows, step counts, step sizes and loading parameters are reported in the tables.''','stationary-final-note')
        code(plot_source('final_time_selection',STATIONARY_NEW),'stationary-final-selection','final_time_selection')
        md(r'''The paired display uses $n=64$ at $t=200$, with reference-minus-posterior-mean differences. All three dimensions and all original/new methods, including negative differences, remain in the following tables.''','stationary-pairs-note')
        code(plot_source('paired_final_time',STATIONARY_PAIRS),'stationary-paired-final','paired_final_time')
        code(STYLE+'\n'+STATIONARY_TABLES,'all-stationary-comparisons')
    md(r'''## Independent memory selection for the posterior mean

This audit searches only the original grid $\beta\in\{1,0.99,0.97,0.9\}$. Define $\widehat V(\beta)$ as the average measured dB output power across validation simulations, averaging the primary reporting measurements inside each simulation. For moving data these are all 32 reports; for stationary data only $t=200$ is used. The minimum on the validation grid selects $\widehat\beta$. Test data and secondary outcomes do not participate in selection.

The open marker identifies the selected grid point. The labels give its actual $\widehat\beta$. Confidence bands summarize validation-simulation variability and do not change the selection rule.''','mean-beta-note')
    code(plot_source('posterior_mean_beta',BETA_VALIDATION),'mean-beta-validation','posterior_mean_beta')
    code(BETA_RESULTS,'mean-beta-test-results')
    md(r'''## EW objective, convergence and task loss

Let $C_t=\beta C_{t-1}+\widetilde u_t\widetilde u_t^H$ be the covariance statistic of the likelihood-scaled observations used by GREAT EW. Under the common uniform-prior, fixed-scale construction, its dominant eigenspace equals the mode of the Bingham parameter: adding a scalar identity matrix or multiplying by a positive scalar does not change that eigenspace. This is an equality of objectives. A finite number of gradient steps need not attain that mode.

For the current unit GREAT vector $x_t$, define the scaled Rayleigh gap
\[
\delta_t=\frac{\lambda_{\max}(C_t)-x_t^HC_tx_t}{\lambda_{\max}(C_t)-\lambda_{\min}(C_t)}.
\]
The exact mode has $\delta_t=0$. The reported chordal discrepancy is $d_c(P_{x_t},P_{\widehat{\mathcal X}_t})=\sqrt{1-|x_t^H\widehat x_t|^2}$, a distance to the exact modal eigenspace, not an estimation error against the generating subspace. With ordered eigenvalues $\lambda_1\ge\cdots\ge\lambda_n$ of $C_t$, the unscaled gap $g_t=\lambda_1-x_t^HC_tx_t$ satisfies
\[
(\lambda_1-\lambda_2)d_c^2\le g_t\le(\lambda_1-\lambda_n)d_c^2.
\]
A small scaled objective gap need not imply a small direction error when the leading eigengap is small. Optimizing the EW residual also does not guarantee the lowest true output power. The audit retains the original $s=3$ gradient steps per measurement and the separately validation-selected count from $s\in\{1,3,10,30\}$, keeping the original memory and step size fixed. Selection uses measured log power, without covariance truth.

For moving data the objective distribution averages $\delta_t$ over all 32 reports inside each simulation, and measured validation averages those same reports. For stationary data the displayed distribution is $\delta_T$ at each $n=32,64,128$ and $T=200$, and the count is selected using only the measured $t=200$ validation loss. The boxes show medians and interquartile ranges across simulations, with 1.5-IQR whiskers and all outliers; these are distributions, not confidence intervals for a mean. All four saved stationary times remain in the objective diagnostics. The table retains maximum transient errors, so a small average does not conceal poor optimization at particular measurements.''','objective-definition')
    code(plot_source('ew_objective_gap',OBJECTIVE_GAP),'ew-objective-gap','ew_objective_gap')
    code(OBJECTIVE_TABLE,'ew-objective-all-checks')
    md(r'''The next display reports paired test-loss differences from the exact common-EW mode and includes the original posterior mean as context. The zero line is the exact-mode reference. Its intervals are pointwise 95% intervals for paired Monte Carlo mean differences. A gap between GREAT EW and that exact mode is an optimization effect. The posterior mean versus exact-mode comparison is the relevant uncertainty/shrinkage comparison under common parameters; neither a finite-step optimization gap nor a separately selected memory can be attributed entirely to anisotropy.''','objective-task-note')
    code(plot_source('ew_task_power',OBJECTIVE_TASK),'ew-task-power','ew_task_power')
    code(OPTIMIZATION_PAIRS_TO_MEAN,'optimizer-paired-to-posterior-mean')
    md(r'''## Cost of the finite-step comparison

The timings were collected in a separate isolated, warmed single-thread run. Each moving workload processes 256 measurements and 32 task decisions; each stationary workload processes 200 measurements and its four original task reports. All GREAT updates are included. The exact-mode workload updates the same covariance and computes its eigenspace only when a task decision is reported. Diagnostic eigendecompositions used to measure optimization error are excluded from GREAT timings.

These clocks compare GREAT EW with the exact mode; they do not time the posterior mean. Repeated clock measurements describe execution variability, not independent Monte Carlo task simulations or a worst-case deadline. The boxes show medians and interquartile ranges, with 1.5-IQR whiskers and all outliers.''','timing-definition')
    code(plot_source('computation_time',TIMING),'optimization-computation-time','computation_time')
    code("clock_summary=optimization['timing']['summary'].copy()\noperations=optimization['operation_counts'].copy()\nfor table in [clock_summary,operations]:\n    table.loc[table.method.eq('mode_exact'),'inner_steps']=np.nan\n    table['method']=table.method.replace({'mode_exact':'Exact mode','great_ew':'GREAT (EW)'})\ndisplay(clock_summary)\ndisplay(operations)",'timing-counts')
    md(r'''## Reading the comparisons

The posterior-mean MVDR decision minimizes posterior expected linear output power. That statement does not imply optimality for an average individual-dB loss, or for every fixed generating parameter or moving trajectory. The same-parameter mean/mode/isotropic results, finite-step optimization results and independently selected parameter results answer different questions. Positive and negative paired differences are retained throughout.

The original GREAT EW implementation performs a finite optimization update. Its distance from the exact common-EW mode must be checked before using a mean-versus-GREAT difference as evidence for posterior uncertainty. Sliding-window GREAT can also use a different observation history. The independent validation audits separate those effects while preserving the original comparisons.

The primary outcome was defined before examining the rankings. Secondary averaging conventions remain visible because they can reverse a small comparison. All pointwise intervals and raw paired simulation differences are saved alongside this notebook's figures; none of these displays establishes universal superiority or posterior coverage under an unmodelled transition.''','audit-interpretation')
    links=('## Saved audit records\n\n'
        +'[Figures](../results/beamforming_comparison_audit/'+scope+'/figures/) · '
        +'[Original-setting summaries](../results/beamforming_comparison_audit/'+scope+'/existing/) · '
        +'[Mean-memory validation](../results/beamforming_comparison_audit/'+scope+'/mean_validation/) · '
        +'[EW objective and isolated timings](../results/beamforming_comparison_audit/'+scope+'/objective/)')
    if scope=='stationary':
        links+=' · [Final-measurement selection](../results/beamforming_comparison_audit/stationary/selection/)'
    md(links,'audit-record-links')
    nb=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},
        'editable_plotting':{'schema':1,'mode':'inline_matplotlib','figure_count':len(FIGURE_NAMES[scope]),
            'rendered_figure_count':len(FIGURE_NAMES[scope]),'output_directory':str(OUTPUT/scope/'figures'),
            'input_hashes':inputs,'loader_sources':sources,
            'controls':'Visible Matplotlib code,15x5,300dpi,LaTeX,fonts25/23/22; local exports only.'},
        'beamforming_comparison_audit':{'schema':1,'scope':scope,'config':config,
            'producer':'scripts/make_beamforming_comparison_audit_notebooks.py','producer_sha256':sha(__file__),
            'input_hashes':inputs,'loader_sources':sources,
            'historical_producer_sources':evidence['existing']['source_hashes'],
            'original_settings_unchanged':True,'new_parameters_selected_on_validation_only':True}})
    import nbformat
    nbformat.write(nb,root/NOTEBOOKS[scope])
    return nb


def execute(root,notebook,timeout):
    import nbformat
    from nbclient import NotebookClient
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
    os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
    NotebookClient(notebook,timeout=timeout,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(notebook,root/NOTEBOOKS[notebook.metadata['beamforming_comparison_audit']['scope']])


def write_manifest(root,scope,notebook,protection):
    directory=root/OUTPUT/scope;files={};figures={}
    for stem in FIGURE_NAMES[scope]:
        pdf=OUTPUT/scope/'figures'/(stem+'.pdf');png=pdf.with_suffix('.png')
        for path in (pdf,png):
            if not (root/path).is_file():raise RuntimeError('Missing audit figure: '+str(path))
            files[str(path)]=sha(root/path)
        figures[stem+'.pdf']=dict(pdf=str(pdf),png=str(png),figure_size_in=[15,5],dpi=300,text_usetex=True,
                                  pdf_sha256=files[str(pdf)],png_sha256=files[str(png)])
    for path in directory.rglob('*'):
        unpublished=any(part.startswith('unpublished_') or part=='archive' for part in path.relative_to(directory).parts)
        if (path.is_file() and not unpublished and path.suffix in ('.csv','.gz','.json','.npz')
                and path.name!='figure_manifest.json'):
            files[str(path.relative_to(root))]=sha(path)
    meta=notebook.metadata['beamforming_comparison_audit']
    record=dict(schema=1,status='completed',scope=scope,notebook=str(NOTEBOOKS[scope]),
        notebook_sha256=sha(root/NOTEBOOKS[scope]),producer=meta['producer'],producer_sha256=sha(__file__),
        config=meta['config'],input_hashes=meta['input_hashes'],loader_sources=meta['loader_sources'],
        historical_producer_sources=meta['historical_producer_sources'],figures=figures,files=files,
        preservation=protection,scope_note='Separate audit exports only; existing notebooks and results unchanged relative to the pre-render preservation snapshot.')
    (directory/'figure_manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--scope',choices=['moving','stationary','both'],default='both')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args();root=args.root.resolve();folder=root/OUTPUT;folder.mkdir(parents=True,exist_ok=True)
    import fcntl
    with open('/tmp/bayesian-st-notebook-render.lock','a+') as render_lock:
        fcntl.flock(render_lock,fcntl.LOCK_EX)
        baseline=protected_files(root)
        (folder/'notebook_preservation_baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        scopes=('moving','stationary') if args.scope=='both' else (args.scope,)
        reports=[]
        for scope in scopes:
            notebook=build(root,scope)
            if args.execute:
                execute(root,notebook,args.timeout)
                protection=check_preservation(root,baseline)
                report=write_manifest(root,scope,notebook,protection)
                reports.append({'scope':scope,'status':'completed','figures':len(report['figures'])})
            else:reports.append({'scope':scope,'status':'prepared'})
        print(json.dumps({'status':'passed','notebooks':reports,'preservation':check_preservation(root,baseline)},indent=2),flush=True)



if __name__=='__main__':main()
