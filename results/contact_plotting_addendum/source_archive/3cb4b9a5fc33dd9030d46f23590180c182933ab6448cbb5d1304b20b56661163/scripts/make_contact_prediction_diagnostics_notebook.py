"""Build and execute a separate contact prediction diagnostics notebook.

Only saved observations, predictions and posterior parameters are read. Existing
application notebooks, beamforming artifacts and contact figures are protected
by hashes; every new figure exposes its complete Matplotlib source in its cell.
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
NOTEBOOK=Path('notebooks/contact_prediction_diagnostics.ipynb')
OUTPUT=Path('results/contact_prediction_diagnostics')
FIGURES=[('prediction_error_stationary','main'),('prediction_error_sphere','main'),
         ('prediction_error_references','supplementary'),('moving_sphere_posterior_spheres','supplementary')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_files(root):
    paths=set((root/'notebooks').glob('*.ipynb'))-{root/NOTEBOOK}
    for directory in (root/'results').glob('beamforming*'):
        if directory.is_dir():
            paths.update(path for path in directory.rglob('*') if path.is_file())
    for directory in ('results/main/beamforming','results/supplementary/beamforming',
                      'results/notebook_exports/01_array','results/main/contact',
                      'results/supplementary/contact','results/contact',
                      'results/notebook_exports/03_contact'):
        folder=root/directory
        for suffix in ('*.pdf','*.png'):
            paths.update(folder.rglob(suffix))
    owned={root/'results'/section/'contact'/(stem+suffix)
           for stem,section in FIGURES for suffix in ('.pdf','.png')}
    paths.difference_update(owned)
    return {str(path.relative_to(root)):sha(path) for path in sorted(paths)}


def verify_protected(root,baseline):
    changed=[name for name,digest in baseline.items() if not (root/name).is_file() or sha(root/name)!=digest]
    if changed:
        raise RuntimeError('An existing notebook or protected result changed: '+', '.join(changed))
    return dict(status='passed',files_checked=len(baseline),existing_notebooks=sum(Path(name).parent==Path('notebooks') and name.endswith('.ipynb') for name in baseline),
                changed=[],baseline_sha256=hashlib.sha256(json.dumps(baseline,sort_keys=True).encode()).hexdigest())


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
from matplotlib import colors
from matplotlib.lines import Line2D
from IPython import get_ipython
from IPython.display import Markdown, display
from lib.contact_prediction_diagnostics import load_contact_prediction_diagnostics
from lib.contact_moving_snapshot import load_moving_sphere_snapshot

# Data readers validate the saved experiment; neither reruns the estimator.
data = load_contact_prediction_diagnostics(ROOT)
snapshot = load_moving_sphere_snapshot(ROOT, t=96)
EDITABLE_OUTPUT = ROOT / 'results/contact_prediction_diagnostics/figures'
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
    get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': None}")
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
    'font.family': 'serif', 'lines.linewidth': 1.8, 'lines.markersize': 5,
    'savefig.bbox': None, 'figure.max_open_warning': 0,
})
print('Saved contact predictions and the corrected sphere posterior at t=96 are verified.')
'''

CONTROLS=r'''# Edit this figure's canvas, text, colors and save commands directly.
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

STYLE=r'''LABELS = {
    'mean': r'$\overline P_t^+$', 'mode': r'$P_{\widehat{\mathcal X}_t}$',
    'iso': r'$\overline P_t^{\mathrm{iso}}$',
    'regularized': r'$\widetilde M_t$',
    'great_window': 'GREAT (sliding window)', 'great_ew': 'GREAT (exponential weighting)',
}
COLORS = {'mean':'#0072B2', 'mode':'#D55E00', 'iso':'#CC79A7',
          'regularized':'#8C564B', 'great_window':'#009E73', 'great_ew':'#56B4E9'}
MARKERS = {'mean':'o', 'mode':'s', 'iso':'^', 'regularized':'D',
           'great_window':'D', 'great_ew':'v'}
'''

SAVE=r'''# Vector text/curves remain in the PDF; surfaces are rasterized at DPI.
fig.get_layout_engine().set(h_pad=.18, w_pad=.16)
if SAVE_PDF:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, dpi=DPI, bbox_inches=None)
    PNG_PATH = PDF_PATH.with_suffix('.png')
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches=None)
    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())
    DRAFT_PNG_PATH = DRAFT_PATH.with_suffix('.png')
    DRAFT_PNG_PATH.write_bytes(PNG_PATH.read_bytes())
plt.show()
'''


def plotting_source(stem,section,body):
    return (CONTROLS+f"PDF_PATH = EDITABLE_OUTPUT / '{stem}.pdf'\nDRAFT_SECTION = '{section}'\n"
        +"DRAFT_PATH = ROOT / 'results' / DRAFT_SECTION / 'contact' / PDF_PATH.name\n\n"
        +STYLE+'\n'+dedent(body).strip()+'\n\n'+SAVE)


HISTORY=r'''
condition = CONDITION
record = data['conditions'][condition]
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for curve in record['curves']:
    method = curve['method']
    points = np.asarray(curve['points'], dtype=float)
    ax.plot(points[:,0], points[:,1], color=COLORS[method], marker=MARKERS[method],
            linestyle='--' if method == 'iso' else '-', label=LABELS[method])
    if method == 'mean':
        ax.fill_between(points[:,0], points[:,2], points[:,3],
                        color=COLORS[method], alpha=.14, linewidth=0)
ax.set(xlabel=r'$t$', ylabel=r'$\mathcal E_t(D_t)$ $((\mathrm{mm}/\mathrm{s})^2)$',
       xlim=(1, int(record['T'])))
ax.spines[['top','right']].set_visible(False)
ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
           ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE)
'''

SPHERE_HISTORY = HISTORY.replace(
    "for curve in record['curves']:",
    """# Keep the measured-frame curves and add the saved identity-transport control.
LABELS['mean_identity'] = r'Posterior mean, $G_t=I_3$'
COLORS['mean_identity'] = COLORS['mean']
MARKERS['mean_identity'] = 'x'
curves = list(record['curves'])
identity_curves = [curve for curve in record['all_curves']
                   if curve['method'] == 'mean_identity']
if len(identity_curves) != 1:
    raise ValueError('Expected one saved identity-transport posterior-mean curve')
curves += identity_curves
for curve in curves:""").replace(
    "linestyle='--' if method == 'iso' else '-'",
    "linestyle='--' if method in ('iso', 'mean_identity') else '-'")

REFERENCES=r'''
# This endpoint averages all 48 scheduled action times inside each simulation.
summary = data['episode_average_summary']
summary = summary[summary.condition.eq('sphere')]
methods = ['mean','mode','iso','regularized','great_window','great_ew']
methods = [method for method in methods if summary.method.eq(method).any()]
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for position, method in enumerate(methods):
    rows = summary[summary.method.eq(method)]
    if len(rows) != 1:
        raise ValueError('Expected one saved sphere average per method')
    row = rows.iloc[0]
    ax.errorbar(row['mean'], position,
        xerr=[[row['mean']-row['lower']], [row['upper']-row['mean']]],
        fmt=MARKERS[method], color=COLORS[method], capsize=4., markersize=7.)
ax.set(yticks=np.arange(len(methods)), yticklabels=[LABELS[method] for method in methods],
       xlabel=r'$\overline{\mathcal E}(D)$ $((\mathrm{mm}/\mathrm{s})^2)$')
ax.tick_params(axis='y', labelsize=LEGEND_FONT_SIZE)
ax.invert_yaxis()
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x', color='#BFC4C8', alpha=.30, linewidth=.45)
'''

SPHERES=r'''
# The same native sensor-coordinate sphere mesh and density scale are used twice.
nodes = np.asarray(snapshot['nodes'])
log_densities = [snapshot['log_density_full'], snapshot['log_density_isotropic']]
mode = np.asarray(snapshot['mode'], dtype=float)
true_normal = np.asarray(snapshot['true_normal'], dtype=float)
ELEVATION, AZIMUTH = 25., -50.
el, az = np.deg2rad([ELEVATION, AZIMUTH])
view_direction = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
# Apply one common antipodal sign to both markers; no coordinate rotation is used.
display_sign = 1. if mode @ view_direction >= 0. else -1.
display_mode, display_truth = display_sign*mode, display_sign*true_normal
cmap = colors.LinearSegmentedColormap.from_list('blue_purple_pink_yellow', [
    (0.,'#263a8b'), (.28,'#55209a'), (.53,'#b33695'),
    (.72,'#ed78a4'), (.88,'#ffc85a'), (1.,'#f4f52b')], N=256)
norm = colors.LogNorm(vmin=snapshot['density_limits'][0],
                      vmax=snapshot['density_limits'][1], clip=True)
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout='constrained')
axes = [fig.add_subplot(1,2,index+1,projection='3d') for index in range(2)]
for ax, log_density in zip(axes, log_densities):
    ax.computed_zorder = False
    surface = ax.plot_surface(nodes[...,0], nodes[...,1], nodes[...,2],
        facecolors=cmap(norm(np.exp(log_density))), shade=False, linewidth=0,
        antialiased=False, rstride=1, cstride=1, zorder=1)
    surface.set_rasterized(True)
    phi = np.linspace(0., 2*np.pi, 241)
    for plane in [(0,1),(0,2),(1,2)]:
        ring = np.zeros((len(phi),3))
        ring[:,plane[0]], ring[:,plane[1]] = 1.006*np.cos(phi), 1.006*np.sin(phi)
        visible = np.where((ring @ view_direction >= 0.)[:,None], ring, np.nan)
        ax.plot(*visible.T, color='#B7C0CB', linewidth=.5, alpha=.55, zorder=3)
    ax.scatter(*display_mode, color='#D55E00', marker='o', s=55,
               depthshade=False, zorder=5)
    ax.scatter(*display_truth, color='#111111', marker='*', s=150,
               depthshade=False, zorder=6)
    ax.set_proj_type('ortho')
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_box_aspect((1.,1.,1.), zoom=.84)
    ax.set(xlim=(-1.12,1.12), ylim=(-1.12,1.12), zlim=(-1.12,1.12),
           xticks=(-1.,0.,1.), yticks=(-1.,0.,1.), zticks=(-1.,0.,1.))
    ax.set_xlabel(r'$n_1$', labelpad=8.)
    ax.set_ylabel(r'$n_2$', labelpad=8.)
    ax.set_zlabel(r'$n_3$', labelpad=8.)
    ax.tick_params(labelsize=TICK_FONT_SIZE, pad=3, length=1)
    ax.grid(False)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.set_facecolor((1.,1.,1.,0.))
        axis.pane.set_edgecolor((.65,.68,.72,.35))
        axis.line.set_color('#70777D')
        axis.line.set_linewidth(.6)
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), ax=axes,
             label=r'$q_t([n])$', shrink=.75, pad=.03)
fig.legend(handles=[
    Line2D([],[],marker='o',linestyle='none',color='#D55E00',markersize=7,label=r'$\widehat n_t$'),
    Line2D([],[],marker='*',linestyle='none',color='#111111',markersize=11,label=r'$n_t$')],
    loc='outside lower center', ncol=2, frameon=False, fontsize=LEGEND_FONT_SIZE)
'''

PAIRED_TABLES=r'''# These saved paired summaries use the original simulation identifiers.
comparators = ['mode','iso','regularized','great_window','great_ew']
for label, frame in [
    ('Stationary concentrated inputs at $t=16$', data['paired_checkpoint_summary']),
    ('Moving sphere: all 48 action times', data['paired_episode_average_summary'])]:
    if 'Stationary' in label:
        frame = frame[frame.condition.eq('static_concentrated') & frame.t.eq(16)]
    else:
        frame = frame[frame.condition.eq('sphere')]
    frame = frame[frame.reference.isin(comparators)]
    rows = ['| Reference | Mean difference [95% CI] | Simulations |', '| --- | --- | --- |']
    for row in frame.itertuples():
        rows.append('| '+LABELS[row.reference]+' | '+f'{row.gain:.4g} [{row.lower:.4g}, {row.upper:.4g}]'
                    +' | '+str(row.simulations)+' |')
    display(Markdown('**'+label+'**\n\n'+'\n'.join(rows)))
'''

SNAPSHOT_TABLE=r'''full = np.asarray(snapshot['curvature_eigenvalues'])
isotropic = np.asarray(snapshot['isotropic_curvature_eigenvalues'])
mode = np.asarray(snapshot['mode'])
M = np.asarray(snapshot['normal_mean'])
M_iso = np.asarray(snapshot['normal_mean_isotropic'])
full_radius = 1.-float(mode @ M @ mode)
iso_radius = 1.-float(mode @ M_iso @ mode)
rows = [
    ['First tangent curvature', full[0], isotropic[0]],
    ['Second tangent curvature', full[1], isotropic[1]],
    ['Mean squared chordal distance from the mode', full_radius, iso_radius],
]
display(pd.DataFrame(rows, columns=['Quantity','Full posterior','Isotropic comparison']))
print('Absolute mean squared chordal-distance matching error:', abs(full_radius-iso_radius))
print('Full-posterior tangent curvature ratio:', float(full.max()/full.min()))
print('Antipodal angle between the mode and generating normal (degrees):',
      snapshot['checks']['antipodal_mode_true_angle_deg'])
display(Markdown('The tangent curvatures are '+f'{full[0]:.2f} and {full[1]:.2f}'+
    ', with ratio '+f'{full.max()/full.min():.3f}'+
    '. This moving posterior has modest anisotropy. The common density scale and these numerical values describe the actual saved checkpoint.'))
'''


def build(root):
    import nbformat
    from lib.contact_prediction_diagnostics import load_contact_prediction_diagnostics
    from lib.contact_moving_snapshot import load_moving_sphere_snapshot
    root=Path(root);data=load_contact_prediction_diagnostics(root);snapshot=load_moving_sphere_snapshot(root,t=96)
    inputs={**data['input_hashes'],**snapshot['input_hashes']}
    data_manifest=OUTPUT/'data/manifest.json'
    if not (root/data_manifest).is_file():
        raise RuntimeError('Export verified contact diagnostics before building the notebook')
    inputs[str(data_manifest)]=sha(root/data_manifest)
    sources={**data['source_hashes'],**snapshot['source_hashes']}
    cells=[]
    def markdown(value,identifier):
        cells.append(nbformat.v4.new_markdown_cell(dedent(value).strip(),id=identifier))
    def code(value,identifier,metadata=None):
        cells.append(nbformat.v4.new_code_cell(value,id=identifier,metadata=metadata or {}))
    markdown(r'''# Contact prediction in stationary and moving environments

This notebook follows the existing contact experiments from their physical interpretation to prediction error, reference methods and one moving posterior. It reads the original saved measurements and posterior parameters. Model settings, selected estimator parameters and simulation identifiers are unchanged.

The stationary concentrated-input history below uses the controlled `static_concentrated` fixture: both the physical plane and the sensor frame are fixed. It is distinct from the saved `plane` transport experiment, where the plane is fixed in world coordinates and the sensor rotates. In that experiment, the measured $G_t$ describes only the sensor-coordinate change.

Let $R_t$ map sensor coordinates to world coordinates. Then
$P_t^{\mathrm{sensor}}=R_t^\top P_t^{\mathrm{world}}R_t$ and $G_t=R_t^\top R_{t-1}$.

For the sphere, the physical tangent plane changes as the contact point moves, and the sensor frame also rotates. The measured $G_t$ still accounts only for sensor-frame rotation. It does not supply the changing physical tangent plane to the tracker.

The application is the maintained bilateral contact model already used in the original notebook. These diagnostics do not introduce another physical plane model, friction model or unilateral contact rule.''','contact-diagnostics-introduction')
    code(SETUP,'contact-diagnostics-setup',{'editable_plotting_role':'setup'})
    markdown(r'''## Prediction error

The saved clean-velocity error is
\[
\mathcal E_t(D_t)=\frac13\sum_{j=1}^{3}\|D_t v_t^{(j)}-P_{\mathcal X_t}v_t^{(j)}\|^2,
\qquad v_t^{(j)}=v_0 R_t^\top e_j,\quad v_0=0.02\,\mathrm{m/s}.
\]
The squared norm sums the velocity coordinates; there is no second division by three. Multiplication by $10^6$ converts the saved SI errors to $(\mathrm{mm}/\mathrm{s})^2$.

The posterior-mean response uses $D_t=\overline P_t^+$, the mode uses $P_{\widehat{\mathcal X}_t}$, and the isotropic comparison uses $\overline P_t^{\mathrm{iso}}$. Each point is a Monte Carlo mean over the same 256 simulations. Blue shading is a pointwise Student 95% confidence interval for that mean. It is not posterior uncertainty about the error of one simulation.''','prediction-error-definition')
    markdown(r'''### Stationary plane with concentrated inputs

Both the plane and sensor frame remain fixed in this controlled fixture. Concentrated applied inputs reveal some normal rotations more clearly than others. Only the saved checkpoints $t=1,4,16,32$ are shown. Differences among these actions test the value of the posterior mean for predicting clean velocities under this observation model.''','stationary-prediction-note')
    code(plotting_source(FIGURES[0][0],FIGURES[0][1],HISTORY.replace('CONDITION',repr('static_concentrated'))),
         'prediction-error-stationary',{'editable_plotting_role':'plot','figure_id':'prediction_error_stationary','source_page':1})
    markdown(r'''### Contact moving over a sphere

Here the physical tangent plane moves as well as changing coordinates with the sensor. The history uses the actual saved checkpoints $t=4,16,32,64,128,192$. Lines connect those saved points; they do not reconstruct errors at unrecorded checkpoints. The test reveals tracking performance under the already selected estimator settings, including any effects of physical motion not represented by measured frame transport. The dashed blue control uses the saved posterior-mean predictions with $G_t=I_3$, omitting measured sensor-frame transport while retaining its original settings.''','sphere-prediction-note')
    code(plotting_source(FIGURES[1][0],FIGURES[1][1],SPHERE_HISTORY.replace('CONDITION',repr('sphere'))),
         'prediction-error-sphere',{'editable_plotting_role':'plot','figure_id':'prediction_error_sphere','source_page':1})
    markdown(r'''## Reference algorithms and paired comparisons

The next plot retains the regularized mode and both GREAT references. Its endpoint is the saved time average over all 48 action times $t=4,8,\ldots,192$ in each sphere simulation,
$\overline{\mathcal E}(D)=48^{-1}\sum_{j=1}^{48}\mathcal E_{4j}(D_{4j})$.
It is not the average of the six displayed checkpoints. The horizontal intervals are pointwise Student 95% intervals for Monte Carlo means. Reference algorithms retain their original selected parameters. The regularized-mode predictor is
$\widetilde M_t=\rho P_{\widehat{\mathcal X}_t}+(1-\rho)\tfrac23I_3$, with the original validation-selected $\rho$. It is generally not a projector.

The mean, mode and regularized mode use the measured velocity, known applied input and measured sensor-frame changes. The isotropic mean uses the same posterior parameter as the full mean. GREAT uses measured velocities and frame changes, without the known applied input. Its comparison therefore changes both the estimator and the available information.''','reference-comparisons-note')
    code(plotting_source(FIGURES[2][0],FIGURES[2][1],REFERENCES),
         'prediction-error-references',{'editable_plotting_role':'plot','figure_id':'prediction_error_references','source_page':1})
    markdown(r'''The tables report reference error minus posterior-mean error using the same simulation in each pair, in $(\mathrm{mm}/\mathrm{s})^2$. Positive differences favor the posterior mean. A confidence interval containing zero does not resolve an advantage for that reference and condition. Paired comparisons provide different evidence from overlap of separate mean intervals.''','paired-comparisons-note')
    code(STYLE+'\n'+PAIRED_TABLES,'contact-paired-tables')
    markdown(r'''## The moving posterior at $t=96$

This checkpoint was selected before inspecting its density. It uses the corrected posterior from one saved sphere simulation, in native sensor coordinates. If the plane parameter is $A_t^+$, its unoriented normal has Bingham parameter $B_t=-A_t^+$. The normal $n$ and its antipode $-n$ describe the same plane.

The left sphere shows this full normal posterior. The right sphere is its matched isotropic Bingham comparison, with the same modal normal and mean squared chordal distance from that mode. Both use the same absolute density scale, logarithmic color normalization, sphere mesh and camera. The circle denotes the modal normal $\widehat n_t$ and the star the generating normal $n_t$; one common antipodal sign is used for both markers. The generating normal is used only for evaluation.

This moving checkpoint need not reproduce the strong anisotropy of the controlled concentrated-input example. The actual tangent curvatures and matching error below quantify its shape without changing the experiment.''','moving-posterior-note')
    code(plotting_source(FIGURES[3][0],FIGURES[3][1],SPHERES),
         'moving-sphere-posterior-spheres',{'editable_plotting_role':'plot','figure_id':'moving_sphere_posterior_spheres','source_page':1})
    code(SNAPSHOT_TABLE,'moving-sphere-curvature-table')
    averages=data['episode_average_summary']
    averages=averages[averages.condition.eq('sphere')].set_index('method')
    paired=data['paired_episode_average_summary']
    paired=paired[paired.condition.eq('sphere')].set_index('reference')
    results=(r'Across all 48 sphere action times, the mean errors are '
        +f"{averages.loc['mean','mean']:.4f}, {averages.loc['mode','mean']:.4f} and {averages.loc['iso','mean']:.4f}"
        +r' $(\mathrm{mm}/\mathrm{s})^2$ for the posterior mean, mode and isotropic mean. '
        +'The paired mode-minus-mean difference is '
        +f"{paired.loc['mode','gain']:.4f} [{paired.loc['mode','lower']:.4f}, {paired.loc['mode','upper']:.4f}]"
        +', and the isotropic-minus-mean difference is '
        +f"{paired.loc['iso','gain']:.4f} [{paired.loc['iso','lower']:.4f}, {paired.loc['iso','upper']:.4f}]"
        +r' in the same units (pointwise 95% intervals). '
        +'The positive paired difference against the matched isotropic mean supports the value of directional posterior moments for this saved condition. It does not imply the same ordering at every time.'
        +'\n\nAcross the same 48 action times, the paired identity-transport-minus-measured-transport posterior-mean error is '
        +f"{paired.loc['mean_identity','gain']:.4f} [{paired.loc['mean_identity','lower']:.4f}, {paired.loc['mean_identity','upper']:.4f}]"
        +r' $(\mathrm{mm}/\mathrm{s})^2$ (pointwise 95% interval across 256 simulations). '
        +'This uses the saved whole-run episode-average records, rather than an average of the six plotted checkpoints.')
    markdown('## What the prediction comparisons show\n\n'+results,'contact-prediction-results')
    markdown(r'''## Interpretation and links to the controlled example

A lower clean-velocity error supports a prediction advantage for the particular condition and selected settings. These comparisons do not establish a general coverage guarantee under physical drift. The moving sphere and stationary concentrated-input experiments answer different questions; the same ranking need not hold at every time or against every reference.

The two posterior laws can also have different conditional response variances,
$\operatorname{tr}\operatorname{Cov}_{q_t}(P_{\mathcal X_t}v)=v^\top[\overline P_t-(\overline P_t)^2]v$. Each such value belongs to its own law and its own posterior mean. It is not the empirical prediction error against a common generating trajectory. This notebook keeps those interpretations separate.

The existing [controlled normal-rotation calculation](../results/main/contact/information_rotation.pdf) and [concentrated-input posterior spheres](../results/main/contact/posterior_anisotropy_spheres.pdf) remain unchanged. There, $p_Z$ denotes the velocity observation law at the plane specified by graph coordinate $Z$; its divergence compares observation laws, rather than posterior credible regions. That controlled example establishes how directional information can differ. The moving snapshot here reports the shape actually obtained during the saved physical trajectory.

The figure code is fully visible above. `SAVE_PDF` controls both local PDF/PNG exports and copies into the manuscript's main or supplementary folder.''','contact-diagnostics-interpretation')
    nb=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},
        'editable_plotting':{'schema':1,'mode':'inline_matplotlib','figure_count':4,'rendered_figure_count':4,
            'output_directory':str(OUTPUT/'figures'),'input_hashes':inputs,
            'loader_sources':{'lib/contact_prediction_diagnostics.py':sha(root/'lib/contact_prediction_diagnostics.py'),
                              'lib/contact_moving_snapshot.py':sha(root/'lib/contact_moving_snapshot.py')},
            'controls':'Full visible Matplotlib source in each figure cell; DPI=300 and FIGSIZE=(15,5).'},
        'contact_prediction_diagnostics':{'schema':1,'producer':'scripts/make_contact_prediction_diagnostics_notebook.py',
            'producer_sha256':sha(__file__),'input_hashes':inputs,'source_hashes':sources,
            'config':data['config'],'snapshot_time':96,'numerical_scope':'Saved results only; no simulation, inference or parameter selection.'}})
    path=root/NOTEBOOK;path.parent.mkdir(exist_ok=True)
    nbformat.write(nb,path)
    return nb


def execute(root,notebook,timeout):
    from nbclient import NotebookClient
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
    os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
    NotebookClient(notebook,timeout=timeout,kernel_name='python3',
        resources={'metadata':{'path':str(root)}}).execute()
    import nbformat
    nbformat.write(notebook,root/NOTEBOOK)


def manifest(root,notebook,protection):
    info=notebook.metadata['contact_prediction_diagnostics'];records={};files={}
    for stem,section in FIGURES:
        local_pdf=OUTPUT/'figures'/(stem+'.pdf');local_png=local_pdf.with_suffix('.png')
        draft_pdf=Path('results')/section/'contact'/local_pdf.name;draft_png=draft_pdf.with_suffix('.png')
        for relative in (local_pdf,local_png,draft_pdf,draft_png):
            if not (root/relative).is_file():raise RuntimeError('Missing figure: '+str(relative))
            files[str(relative)]=sha(root/relative)
        if files[str(local_pdf)]!=files[str(draft_pdf)] or files[str(local_png)]!=files[str(draft_png)]:
            raise RuntimeError('Local and manuscript figures differ: '+stem)
        records[stem+'.pdf']=dict(section=section,figure_size_in=[15,5],dpi=300,text_usetex=True,
            local_pdf=str(local_pdf),local_png=str(local_png),draft_pdf=str(draft_pdf),draft_png=str(draft_png),
            hashes={str(p):files[str(p)] for p in (local_pdf,local_png,draft_pdf,draft_png)})
    for folder in ('data','tables'):
        for path in (root/OUTPUT/folder).glob('*'):
            if path.is_file():files[str(path.relative_to(root))]=sha(path)
    record=dict(schema=1,status='completed',notebook=str(NOTEBOOK),notebook_sha256=sha(root/NOTEBOOK),
        producer='scripts/make_contact_prediction_diagnostics_notebook.py',producer_sha256=sha(__file__),
        config=info['config'],input_hashes=info['input_hashes'],source_hashes=info['source_hashes'],
        snapshot_time=96,figures=records,files=files,preservation=protection,
        scope='Saved contact predictions and native moving posterior; no new experiments or numerical fitting.')
    (root/OUTPUT/'figure_manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args();root=args.root.resolve()
    folder=root/OUTPUT;folder.mkdir(parents=True,exist_ok=True)
    baseline=protected_files(root)
    (folder/'preservation_baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    from lib.contact_moving_snapshot import export_moving_sphere_snapshot
    from scripts.export_contact_prediction_diagnostics import export
    export_moving_sphere_snapshot(root,t=96)
    export(root)
    nb=build(root)
    if args.execute:
        execute(root,nb,args.timeout)
        protection=verify_protected(root,baseline)
        record=manifest(root,nb,protection)
        print(json.dumps({'status':'completed','notebook':str(NOTEBOOK),'figures':list(record['figures']),
                          'preservation':protection},indent=2),flush=True)
    else:
        print(json.dumps({'status':'prepared','notebook':str(NOTEBOOK),'preservation':verify_protected(root,baseline)},indent=2),flush=True)


if __name__=='__main__':
    main()
