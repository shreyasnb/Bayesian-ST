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
TRANSPORT_OUTPUT=Path('results/contact_transport_diagnostics')
TRANSPORT_FIGURES=[('transport_paired_rise','supplementary'),
                   ('transport_projector_change','supplementary'),
                   ('transport_excitation','supplementary'),
                   ('transport_curvature','supplementary')]

def figure_output(stem):
    return (TRANSPORT_OUTPUT if stem.startswith('transport_') else OUTPUT)/'figures'/(stem+'.pdf')


def load_transport_evidence(root):
    """Read the separately sealed replay; never start inference from a cell."""
    from lib.contact_transport_diagnostics import load_contact_transport_diagnostics
    return load_contact_transport_diagnostics(Path(root))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_caption_metadata(root):
    path=Path(root)/'results/solid_trajectory_update/caption_inventory.json'
    if not path.is_file():return {}
    inventory=json.loads(path.read_text())['captions_by_path']
    captions={}
    for stem,section in FIGURES[:3]+TRANSPORT_FIGURES:
        name='results/'+section+'/contact/'+stem+'.pdf'
        captions[stem+'.pdf']=inventory[name]['latex_caption']
    return {'figure_captions':captions,'caption_inventory':'results/solid_trajectory_update/caption_inventory.json',
            'caption_inventory_sha256':sha(path)}


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
           for stem,section in FIGURES+TRANSPORT_FIGURES for suffix in ('.pdf','.png')}
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
    style=STYLE
    if stem in {'prediction_error_stationary','prediction_error_sphere'} or stem.startswith('transport_'):
        import ast
        lines=style.splitlines(keepends=True)
        removed=set()
        for node in ast.parse(style).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MARKERS' for t in node.targets):
                removed.update(range(node.lineno-1,node.end_lineno))
        style=''.join(line for i,line in enumerate(lines) if i not in removed)
    path_source=("TRANSPORT_OUTPUT = ROOT / 'results/contact_transport_diagnostics/figures'\n"
                 +f"PDF_PATH = TRANSPORT_OUTPUT / '{stem}.pdf'\n") if stem.startswith('transport_') else f"PDF_PATH = EDITABLE_OUTPUT / '{stem}.pdf'\n"
    return (CONTROLS+path_source+f"DRAFT_SECTION = '{section}'\n"
        +"DRAFT_PATH = ROOT / 'results' / DRAFT_SECTION / 'contact' / PDF_PATH.name\n\n"
        +style+'\n'+dedent(body).strip()+'\n\n'+SAVE)


HISTORY=r'''
condition = CONDITION
record = data['conditions'][condition]
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for curve in record['curves']:
    method = curve['method']
    points = np.asarray(curve['points'], dtype=float)
    ax.plot(points[:,0], points[:,1], color=COLORS[method], linestyle='-', label=LABELS[method])
    if method == 'mean':
        ax.fill_between(points[:,0], points[:,2], points[:,3],
                        color=COLORS[method], alpha=.14, linewidth=0)
ax.set(xlabel=r'$t$', ylabel=r'$\mathcal E_t(D_t)$ $((\mathrm{mm}/\mathrm{s})^2)$',
       xlim=(1, int(record['T'])))
BORDER_WIDTH = .9
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(BORDER_WIDTH)
ax.tick_params(axis='both', which='both', direction='in',
               top=True, right=True, bottom=True, left=True,
               labeltop=False, labelright=False, labelbottom=True, labelleft=True,
               width=BORDER_WIDTH)
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_label_position('left')
ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
           ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE)
'''

SPHERE_HISTORY = HISTORY.replace(
    "record = data['conditions'][condition]",
    "from lib.contact_transport_diagnostics import load_contact_transport_diagnostics\n"
    "transport = load_contact_transport_diagnostics(ROOT)\n"
    "record = {'T': transport['config']['T'], 'curves': transport['sphere_curves']}"
).replace(
    "for curve in record['curves']:",
    "LABELS['mean_identity'] = r'Posterior mean, $G_t=I_3$'\n"
    "COLORS['mean_identity'] = '#484848'\n"
    "for curve in record['curves']:")

TRANSPORT_AXES = r'''
BORDER_WIDTH = .9
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(BORDER_WIDTH)
ax.tick_params(axis='both', which='both', direction='in',
               top=True, right=True, bottom=True, left=True,
               labeltop=False, labelright=False, labelbottom=True, labelleft=True,
               width=BORDER_WIDTH)
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_label_position('left')
ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
           ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE)
'''

TRANSPORT_READ = r'''from lib.contact_transport_diagnostics import load_contact_transport_diagnostics
transport = load_contact_transport_diagnostics(ROOT)
'''

TRANSPORT_RISE = TRANSPORT_READ + r'''
# Each difference uses the same simulation at the two recorded action times.
frame = transport['late_rise_trials']
LABELS['mean_identity'] = r'Posterior mean, $G_t=I_3$'
COLORS['mean_identity'] = '#484848'
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for method in ['mean', 'mean_identity']:
    values = np.sort(frame.loc[frame.method.eq(method), 'difference_mm2_s2'].to_numpy())
    probability = np.arange(1, len(values)+1, dtype=float)/len(values)
    ax.step(values, probability, where='post', color=COLORS[method],
            linestyle='-', label=LABELS[method])
ax.axvline(0., color='#777777', linestyle='--', linewidth=1.)
ax.set(xlabel=r'$\Delta\mathcal E$ $((\mathrm{mm}/\mathrm{s})^2)$',
       ylabel=r'$\widehat F_{\Delta\mathcal E}(x)$', ylim=(0., 1.))
''' + TRANSPORT_AXES

TRANSPORT_PROJECTOR = TRANSPORT_READ + r'''
# These are Monte Carlo means of the generating geometric increments.
frame = transport['projector_changes']
metric_styles = {
    'sensor_increment_fro': (r'$\|P_t-P_{t-1}\|_{\mathrm F}$', '#0072B2'),
    'transported_increment_fro':
        (r'$\|P_t-G_tP_{t-1}G_t^\top\|_{\mathrm F}$', '#484848'),
}
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for metric, (label, color_) in metric_styles.items():
    rows = frame[frame.metric.eq(metric)].sort_values('t')
    ax.plot(rows.t, rows['mean'], color=color_, linestyle='-', label=label)
ax.set(xlabel=r'$t$', ylabel='Frobenius norm', xlim=(1, transport['config']['T']))
''' + TRANSPORT_AXES

TRANSPORT_EXCITATION = TRANSPORT_READ + r'''
# H_t is the normalized, exponentially weighted Gram matrix of known inputs.
frame = transport['excitation']
metric_styles = {
    'input_gram_lambda_1': (r'$\lambda_1(H_t)$', '#0072B2'),
    'input_gram_lambda_2': (r'$\lambda_2(H_t)$', '#D55E00'),
    'input_gram_lambda_3': (r'$\lambda_3(H_t)$', '#CC79A7'),
}
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for metric, (label, color_) in metric_styles.items():
    rows = frame[frame.metric.eq(metric)].sort_values('t')
    ax.plot(rows.t, rows['mean'], color=color_, linestyle='-', label=label)
    if not rows.deterministic.all():
        ax.fill_between(rows.t, rows.lower, rows.upper, color=color_, alpha=.12, linewidth=0)
ax.set(xlabel=r'$t$', ylabel=r'$\lambda_j(H_t)$', xlim=(1, transport['config']['T']))
ax.set_yscale('log')
''' + TRANSPORT_AXES

TRANSPORT_CURVATURE = TRANSPORT_READ + r'''
# Curvatures belong to the corresponding full posterior, not to the prediction errors.
frame = transport['curvature']
metric_styles = {
    'curvature_min': (r'$c_{\min,t}$', '#0072B2'),
    'curvature_max': (r'$c_{\max,t}$', '#D55E00'),
    'curvature_identity_min': (r'$c_{\min,t}^{G=I}$', '#484848'),
    'curvature_identity_max': (r'$c_{\max,t}^{G=I}$', '#999999'),
}
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for metric, (label, color_) in metric_styles.items():
    rows = frame[frame.metric.eq(metric)].sort_values('t')
    ax.plot(rows.t, rows['mean'], color=color_, linestyle='-', label=label)
    ax.fill_between(rows.t, rows.lower, rows.upper, color=color_, alpha=.12, linewidth=0)
ax.set(xlabel=r'$t$', ylabel='Tangent curvature', xlim=(1, transport['config']['T']))
''' + TRANSPORT_AXES


def transport_cells(evidence=None):
    """Supplementary source cells, appended after all existing notebook cells."""
    import nbformat
    blocks=[
        ('transport_paired_rise',r'''## Additional diagnostics for measured transport

These plots use a separately verified replay of the same 256 sphere simulations, with all original settings and action times unchanged. No event is introduced at $t=128$.

For each simulation, define $\Delta\mathcal E=\mathcal E_{192}(D_{192})-\mathcal E_{128}(D_{128})$. The empirical cumulative distribution below retains both positive and negative changes. It compares the posterior mean with measured transport to the same estimator with $G_t=I_3$. A time increase for the identity control alone does not identify a universal cause of error under physical drift.''',TRANSPORT_RISE),
        ('transport_projector_change',r'''### Physical motion and coordinate changes

Here $P_t=R_t^\top P_t^{\mathrm{world}}R_t$ is the generating sensor-frame projector, used only for evaluation. The first curve includes the change of sensor coordinates. The second transports the preceding projector with the measured $G_t$ before comparing it to the current one, isolating the physical projector increment. Curves are means over the same 256 simulations, whose initial normals and physical paths vary; their confidence intervals remain in the exported table and are omitted here. The transported increment agrees with the world-coordinate increment in each simulation, which is retained as a numerical check.''',TRANSPORT_PROJECTOR),
        ('transport_excitation',r'''### Information supplied by the known inputs

Let $S_t=\beta G_tS_{t-1}G_t^\top+v_tv_t^\top$ and $s_t=\beta s_{t-1}+1$, where $v_t$ is the known applied input. The dimensionless matrix $H_t=S_t/(\sigma^2s_t)$ uses the original sphere setting $\beta=0.9$; it introduces no additional moving window. Its eigenvalues are ordered increasingly. They describe directional input excitation, not prediction error or posterior coverage. The applied inputs and hence these eigenvalues are deterministic for the shared trajectory; their Monte Carlo intervals collapse.''',TRANSPORT_EXCITATION),
        ('transport_curvature',r'''### Posterior concentration during motion

For ascending eigenvalues of the plane posterior parameter $A_t^+$, define $c_{\min,t}=2[\lambda_2(A_t^+)-\lambda_1(A_t^+)]$ and $c_{\max,t}=2[\lambda_3(A_t^+)-\lambda_1(A_t^+)]$. These are the two angular tangent curvatures of the normal posterior on the unit sphere. Superscript $G=I$ denotes the identity-transport control. Curves show means over the same 256 simulations, and shading gives pointwise Student 95% intervals. Concentration of a model posterior does not by itself establish accuracy under unmodelled physical motion.''',TRANSPORT_CURVATURE),
    ]
    cells=[]
    for stem,note,body in blocks:
        cells.append(nbformat.v4.new_markdown_cell(dedent(note).strip(),id=stem.replace('_','-')+'-note'))
        cells.append(nbformat.v4.new_code_cell(plotting_source(stem,'supplementary',body),
            id=stem.replace('_','-'),metadata={'editable_plotting_role':'plot','figure_id':stem,'source_page':1,
                'source_catalogue_page':False,'example':'contact','section':'supplementary'}))
    if evidence is not None:
        rise=evidence['paired_rise'].set_index('method')
        measured=rise.loc['mean'];identity=rise.loc['mean_identity']
        text=(r'### What changes after $t=128$'
            +'\n\nThe measured-transport posterior mean has paired change '
            +f"{measured['mean']:.3f} [{measured['lower']:.3f}, {measured['upper']:.3f}]"
            +r' $(\mathrm{mm}/\mathrm{s})^2$ from $t=128$ to $t=192$; its mean error decreases. '
            +'The identity-transport control has change '
            +f"{identity['mean']:.3f} [{identity['lower']:.3f}, {identity['upper']:.3f}]"
            +r' in the same units, with a positive change in '
            +f"{100*identity['fraction_positive']:.1f}% of the 256 simulations. "
            +'All simulations, including the tails, are retained. The CDF makes the distinction between a change of the Monte Carlo mean and a change shared by individual simulations visible.'
            +'\n\nThere is no programmed event at $t=128$. The generating sphere motion, measured sensor rotation and applied inputs continue under their original definitions. These diagnostics distinguish physical motion, coordinate changes, input excitation and posterior concentration; they do not establish a general causal explanation or a coverage guarantee under unmodelled physical drift.')
        cells.append(nbformat.v4.new_markdown_cell(text,id='transport-diagnostics-interpretation'))
    return cells


def dense_sphere_source(source):
    """Keep current user controls while changing only the verified data source."""
    from scripts.make_beamforming_comparison_audit_notebooks import solid_trajectory_source
    source=solid_trajectory_source(source)
    if "'curves': transport['sphere_curves']" in source:
        return source.replace("COLORS['mean_identity'] = COLORS['mean']", "COLORS['mean_identity'] = '#484848'")
    old="record = data['conditions'][condition]"
    if source.count(old)!=1:
        raise RuntimeError('Expected the current sphere-history data declaration')
    source=source.replace(old,
        "from lib.contact_transport_diagnostics import load_contact_transport_diagnostics\n"
        "transport = load_contact_transport_diagnostics(ROOT)\n"
        "record = {'T': transport['config']['T'], 'curves': transport['sphere_curves']}")
    start=source.index('# Keep the measured-frame curves and add the saved identity-transport control.')
    end=source.index('for curve in curves:',start)+len('for curve in curves:')
    source=source[:start]+(
        "# All 48 original action times; no interpolation or new simulation settings.\n"
        "LABELS['mean_identity'] = r'Posterior mean, $G_t=I_3$'\n"
        "COLORS['mean_identity'] = '#484848'\n"
        "for curve in record['curves']:")+source[end:]
    return source


TRANSPORT_PROOF = r'''
def _transport_dense_check(figure):
    expected={curve['method']:np.asarray(curve['points'],dtype=float)
              for curve in transport['sphere_curves']}
    assert set(expected)=={'mean','mode','iso','mean_identity'}
    assert all(np.array_equal(points[:,0],np.arange(4,193,4)) for points in expected.values())
    axis=figure.axes[0]
    for method,points in expected.items():
        matches=[line for line in axis.lines if line.get_label()==LABELS[method]]
        assert len(matches)==1
        line=matches[0]
        assert np.array_equal(np.asarray(line.get_xdata(),dtype=float),points[:,0])
        assert np.array_equal(np.asarray(line.get_ydata(),dtype=float),points[:,1])
        assert line.get_marker() in ('None',None,'','none',' ') and line.get_linestyle()=='-'
        assert line.get_markevery() is None and str(line.get_color())==COLORS[method]
    old=data['conditions']['sphere']['all_curves']
    for curve in old:
        method=curve['method']
        if method not in expected:continue
        points=np.asarray(curve['points'],dtype=float)
        recovered=expected[method]
        selected=np.array([np.flatnonzero(recovered[:,0]==t)[0] for t in points[:,0]])
        assert np.allclose(recovered[selected],points,rtol=2e-12,atol=5e-10),method
    old_state=_style_states['sphere_previous'];new_state=_style_state(figure)
    assert old_state['figsize']==new_state['figsize'] and old_state['dpi']==new_state['dpi']
    assert old_state['legend_text']==new_state['legend_text']
    assert old_state['legend_fonts']==new_state['legend_fonts']
    for key in ('xlabel','ylabel','xlim','ylim','xscale','yscale','label_fonts'):
        if key=='ylim':continue # The recovered intermediate values can extend the visible range.
        assert old_state['axes'][0][key]==new_state['axes'][0][key],key
    for old_line,new_line in zip(old_state['axes'][0]['lines'],new_state['axes'][0]['lines']):
        for key in ('label','linewidth','alpha'):
            assert old_line[key]==new_line[key],(key,old_line['label'])
        if old_line['label']!=LABELS['mean_identity']:
            assert old_line['color']==new_line['color']
    old_bands=old_state['axes'][0]['bands'];new_bands=new_state['axes'][0]['bands']
    assert len(old_bands)==len(new_bands)==1
    for key in ('facecolors','alpha','linewidths'):
        assert old_bands[0][key]==new_bands[0][key],key
    # The band is Matplotlib's unmodified polygon of the verified Student limits.
    reference=plt.figure();reference_ax=reference.add_subplot()
    points=expected['mean']
    reference_band=reference_ax.fill_between(points[:,0],points[:,2],points[:,3],
                                             color=COLORS['mean'],alpha=.14,linewidth=0)
    assert [_style_array(p.vertices) for p in reference_band.get_paths()]==new_bands[0]['paths']
    plt.close(reference)
    _style_reports.append({'figure':'prediction_error_sphere','action_times':48,'simulations':256,
        'all_verified_means_exact':True,'all_original_checkpoint_means_and_CIs_unchanged':True,
        'mean_Student_CI_polygon_exact':True,'original_colors_except_gray_identity':True,
        'line_widths_labels_fonts_figsize_and_band_opacity_unchanged':True,
        'solid_markerless_trajectories':4,'event_at_128_added':False})

def _transport_diagnostic_check(figure,key):
    for axis in figure.axes:
        for line in axis.lines:
            if line.get_label().startswith('_'):continue # The zero reference is not a method trajectory.
            assert line.get_marker() in ('None',None,'','none',' ') and line.get_linestyle()=='-'
            assert line.get_markevery() is None
        assert all(spine.get_visible() for spine in axis.spines.values())
        assert all(abs(spine.get_linewidth()-.9)<1e-12 for spine in axis.spines.values())
    assert list(figure.get_size_inches())==[15.,5.] and figure.dpi==300
    _style_reports.append({'figure':key,'verified_saved_data_only':True,
        'solid_markerless_trajectories':True,'four_borders':True,'figure_size_in':[15,5],'dpi':300})
'''

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
BORDER_WIDTH = .9
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(BORDER_WIDTH)
ax.tick_params(axis='both', which='both', direction='in',
               top=True, right=True, bottom=True, left=True,
               labeltop=False, labelright=False, labelbottom=True, labelleft=True,
               width=BORDER_WIDTH)
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_label_position('left')
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
    transport=load_transport_evidence(root)
    transport_manifest=TRANSPORT_OUTPUT/'data/manifest.json'
    if not (root/transport_manifest).is_file():raise RuntimeError('Export the verified dense transport replay first')
    inputs={**data['input_hashes'],**snapshot['input_hashes'],**transport['input_hashes']}
    inputs[str(transport_manifest)]=sha(root/transport_manifest)
    data_manifest=OUTPUT/'data/manifest.json'
    if not (root/data_manifest).is_file():
        raise RuntimeError('Export verified contact diagnostics before building the notebook')
    inputs[str(data_manifest)]=sha(root/data_manifest)
    sources={**data['source_hashes'],**snapshot['source_hashes'],**transport.get('source_hashes',{})}
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

Here the physical tangent plane moves as well as changing coordinates with the sensor. The history uses all 48 original action times $t=4,8,\ldots,192$, recovered by a verified deterministic replay of the same 256 simulations. Lines join these recorded action errors without interpolation or an assumed event at $t=128$. The test reveals tracking performance under the already selected estimator settings, including any effects of physical motion not represented by measured frame transport. The gray control uses the saved posterior-mean predictions with $G_t=I_3$, omitting measured sensor-frame transport while retaining its original settings.''','sphere-prediction-note')
    code(plotting_source(FIGURES[1][0],FIGURES[1][1],SPHERE_HISTORY.replace('CONDITION',repr('sphere'))),
         'prediction-error-sphere',{'editable_plotting_role':'plot','figure_id':'prediction_error_sphere','source_page':1})
    markdown(r'''## Reference algorithms and paired comparisons

The next plot retains the regularized mode and both GREAT references. Its endpoint is the saved time average over all 48 action times $t=4,8,\ldots,192$ in each sphere simulation,
$\overline{\mathcal E}(D)=48^{-1}\sum_{j=1}^{48}\mathcal E_{4j}(D_{4j})$.
This average uses the same 48 action times shown in the sphere history. The horizontal intervals are pointwise Student 95% intervals for Monte Carlo means. Reference algorithms retain their original selected parameters. The regularized-mode predictor is
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
        +'This uses the original whole-run episode-average records, which agree with the recovered 48-action histories.')
    markdown('## What the prediction comparisons show\n\n'+results,'contact-prediction-results')
    markdown(r'''## Interpretation and links to the controlled example

A lower clean-velocity error supports a prediction advantage for the particular condition and selected settings. These comparisons do not establish a general coverage guarantee under physical drift. The moving sphere and stationary concentrated-input experiments answer different questions; the same ranking need not hold at every time or against every reference.

The two posterior laws can also have different conditional response variances,
$\operatorname{tr}\operatorname{Cov}_{q_t}(P_{\mathcal X_t}v)=v^\top[\overline P_t-(\overline P_t)^2]v$. Each such value belongs to its own law and its own posterior mean. It is not the empirical prediction error against a common generating trajectory. This notebook keeps those interpretations separate.

The existing [controlled normal-rotation calculation](../results/main/contact/information_rotation.pdf) and [concentrated-input posterior spheres](../results/main/contact/posterior_anisotropy_spheres.pdf) remain unchanged. There, $p_Z$ denotes the velocity observation law at the plane specified by graph coordinate $Z$; its divergence compares observation laws, rather than posterior credible regions. That controlled example establishes how directional information can differ. The moving snapshot here reports the shape actually obtained during the saved physical trajectory.

The figure code is fully visible above. `SAVE_PDF` controls both local PDF/PNG exports and copies into the manuscript's main or supplementary folder.''','contact-diagnostics-interpretation')
    cells.extend(transport_cells(transport))
    nb=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},
        'editable_plotting':{'schema':1,'mode':'inline_matplotlib','figure_count':8,'rendered_figure_count':8,
            'output_directory':str(OUTPUT/'figures'),'input_hashes':inputs,
            'loader_sources':{'lib/contact_prediction_diagnostics.py':sha(root/'lib/contact_prediction_diagnostics.py'),
                              'lib/contact_moving_snapshot.py':sha(root/'lib/contact_moving_snapshot.py'),
                              'lib/contact_transport_diagnostics.py':sha(root/'lib/contact_transport_diagnostics.py')},
            'controls':'Full visible Matplotlib source in each figure cell; DPI=300 and FIGSIZE=(15,5).'},
        'contact_prediction_diagnostics':{'schema':1,'producer':'scripts/make_contact_prediction_diagnostics_notebook.py',
            'producer_sha256':sha(__file__),'input_hashes':inputs,'source_hashes':sources,
            'config':data['config'],'snapshot_time':96,'numerical_scope':'Saved results only; no simulation, inference or parameter selection.',
            **contact_caption_metadata(root)},
        'contact_transport_diagnostics':{'schema':1,'manifest':str(transport_manifest),
            'manifest_sha256':sha(root/transport_manifest),'input_hashes':transport['input_hashes'],
            'source_hashes':transport.get('source_hashes',{}),'checks':transport['checks'],
            'action_times':list(range(4,193,4)),'simulations':256,
            'identity_color':'#484848','figures':[stem for stem,_ in TRANSPORT_FIGURES]}})
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
    present={cell.metadata.get('figure_id') for cell in notebook.cells}
    figure_specs=FIGURES+[item for item in TRANSPORT_FIGURES if item[0] in present]
    for stem,section in figure_specs:
        local_pdf=figure_output(stem);local_png=local_pdf.with_suffix('.png')
        draft_pdf=Path('results')/section/'contact'/local_pdf.name;draft_png=draft_pdf.with_suffix('.png')
        for relative in (local_pdf,local_png,draft_pdf,draft_png):
            if not (root/relative).is_file():raise RuntimeError('Missing figure: '+str(relative))
            files[str(relative)]=sha(root/relative)
        if files[str(local_pdf)]!=files[str(draft_pdf)] or files[str(local_png)]!=files[str(draft_png)]:
            raise RuntimeError('Local and manuscript figures differ: '+stem)
        records[stem+'.pdf']=dict(section=section,figure_size_in=[15,5],dpi=300,text_usetex=True,
            local_pdf=str(local_pdf),local_png=str(local_png),draft_pdf=str(draft_pdf),draft_png=str(draft_png),
            hashes={str(p):files[str(p)] for p in (local_pdf,local_png,draft_pdf,draft_png)})
        if stem+'.pdf' in info.get('figure_captions',{}):
            records[stem+'.pdf']['latex_caption']=info['figure_captions'][stem+'.pdf']
    for folder in ('data','tables'):
        for path in (root/OUTPUT/folder).glob('*'):
            if path.is_file():files[str(path.relative_to(root))]=sha(path)
    record=dict(schema=1,status='completed',notebook=str(NOTEBOOK),notebook_sha256=sha(root/NOTEBOOK),
        producer='scripts/make_contact_prediction_diagnostics_notebook.py',producer_sha256=sha(__file__),
        config=info['config'],input_hashes=info['input_hashes'],source_hashes=info['source_hashes'],
        snapshot_time=96,figures=records,files=files,preservation=protection,
        scope='Saved contact predictions and native moving posterior; no new experiments or numerical fitting.',
        **{key:info[key] for key in ('figure_captions','caption_inventory','caption_inventory_sha256') if key in info})
    (root/OUTPUT/'figure_manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


def restyle_contact_diagnostics(root,timeout=900):
    from scripts.make_beamforming_comparison_audit_notebooks import restyle_existing_notebook
    def refresh(root,notebook,protection):
        from scripts.check_manuscript_plot_exports import audit_pdf
        path=root/OUTPUT/'pdf_layout_checks.json'
        layout=json.loads(path.read_text())
        layout['pdfs']={name:audit_pdf(root/name,[[15,5]],
            primary_fonts={'LABEL_FONT_SIZE':25,'TICK_FONT_SIZE':23,'LEGEND_FONT_SIZE':22})
            for name in layout['pdfs']}
        if any(record['errors'] for record in layout['pdfs'].values()):
            raise RuntimeError('Restyled contact PDF layout failed')
        layout['status']='passed'
        layout['audit_source_sha256']=sha(root/'scripts/check_manuscript_plot_exports.py')
        path.write_text(json.dumps(layout,indent=2)+'\n')
        captions=contact_caption_metadata(root)
        notebook.metadata['contact_prediction_diagnostics'].update(captions)
        notebook_path=root/NOTEBOOK
        saved=json.loads(notebook_path.read_text())
        saved['metadata']['contact_prediction_diagnostics'].update(captions)
        temporary=notebook_path.with_suffix('.captions.tmp')
        temporary.write_text(json.dumps(saved,indent=1,ensure_ascii=False)+'\n');temporary.replace(notebook_path)
        return manifest(root,notebook,protection)
    return restyle_existing_notebook(root,NOTEBOOK,
        {'prediction_error_stationary','prediction_error_sphere'},
        'contact_prediction_diagnostics',refresh,timeout=timeout)


def update_contact_transport_notebook(root,timeout=900):
    """Update two histories and append diagnostics without rerunning old cells."""
    import fcntl
    import tempfile
    from copy import deepcopy
    import nbformat
    from nbclient import NotebookClient
    from scripts.make_beamforming_comparison_audit_notebooks import (
        solid_trajectory_source,_STYLE_PROOF_SETUP)
    from scripts.check_manuscript_plot_exports import audit_pdf
    root=Path(root).resolve();path=root/NOTEBOOK
    with open('/tmp/bayesian-st-notebook-render.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        original_bytes=path.read_bytes();before=json.loads(original_bytes);notebook=deepcopy(before)
        evidence=load_transport_evidence(root)
        manifest_path=TRANSPORT_OUTPUT/'data/manifest.json'
        if not (root/manifest_path).is_file():
            raise RuntimeError('Missing sealed transport replay manifest')
        numeric={**notebook['metadata']['editable_plotting']['input_hashes'],
                 **evidence['input_hashes'],str(manifest_path):sha(root/manifest_path)}
        for name,digest in list(numeric.items()):
            if isinstance(digest,dict):digest=digest['sha256'];numeric[name]=digest
            if sha(root/name)!=digest:raise RuntimeError('Numerical input changed: '+name)
            candidate=root/name
            if candidate.name=='manifest.json':
                for relative in json.loads(candidate.read_text()).get('artifacts',{}):
                    artifact=root/relative
                    if not artifact.is_file():artifact=candidate.parent/relative
                    if artifact.is_file():numeric[str(artifact.relative_to(root))]=sha(artifact)
        old_manifest=json.loads((root/OUTPUT/'figure_manifest.json').read_text())
        protected={}
        selected={'prediction_error_stationary','prediction_error_sphere'}
        for name,record in old_manifest['figures'].items():
            if Path(name).stem in selected:continue
            for key in ('local_pdf','local_png','draft_pdf','draft_png'):
                if key in record:protected[record[key]]=sha(root/record[key])
        descriptor,baseline=tempfile.mkstemp(prefix='contact-dense-transport-baseline-',suffix='.json',dir='/tmp')
        os.close(descriptor);Path(baseline).write_text(json.dumps(
            {'notebook':before,'numeric':numeric,'protected_figures':protected},indent=1)+'\n')
        def source(cell):return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
        indices={cell.get('metadata',{}).get('figure_id'):i for i,cell in enumerate(notebook['cells'])}
        stationary=indices['prediction_error_stationary'];sphere=indices['prediction_error_sphere']
        notebook['cells'][stationary]['source']=solid_trajectory_source(source(notebook['cells'][stationary])).splitlines(keepends=True)
        notebook['cells'][sphere]['source']=dense_sphere_source(source(notebook['cells'][sphere])).splitlines(keepends=True)
        changed_prose=[]
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='markdown':continue
            old=source(cell);new=old
            new=new.replace('The history uses the actual saved checkpoints $t=4,16,32,64,128,192$. Lines connect those saved points; they do not reconstruct errors at unrecorded checkpoints.',
                'The history uses all 48 original action times $t=4,8,\\ldots,192$, recovered by a verified deterministic replay of the same 256 simulations. Lines join these recorded action errors without interpolation or an assumed event at $t=128$.')
            new=new.replace('The dashed blue control uses','The gray control uses').replace('The blue control uses','The gray control uses')
            new=new.replace('It is not the average of the six displayed checkpoints.',
                'This average uses the same 48 action times shown in the sphere history.')
            new=new.replace('This uses the saved whole-run episode-average records, rather than an average of the six plotted checkpoints.',
                'This uses the original whole-run episode-average records, which agree with the recovered 48-action histories.')
            if new!=old:cell['source']=new.splitlines(keepends=True);changed_prose.append(i)
        appended=transport_cells(evidence)
        old_ids={cell.get('id') for cell in notebook['cells']}
        existing_append=[cell.get('id') in old_ids for cell in appended]
        if any(existing_append) and not all(existing_append):
            raise RuntimeError('Only part of the transport appendix is present')
        append_start=len(notebook['cells'])
        if not all(existing_append):notebook['cells'].extend(json.loads(json.dumps(appended)))
        setup=next(cell for cell in notebook['cells'] if cell.get('id')=='contact-diagnostics-setup')
        work=[deepcopy(setup),nbformat.v4.new_code_cell(_STYLE_PROOF_SETUP+TRANSPORT_PROOF,id='private-contact-style-proof')]
        for index,key in ((stationary,'prediction_error_stationary'),(sphere,'sphere_previous')):
            old=source(before['cells'][index]).replace('SAVE_PDF = True','SAVE_PDF = False')
            old=old.replace('plt.show()',f"_style_states[{key!r}]=_style_state(fig)\nplt.close(fig)")
            work.append(nbformat.v4.new_code_cell(old,id='private-old-'+key.replace('_','-')))
        destinations=[]
        for index in [stationary,sphere]+[i for i in range(append_start,len(notebook['cells'])) if notebook['cells'][i]['cell_type']=='code']:
            fresh=deepcopy(notebook['cells'][index]);key=fresh['metadata']['figure_id']
            fresh['outputs']=[];fresh['execution_count']=None
            if key=='prediction_error_stationary':proof="_style_check(fig,'prediction_error_stationary')"
            elif key=='prediction_error_sphere':proof='_transport_dense_check(fig)'
            else:proof=f'_transport_diagnostic_check(fig,{key!r})'
            fresh['source']=source(fresh).replace('plt.show()',proof+'\nplt.show()')
            destinations.append((index,len(work)));work.append(fresh)
        work.append(nbformat.v4.new_code_cell("print('CONTACT_TRANSPORT_PROOF='+_style_json.dumps(_style_reports))",id='private-contact-transport-proof-report'))
        mini=nbformat.reads(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=work,
            metadata={'kernelspec':notebook['metadata']['kernelspec']})),as_version=4)
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
        os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
        os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
        os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
        NotebookClient(mini,timeout=timeout,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
        if path.read_bytes()!=original_bytes:raise RuntimeError('The contact notebook changed concurrently')
        reports=None
        for item in mini.cells[-1].outputs:
            if 'CONTACT_TRANSPORT_PROOF=' in item.get('text',''):
                reports=json.loads(item['text'].split('CONTACT_TRANSPORT_PROOF=',1)[1].strip())
        if reports is None:raise RuntimeError('Missing transport numerical-artist proof')
        for index,position in destinations:
            cell=notebook['cells'][index];executed=mini.cells[position]
            cell['outputs']=json.loads(json.dumps(executed.outputs));cell['execution_count']=executed.execution_count
            if 'execution' in executed.metadata:cell['metadata']['execution']=json.loads(json.dumps(executed.metadata['execution']))
        changed={stationary,sphere,*changed_prose}
        if any(cell!=before['cells'][i] for i,cell in enumerate(notebook['cells'][:append_start]) if i not in changed):
            raise RuntimeError('An unrelated original contact cell changed')
        if any(sha(root/name)!=digest for name,digest in {**numeric,**protected}.items()):
            raise RuntimeError('An original numerical input or untouched forest/3D asset changed')
        editable=notebook['metadata']['editable_plotting'];editable['input_hashes'].update(evidence['input_hashes'])
        editable['input_hashes'][str(manifest_path)]=sha(root/manifest_path)
        editable['loader_sources']['lib/contact_transport_diagnostics.py']=sha(root/'lib/contact_transport_diagnostics.py')
        editable['figure_count']=editable['rendered_figure_count']=len(FIGURES)+len(TRANSPORT_FIGURES)
        info=notebook['metadata']['contact_prediction_diagnostics']
        info['producer_sha256']=sha(__file__);info['input_hashes']=dict(editable['input_hashes'])
        info['source_hashes'].update(evidence.get('source_hashes',{}));info.update(contact_caption_metadata(root))
        notebook['metadata']['contact_transport_diagnostics']={'schema':1,'manifest':str(manifest_path),
            'manifest_sha256':sha(root/manifest_path),'input_hashes':evidence['input_hashes'],
            'source_hashes':evidence.get('source_hashes',{}),'checks':evidence['checks'],
            'action_times':list(range(4,193,4)),'simulations':256,
            'identity_color':'#484848','figures':[stem for stem,_ in TRANSPORT_FIGURES]}
        notebook['metadata']['solid_trajectory_style']={'producer':'scripts/make_contact_prediction_diagnostics_notebook.py',
            'producer_sha256':sha(__file__),'markers':False,'linestyle':'-',
            'original_checkpoint_statistics_unchanged':True,'untouched_forest_and_spheres_unchanged':True}
        temporary=path.with_suffix('.transport.tmp');temporary.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n');temporary.replace(path)
        layout={'status':'passed','audit_source_sha256':sha(root/'scripts/check_manuscript_plot_exports.py'),'pdfs':{}}
        for stem,section in FIGURES+TRANSPORT_FIGURES:
            for relative in (figure_output(stem),Path('results')/section/'contact'/(stem+'.pdf')):
                layout['pdfs'][str(relative)]=audit_pdf(root/relative,[[15,5]],
                    primary_fonts={'LABEL_FONT_SIZE':25,'TICK_FONT_SIZE':23,'LEGEND_FONT_SIZE':22})
        if any(record['errors'] for record in layout['pdfs'].values()):raise RuntimeError('A contact diagnostic PDF failed the layout audit')
        (root/OUTPUT/'pdf_layout_checks.json').write_text(json.dumps(layout,indent=2)+'\n')
        protection={'status':'passed','numeric_inputs':len(numeric),'unselected_assets':len(protected),
            'unchanged_original_cell_objects':append_start-len(changed),'baseline':baseline,'reports':reports,
            'scope':'Original numerical evidence and all untouched contact forest/3D exports; only two histories and appended diagnostics rendered.'}
        record=manifest(root,nbformat.read(path,as_version=4),protection)
        transport_records={name:entry for name,entry in record['figures'].items() if name.startswith('transport_')}
        transport_files={name:digest for name,digest in record['files'].items()
                         if name.startswith(str(TRANSPORT_OUTPUT)) or Path(name).name.startswith('transport_')}
        (root/TRANSPORT_OUTPUT/'figure_manifest.json').write_text(json.dumps(dict(
            schema=1,status='completed',notebook=str(NOTEBOOK),notebook_sha256=sha(path),
            producer='scripts/make_contact_prediction_diagnostics_notebook.py',producer_sha256=sha(__file__),
            data_manifest=str(manifest_path),data_manifest_sha256=sha(root/manifest_path),
            input_hashes=evidence['input_hashes'],source_hashes=evidence.get('source_hashes',{}),
            figures=transport_records,files=transport_files,preservation=protection),indent=2,allow_nan=False)+'\n')
        report={'status':'passed','notebook':str(NOTEBOOK),'updated_original_plot_cells':[stationary,sphere],
            'appended_figures':[stem for stem,_ in TRANSPORT_FIGURES],'protection':protection}
        report_path=Path('/tmp/contact-dense-transport-render.json');report_path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':'passed','notebook':str(NOTEBOOK),'report':str(report_path)},indent=2),flush=True)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--restyle',action='store_true',help='Only redraw the existing histories as solid markerless curves; never export numerical data.')
    parser.add_argument('--transport',action='store_true',help='Read the verified dense replay, update histories, and append transport diagnostics without rewriting original numerical evidence.')
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args();root=args.root.resolve()
    folder=root/OUTPUT;folder.mkdir(parents=True,exist_ok=True)
    if args.transport:
        update_contact_transport_notebook(root,timeout=args.timeout)
        return
    if args.restyle:
        restyle_contact_diagnostics(root,timeout=args.timeout)
        return
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
