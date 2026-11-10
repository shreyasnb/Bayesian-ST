"""Title-free paper figures from saved, source-verified numerical evidence.

No simulation, fitting, posterior integration, posterior sampling, or tuning is
performed here. Geometry evaluates supplied density parameters and normalizers
on a display mesh only; that mesh is never used as an integration rule.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .plotting import plt, DPI
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .distribution_geometry_visuals import (
    bingham_sphere_mesh, sphere_mesh,
    real_bingham_log_density, restriction_log_density, shared_density_norm)


METHODS = {
    'mean': r'$\overline P_t^+$',
    'mode': r'$P_{\widehat{\mathcal X}_t}$',
    'iso': r'$\overline P_t^{\mathrm{iso}}$',
    'isotropic': r'$\overline P_t^{\mathrm{iso}}$',
    'mode_loaded': 'Loaded mode',
    'modal_loading': 'Loaded modal beamformer',
    'covariance_loading': 'Loaded EW covariance',
    'regularized': 'Regularized\nprojector',
    'scm': 'Loaded EW covariance',
    'scm_loaded': 'Loaded EW covariance',
    'ew_covariance': 'EW covariance',
    'lw': 'Ledoit--Wolf',
    'grouse': 'GROUSE',
    'great': 'GREAT',
    'great_window': 'GREAT',
    'great_ew': 'GREAT EW',
    'oracle': 'Known covariance',
    'known': 'True variances',
    'exact_known': r'True variances, $\beta=1$',
    'known_forgetting': 'True variances, forgetting',
    'misspecified': 'Fixed variances',
    'learned': 'Learned variances',
    'block_em': 'Block EM',
    'residual': 'Residual fit',
    'ridge': 'Ridge regression',
    'gaussian': 'Extended Kalman filter',
    'laplace': r'$\mathcal N(0,H_A^{-1})$',
    'residual_pca': 'PCA of responses',
}
COLORS = {
    'mean': '#0072B2', 'mode': '#D55E00', 'iso': '#CC79A7', 'isotropic': '#CC79A7',
    'great': '#009E73', 'great_window': '#009E73', 'great_ew': '#56B4E9',
    'mode_loaded': '#8C564B', 'regularized': '#8C564B',
    'modal_loading': '#8C564B', 'covariance_loading': '#484848',
    'scm': '#484848', 'scm_loaded': '#484848', 'lw': '#8172B2',
    'grouse': '#B07C49', 'oracle': '#111111',
    'known': '#111111', 'exact_known': '#111111', 'known_forgetting': '#8172B2',
    'misspecified': '#999999', 'learned': '#0072B2', 'block_em': '#0072B2',
    'residual': '#D55E00', 'ridge': '#B07C49', 'gaussian': '#009E73',
    'laplace': '#8172B2', 'residual_pca': '#B07C49',
}
MARKERS = {'mean': 'o', 'mode': 's', 'iso': '^', 'isotropic': '^',
           'great': 'D', 'great_window': 'D', 'great_ew': 'v', 'mode_loaded': 'P',
           'scm': 'X', 'scm_loaded': 'X', 'oracle': '*',
           'known': 's', 'misspecified': '^', 'learned': 'o'}
MARKERS.update(modal_loading='P', covariance_loading='X', exact_known='s', known_forgetting='D')
METHODS['mode_loaded'] = 'Loaded modal beamformer'
GREY = '#70777D'
BLACK = '#222222'
CAMERA = dict(elev=25., azim=-50.)
FIGURE_WIDTH = 7.0
# Only color changes: the original shared density normalization is retained.
# These knots are positions on that color scale, not probability quantiles.
DENSITY_CMAP = colors.LinearSegmentedColormap.from_list(
    'density_blue_pink_yellow', [(0., '#263a8b'), (.28, '#55209a'),
        (.53, '#b33695'), (.72, '#ed78a4'), (.88, '#ffc85a'), (1., '#f4f52b')], N=256)
DENSITY_CMAP.set_under(DENSITY_CMAP(0.))
DENSITY_CMAP.set_over(DENSITY_CMAP(1.))
PAPER_STYLE = {
    'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
    'font.family': 'serif', 'font.size': 11.,
    'axes.labelsize': 11., 'legend.fontsize': 10.,
    'xtick.labelsize': 10., 'ytick.labelsize': 10.,
    'lines.linewidth': 1.6, 'lines.markersize': 4.,
    'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': None,
    'figure.figsize': (FIGURE_WIDTH, 3.2),
    'figure.max_open_warning': 0,
    'axes.spines.top': False, 'axes.spines.right': False,
}


@contextmanager
def paper_style():
    with plt.rc_context(PAPER_STYLE):
        yield


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def method_label(method, example=None):
    key = str(method)
    return METHODS.get(key, key.replace('_', ' '))


def color(method):
    return COLORS.get(str(method), GREY)


def marker(method):
    return MARKERS.get(str(method), 'o')


def clean(ax, which='both'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=which, color='#BFC4C8', alpha=.30, linewidth=.45)


def letter(ax, value):
    # Short panel letters are the only free-text figure annotations.
    if getattr(ax, 'name', '') == '3d':
        ax.text2D(.02, .97, value, transform=ax.transAxes, ha='left', va='top')
    else:
        ax.text(.015, .985, value, transform=ax.transAxes, ha='left', va='top')


def verify_title_free(fig):
    if fig._suptitle is not None and fig._suptitle.get_text():
        raise RuntimeError('Paper figures must not contain a figure title')
    for ax in fig.axes:
        if any(ax.get_title(loc=loc) for loc in ('left', 'center', 'right')):
            raise RuntimeError('Paper figures must not contain plot titles')
    if (plt.rcParams['text.usetex'] is not True or fig.dpi != DPI
            or plt.rcParams['savefig.dpi'] != DPI
            or not np.isclose(fig.get_figwidth(), FIGURE_WIDTH)):
        raise RuntimeError('Actual LaTeX, seven-inch width and 300-dpi display layers are required')


def prepare_paper_axes(fig):
    """Keep logarithmic panels legible at their actual seven-inch width."""
    if not getattr(fig, '_implementation_spacing', False):
        fig.set_figheight(max(3.35, float(fig.get_figheight())*1.14))
        fig._implementation_spacing = True
    from matplotlib.ticker import FixedLocator, NullLocator, LogFormatterSciNotation, ScalarFormatter, FuncFormatter
    for ax in fig.axes:
        for name in ('x','y'):
            if getattr(ax,'get_'+name+'scale')() != 'log':
                continue
            low,high=getattr(ax,'get_'+name+'lim')()
            if low <= 0 or not np.isfinite([low,high]).all():
                raise ValueError('Logarithmic paper axis needs finite positive limits')
            first,last=int(np.ceil(np.log10(low)-1e-10)),int(np.floor(np.log10(high)+1e-10))
            decades=np.unique(np.round(np.linspace(first,last,min(3,max(1,last-first+1))))).astype(int)
            ticks=10.**decades if first<=last else np.geomspace(low,high,3)
            axis=getattr(ax,name+'axis')
            label=getattr(ax,'get_'+name+'label')().lower()
            if 'dimension' in label or 'sensors' in label:
                raw=[]
                for line in ax.lines:
                    values=np.asarray(line.get_xdata() if name=='x' else line.get_ydata(),float)
                    raw.extend(values[np.isfinite(values)&(values>0)])
                values=np.unique(raw)
                if len(values):ticks=values[np.unique(np.round(np.linspace(0,len(values)-1,min(3,len(values)))).astype(int))]
                axis.set_major_formatter(ScalarFormatter())
            elif first>last:
                axis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:.2g}'))
            else:
                axis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
            axis.set_major_locator(FixedLocator(ticks));axis.set_minor_locator(NullLocator())


@dataclass
class Evidence:
    root: Path
    example: str
    stage: str
    components: dict
    tables: dict
    arrays: dict
    inputs: dict
    references: dict = field(default_factory=dict)

    def config(self, component):
        return self.components[component].get('config', {})

    def frame(self, component, *suffixes, optional=False):
        group = self.tables.get(component, {})
        for suffix in suffixes:
            exact = [frame for name, frame in group.items() if name == suffix]
            if len(exact) == 1:
                return exact[0].copy()
            candidates = [frame for name, frame in group.items()
                          if name.endswith('_' + suffix) or name.endswith(suffix)]
            if len(candidates) == 1:
                return candidates[0].copy()
        if optional:
            return pd.DataFrame()
        raise KeyError(f'{self.example}/{component}: missing unique table {suffixes}; '
                       f'available {sorted(group)}')

    def array(self, component, *suffixes, optional=False):
        group = self.arrays.get(component, {})
        for suffix in suffixes:
            if suffix in group:
                return group[suffix]
            candidates = [record for name, record in group.items()
                          if name.endswith('_' + suffix) or name.endswith(suffix)]
            if len(candidates) == 1:
                return candidates[0]
        if optional:
            return {}
        raise KeyError(f'{self.example}/{component}: missing saved array {suffixes}; '
                       f'available {sorted(group)}')


def load_evidence(root, example, stage, *, components=None, require_general=True):
    """Load only manifest-declared, content-verified numerical artifacts."""
    from .final_validation import require_gate, verify_component
    root = Path(root).resolve()
    if require_general:
        require_gate(root)
    requested = tuple(components) if components is not None else ('theory', 'performance', 'prediction', 'computation')
    components, tables, arrays, inputs = {}, {}, {}, {}
    for component in requested:
        meta = verify_component(root, example, stage, component)
        components[component] = meta
        path = root / 'results' / example / 'data' / stage / component / 'manifest.json'
        inputs[str(path.relative_to(root))] = sha(path)
        tables[component], arrays[component] = {}, {}
        for name in meta['artifacts']:
            artifact = root / name
            if artifact.suffix == '.csv' or artifact.name.endswith('.csv.gz'):
                try:
                    stem = artifact.name[:-7] if artifact.name.endswith('.csv.gz') else artifact.stem
                    tables[component][stem] = pd.read_csv(artifact)
                except pd.errors.EmptyDataError:
                    tables[component][artifact.stem] = pd.DataFrame()
            elif artifact.suffix == '.npz':
                with np.load(artifact, allow_pickle=False) as record:
                    arrays[component][artifact.stem] = {key: record[key] for key in record.files}
    return Evidence(root, example, stage, components, tables, arrays, inputs)


def _column(frame, *names):
    for name in names:
        if name in frame:
            return name
    raise KeyError(f'Missing column {names}; available {list(frame)}')


def _finite_column(frame, *names):
    """Choose aliases of one declared quantity, never an all-missing column."""
    for name in names:
        if name in frame and np.isfinite(pd.to_numeric(frame[name], errors='coerce')).any():
            return name
    raise KeyError(f'No finite saved values for declared quantity {names}')


def _normalize(frame):
    frame = frame.copy()
    mapping = {}
    for destination, candidates in {
        'episode': ('trial',), 'n': ('N',), 'sigma2': ('s',),
        'condition': ('scenario', 'regime'), 't': ('time',),
    }.items():
        if destination not in frame:
            for source in candidates:
                if source in frame:
                    mapping[source] = destination
                    break
    frame = frame.rename(columns=mapping)
    if 'method' in frame:
        frame['method'] = frame.method.replace({
            'isotropic': 'iso', 'modal_loading': 'mode_loaded',
            'covariance_loading': 'scm_loaded', 'ledoit_wolf': 'lw'})
    return frame


def _primary(frame, control='mean'):
    """Select declared primary means without treating controls as repetitions."""
    frame = _normalize(frame)
    if 'split' in frame and 'test' in set(frame.split.astype(str)):
        frame = frame[frame.split.eq('test')]
    if 'control' in frame:
        frame = frame[frame.control.eq(control)]
    return frame


def _mean_interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan
    half = float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean-half, mean+half


def _identity(ax, x, y):
    values = np.r_[np.asarray(x, float), np.asarray(y, float)]
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('An identity comparison needs finite saved values')
    low, high = values.min(), values.max()
    padding = max(.06 * (high-low), .01 * max(abs(low), abs(high)), 1e-10)
    lower = max(low-padding, .8*low) if low > 0 else low-padding
    ax.plot([lower, high+padding], [lower, high+padding], '--',
            color=GREY, linewidth=1.)
    ax.set(xlim=(lower, high+padding), ylim=(lower, high+padding))


def _saved(record, *keys):
    for key in keys:
        if key in record:
            return np.asarray(record[key])
    raise KeyError(f'Missing saved array {keys}; available {sorted(record)}')


def _optional(record, *keys):
    for key in keys:
        if key in record:
            return np.asarray(record[key])
    return None


def display_sphere_geometry(A, nodes=None, rotation=None):
    """Rigid display coordinates and exact quadratic-density level curves.

    Native coordinates are the default (R=I). Optional rigid changes of
    coordinates exercise numerical invariance in the independent audit only;
    final figures retain the saved physical/coefficient frames. The returned
    exact level curves are mathematical audit targets and are not drawn over
    the final densities. Display nodes are never quadrature.
    """
    A = np.asarray(A)
    if (A.shape != (3, 3) or not np.isfinite(A).all()
            or np.max(np.abs(A.imag)) > 1e-10
            or not np.allclose(A, A.T, atol=1e-10, rtol=1e-12)):
        raise ValueError('Display geometry needs a real symmetric 3 by 3 parameter')
    A = (A.real + A.real.T)/2
    values, U = np.linalg.eigh(A)
    mode, soft = U[:, -1].copy(), U[:, -2].copy()
    for vector in (mode, soft):
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    hard = np.cross(mode, soft)
    gaps = values[-1]-values[-2::-1]
    if rotation is None:
        rotation = np.eye(3)
    R = np.asarray(rotation, float)
    if (R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-12)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-12)):
        raise ValueError('A display change must be a proper rigid rotation')
    native = bingham_sphere_mesh(A, 161, 321).nodes if nodes is None else np.asarray(nodes, float)
    if native.shape[-1] != 3 or not np.allclose(np.sum(native*native, axis=-1), 1., atol=1e-10):
        raise ValueError('Display coordinates must remain on the unit sphere')
    phi = np.linspace(0., 2*np.pi, 241)
    contours = {}
    for level in (1., 2.):
        if level >= gaps[0]:
            continue
        s = np.sqrt(level/gaps[0])*np.cos(phi)
        h = np.sqrt(level/gaps[1])*np.sin(phi)
        m = np.sqrt(np.maximum(0., 1-s*s-h*h))
        cap = m[:, None]*mode + s[:, None]*soft + h[:, None]*hard
        contours[level] = np.stack([cap @ R.T, -cap @ R.T])
    return dict(rotation=R, parameter=R @ A @ R.T, mode=R @ mode,
                native_mode=mode, eigenvalues=values, gaps=gaps,
                native_nodes=native, nodes=native @ R.T, contours=contours)


def density_sphere(ax, nodes, log_density, norm, coordinate='n', mode=None, truth=None):
    """True orthographic fixed-radius sphere with visible native 3D axes."""
    # All guide curves and markers are front-masked below. Explicit layers
    # prevent Axes3D's collection sort from hiding them behind the surface.
    ax.computed_zorder = False
    nodes, log_density = np.asarray(nodes), np.asarray(log_density)
    if nodes.shape[-1] != 3 or not np.allclose(np.sum(nodes*nodes, axis=-1), 1., atol=1e-10):
        raise ValueError('Density display vertices must remain on the unit sphere')
    rgba = DENSITY_CMAP(norm(np.exp(log_density)))
    surface = ax.plot_surface(nodes[..., 0], nodes[..., 1], nodes[..., 2],
        facecolors=rgba, shade=False, linewidth=0, antialiased=False,
        rstride=1, cstride=1, zorder=1)
    # Only the colored surface is rasterized; text, axes and curves stay vector.
    surface.set_rasterized(True)
    phi = np.linspace(0., 2*np.pi, 241)
    el, az = np.deg2rad([CAMERA['elev'], CAMERA['azim']])
    facing = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    for plane in ((0, 1), (0, 2), (1, 2)):
        ring = np.zeros((len(phi), 3))
        ring[:, plane[0]], ring[:, plane[1]] = 1.006*np.cos(phi), 1.006*np.sin(phi)
        # Show only the near hemisphere: a far-side guide must not be painted
        # over the visible surface. These are coordinate great circles, not
        # density contours, intervals or a deformation of the unit sphere.
        visible = np.where((ring @ facing >= 0.)[:, None], ring, np.nan)
        ax.plot(*visible.T, color='#B7C0CB', linewidth=.5, alpha=.55, zorder=3)
    for vector, symbol, edge in ((mode, 'o', '#D55E00'), (truth, 'x', BLACK)):
        if vector is not None:
            vector = np.asarray(vector, float)
            vector /= np.linalg.norm(vector)
            visible = vector if vector @ facing >= 0. else -vector
            ax.scatter(*visible, marker=symbol, s=12, color=edge, depthshade=False, zorder=4)
    ax.set_proj_type('ortho')
    ax.view_init(**CAMERA)
    ax.set_box_aspect((1., 1., 1.))
    ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), zlim=(-1.12, 1.12),
           xticks=(-1., 0., 1.), yticks=(-1., 0., 1.), zticks=(-1., 0., 1.))
    ax.set_xlabel(r'$' + coordinate + r'_1$', labelpad=-2)
    ax.set_ylabel(r'$' + coordinate + r'_2$', labelpad=-2)
    ax.set_zlabel('')
    ax.text2D(.95, .67, r'$' + coordinate + r'_3$', transform=ax.transAxes,
              rotation=90, ha='center', va='center', fontsize=11.)
    ax.tick_params(labelsize=10., pad=-2, length=1)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1., 1., 1., 0.))
        axis.pane.set_edgecolor((.65, .68, .72, .35))
        axis.line.set_color('#70777D')
        axis.line.set_linewidth(.65)
    ax._sphere_display = dict(coordinates='Saved native physical or real-coefficient frame',
        rotation_native_to_display=np.eye(3).tolist(), fixed_radius=1.,
        mode_centered=False, density_contours=False,
        depth_cues='Front-visible native coordinate great circles; no density shading',
        cmap=DENSITY_CMAP.name, display_mesh_only=True)
    return surface


def _draw_task_dimensions(ax, evidence):
    frame = _primary(evidence.frame('performance', 'stationary', 'metrics'))
    xname = 'ratio' if 'ratio' in frame else None
    if xname is None:
        Lname = _column(frame, 'L', 't', 'observations')
        frame['ratio'] = frame[Lname] / frame.n
    value = _column(frame, 'heldout_ratio_db', 'output_oracle_db',
                    'heldout_power_ratio_db', 'response_mse')
    if evidence.example == 'contact':
        value = _column(frame, 'response_mse', 'response_loss', 'heldout_response_mse')
    methods = [method for method in ('mean','mode','scm_loaded','great_window')
        if method in set(frame.method.astype(str))]
    # The principal panel uses the four declared ratios and all new dimensions.
    ratios = sorted(frame.ratio.unique())
    styles = ['-', '--', '-.', ':']
    for index, (ratio, group) in enumerate(frame.groupby('ratio', sort=True)):
        if ratio not in (1.,):
            continue
        for method in methods:
            g = group[group.method.eq(method)]
            means, lows, highs, dimensions = [], [], [], []
            for n, h in g.groupby('n', sort=True):
                h = h.groupby('episode', sort=False)[value].mean()
                mean, low, high = _mean_interval(h)
                dimensions.append(n); means.append(mean); lows.append(low); highs.append(high)
            ax.plot(dimensions, means, linestyle='-', marker=marker(method),
                    color=color(method), label=method_label(method, evidence.example))
            if method=='mean':ax.fill_between(dimensions,lows,highs,color=color(method),alpha=.10)
    ax.set_xlabel('Number of sensors' if evidence.example == 'beamforming' else 'Number of observations')
    ax.set_ylabel('Output power / oracle power (dB)' if evidence.example == 'beamforming' else 'Response MSE')
    clean(ax)
    ax._final_selection=dict(cohort='stationary',observations_per_sensor=1,
        dimensions=sorted(frame.n.unique().tolist()),episode_scope='All independent test episodes')


def _episode_losses(frame, endpoint):
    frame = _primary(frame)
    if 'endpoint' in frame and frame.endpoint.eq('episode_average').any():
        frame = frame[frame.endpoint.eq('episode_average')]
    grouping = [name for name in ('condition', 'n', 'method', 'episode') if name in frame]
    return frame.groupby(grouping, sort=False, dropna=False)[endpoint].mean().reset_index()


def _boxplot(ax,data,**kwargs):
    """Keep the declared Matplotlib 3.8 minimum and newer spelling compatible."""
    from inspect import signature
    parameters=signature(ax.boxplot).parameters
    if 'orientation' in kwargs and 'orientation' not in parameters:
        kwargs['vert']=kwargs.pop('orientation')=='vertical'
    if 'tick_labels' in kwargs and 'tick_labels' not in parameters:
        kwargs['labels']=kwargs.pop('tick_labels')
    return ax.boxplot(data,**kwargs)


def _draw_paired_tasks(ax, evidence, all_references=False, condition=None, references=None, confidence=False):
    if evidence.example == 'beamforming':
        frame = evidence.frame('performance', 'moving_episodes', 'episode_metrics', 'moving', 'stationary')
        endpoint = _column(frame, 'heldout_ratio_db', 'output_oracle_db', 'heldout_power')
    else:
        frame = evidence.frame('performance', 'episode_metrics', 'metrics')
        endpoint = _column(frame, 'response_mse', 'response_loss', 'heldout_response_mse')
    frame = _episode_losses(frame, endpoint)
    if 'condition' in frame:
        # Configuration order is fixed before outcomes, not selected by gain.
        if condition is None:
            condition = ('sphere' if evidence.example == 'contact' and 'sphere' in set(frame.condition)
                         else 'smooth' if evidence.example == 'beamforming' and 'smooth' in set(frame.condition)
                         else frame.condition.drop_duplicates().iloc[0])
        frame = frame[frame.condition.eq(condition)]
    if 'n' in frame:
        n = 32 if 32 in set(frame.n) else int(frame.n.iloc[0])
        frame = frame[frame.n.eq(n)]
    base = frame[frame.method.eq('mean')].set_index('episode')[endpoint]
    if references is None:
        references = ['mode', 'iso', 'isotropic', 'great_window', 'great', 'great_ew',
                      'mode_loaded', 'regularized', 'scm_loaded']
        if not all_references:
            references = references[:6]
    values, labels, colors_ = [], [], []
    for method in references:
        other = frame[frame.method.eq(method)].set_index('episode')[endpoint]
        indices = base.index.intersection(other.index)
        if not len(indices):
            continue
        delta = (other.loc[indices]-base.loc[indices]).to_numpy()
        if evidence.example == 'contact':
            delta = delta*1e6
        values.append(delta); labels.append(method_label(method, evidence.example)); colors_.append(color(method))
    if not values:
        raise ValueError('Paired task plot needs same-episode mean and comparator records')
    if confidence:
        for position,(sample,c) in enumerate(zip(values,colors_),1):
            mean,lower,upper=_mean_interval(sample)
            ax.errorbar(mean,position,xerr=[[mean-lower],[upper-mean]],fmt='o',
                color=c,capsize=3.,markersize=4.)
        ax.set(yticks=np.arange(1,len(labels)+1),yticklabels=labels,ylim=(.5,len(labels)+.5))
    else:
        box = _boxplot(ax,values, orientation='horizontal', tick_labels=labels,
            patch_artist=True, medianprops={'color': 'white', 'linewidth': 1.2},
            flierprops={'markersize': 1.8, 'alpha': .35})
        for patch, c in zip(box['boxes'], colors_):
            patch.set_facecolor(c)
            patch.set_alpha(.85)
    ax.axvline(0., linestyle='--', color=BLACK, linewidth=1.)
    ax.set_xlabel('Paired loss difference (dB)' if evidence.example == 'beamforming'
                  else r'$\mathrm{MSE}_{\mathrm{ref}}-\mathrm{MSE}_{\mathrm{B}}$ $((\mathrm{mm}/\mathrm{s})^2)$')
    clean(ax, 'x')
    ax._final_selection=dict(condition=condition,n=n if 'n' in frame else None,
        endpoint=endpoint,displayed_references=list(references),
        uncertainty='Pointwise paired Student95% confidence interval; all episode outliers included' if confidence else 'Episode-difference distribution box plot',
        omitted_references='Every remaining comparison is retained in additional_comparisons.pdf and paired raw summaries',
        statistical_unit='Independent episode; within-episode losses averaged')


def task_performance(evidence):
    if evidence.example == 'beamforming':
        fig, axes = plt.subplots(1,2,figsize=(FIGURE_WIDTH,4.1),layout='constrained',gridspec_kw={'width_ratios':(1.,1.1)})
        _draw_task_dimensions(axes[0], evidence)
        _draw_paired_tasks(axes[1], evidence)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=10.)
    else:
        frame = _primary(evidence.frame('performance', 'metrics', 'episode_metrics'))
        if 'endpoint' in frame:
            frame = frame[frame.endpoint.eq('checkpoint')]
        if 'condition' in frame:
            condition = ('static_concentrated' if 'static_concentrated' in set(frame.condition)
                         else frame.condition.iloc[0])
            frame = frame[frame.condition.eq(condition)]
        endpoint = _column(frame, 'response_mse', 'response_loss', 'heldout_response_mse')
        fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.8), layout='constrained')
        if 't' in frame:
            for method in ('mean', 'mode', 'regularized', 'great_window'):
                group = frame[frame.method.eq(method)]
                if group.empty:
                    continue
                points = []
                for t, g in group.groupby('t', sort=True):
                    mean, lower, upper = _mean_interval(g.groupby('episode')[endpoint].mean()*1e6)
                    points.append((t, mean, lower, upper))
                points = np.asarray(points)
                axes[0].plot(points[:, 0], points[:, 1], marker=marker(method),
                             color=color(method), label=method_label(method))
                if method=='mean':
                    axes[0].fill_between(points[:, 0], points[:, 2], points[:, 3],
                                         color=color(method), alpha=.10)
            axes[0].set_xlabel(r'$t$ (observations)')
            axes[0].set_ylabel(r'$\mathrm{MSE}(\widehat u_t)$ $((\mathrm{mm}/\mathrm{s})^2)$')
            axes[0].margins(y=.12)
            clean(axes[0])
        else:
            _draw_paired_tasks(axes[0], evidence)
        _draw_contact_absolute_tasks(axes[1], evidence)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=10.)
        axes[0]._final_selection=dict(condition=condition,displayed_methods=['mean','mode','regularized','great_window'],
            omitted_methods='All remaining methods are retained in additional_comparisons.pdf')
    for ax, tag in zip(axes, ('(a)', '(b)')):
        letter(ax, tag)
    if evidence.example == 'contact':
        paired, ax = plt.subplots(figsize=(FIGURE_WIDTH,3.3),layout='constrained')
        _draw_paired_tasks(ax,evidence,references=['mode','iso','isotropic','regularized'],confidence=True)
        letter(ax,'(a)')
        paired._final_selection=dict(condition='sphere',diagnostic='paired_mse_difference',
            self_difference='The posterior mean minus itself is exactly zero; no artificial self confidence interval')
        return [fig,paired]
    return [fig]


def _draw_contact_absolute_tasks(ax, evidence):
    frame = _episode_losses(evidence.frame('performance','episode_metrics','metrics'),'response_mse')
    frame = frame[frame.condition.eq('sphere')]
    methods = [key for key in ('mean','mode','iso','isotropic','regularized') if frame.method.eq(key).any()]
    samples = {key:frame[frame.method.eq(key)].set_index('episode').response_mse for key in methods}
    common = samples[methods[0]].index
    for sample in samples.values():common=common.intersection(sample.index)
    if len(common)<2:raise ValueError('Absolute MSE intervals need common independent test episodes')
    for row,method in enumerate(methods,1):
        mean,low,high=_mean_interval(samples[method].loc[common].to_numpy()*1e6)
        ax.errorbar(mean,row,xerr=[[mean-low],[high-mean]],fmt=marker(method),
            color=color(method),capsize=3.,markersize=5.)
    ax.set(yticks=np.arange(1,len(methods)+1),
        yticklabels=[method_label(key) for key in methods],ylim=(.5,len(methods)+.5),
        xlabel=r'$\mathrm{MSE}(\widehat u)$ $((\mathrm{mm}/\mathrm{s})^2)$')
    clean(ax,'x')
    ax._final_selection=dict(condition='sphere',endpoint='episode_average',
        displayed_methods=methods,episodes=len(common),diagnostic='absolute_response_mse',
        uncertainty='Pointwise Student95% intervals of each method mean across the same independent episodes; paired differences retained on page2')


def setup(evidence):
    record = evidence.array('performance', 'stories', 'story')
    fig = plt.figure(figsize=(FIGURE_WIDTH, 3.0), layout='constrained')
    if evidence.example == 'beamforming':
        grid = fig.add_gridspec(1, 3, width_ratios=(1., 1.3, 1.))
        ax = fig.add_subplot(grid[0, 0])
        U = _saved(record, 'setup_U', 'U')
        n = U.shape[-1]
        ax.plot(np.arange(n), np.zeros(n), '-', color=GREY, linewidth=1.)
        ax.scatter(np.arange(n), np.zeros(n), s=10, facecolors='white', edgecolors=BLACK, zorder=3)
        centre = (n-1)/2
        for start, c, label in [((centre, 1.0), '#009E73', r'$a$'),
                                ((centre-.35*n, 1.0), '#D55E00', r'$h_t$')]:
            ax.annotate('', xy=(centre, .06), xytext=start,
                        arrowprops={'arrowstyle': '->', 'color': c, 'linewidth': 1.4})
            ax.text(start[0], start[1]+.04, label, color=c, ha='center')
        ax.set(xlim=(-1., n), ylim=(-.18, 1.2), yticks=[], xticks=[0, n-1],
               xlabel='Sensor index')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax = fig.add_subplot(grid[0, 1])
        image = ax.imshow(abs(U[:min(32, len(U))]).T, origin='lower',
                          aspect='auto', cmap='Greys', interpolation='nearest', rasterized=True)
        ax.set_xlabel('Observation index'); ax.set_ylabel('Sensor index')
        fig.colorbar(image, ax=ax, label=r'$|u_{t,j}|$', shrink=.85)
        ax = fig.add_subplot(grid[0, 2])
        scan = _saved(record, 'scan_angles', 'scan')
        patterns = _optional(record, 'beampatterns', 'setup_beampatterns')
        if patterns is None:
            weights = _saved(record, 'setup_weights')
            signatures = _saved(record, 'scan_signatures')
            patterns = abs(weights.conj() @ signatures.T)**2
        methods = _optional(record, 'beampattern_methods', 'setup_methods')
        if methods is None:
            methods = np.array(['mean', 'mode', 'iso'][:len(patterns)])
        for method, pattern in zip(methods, patterns):
            if method not in ('mean','mode','iso','isotropic'):
                continue
            ax.plot(np.rad2deg(scan), 10*np.log10(np.maximum(pattern, 1e-12)),
                    color=color(method), label=method_label(method, evidence.example))
        desired = float(np.asarray(_optional(record, 'desired_angle')).ravel()[0]) if 'desired_angle' in record else 0.
        interfering = _optional(record, 'interference_angle', 'setup_interference_angle', 'setup_interferer_angle')
        ax.axvline(np.rad2deg(desired), color='#009E73', linestyle=':', linewidth=1.)
        if interfering is not None:
            ax.axvline(np.rad2deg(float(interfering)), color='#D55E00', linestyle=':', linewidth=1.)
        ax.set_xlabel('Arrival angle (deg)'); ax.set_ylabel(r'$|w^H a(\theta)|^2$ (dB)')
        ax.legend(frameon=False, fontsize=10.); clean(ax)
    else:
        grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.))
        ax = fig.add_subplot(grid[0, 0], projection='3d')
        positions = _saved(record, 'positions', 'setup_positions', 'sphere_positions')
        normal = _saved(record, 'world_normal', 'normal', 'sphere_world_normal')
        radius = float(evidence.config('performance').get('radius', .05))
        mesh = sphere_mesh(41, 81).nodes
        surface = ax.plot_surface(*(radius*1e3*mesh).transpose(2, 0, 1),
            color='#CAD6E4', alpha=.22, linewidth=0., shade=False)
        surface.set_rasterized(True)
        ax.plot(*(positions*1e3).T, color=BLACK, linewidth=1.2)
        indices = np.linspace(0, len(positions)-1, 4, dtype=int)
        for i in indices:
            p, n = positions[i]*1e3, normal[i]
            ax.quiver(*p, *n, length=8., normalize=True, color='#D55E00', linewidth=.8)
        ax.set_proj_type('ortho'); ax.view_init(**CAMERA); ax.set_box_aspect((1., 1., 1.))
        ax.set_xlabel(r'$q_1$ (mm)', labelpad=-1); ax.set_ylabel(r'$q_2$ (mm)', labelpad=-1)
        ax.set_zlabel(r'$q_3$ (mm)', labelpad=-1)
        ax.tick_params(labelsize=10., pad=-2)
        ax = fig.add_subplot(grid[0, 1])
        V = _saved(record, 'world_V', 'V', 'sphere_world_V')
        U = _saved(record, 'world_U', 'U', 'sphere_world_U')
        clean_u = _saved(record, 'clean', 'world_clean', 'sphere_world_clean', 'sphere_clean')
        n = normal[0]/np.linalg.norm(normal[0])
        tangent = V[0]-(V[0]@n)*n
        tangent /= np.linalg.norm(tangent)
        for vector, c, label in [(V[0], '#0072B2', r'$v_t$'),
                                  (clean_u[0], '#D55E00', r'$P_{\mathcal X_t}v_t$')]:
            x, y = 1e3*(vector@tangent), 1e3*(vector@n)
            ax.annotate('', xy=(x, y), xytext=(0., 0.),
                        arrowprops={'arrowstyle': '->', 'color': c, 'linewidth': 1.5})
            ax.plot([], [], color=c, label=label)
        ax.scatter(1e3*(U[0]@tangent), 1e3*(U[0]@n), marker='o', s=18,
                   facecolors='white', edgecolors='#CC79A7', label=r'$u_t$')
        vectors=np.asarray([np.zeros(3),V[0],clean_u[0],U[0]])
        ax.update_datalim(1e3*np.column_stack((vectors@tangent,vectors@n)))
        ax.autoscale_view()
        ax.axhline(0., color=GREY, linewidth=.8)
        ax.set_xlabel('Tangential velocity (mm/s)'); ax.set_ylabel('Normal velocity (mm/s)')
        ax.legend(frameon=False, fontsize=10.); clean(ax)
    for i, ax in enumerate([ax for ax in fig.axes if ax.get_label() != '<colorbar>']):
        letter(ax, f'({chr(97+i)})')
    fig._final_selection=dict(episode=0,story=evidence.config('performance').get('story',{}),
        contact_condition='sphere' if evidence.example=='contact' else None)
    return [fig]


def _draw_contact_risk(ax, record):
    query = _saved(record, 'queries', 'unit_queries')
    risk = _saved(record, 'risks_full', 'risk_full')
    iso = _saved(record, 'risks_iso', 'risk_iso', 'risks_isotropic')
    norm2 = np.sum(abs(query)**2, axis=-1)
    angles = _optional(record, 'query_angles')
    family = _optional(record, 'query_family')
    if family is not None:
        selected = family == 1
        query, risk, iso, norm2 = query[selected], risk[selected], iso[selected], norm2[selected]
        if angles is not None:
            angles = angles[selected]
    if angles is None:
        angles = np.linspace(0., 2*np.pi, len(query), endpoint=False)
    ax.plot(angles, risk/norm2, color=color('mean'), label=r'$\overline P_t^+$')
    ax.plot(angles, iso/norm2, '--', color=color('iso'), label=r'$\overline P_t^{\mathrm{iso}}$')
    ax.set_xlabel(r'$\theta$ (rad)'); ax.set_ylabel(r'$v^\top(\overline P-\overline P^2)v/\|v\|^2$')
    ax.set(xticks=np.linspace(0., 2*np.pi, 5),
           xticklabels=[r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
    ax.legend(frameon=False, fontsize=10.); clean(ax)
    ax._final_selection=dict(query_family=1,normalization='Squared norm of saved query')


def posterior_anisotropy(evidence):
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.3), layout='constrained')
    grid = fig.add_gridspec(2,2,height_ratios=(1.25,1.))
    if evidence.example == 'contact':
        axes = [fig.add_subplot(grid[0, i], projection='3d') for i in range(2)]
        record = evidence.array('performance', 'anisotropy')
        A = _saved(record, 'normal_A')
        iso_A = _saved(record, 'iso_normal_A', 'isotropic_normal_A')
        logz = float(_saved(record, 'normal_logz'))
        iso_logz = float(_saved(record, 'iso_normal_logz', 'iso_logz'))
        first = bingham_sphere_mesh(A, 161, 321).nodes
        second = bingham_sphere_mesh(iso_A, 161, 321).nodes
        densities = [real_bingham_log_density(A, logz, first),
                     real_bingham_log_density(iso_A, iso_logz, second)]
        norm = shared_density_norm(densities)
        mode = _saved(record, 'mode_normal')
        for ax, nodes, density in zip(axes, [first, second], densities):
            density_sphere(ax, nodes, density, norm, coordinate='n', mode=mode)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DENSITY_CMAP),
                     ax=axes, label=r'$q([n])$', shrink=.64, pad=.015)
        risk_ax = fig.add_subplot(grid[1,:]); _draw_contact_risk(risk_ax, record)
    else:
        axes = [fig.add_subplot(grid[0, 0], projection='3d')]
        record = evidence.array('performance', 'stories', 'story')
        A = _saved(record, 'section_A', 'setup_A')
        frame = _saved(record, 'section_frame', 'setup_section_frame')
        restricted = frame.conj().T @ A @ frame
        mesh = bingham_sphere_mesh(restricted.real, 161, 321).nodes
        eigenvalues = _saved(record, 'section_eigenvalues', 'setup_eigenvalues')
        density = np.einsum('...i,ij,...j->...', mesh, restricted.real, mesh)-float(np.max(eigenvalues))
        norm = shared_density_norm([density])
        density_sphere(axes[0], mesh, density, norm, coordinate='x',
                       mode=np.array([1., 0., 0.]))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DENSITY_CMAP), ax=axes[0],
                     label=r'$q(U_3x)/q(\widehat{\mathcal X})$', shrink=.64, pad=.015)
        ax = fig.add_subplot(grid[0, 1])
        axes.append(ax)
        p = _saved(record, 'section_mean_eigenvalues', 'setup_mean_eigenvalues')
        pi = _saved(record, 'section_iso_mean_eigenvalues', 'setup_iso_mean_eigenvalues')
        ax.plot(np.arange(2, len(p)+1), p[1:], 'o-', color=color('mean'), label=r'$\overline P_t^+$')
        ax.plot(np.arange(2, len(pi)+1), pi[1:], '^--', color=color('iso'), label=r'$\overline P_t^{\mathrm{iso}}$')
        ax.set_yscale('log'); ax.set_xlabel('Complementary direction')
        ax.set_ylabel('Mean eigenvalue'); ax.legend(frameon=False, fontsize=10.); clean(ax)
        risk_ax = fig.add_subplot(grid[1,:])
        frame_data = _normalize(evidence.frame('performance', 'stationary'))
        frame_data = frame_data[frame_data.n.eq(64)]
        horizon = _column(frame_data, 'L', 't', 'observations')
        frame_data = frame_data[frame_data[horizon].eq(64)]
        value = _column(frame_data, 'heldout_ratio_db', 'output_oracle_db')
        full = frame_data[frame_data.method.eq('mean')].set_index('episode')[value]
        iso = frame_data[frame_data.method.isin(['iso', 'isotropic'])].set_index('episode')[value]
        indices = full.index.intersection(iso.index)
        delta = np.sort((iso.loc[indices]-full.loc[indices]).to_numpy())
        risk_ax.step(delta, np.arange(1, len(delta)+1)/len(delta), color=color('mean'), where='post')
        risk_ax.axvline(0., color=GREY, linestyle='--', linewidth=1.)
        risk_ax.set_xlabel('Isotropic minus full (dB)'); risk_ax.set_ylabel('Fraction of episodes')
        clean(risk_ax)
    for ax, tag in zip(axes+[risk_ax], ['(a)', '(b)', '(c)']):
        letter(ax, tag)
    fig._final_selection=(dict(condition=evidence.config('performance').get('anisotropy_input','static_concentrated'),
        episode=0,t=evidence.config('performance').get('anisotropy_time',16),query_family=1)
        if evidence.example=='contact' else dict(cohort='stationary',n=64,t=64,
        section='Real coefficient great sphere in the saved complex three-frame; not a normalized marginal'))
    figures=[fig]
    if evidence.example=='contact':
        figures.append(_constructed_anisotropy(evidence))
    return figures


def _constructed_anisotropy(evidence):
    """Retain the verified equal-distance-law counterexample as its own page."""
    from .simulation_validation import verify_component
    meta=verify_component(evidence.root,'paper','geometry')
    folder=evidence.root/'results/simulation_redesign/paper/geometry'
    manifest=folder/'manifest.json'
    evidence.inputs[str(manifest.relative_to(evidence.root))]=sha(manifest)
    for name in ('matrices.npz','directional.csv','distance_cdf.csv'):
        evidence.inputs[str((folder/name).relative_to(evidence.root))]=sha(folder/name)
    evidence.references['historical_geometry']=dict(config=meta['config'],sources=meta['sources'],
        checks=meta['checks'],manifest=str(manifest.relative_to(evidence.root)),
        tables={'scope':'Constructed deterministic identities, not independent experimental repetitions'})
    with np.load(folder/'matrices.npz',allow_pickle=False) as saved:
        parameters=saved['A'][:2];logz=saved['log_normalizer'][:2];modes=saved['mode'][:2]
    directional=pd.read_csv(folder/'directional.csv');cdf=pd.read_csv(folder/'distance_cdf.csv')
    fig=plt.figure(figsize=(FIGURE_WIDTH,5.5),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=(1.2,1.))
    spheres=[fig.add_subplot(grid[0,i],projection='3d') for i in range(2)]
    meshes=[bingham_sphere_mesh(A, 161, 321).nodes for A in parameters]
    densities=[real_bingham_log_density(A,z,mesh) for A,z,mesh in zip(parameters,logz,meshes)]
    norm=shared_density_norm(densities)
    for ax,mesh,q,mode in zip(spheres,meshes,densities,modes):density_sphere(ax,mesh,q,norm,mode=mode)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DENSITY_CMAP),ax=spheres,label=r'$q([n])$',shrink=.7,pad=.015)
    lower=[fig.add_subplot(grid[1,i]) for i in range(2)]
    for case,c,style in [(0,color('mean'),'-'),(1,color('iso'),'--')]:
        g=cdf[cdf.case.eq(case)];label=r'$q_{A^{('+str(case+1)+r')}}$'
        lower[0].plot(g.radius,g.cdf,color=c,linestyle=style,label=label)
        g=directional[directional.case.eq(case)&directional.angle.notna()]
        lower[1].plot(g.angle,g.risk,color=c,linestyle=style,label=label)
    lower[0].set(xlabel='Chordal distance',ylabel='Cumulative probability')
    lower[1].set(xlabel=r'$\theta$ (rad)',ylabel=r'$v^\top(\overline P-\overline P^2)v$')
    for ax in lower:ax.legend(frameon=False,fontsize=10.);clean(ax)
    for ax,tag in zip(spheres+lower,['(a)','(b)','(c)','(d)']):letter(ax,tag)
    fig._final_selection=dict(scope='Constructed swapped-eigenvalue identity, not data-generated performance',
        normal_parameters=parameters.tolist(),same_mode=True,same_entire_centered_chordal_distance_law=True)
    fig._final_repetition_unit='Deterministic continuous integration and identities; no independent episode interpretation'
    return fig


def _prediction_rows(evidence):
    frame = _normalize(evidence.frame('prediction', 'prediction_adaptation', 'adaptation'))
    if 'split' in frame and frame.split.eq('test').any():
        frame = frame[frame.split.eq('test')]
    if 'control' in frame:
        frame = frame[frame.control.eq('mean')]
    return frame


def _condition_groups(frame):
    if 'condition' not in frame:
        return [('matched', frame)]
    order = ['noise_only', 'motion_only', 'power_only', 'noise', 'motion', 'power']
    names = list(frame.condition.drop_duplicates())
    ordered = [name for name in order if name in names]+[name for name in names if name not in order]
    return [(name, frame[frame.condition.eq(name)]) for name in ordered]


def _draw_variance_time(ax, frame, example, signal=False, legend=True):
    time = _column(frame, 'time_s', 't')
    true = 'tau2_true' if signal else 'sigma2_true'
    fitted = 'tau2_estimate' if signal else 'sigma2_estimate'
    scale = 1e6 if example == 'contact' and not signal else 1.
    if true not in frame or fitted not in frame:
        raise KeyError(f'Causal variance trajectories need {true} and {fitted}')
    for method in ['known', 'exact_known', 'known_forgetting', 'misspecified', 'learned']:
        group = frame[frame.method.eq(method)]
        if group.empty:
            continue
        grouped = group.groupby(time)[fitted]
        points = grouped.median()
        c = color(method)
        ax.step(points.index, points*scale, where='post', color=c,
                linestyle='--' if method == 'misspecified' else '-',
                marker=marker(method), label=method_label(method))
        if method == 'learned':
            low, high = grouped.quantile(.25), grouped.quantile(.75)
            ax.fill_between(points.index, low*scale, high*scale, color=c, alpha=.13, step='post')
    truth = frame.groupby(time)[true].first()
    ax.step(truth.index, truth*scale, where='post', color=BLACK, linestyle=':',
            linewidth=1.35, label=r'$\tau_t^2$' if signal else r'$\sigma_t^2$')
    differences = np.flatnonzero(np.abs(np.diff(truth.to_numpy())) > 1e-12)
    for index in differences:
        ax.axvline(truth.index[index+1], color=GREY, linestyle=':', linewidth=.7)
    ax.set_xlabel('Time (s)' if time == 'time_s' else 'Number of observations')
    ax.set_ylabel(r'Signal variance $\tau^2$' if signal else
        r'Noise variance $((\mathrm{mm}/\mathrm{s})^2)$' if example == 'contact'
        else r'Noise variance $\sigma^2$')
    if legend:ax.legend(frameon=False, fontsize=10., ncol=1,loc='upper left')
    clean(ax)


def _draw_future_time(ax, frame, example, scores=False, legend=True):
    time = _column(frame, 'time_s', 't')
    if scores:
        endpoint = _finite_column(frame, 'prequential_score', 'mixture_log_score')
    elif example == 'beamforming':
        endpoint = _finite_column(frame, 'heldout_output_oracle_db', 'heldout_ratio_db')
    else:
        endpoint = _finite_column(frame, 'future_task_loss', 'observed_response_mse')
    scale = 1e6 if example == 'contact' and not scores else 1.
    for method in ['known', 'exact_known', 'known_forgetting', 'misspecified', 'learned']:
        group = frame[frame.method.eq(method)]
        if group.empty:
            continue
        rows = []
        for t, g in group.groupby(time, sort=True):
            values = (g.groupby('episode')[endpoint].mean()*scale).dropna()
            if not len(values):
                continue
            mean, low, high = _mean_interval(values)
            rows.append((t, mean, low, high))
        if not rows:
            continue
        points = np.asarray(rows)
        ax.plot(points[:, 0], points[:, 1], marker=marker(method), color=color(method),
                linestyle='--' if method == 'misspecified' else '-',
                label=method_label(method))
        ax.fill_between(points[:, 0], points[:, 2], points[:, 3], color=color(method), alpha=.09)
    ax.set_xlabel('Time (s)' if time == 'time_s' else 'Number of observations')
    ax.set_ylabel('Predictive log score' if scores else
        'Output power / oracle power (dB)' if example == 'beamforming' else
        r'Observed future response MSE $((\mathrm{mm}/\mathrm{s})^2)$')
    if legend:ax.legend(frameon=False, fontsize=10.)
    clean(ax)


def noise_adaptation(evidence):
    frame = _prediction_rows(evidence)
    figures = []
    for condition, group in _condition_groups(frame):
        if evidence.example == 'beamforming':
            fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.4), layout='constrained')
            _draw_variance_time(axes[0, 0], group, evidence.example,legend=False)
            _draw_variance_time(axes[0, 1], group, evidence.example, signal=True,legend=False)
            _draw_future_time(axes[1, 0], group, evidence.example,legend=False)
            _draw_future_time(axes[1, 1], group, evidence.example, scores=True,legend=False)
            flattened = axes.ravel()
        else:
            fig = plt.figure(figsize=(FIGURE_WIDTH, 5.1), layout='constrained')
            grid = fig.add_gridspec(2, 2)
            flattened = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
            _draw_variance_time(flattened[0], group, evidence.example,legend=False)
            _draw_future_time(flattened[1], group, evidence.example,legend=False)
            _draw_future_time(flattened[2], group, evidence.example, scores=True,legend=False)
        for ax, tag in zip(flattened, ['(a)', '(b)', '(c)', '(d)']):
            letter(ax, tag)
        handles={}
        for ax in flattened:
            for handle,label in zip(*ax.get_legend_handles_labels()):handles.setdefault(label,handle)
        fig.legend(list(handles.values()),list(handles),loc='outside lower center',ncol=3,frameon=False,fontsize=10.)
        fig._final_condition = str(condition)
        figures.append(fig)
    if evidence.example == 'contact':
        figures.extend(_contact_adaptive_beliefs(evidence))
    return figures


def _contact_adaptive_beliefs(evidence):
    """Render pre-correction beliefs recorded on one declared causal episode."""
    story=evidence.array('prediction','stories')
    prefix='contact_noise_change_'
    checkpoints=(32,96,192)
    parameters=[];densities=[];meshes=[];risks=[]
    for t in checkpoints:
        name=prefix+'t'+str(t)+'_'
        if not bool(np.asarray(_saved(story,name+'mean_numerical_ok')).item()):
            raise ValueError('Adaptive belief display requires a resolved saved normalizer')
        plane_A=_saved(story,name+'A')
        normal_A=-plane_A
        normal_logz=float(_saved(story,name+'log_normalizer'))-float(np.trace(plane_A))
        mesh=bingham_sphere_mesh(normal_A, 161, 321).nodes
        parameters.append(normal_A);meshes.append(mesh)
        densities.append(real_bingham_log_density(normal_A,normal_logz,mesh))
        M=_saved(story,name+'M');v=_saved(story,prefix+'V')[t-1]
        risks.append(float(v@(M-M@M)@v)*1e6)
    norm=shared_density_norm(densities)
    fig=plt.figure(figsize=(FIGURE_WIDTH,3.6),layout='constrained')
    spheres=[fig.add_subplot(1,2,i+1,projection='3d') for i in range(2)]
    for ax,mesh,q,tag in zip(spheres,meshes[:2],densities[:2],['(a)','(b)']):
        density_sphere(ax,mesh,q,norm,coordinate='n');letter(ax,tag)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DENSITY_CMAP),ax=spheres,
        label=r'$q_t^-([n])$',shrink=.7,pad=.07)
    first=fig
    fig=plt.figure(figsize=(FIGURE_WIDTH,4.1),layout='constrained')
    grid=fig.add_gridspec(1,2,width_ratios=(1.15,1.))
    sphere=fig.add_subplot(grid[0,0],projection='3d')
    density_sphere(sphere,meshes[2],densities[2],norm,coordinate='n');letter(sphere,'(a)')
    ax=fig.add_subplot(grid[0,1]);dt=float(evidence.config('prediction')['primary_contact_dt'])
    ax.plot((np.asarray(checkpoints)-1)*dt,risks,'o-',color=color('mean'))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Response risk $((\mathrm{mm}/\mathrm{s})^2)$')
    ax.yaxis.tick_right();ax.yaxis.set_label_position('right')
    clean(ax);letter(ax,'(b)')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=DENSITY_CMAP),ax=[sphere,ax],
        label=r'$q_t^-([n])$',orientation='horizontal',shrink=.65,fraction=.06,pad=.08)
    for page,times in [(first,[32,96]),(fig,[192])]:
        page._final_condition='contact_noise_change_individual_beliefs'
        page._final_selection=dict(condition='noise_change',episode=0,checkpoints=times,
            shared_density_scale_checkpoints=list(checkpoints),
            timing='Pre-correction A_t^- and M_t^-; before assimilating observation t',
            coordinates='Measured wrist frame at each checkpoint',
            parameter_conversion='A_normal=-A_plane; psi_normal=psi_plane-trace(A_plane)',
            response_risk='v_t^T(M_t^--(M_t^-)^2)v_t for the recorded physical input at t')
        page._final_repetition_unit='One predeclared test episode; individual posterior densities, no repeated-experiment calibration claim'
    return [first,fig]


def _reference_frame(evidence, component, name):
    """Read-only use of a verified historical mathematical reference cohort."""
    from .implementation_validation import verify_component
    meta = verify_component(evidence.root, 'paper', component)
    folder = evidence.root/'results/implementation_audit/paper'/component
    path, manifest = folder/(name+'.csv'), folder/'manifest.json'
    evidence.inputs[str(path.relative_to(evidence.root))] = sha(path)
    evidence.inputs[str(manifest.relative_to(evidence.root))] = sha(manifest)
    frame = pd.read_csv(path)
    reference = evidence.references.setdefault('historical_'+component, dict(
        config=meta.get('config', {}), sources=meta['sources'], checks=meta.get('checks', {}),
        manifest=str(manifest.relative_to(evidence.root)), tables={}))
    reference['tables'][name] = dict(rows=len(frame),
        distinct_trial=int(frame.trial.nunique()) if 'trial' in frame else None)
    return frame


def variance_estimation(evidence):
    iterations = _reference_frame(evidence, 'em', 'iterations')
    fits = _reference_frame(evidence, 'em', 'fits')
    model = ('complex_marginal_rank_one' if evidence.example == 'beamforming'
             else 'conditional_contact_plane')
    iterations = iterations[iterations.model.eq(model)]
    fits = fits[fits.model.eq(model)]
    trial = int(iterations.trial.min())
    story = iterations[iterations.trial.eq(trial)]
    columns = 2 if evidence.example == 'beamforming' else 3
    fig, axes = plt.subplots(1, columns, figsize=(FIGURE_WIDTH, 2.8), layout='constrained')
    starts = list(story.start.drop_duplicates())
    palette = [color('mean'), color('iso'), color('mode'), GREY]
    labels = []
    for index, (start, group) in enumerate(story.groupby('start', sort=False)):
        group = group.sort_values('iteration')
        initial = group.iloc[0]
        label = (r'$\sigma_0^2='+f'{initial.s:g}'+r',\ \tau_0^2='+f'{initial.tau2:g}'+r'$'
                 if evidence.example == 'beamforming' else r'$\sigma_0^2='+f'{initial.s:g}'+r'$')
        labels.append(label)
        axes[0].plot(group.iteration, group.log_evidence-group.log_evidence.iloc[0],
                     'o-', color=palette[index % len(palette)], markersize=2.5, label=label)
    axes[0].set_xlabel('EM iteration'); axes[0].set_ylabel('Log-evidence increase')
    axes[0].legend(frameon=False, fontsize=10.); clean(axes[0])
    data, ticks, positions, boxcolors = [], [], [], []
    for index, start in enumerate(starts):
        group = fits[fits.start.eq(start)]
        data.append((group.s/group.true_s).to_numpy())
        positions.append(index-.12 if evidence.example == 'beamforming' else index)
        ticks.append(index); boxcolors.append(color('mean'))
        if evidence.example == 'beamforming':
            data.append((group.tau2/group.true_tau2).to_numpy())
            positions.append(index+.12); boxcolors.append(color('iso'))
    boxes = axes[1].boxplot(data, positions=positions, widths=.20, patch_artist=True,
        manage_ticks=False, medianprops={'color': 'white'}, flierprops={'markersize': 1.8, 'alpha': .4})
    for box, c in zip(boxes['boxes'], boxcolors):
        box.set_facecolor(c)
    axes[1].axhline(1., linestyle='--', color=BLACK, linewidth=1.)
    axes[1].set(xticks=ticks, xticklabels=[str(i+1) for i in ticks])
    axes[1].set_xlabel('Initial-value index'); axes[1].set_ylabel('Fitted / true variance')
    handles = [Patch(facecolor=color('mean'), label=r'$\widehat\sigma^2/\sigma^2$')]
    if evidence.example == 'beamforming':
        handles.append(Patch(facecolor=color('iso'), label=r'$\widehat\tau^2/\tau^2$'))
    axes[1].legend(handles=handles, frameon=False, fontsize=10.); clean(axes[1], 'y')
    if evidence.example == 'contact':
        residual = _reference_frame(evidence, 'em', 'residuals')
        residual = residual[residual.model.eq(model)]
        lookup = fits[['trial', 'start', 'true_s']].drop_duplicates()
        residual = residual.merge(lookup, on=['trial', 'start'], how='left')
        x = residual.mean_only_update/residual.true_s
        y = residual.expected_update/residual.true_s
        axes[2].scatter(x, y, color=color('mean'), s=10, alpha=.45)
        _identity(axes[2], x, y)
        axes[2].set_xlabel(r'Residual-only variance$/\sigma^2$')
        axes[2].set_ylabel(r'Uncertainty-corrected variance$/\sigma^2$')
        clean(axes[2])
    for ax, tag in zip(axes, ['(a)', '(b)', '(c)']):
        letter(ax, tag)
    return [fig]


def predictive_calibration(evidence):
    frame = _normalize(evidence.frame('prediction', 'prediction_calibration_summary',
                                      'calibration_summary', 'prediction_calibration', 'calibration'))
    if 'split' in frame and frame.split.eq('test').any():
        frame = frame[frame.split.eq('test')]
    if 'stage' in frame and frame.stage.eq('test').any():
        frame = frame[frame.stage.eq('test')]
    target = _column(frame, 'target', 'kind')
    regime = _column(frame, 'regime', 'condition')
    figures = []
    grouping = [regime]+(['t'] if 't' in frame else [])
    for value, group in frame.groupby(grouping, sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 3.3), layout='constrained')
        for ax, kind, tag in zip(axes, ['state_hpd', 'clean_response', 'noisy_observation'],
                                 ['(a)', '(b)', '(c)']):
            subset = group[group[target].eq(kind)]
            if subset.empty and kind == 'state_hpd':
                subset = group[group[target].eq('state_ball')]
            for method in ['exact_known', 'known', 'known_forgetting', 'misspecified', 'learned']:
                selected = subset[subset.method.eq(method)]
                if selected.empty:
                    continue
                xs, ys, lows, highs = [], [], [], []
                for probability, g in selected.groupby('nominal', sort=True):
                    if 'coverage' in g:
                        if len(g) != 1:
                            raise ValueError('Calibration summary has duplicated probability cells')
                        row = g.iloc[0]
                        inclusion = row.coverage
                        lower, upper = row['lower'], row['upper']
                    else:
                        from scipy.stats import beta as beta_distribution
                        per_episode = g.groupby('episode').covered.mean()
                        if not np.isin(per_episode, [0., 1.]).all():
                            raise ValueError('Binomial calibration requires one binary inclusion per episode/target')
                        count, successes = len(per_episode), int(per_episode.sum())
                        inclusion = successes/count
                        lower = 0. if successes == 0 else beta_distribution.ppf(.025, successes, count-successes+1)
                        upper = 1. if successes == count else beta_distribution.ppf(.975, successes+1, count-successes)
                    xs.append(probability); ys.append(inclusion); lows.append(lower); highs.append(upper)
                ax.plot(xs, ys, marker=marker(method), color=color(method),
                        linestyle='--' if method == 'misspecified' else '-', label=method_label(method))
                ax.errorbar(xs,ys,yerr=[np.asarray(ys)-np.asarray(lows),np.asarray(highs)-np.asarray(ys)],
                    fmt='none',ecolor=color(method),alpha=.65,elinewidth=.7,capsize=2.)
            ax.plot([0., 1.], [0., 1.], '--', color=GREY, linewidth=1.)
            ax.set(xlim=(.43,1.02),ylim=(-.01,1.03),xticks=[.5,.8,.95])
            ax.set_xlabel('Nominal probability'); ax.set_ylabel('Observed inclusion')

            clean(ax); letter(ax, tag)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=10.)
        fig._final_condition = str(value[0] if isinstance(value, tuple) else value)
        fig._final_selection = dict(zip(grouping,value if isinstance(value,tuple) else [value]))
        fig._final_selection['diagnostic']='inclusion'
        figures.append(fig)
    figures.extend(_calibration_widths(evidence))
    return figures


def _calibration_widths(evidence):
    """Show the recorded 95% report sizes without treating HPD energy as size."""
    frame=_normalize(evidence.frame('prediction','calibration'))
    frame=frame[frame.split.eq('test')&frame.cohort.eq('calibration')
        &frame.stress.eq('matched')&frame.t.eq(64)&np.isclose(frame.nominal,.95)]
    methods=['exact_known','known_forgetting','misspecified','learned']
    count=int(evidence.config('prediction')['calibration_test_episodes'])
    quantities=[('state_ball','Chordal radius'),
        ('clean_response',r'Interval width of $q^H Pq$' if evidence.example=='beamforming'
            else r'Interval width of $e_1^T Pq$'),
        ('noisy_observation',r'Interval width of $\operatorname{Re}(u_{t,1})$')]
    figures=[]
    for regime in ['prior_predictive','fixed_state','drifting']:
        group=frame[frame.condition.eq(regime)]
        fig,axes=plt.subplots(3,1,figsize=(FIGURE_WIDTH,6.8),layout='constrained')
        for ax,(kind,xlabel),tag in zip(axes,quantities,['(a)','(b)','(c)']):
            arrays=[]
            for method in methods:
                rows=group[group.kind.eq(kind)&group.method.eq(method)].sort_values('episode')
                if len(rows)!=count or rows.episode.nunique()!=count or not rows.numerical_ok.eq(True).all():
                    raise ValueError('Report-width cohort is missing, duplicated or unresolved: '+regime+'/'+kind+'/'+method)
                lower=rows.lower.to_numpy();upper=rows.upper.to_numpy()
                width=upper-lower
                if not np.isfinite(width).all() or np.any(width<0) or not np.allclose(width,rows.width,rtol=1e-12,atol=1e-12):
                    raise ValueError('Saved interval endpoints and width disagree')
                if kind=='state_ball':
                    if not np.allclose(lower,0.,atol=1e-14) or np.any(upper>1.+1e-10):
                        raise ValueError('The rank/codimension-one state ball must use squared chordal distance')
                    width=np.sqrt(upper)
                arrays.append(width)
            boxes=_boxplot(ax,arrays,orientation='horizontal',
                tick_labels=[method_label(method) for method in methods],patch_artist=True,
                medianprops={'color':'white','linewidth':1.2},
                flierprops={'markersize':1.8,'alpha':.35})
            for box,method in zip(boxes['boxes'],methods):box.set_facecolor(color(method))
            ax.set_xlabel(xlabel);ax.set_xlim(left=0.)
            clean(ax,'x');letter(ax,tag)
        fig._final_condition=regime
        fig._final_selection=dict(condition=regime,t=64,nominal=.95,diagnostic='width',
            split='test',cohort='calibration',stress='matched',methods=methods,
            targets=[kind for kind,_ in quantities],episodes_per_method=count,
            state_size='Per-episode chordal radius sqrt(upper), where upper is the squared-distance state-ball threshold',
            scalar_size='Per-episode upper minus lower; n=3 reliability-fixture units',
            distribution='Median, interquartile box, 1.5-IQR whiskers and every episode outlier; not a confidence interval or posterior distribution over width')
        fig._final_repetition_unit='Independent test episode; each value is one recorded 95% report size at t=64'
        figures.append(fig)
    return figures


def predictive_scores(evidence):
    frame = _normalize(evidence.frame('prediction', 'prediction_scores', 'scores',
                                      'prediction_adaptation', 'adaptation'))
    if 'split' in frame and frame.split.eq('test').any():
        frame = frame[frame.split.eq('test')]
    if 'stage' in frame and frame.stage.eq('test').any():
        frame = frame[frame.stage.eq('test')]
    if 'control' in frame:
        frame = frame[frame.control.eq('mean')]
    if 'cohort' in frame:
        frame = frame[frame.cohort.eq('calibration')]
    if 'stress' in frame:
        frame = frame[frame.stress.eq('matched')]
    if frame.empty:
        raise ValueError('Predictive score comparison needs matched calibration-cohort observations')
    if 'method' in frame and frame.method.eq('learned').any():
        frame = frame[frame.method.eq('learned')]
    gaussian = _column(frame, 'gaussian_score', 'gaussian_log_score')
    mixture = _column(frame, 'prequential_score', 'mixture_log_score', 'mixture_score')
    predicted = next((name for name in ('risk_predicted', 'predicted_risk', 'posterior_risk')
                      if name in frame), None)
    realized = next((name for name in ('risk_realized', 'realized_risk', 'response_loss', 'future_task_loss')
                     if name in frame), None)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.8), layout='constrained')
    keys = [name for name in ('condition', 'regime', 'episode') if name in frame]
    columns = [gaussian, mixture]+[name for name in (predicted, realized) if name is not None]
    episode = frame.groupby(keys, sort=False)[list(dict.fromkeys(columns))].mean().reset_index()
    axes[0].scatter(episode[gaussian], episode[mixture], s=10, alpha=.45, color=color('mean'))
    _identity(axes[0], episode[gaussian], episode[mixture])
    axes[0].set_xlabel('Gaussian predictive log score')
    axes[0].set_ylabel('Mixture predictive log score')
    clean(axes[0])
    if predicted is None or realized is None:
        raise KeyError('Predictive scores need saved predicted/realized clean-response risks')
    axes[1].scatter(episode[predicted], episode[realized], s=10, alpha=.45, color=color('mean'))
    _identity(axes[1], episode[predicted], episode[realized])
    axes[1].set_xlabel('Predicted total response risk')
    axes[1].set_ylabel('Realized squared response error')
    clean(axes[1])
    for ax, tag in zip(axes, ['(a)', '(b)']):
        letter(ax, tag)
    fig._final_selection=dict(cohort='calibration',stress='matched',method='learned',
        averaging='Reporting times averaged within episode and regime; regimes remain distinct points')
    return [fig]


def model_mismatch(evidence):
    frame = _normalize(evidence.frame('prediction', 'prediction_mismatch', 'mismatch'))
    if 'method' in frame and frame.method.eq('learned').any():
        frame = frame[frame.method.eq('learned')]
    if 'control' in frame:
        frame = frame[frame.control.eq('mean')]
    stress = _column(frame, 'stress', 'mismatch', 'condition')
    if evidence.example == 'beamforming':
        numerator = _finite_column(frame, 'task_loss_mean')
        denominator = _finite_column(frame, 'oracle_power')
        eligible = np.isfinite(frame[numerator]) & np.isfinite(frame[denominator]) & frame[denominator].gt(0) & frame[numerator].gt(0)
        if not eligible.any():
            raise ValueError('Mismatch power ratio has no positive saved oracle/output pairs')
        endpoint = 'observed_oracle_ratio_db'
        frame[endpoint] = np.nan
        frame.loc[eligible, endpoint] = 10*np.log10(frame.loc[eligible,numerator]/frame.loc[eligible,denominator])
    else:
        endpoint = _finite_column(frame, 'response_loss_mean', 'response_loss')
    score = _column(frame, 'prequential_score', 'mixture_log_score', 'logscore')
    names = list(frame[stress].drop_duplicates())
    labels = {
        'matched': 'Matched',
        'desired_contamination': 'Desired signal',
        'unequal_rank_two_powers': 'Unequal powers',
        'correlated_noise': 'Correlated noise',
        'anisotropic_noise': 'Anisotropic noise',
        'fast_drift': 'Faster motion',
        'additional_interference': 'Additional interference',
        'known_resistance_correct_noise': 'Known resistance',
        'known_resistance_raw_isotropic': 'Wrong noise covariance',
        'anisotropic_damping': 'Wrong resistance',
        'wrong_resistance': 'Wrong resistance',
        'known_anisotropic_resistance': 'Known resistance',
        'friction': 'Friction and sticking',
        'contact_loss': 'Contact loss',
        'position_differences': 'Position differences',
    }
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.0), layout='constrained')
    for ax, column, scale in [(axes[0], endpoint, 1e6 if evidence.example == 'contact' else 1.),
                               (axes[1], score, 1.)]:
        arrays = [frame[frame[stress].eq(name)].groupby('episode')[column].mean().to_numpy()*scale
                  for name in names]
        boxes = _boxplot(ax,arrays, orientation='horizontal',
            tick_labels=[labels.get(name, str(name).replace('_', ' ')) for name in names],
            patch_artist=True, medianprops={'color': 'white'},
            flierprops={'markersize': 1.5, 'alpha': .35})
        for index, box in enumerate(boxes['boxes']):
            box.set_facecolor([color('mean'), color('mode'), color('iso'), GREY][index % 4])
        clean(ax, 'x')
    axes[0].set_xlabel('Output power / oracle power (dB)' if evidence.example == 'beamforming'
                       else r'Response MSE $((\mathrm{mm}/\mathrm{s})^2)$')
    axes[1].set_xlabel('Predictive log score')
    for ax, tag in zip(axes, ['(a)', '(b)']):
        letter(ax, tag)
    fig._final_selection=dict(cohort='mismatch',method='learned',control='mean',
        displayed_stresses=names,endpoint=endpoint)
    return [fig]


def additional_comparisons(evidence):
    """Full paired grids without seven overlapping trajectory curves."""
    field='stationary' if evidence.example=='beamforming' else 'metrics'
    frame=_normalize(evidence.frame('performance',field))
    if 'split' in frame:frame=frame[frame.split.eq('test')]
    if evidence.example=='contact':frame=frame[frame.endpoint.eq('checkpoint')]
    endpoint=_finite_column(frame,'heldout_ratio_db') if evidence.example=='beamforming' else _finite_column(frame,'response_mse')
    axis='t';scale=1. if evidence.example=='beamforming' else 1e6
    groups=list(frame.groupby('n',sort=True)) if evidence.example=='beamforming' else list(frame.groupby('condition',sort=False))
    prepared=[];maximum=0.
    for identifier,group in groups:
        pair=group.pivot_table(index=['episode',axis],columns='method',values=endpoint,aggfunc='mean')
        methods=[m for m in ['mode','iso','mode_loaded','scm_loaded','great_window','great_ew','lw','regularized','mean_identity','mode_identity','oracle'] if m in pair]
        methods.extend(m for m in pair.columns if m not in methods and m!='mean')
        times=sorted(group[axis].unique());gain=np.full((len(methods),len(times)),np.nan);lower=gain.copy();upper=gain.copy()
        for i,method in enumerate(methods):
            for j,time in enumerate(times):
                values=pair.xs(time,level=axis)[[method,'mean']].dropna()
                if len(values):gain[i,j],lower[i,j],upper[i,j]=_mean_interval(scale*(values[method]-values['mean']))
        finite=np.abs(gain[np.isfinite(gain)])
        maximum=max(maximum,float(finite.max()) if finite.size else 0.)
        prepared.append((identifier,methods,times,gain,lower,upper))
    norm=colors.TwoSlopeNorm(vmin=-max(maximum,1e-8),vcenter=0.,vmax=max(maximum,1e-8))
    figures=[]
    from matplotlib.patches import Rectangle
    for identifier,methods,times,gain,lower,upper in prepared:
        fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,4.4),layout='constrained')
        image=ax.imshow(gain,aspect='auto',cmap='RdBu',norm=norm,interpolation='nearest',rasterized=True)
        for i in range(len(methods)):
            for j in range(len(times)):
                if np.isfinite(lower[i,j]) and lower[i,j]<=0<=upper[i,j]:
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='///',edgecolor='#70777D',linewidth=0.))
        labels=[]
        for method in methods:
            labels.append(r'$\overline P_t^+,G_t=I$' if method=='mean_identity' else r'$P_{\widehat{\mathcal X}_t},G_t=I$' if method=='mode_identity' else method_label(method,evidence.example))
        xlabels=[f'{t/identifier:g}' for t in times] if evidence.example=='beamforming' else [str(t) for t in times]
        ax.set(xticks=np.arange(len(times)),xticklabels=xlabels,yticks=np.arange(len(methods)),yticklabels=labels,
            xlabel=r'Observations per sensor $L/n$' if evidence.example=='beamforming' else 'Number of observations')
        fig.colorbar(image,ax=ax,label='Paired loss difference (dB)' if evidence.example=='beamforming' else r'$\mathrm{MSE}_{\mathrm{ref}}-\mathrm{MSE}_{\mathrm{B}}$ $((\mathrm{mm}/\mathrm{s})^2)$',shrink=.85)
        ax.legend(handles=[Patch(facecolor='none',hatch='///',edgecolor=GREY,label='Interval contains zero')],frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=10.)
        fig._final_selection={'n':int(identifier)} if evidence.example=='beamforming' else {'condition':str(identifier)}
        fig._final_selection.update(methods=methods,reporting_times=[int(t) for t in times],paired_interval='Pointwise Student95% interval of independent-episode differences; not simultaneous')
        figures.append(fig)
    if evidence.example=='beamforming':
        moving=_normalize(evidence.frame('performance','moving_episodes','episode_metrics',optional=True))
        if not moving.empty:
            for condition in moving.condition.drop_duplicates():
                fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,3.5),layout='constrained')
                _draw_paired_tasks(ax,evidence,all_references=True,condition=condition)
                fig._final_condition=condition;figures.append(fig)
    return figures


def _computation_summary(evidence):
    return _normalize(evidence.frame('computation', 'summary'))


def _draw_cost_tradeoff(ax, evidence):
    cost=_computation_summary(evidence)
    cost=cost[cost.case.eq('task_stationary')]
    if cost.empty:raise ValueError('Matching statistical-state task timing cohort is missing')
    frame=_normalize(evidence.frame('performance','stationary'))
    horizon=_column(frame,'L','t','observations')
    frame=frame[frame[horizon].eq(frame.n) & frame.episode.eq(0)]
    for method in ('mean','mode','scm_loaded','great_window'):
        x=[];y=[]
        for n,g in cost[cost.method.eq(method)].groupby('n',sort=True):
            g=g[g.action_cadence.eq(n)]
            target=frame[frame.n.eq(n) & frame.method.eq('mean')]
            if g.empty or target.empty:continue
            oracle=target.oracle_power.dropna().unique()
            if len(oracle)!=1 or oracle[0]<=0:raise ValueError('One saved episode-0 oracle power is required')
            loss=float(g.task_loss.mean())
            if not np.isfinite(loss) or loss<=0:continue
            x.append(float(g.median_ms.median()));y.append(10*np.log10(loss/oracle[0]))
        if x:ax.plot(x,y,marker=marker(method),color=color(method),label=method_label(method,evidence.example))
    ax.set(xscale='log',xlabel='Episode computation time (ms)',ylabel='Output power / oracle power (dB)')
    clean(ax)
    ax._final_selection=dict(timing_case='task_stationary',performance_episode=0,
        task_loss='True-covariance loss of the same predeclared timing episode, not a population average',
        timing_repetitions='Repeated clocks on one saved episode')


def computation_time(evidence):
    summary = _computation_summary(evidence)
    memory = _normalize(evidence.frame('computation', 'memory'))
    fig=plt.figure(figsize=(FIGURE_WIDTH,6.2),layout='constrained')
    grid=fig.add_gridspec(2,1)
    memory_fig,memory_ax=plt.subplots(figsize=(FIGURE_WIDTH,3.3),layout='constrained')
    axes=[fig.add_subplot(grid[0]),fig.add_subplot(grid[1]),memory_ax]
    cadence = summary[summary.case.eq('data_T64')]
    if cadence.empty:raise ValueError('The saved data_T64 cadence cohort is missing')
    elapsed = _column(cadence, 'wall_median_ms', 'median_ms')
    for method, interval, style in [('mean', 1, '-'), ('mode', 1, '--'),
                                    ('mean', 16, '-.'), ('mode', 16, ':')]:
        g = cadence[cadence.method.eq(method)]
        if 'report_every' in g:g=g[g.report_every.eq(0)]
        if 'action_cadence' in g:
            g = g[g.action_cadence.eq(interval)]
        if g.empty:
            continue
        g = g.sort_values('n')
        label=method_label(method)+(', every observation' if interval==1 else ', every '+str(interval)+' observations')
        axes[0].plot(g.n, g[elapsed], marker=marker(method), linestyle=style,
                     color=color(method), label=label)
    axes[0].set_xscale('log', base=2); axes[0].set_yscale('log')
    axes[0].set_xlabel('Ambient dimension' if evidence.example == 'contact' else 'Number of sensors')
    axes[0].set_ylabel('Time for 64 observations (ms)')
    axes[0].legend(frameon=False,fontsize=10.,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02));clean(axes[0])
    n = 3 if evidence.example=='contact' else 32 if 32 in set(cadence.n) else int(cadence.n.max())
    group = cadence[cadence.n.eq(n)]
    if group.method.eq('adaptive_mean').any():
        group = group[group.method.isin(['mean','mean_report','adaptive_mean'])]
        group=group[(group.method.ne('mean')) | group.action_cadence.isin([8,32])]
    else:
        group = group[group.method.eq('mean')]
    phase_names = [('correction', 'Parameter update', color('mean')),
                   ('action', 'Task decision', color('mode')),
                   ('report', 'Uncertainty report', color('iso')),
                   ('fitting', 'EM', '#009E73'),
                   ('overhead', 'Other', '#B8BEC4')]
    group = group.sort_values(['method', 'action_cadence'])
    positions = np.arange(len(group))
    bottom = np.zeros(len(group))
    for key, label, c in phase_names:
        column = next((name for name in (key+'_median_ms', key+'_ms', key+'_seconds')
                       if name in group), None)
        if column is None:
            continue
        values = group[column].to_numpy()*(1000 if column.endswith('_seconds') else 1.)
        axes[1].barh(positions,values,left=bottom,color=c,height=.7,label=label)
        bottom += values
    labels=[]
    for r in group.to_dict('records'):
        name=method_label('mean')
        if r['method']=='mean_report':name+=' + uncertainty'
        if r['method']=='adaptive_mean':name+=' + EM ('+str(int(r['em_iterations']))+' iterations)'
        labels.append(name+'\ndecisions every '+str(int(r['action_cadence']))+' obs.')
    axes[1].set(yticks=positions,yticklabels=labels)
    axes[1].set_xlabel('Time for 64 observations (ms)'); axes[1].set_ylabel('')
    fig.legend(*axes[1].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=10.)
    clean(axes[1], 'x')
    if 'case' in memory:
        g = memory[memory.case.eq('data_T64')]
        if not g.empty:
            memory = g
    if 'method' in memory:memory=memory[memory.method.eq('mean')]
    if 'report_every' in memory:memory=memory[memory.report_every.eq(0)]
    if 'action_cadence' in memory:
        recorded=sorted(memory.action_cadence.dropna().unique())
        if len(recorded)!=1:raise ValueError('Memory plot needs one declared saved action cadence')
        memory=memory[memory.action_cadence.eq(recorded[0])]
        axes[2]._final_selection=dict(memory_action_cadence=int(recorded[0]),method='mean',case='data_T64')
    if memory.empty:raise ValueError('The declared mean data_T64 memory cohort is missing')
    for column, label, c, style in [
            ('persistent_array_bytes', 'Stored arrays', color('mean'), '-'),
            ('python_numpy_peak_bytes', 'Tracked allocation', color('mode'), '--')]:
        if column in memory:
            points = memory.groupby('n')[column].median()/1024
            axes[2].plot(points.index, points, marker='o', linestyle=style, color=c, label=label)
    axes[2].set_xscale('log', base=2); axes[2].set_yscale('log')
    axes[2].set_xlabel('Ambient dimension'); axes[2].set_ylabel('Memory (KiB)')
    axes[2].legend(frameon=False, fontsize=10.); clean(axes[2])
    for ax, tag in zip(axes, ['(a)', '(b)', '(a)']):
        letter(ax, tag)
    fig._final_selection=dict(observations=64,diagnostic='processing_time_and_operations',
        included='All 64 parameter updates and the stated task decisions, uncertainty reports and EM fits')
    memory_fig._final_selection=dict(observations=64,diagnostic='memory')
    return [fig,memory_fig]


def numerical_accuracy(evidence):
    small = evidence.frame('theory', 'numerical_accuracy')
    large = evidence.frame('computation', 'accuracy')
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.8), layout='constrained')
    quantities = [
        ('mean_error', 'Mean', color('mean'), 'o'),
        ('log_normalizer_error', 'Log normalizer', color('mode'), 's'),
        ('second_moment_error', 'Second moment', color('iso'), '^'),
        ('quantile_error', 'Quantile', '#009E73', 'D'),
    ]
    for name, label, c, m in quantities:
        if name not in small:
            continue
        good = np.isfinite(small[name])
        axes[0].scatter(small.loc[good, 'span'], np.maximum(small.loc[good, name], 1e-17),
                        color=c, marker=m, s=18, label=label)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('Spectral span'); axes[0].set_ylabel('Independent-reference discrepancy')
    axes[0].legend(frameon=False, fontsize=10.); clean(axes[0])
    for names, label, c, m in [
            (('mean_reference_error','mean_error', 'selected_mean_error', 'maximum_selected_mean_error'), 'Selected mean', color('mean'), 'o'),
            (('normalizer_reference_error','log_normalizer_error', 'logz_error', 'logz_absolute_error'), 'Log normalizer', color('mode'), 's')]:
        column = next((name for name in names if name in large), None)
        if column is None:
            continue
        good = np.isfinite(large[column])
        axes[1].scatter(large.loc[good, 'n'], np.maximum(large.loc[good, column], 1e-17),
                        color=c, marker=m, s=18, label=label)
    axes[1].set_xscale('log', base=2); axes[1].set_yscale('log')
    axes[1].set_xlabel('Ambient dimension'); axes[1].set_ylabel('Independent-reference discrepancy')
    axes[1].legend(frameon=False, fontsize=10.); clean(axes[1])
    for ax, tag in zip(axes, ['(a)', '(b)']):
        letter(ax, tag)
    figures = [fig]
    # Numerical targets are counted separately. A resolved mean is never
    # substituted for unresolved second moments, quantiles or reference checks.
    flags = [name for name in ('mean_ok', 'normalizer_ok', 'second_moment_ok',
                              'derivative_ok', 'reference_ok', 'quantile_ok') if name in small]
    if flags:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.8), layout='constrained')
        passed = [int(small[name].fillna(False).astype(bool).sum()) for name in flags]
        total = [len(small)]*len(flags)
        x = np.arange(len(flags))
        ax.bar(x, passed, color=color('mean'), label='Resolved')
        ax.bar(x, np.array(total)-passed, bottom=passed, color='#B8BEC4', label='Unresolved')
        ax.set(xticks=x, xticklabels=[name.replace('_ok', '').replace('_', ' ') for name in flags])
        ax.set_xlabel('Numerical target'); ax.set_ylabel('Number of requested cases')
        ax.legend(frameon=False); clean(ax, 'y')
        figures.append(fig)
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,3.5),layout='constrained')
    targets=[('mean_ok','Mean',None),('normalizer_ok','Normalizer',None),
        ('reference_ok','Reference',None),('second_moment_ok','Second moment','second_moment_evaluated'),
        ('tail_ok','Response quantiles','tail_evaluated'),('state_report_ok','State regions',None)]
    labels=[];resolved=[];unresolved=[];not_evaluated=[]
    for flag,label,evaluated in targets:
        if flag not in large:continue
        requested=large[flag].notna()
        if evaluated is not None and evaluated in large:requested=large[evaluated].eq(True)
        if flag=='reference_ok' and 'reference_scope' in large:
            requested &= ~large.reference_scope.fillna('').str.contains('not supplied',case=False)
        good=large[flag].eq(True)&requested
        labels.append(label);resolved.append(int(good.sum()));unresolved.append(int((requested&~good).sum()))
        not_evaluated.append(int((~requested).sum()))
    x=np.arange(len(labels));first=np.array(resolved);second=np.array(unresolved)
    ax.bar(x,resolved,color=color('mean'),label='Resolved')
    ax.bar(x,unresolved,bottom=first,color=color('mode'),label='Unresolved')
    ax.bar(x,not_evaluated,bottom=first+second,color='#B8BEC4',label='Not evaluated')
    ax.set(xticks=x,xticklabels=labels,ylabel='Number of saved numerical targets')
    ax.tick_params(axis='x',rotation=20)
    ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=10.);clean(ax,'y')
    fig._final_selection={'scope':'Large-dimension numerical rows, separate mean/normalizer/reference/second-moment/tail/state flags; unevaluated targets never counted as resolved'}
    figures.append(fig)
    return figures


def paper_summary(evidence):
    if evidence.example == 'beamforming':
        fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH,6.6), layout='constrained')
        _draw_task_dimensions(axes[0, 0], evidence)
        _draw_paired_tasks(axes[0, 1], evidence)
        frame = _prediction_rows(evidence)
        conditions = _condition_groups(frame)
        condition = next((item for item in conditions if item[0] in ('noise_only', 'noise')), conditions[0])
        _draw_variance_time(axes[1, 0], condition[1], evidence.example)
        _draw_cost_tradeoff(axes[1, 1], evidence)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False,fontsize=10.)
        for ax, tag in zip(axes.ravel(), ['(a)', '(b)', '(c)', '(d)']):
            letter(ax, tag)
    else:
        record = evidence.array('performance', 'anisotropy')
        fig = plt.figure(figsize=(FIGURE_WIDTH, 5.4), layout='constrained')
        grid = fig.add_gridspec(2, 2)
        ax0 = fig.add_subplot(grid[0, 0], projection='3d')
        ax1 = fig.add_subplot(grid[0, 1], projection='3d')
        A, Ai = _saved(record, 'normal_A'), _saved(record, 'iso_normal_A')
        mesh0, mesh1 = bingham_sphere_mesh(A, 161, 321).nodes, bingham_sphere_mesh(Ai, 161, 321).nodes
        q0 = real_bingham_log_density(A, float(_saved(record, 'normal_logz')), mesh0)
        q1 = real_bingham_log_density(Ai, float(_saved(record, 'iso_normal_logz')), mesh1)
        norm = shared_density_norm([q0, q1])
        mode = _saved(record, 'mode_normal')
        density_sphere(ax0, mesh0, q0, norm, mode=mode)
        density_sphere(ax1, mesh1, q1, norm, mode=mode)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DENSITY_CMAP), ax=[ax0, ax1],
                     label=r'$q([n])$', shrink=.68, pad=.02)
        ax2, ax3 = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
        _draw_contact_risk(ax2, record)
        _draw_paired_tasks(ax3, evidence)
        for ax, tag in zip([ax0, ax1, ax2, ax3], ['(a)', '(b)', '(c)', '(d)']):
            letter(ax, tag)
    return [fig]


CATALOGUE_NAMES = (
    'setup', 'task_performance', 'posterior_anisotropy', 'noise_adaptation',
    'information', 'posterior_curvature', 'tracking_bound', 'block_updates',
    'additional_comparisons', 'variance_estimation', 'predictive_calibration',
    'predictive_scores', 'model_mismatch', 'computation_time',
    'numerical_accuracy', 'paper_summary')

PLACEMENT = {
    'setup': 'main',
    'task_performance': 'main',
    'posterior_anisotropy': 'contact main; beamforming supplementary',
    'noise_adaptation': 'beamforming main candidate; contact supplementary',
    'information': 'contact main candidate; complete versions supplementary',
    'paper_summary': 'main',
    'computation_time': 'compact beamforming main; complete versions supplementary',
}

BUNDLES = {
    'supp_information_tracking': ('information', 'posterior_curvature', 'tracking_bound'),
    'supp_comparisons': ('task_performance', 'block_updates', 'additional_comparisons', 'posterior_anisotropy'),
    'supp_variance_prediction_cost': ('variance_estimation', 'noise_adaptation',
        'predictive_calibration', 'predictive_scores', 'model_mismatch',
        'computation_time', 'numerical_accuracy'),
}

CAPTIONS = {
    'setup': r'Available measurements and task geometry. Beamforming uses interference-only proper-complex snapshots; the desired unit steering vector enters the distortionless decision. The displayed beampattern uses a predeclared saved episode, not another independent repetition. Contact uses an independently solved maintained bilateral constraint with known resistance, applied input and directly measured noisy velocity. Positions and normals are generation/scoring information.',
    'task_performance': 'Task losses on common independent test episodes. Losses are averaged within each episode before pointwise Student 95\\% intervals. For contact, main panel (a) shows stationary concentrated-input checkpoint MSE versus observation count, with the posterior-mean confidence band; panel (b) shows absolute episode-average MSE for the moving sphere condition, with a confidence interval for every method including the posterior mean. All methods in panel (b) use the same independent episodes. The separate second page retains paired MSE differences; these are absolute differences in squared velocity units, not relative MSE. A posterior mean compared with itself has identically zero paired difference. Beamforming paired boxes show the distribution of same-episode differences, not confidence intervals. GREAT and application-specific controls retain their stated information and independently validated settings. Complex beamforming GREAT is an extension of the real algorithm, as described in the notebook text. Oracle quantities are evaluation-only.',
    'posterior_anisotropy': r'Contact: actual data-generated $q_{-A_t^+}([n])$ and a fitted isotropic Bingham law with equal mode and mean squared chordal distance. Their fixed-radius orthographic views share camera and color normalization; antipodal normals represent one plane. Color clips only display resolution below its logarithmic floor. Panels (a) and (b) show the actual and fitted-isotropic laws; the orange marker identifies the visible representative of their common modal line. Panel (c) uses $v(\theta)=\cos\theta\,e_2+\sin\theta\,e_3$ and plots $v(\theta)^T(M-M^2)v(\theta)$ using each law and its own mean $M$. These are belief-specific predictive risks, not losses of two estimators evaluated against one common truth. A separate constructed swapped-eigenvalue page has identical centered chordal-distance laws but different directional risks. Beamforming: a real-coefficient section $h=U_3x$, $x\in S^2$, of the ambient complex-line density, shown relative to its mode; this is not a normalized marginal or a full posterior. A matched isotropic mean is a task control, not an invented probability law. Visible anisotropy and empirical task benefit are separate comparisons.',
    'noise_adaptation': r'Actual causal variance learning from completed nonoverlapping past blocks. Each observation is corrected once; a fit updates only subsequent variances while retaining the streamed subspace belief. Known coordinate changes are aligned before block fitting. Curves show saved trajectory/episode summaries; variance bands are interquartile repeated-episode spread, not a posterior over variance. Future-task and pre-observation score intervals are pointwise 95\% intervals across independent episodes. Each page retains one predeclared noise-, motion-, or signal-power intervention. The contact supplementary density pages show one predeclared episode: panels (a,b) of the first density page use $t=32,96$, and panel (a) of the following density page uses $t=192$; pre-correction $A_t^-$ is converted to the wrist-frame normal law $q_{-A_t^-}$ with normalizer $\psi_1(-A_t^-)=\psi_2(A_t^-)-\operatorname{tr}(A_t^-)$. The three orthographic spheres share a density scale; panel (b) of the following density page shows the physical-query risk at all three checkpoints, $v_t^T(M_t^--(M_t^-)^2)v_t$ at the recorded physical query. This is individual belief uncertainty, distinct from the repeated-episode variance bands and from calibration. Empirical-Bayes plug-in scales omit their uncertainty and are outside the fixed-true-variance theorem.',
    'information': r'Independent Gaussian Fisher, score and curvature checks. Expected information is distinguished from finite-data posterior curvature. All matrix comparisons use aligned tangent coordinates; proper-complex information uses real/imaginary coordinates and its correct factor. Recorded-input excitation and marginalized-input information use the stated common data law. Monte Carlo dataset count differs from observations per dataset. Reused mathematical reference cohorts keep their actual source hashes and repetition counts.',
    'posterior_curvature': r'Invariant-density Hessian and local accuracy diagnostics. Empirical bounded log-coordinate spread is compared with $H_A^{-1}$ only under recorded locality conditions; broad/nonlocal cases remain visible. The stationary inverse-Fisher study uses fixed known variances and $\beta=1$, reports estimation bias and local fractions, and treats inverse Fisher as a local reference rather than a global intrinsic Cramer--Rao equality. The real graph-density volume term is a separate quantity, retained in the tables. Forgetting sampling references use squared weights.',
    'tracking_bound': r'Recomputed sharp fixed-time envelope $\sqrt{\min(k,n-k)}\,\epsilon_t/(\gamma_t-\epsilon_t)$ under $\epsilon_t<\gamma_t/2$, with the exact aggregate conditional drift and the appropriate real/complex constants. True path, gap and drift are analysis quantities, not online confidence radii. Invalid conditions remain marked and receive no violation test. Error bands describe repeated-episode spread. The geometric ceiling is not used to clip the theorem. Phase tables and exact binomial violation intervals retain every declared condition.',
    'block_updates': r'Exact discounted aggregation, delayed actions and constant-subspace approximation are separate checks. Aggregation retains the boundary factor $\beta^L$, within-block weights, per-observation variances and recorded noncommuting transport. Numerical zeros are displayed at a $10^{-16}$ plotting floor; full tolerances remain in the tables. Delayed actions are evaluated only after a block arrives, with common endpoints and physical output age reported separately. The block-likelihood reference resets to the uniform prior, $A_0=0$, and uses $q_{\sum_{j=t-L+1}^t B_j}$ with $\beta=1$ within the raw $L$-observation window. It does not retain the singleton history. Under motion its one-state likelihood is an approximation, distinct from the algebraic aggregation identity. Processing and task-decision cost is reported in the computation output.',
    'additional_comparisons': r'Complete declared dimension, memory and method comparisons. Heatmaps show comparator-minus-full posterior-mean loss; positive values favor the full mean. Hatching marks cells whose paired pointwise Student 95\% interval includes zero. No simultaneous-coverage claim is made. Test episodes never tune a setting. Paired differences include losses and inconclusive intervals; broad grids are exploratory without simultaneous coverage. Contact GREAT is response-only and cannot isolate the benefit of known inputs. The same-parameter local Gaussian and recursive EKF are distinct supplementary references with their validity diagnostics retained. Identical modal/PCA or conditional spectral least-squares solutions are not counted as independent algorithms.',
    'variance_estimation': r'Fixed-boundary stationary-block EM reference. Every E-step reconstructs the full block from one saved boundary prior. Evidence curves are shifted by their own initial value, so final heights do not rank starts; absolute normalized evidence selects starts. Both positive starts and the uniform-prior zero-signal boundary are retained. Noise floors and pooled isotropic M-steps are tested in the numerical tables. Contact residual updates include posterior response uncertainty, not only a squared residual using the mean projector. These reference blocks are not online adaptive-tracker episodes.',
    'predictive_calibration': r'Actual-recursion calibration with distinct state, clean-response and noisy-next-observation targets. Pages separate joint prior-predictive sampling, fixed-state repetitions and drift diagnostics. Vertical intervals are pointwise exact binomial uncertainty across independent episodes, not credible uncertainty. All intervals are evaluated at their declared time before the scored future observation is seen. Appended pages show 95% report-size distributions at t=64 for every probability regime. State-ball sizes are chordal radii, obtained by taking the square root of each recorded squared-distance upper threshold; HPD energy thresholds are not used as geometric widths. Scalar sizes are upper minus lower in the n=3 fixture units. Boxes show the median and interquartile range across independent episodes, with 1.5-IQR whiskers and every outlier; this spread is not a posterior distribution or a confidence interval over width. Proper-score checks are separate. Validation-fitted widths do not establish a matched-coverage advantage if achieved test coverage misses its target. Numerical mass and quantile precision are separate from mean accuracy.',
    'predictive_scores': r'Excluded-observation proper scores and response risk under the actual causal recursion. The normalized Bingham-mixture predictive density and Gaussian moment surrogate are distinct densities. Snapshot/query errors are averaged within each episode before comparison. The risk identity predicts total squared clean-response error; a full directional covariance requires second projector moments. Neither pointwise identity lines nor a single trajectory establish a universal score improvement or calibration guarantee.',
    'model_mismatch': r'Declared mismatch experiments under the actual recursion. Beamforming changes training contamination, source powers/geometry, sensor covariance or motion speed. Contact changes constrained mechanics or sensor construction, including dissipative friction/sticking and contact loss. Known anisotropic resistance requires transformed measurements and the correct sensor covariance; isotropic physical velocity noise becomes $\sigma^2D$. Position differencing induces correlated velocity errors. Increased posterior spread is not a repair of an incorrect likelihood.',
    'computation_time': 'Time to process exactly 64 observations on one BLAS thread. Panel (a) includes all 64 parameter corrections and task decisions every one or sixteen observations; its displayed curves exclude optional uncertainty reports. Panel (b) separates parameter updates, task decisions, uncertainty reports, EM and other time for the same 64-observation workload. Row labels state task-decision intervals; optional uncertainty reports and EM fits occur every 32 observations. EM rows use two or four iterations. Phase stacks sum per-phase medians and need not equal median total elapsed time. The separate memory page distinguishes stored arrays from tracked allocations and excludes unmeasured native workspaces/RSS. Repeated clocks measure execution variability on a fixed statistical state, not independent task-performance episodes or worst-case deadlines.',
    'numerical_accuracy': r'Independent accuracy at the requested targets. Normalizer, mean, second moments, derivative and quantile diagnostics retain separate flags. Small-state quadrature and high-precision native complex-line references have their own resolution limits. A first-derivative truncation certificate is not a floating-point, second-moment or tail certificate. Broad and highly concentrated unresolved cases are retained; rank/codimension-one feasibility does not establish arbitrary-rank integration.',
    'paper_summary': r'Selected main panels from the same saved arrays used by the individual figures. Contact emphasizes the physical input/constraint model and directional posterior uncertainty; beamforming emphasizes moderate-dimensional task performance, causal variance learning and computational cost with matched task-decision intervals. No simulation, posterior resampling, tuning or trajectory selection occurs during summary composition. Caption conditions and full numerical records remain in the associated figure entries.',
}


def information(evidence):
    if evidence.example == 'contact':
        from .final_information_visuals import contact_information
        return contact_information(evidence)
    frame = evidence.frame('theory', 'information')
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.7), layout='constrained')
    # This independently checked matrix cohort predates the final task study.
    # Its source/table hashes are verified directly, without treating its old
    # global gate as a validation of the revised theorem or new experiments.
    from .implementation_validation import verify_component
    reference = verify_component(evidence.root, 'paper', 'information')
    old_folder = evidence.root/'results/implementation_audit/paper/information'
    old_manifest = old_folder/'manifest.json'
    evidence.inputs[str(old_manifest.relative_to(evidence.root))] = sha(old_manifest)
    old_path = old_folder/'fisher_matrices.csv'
    evidence.inputs[str(old_path.relative_to(evidence.root))] = sha(old_path)
    frame = pd.read_csv(old_path).rename(columns={'s': 'sigma2'})
    evidence.references['historical_information'] = dict(
        config=reference.get('config', {}), sources=reference['sources'],
        checks=reference.get('checks', {}), manifest=str(old_manifest.relative_to(evidence.root)),
        tables={'fisher_matrices': {'rows': len(frame)}})
    frame = frame[frame.complex_data.astype(bool).eq(evidence.example == 'beamforming')]
    frame = frame[frame.channel.eq('marginal' if evidence.example == 'beamforming' else 'conditional')]
    if {'i', 'j', 'expected', 'score_covariance'}.issubset(frame):
        group_names = [name for name in ('case', 'channel', 'sigma2', 'L') if name in frame]
        if not group_names:
            group_names = ['model'] if 'model' in frame else []
        grouped = frame.groupby(group_names, sort=False) if group_names else [(None, frame)]
        all_x, all_y = [], []
        for _, g in grouped:
            expected = g.pivot(index='i', columns='j', values='expected').to_numpy()
            estimated = g.pivot(index='i', columns='j', values='score_covariance').to_numpy()
            x, y = np.linalg.eigvalsh(expected), np.linalg.eigvalsh(estimated)
            axes[0].scatter(x, y, color=color('mean'), s=12, alpha=.6)
            all_x.extend(x); all_y.extend(y)
        _identity(axes[0], all_x, all_y)
        if min(all_x+all_y) > 0:
            axes[0].set_xscale('log'); axes[0].set_yscale('log')
    else:
        expected = _column(frame, 'analytical', 'analytic', 'expected_eigenvalue',
                           'fisher_expected', 'expected', 'reference')
        measured = _column(frame, 'empirical', 'score_covariance', 'estimated_eigenvalue',
                           'fisher_empirical', 'estimate')
        x, y = frame[expected], frame[measured]
        axes[0].scatter(x, y, s=12, color=color('mean'), alpha=.6)
        _identity(axes[0], x, y)
        if min(x.min(), y.min()) > 0:
            axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('Analytical Fisher eigenvalue')
    axes[0].set_ylabel('Estimated Fisher eigenvalue')
    clean(axes[0])
    design = evidence.frame('theory', 'information_design', 'information_loss', optional=True)
    if evidence.example == 'contact':
        record = evidence.array('performance', 'anisotropy')
        eigenvalues = _saved(record, 'fisher_eigenvalues')
        if eigenvalues.ndim == 1:
            axes[1].bar(np.arange(1, len(eigenvalues)+1), eigenvalues,
                        color=color('mean'), width=.65)
            axes[1].set_xlabel('Tangent direction'); axes[1].set_ylabel('Fisher eigenvalue')
        else:
            for j, values in enumerate(eigenvalues):
                axes[1].plot(np.arange(1, len(values)+1), values, marker=('o', '^')[j % 2],
                             color=(color('mean'), color('iso'))[j % 2],
                             label=('Balanced inputs', 'Concentrated input directions')[j % 2])
            axes[1].set_xlabel('Tangent direction'); axes[1].set_ylabel('Fisher eigenvalue')
            axes[1].legend(frameon=False)
    elif not design.empty:
        if 'complex_data' in design:
            design = design[design.complex_data.astype(bool).eq(evidence.example == 'beamforming')]
        identity_columns = [name for name in ('sigma2','L','expected_conditional','expected_marginal') if name in design]
        if identity_columns:
            design = design.drop_duplicates(identity_columns)
        x = _column(design, 'snr', 'signal_to_noise', 'sigma2', 'L')
        for column, label, c in [('conditional', 'Recorded inputs', color('mode')),
                                  ('marginal', 'Inputs marginalized', color('mean'))]:
            name = next((name for name in ('expected_'+column, column, column+'_information', column+'_coefficient')
                         if name in design), None)
            if name is not None:
                axes[1].plot(design[x], design[name], 'o-', color=c, label=label)
        axes[1].set_xlabel(r'$\tau^2/\sigma^2$' if x == 'snr' else r'Noise variance $\sigma^2$' if x=='sigma2' else x.replace('_', ' '))
        axes[1].set_ylabel('Fisher coefficient')
        axes[1].legend(frameon=False, fontsize=10.)
    else:
        x = _column(frame, 'L', 'observations', 'sigma2', 'sigma2')
        y = _column(frame, 'relative_error', 'score_relative_error', 'error', 'mc_error')
        for group, g in frame.groupby('sigma2', sort=True) if 'sigma2' in frame else [(None, frame)]:
            axes[1].plot(g[x], g[y], 'o-', label=(r'$\sigma^2='+f'{group:g}'+r'$') if group is not None else None)
        axes[1].set_xlabel('Number of observations' if x in ('L', 'observations') else r'$\sigma^2$')
        axes[1].set_ylabel('Relative information error')
        axes[1].legend(frameon=False, fontsize=10.)
    clean(axes[1])
    for ax, tag in zip(axes, ['(a)', '(b)']):
        letter(ax, tag)
    return [fig]


def posterior_curvature(evidence):
    curve = evidence.frame('theory', 'curvature')
    accuracy = evidence.frame('theory', 'inverse_fisher_summary', 'inverse_fisher')
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.3), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    x = _column(curve, 'inverseHvariance', 'inverse_h_variance', 'inverse_hessian_variance',
                'inverse_curvature', 'inverse_curvature_variance', 'laplace_variance')
    y = _column(curve, 'empirical_localvariance', 'empirical_local_variance',
                'empirical_variance', 'empirical_log_coordinate_variance', 'local_variance')
    valid_name = next((name for name in ('local_approximation_eligible', 'local_valid', 'local_ok', 'resolved') if name in curve), None)
    valid = np.ones(len(curve), bool) if valid_name is None else curve[valid_name].fillna(False).astype(bool).to_numpy()
    axes[0].scatter(curve.loc[valid, x], curve.loc[valid, y], s=14,
                    color=color('mean'), label='Local regime')
    if (~valid).any():
        axes[0].scatter(curve.loc[~valid, x], curve.loc[~valid, y], s=18,
                        facecolors='none', edgecolors=color('mode'), marker='^', label='Outside local regime')
    _identity(axes[0], curve[x], curve[y])
    if np.min(curve[[x, y]].to_numpy()) > 0:
        axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$H_A^{-1}$ tangent variance')
    axes[0].set_ylabel('Empirical log-coordinate variance')
    axes[0].legend(frameon=False, fontsize=10.); clean(axes[0])
    L = _column(accuracy, 'L', 'observations', 't')
    mse = _column(accuracy, 'mse', 'tangent_mse', 'empirical_mse')
    ref = _column(accuracy, 'reftrace', 'inverse_fisher_trace', 'reference_trace')
    stationary = accuracy[np.isclose(accuracy.beta, 1.)] if 'beta' in accuracy else accuracy
    if 'sigma2' in stationary:
        stationary = stationary[np.isclose(stationary.sigma2, stationary.sigma2.min())]
    for name, values, c, style in [('Empirical MSE', stationary[mse], color('mean'), '-'),
                                  (r'$\operatorname{tr}(\mathcal F^{-1})$', stationary[ref], GREY, '--')]:
        axes[1].plot(stationary[L], values, marker='o', linestyle=style, color=c, label=name)
    if {'tangent_mse_lower', 'tangent_mse_upper'}.issubset(stationary):
        axes[1].fill_between(stationary[L], stationary.tangent_mse_lower,
                             stationary.tangent_mse_upper, color=color('mean'), alpha=.12)
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_xlabel('Number of observations'); axes[1].set_ylabel('Squared tangent error')
    axes[1].legend(frameon=False, fontsize=10.); clean(axes[1])
    fraction = _column(accuracy, 'local_fraction', 'fraction_local')
    bias = _column(accuracy, 'biasnorm', 'bias_norm', 'tangent_bias_norm')
    axes[2].plot(stationary[L], stationary[fraction], 'o-', color=color('mean'), label='Local fraction')
    normalized_bias = stationary[bias]/np.sqrt(stationary[ref])
    axes[2].plot(stationary[L], normalized_bias, '^--', color=color('mode'),
                 label=r'Bias$/\sqrt{\operatorname{tr}(\mathcal F^{-1})}$')
    axes[2].set_xscale('log'); axes[2].set_xlabel('Number of observations')
    axes[2].set_ylabel('Local fraction / normalized bias')
    axes[2].set_ylim(bottom=0)
    axes[2].legend(frameon=False, fontsize=10.); clean(axes[2])
    for ax, tag in zip(axes, ['(a)', '(b)', '(c)']):
        letter(ax, tag)
    return [fig]


def _tracking_plot(ax, rows):
    times=sorted(rows.t.unique());summary=[]
    for t in times:
        group=rows[rows.t.eq(t)]
        # All deterministic bound quantities must identify one cohort.
        for key in ('gamma','epsilon','bound','ceiling','valid'):
            if group[key].dropna().nunique()>1:
                raise ValueError('Tracking display mixed distinct deterministic cohorts: '+key)
        summary.append((t,group.modal_error.median(),group.modal_error.quantile(.25),
                        group.modal_error.quantile(.75),group.bound.iloc[0],
                        group.ceiling.iloc[0],bool(group.valid.iloc[0])))
    points=np.asarray(summary)
    ax.plot(points[:,0],points[:,1],'o-',color=color('mean'),label='Modal error')
    ax.fill_between(points[:,0],points[:,2],points[:,3],color=color('mean'),alpha=.12)
    ax.plot(points[:,0],points[:,4],'s-',color=color('iso'),label='Theorem bound')
    ax.plot(points[:,0],points[:,5],'--',color=BLACK,label='Distance ceiling')
    bad=points[:,6]==0
    if bad.any():ax.scatter(points[bad,0],points[bad,5],marker='x',color=GREY,label=r'$\epsilon_t\ge\gamma_t/2$')
    ax.set(xscale='log' if len(times)>1 else 'linear',
        yscale='log' if np.all(points[:,1:4]>0) else 'linear',
        xlabel='Number of observations',ylabel='Chordal distance')
    if len(times)==1:
        ax.set_xlim(times[0]*.9,times[0]*1.1);ax.set_xticks(times)
    if not np.all(points[:,1:4]>0):ax.set_ylim(-.05,1.08*points[:,5].max())
    clean(ax)


def _beta_label(beta):
    gap=1-float(beta)
    if gap>0 and gap<.001:
        exponent=int(np.round(np.log10(gap)))
        if np.isclose(gap,10.**exponent,rtol=1e-6,atol=0):return r'$1-10^{'+str(exponent)+r'}$'
    return f'{beta:.9g}'


def tracking_bound(evidence):
    all_rows=_normalize(evidence.frame('theory','tracking'))
    phase=_normalize(evidence.frame('theory','tracking_phase'))
    rows=all_rows.copy()
    for flag in ('nonuniform_prior','zero_signal'):
        if flag in rows:rows=rows[~rows[flag].fillna(False).astype(bool)]
    if 'design' in rows:rows=rows[rows.design.eq('balanced')]
    base=rows[rows.condition.eq('static') & np.isclose(rows.beta,1.)]
    if base.empty:raise ValueError('The declared static, balanced, uniform-prior beta=1 tracking cohort is missing')
    fig=plt.figure(figsize=(FIGURE_WIDTH,5.5),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=(1.3,1.))
    axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
    _tracking_plot(axes[0],base)
    axes[0].legend(frameon=False,ncol=2,fontsize=10.,loc='lower center',bbox_to_anchor=(.5,1.02))
    last=base[base.t.eq(base.t.max())]
    axes[1].scatter(last.epsilon,last.actual_perturbation,s=12,color=color('mean'),alpha=.6)
    _identity(axes[1],last.epsilon,last.actual_perturbation)
    axes[1].set(xlabel=r'Theoretical allowance $\epsilon_t$',ylabel='Operator perturbation');clean(axes[1])
    phase=phase.sort_values(['beta','drift_rate'])
    phase['state']=phase.valid.astype(bool).astype(int)
    table=phase.pivot(index='beta',columns='drift_rate',values='state')
    from matplotlib.colors import ListedColormap, BoundaryNorm
    image=axes[2].imshow(table.to_numpy(),origin='lower',aspect='auto',
        cmap=ListedColormap(['#D7DCE1','#0072B2']),norm=BoundaryNorm([-.5,.5,1.5],2),interpolation='nearest')
    axes[2].set(xticks=np.arange(len(table.columns)),xticklabels=[f'{v:g}' for v in table.columns],
        yticks=np.arange(len(table.index)),yticklabels=[_beta_label(v) for v in table.index],
        xlabel='Subspace angular speed',ylabel=r'Forgetting factor $\beta$')
    cb=fig.colorbar(image,ax=axes[2],ticks=[0,1],shrink=.85)
    cb.ax.set_yticklabels([r'$\epsilon\ge\gamma/2$',r'$\epsilon<\gamma/2$'])
    for ax,tag in zip(axes,['(a)','(b)','(c)']):letter(ax,tag)
    fig._final_selection=dict(condition='static',beta=1.,design='balanced',nonuniform_prior=False,zero_signal=False)
    figures=[fig]
    # Absolute allowance contributions remain meaningful when gamma is zero.
    keys=[k for k in ('condition','beta','nonuniform_prior','zero_signal','design') if k in all_rows]
    terminal=all_rows[all_rows.t.eq(all_rows.groupby(keys).t.transform('max'))]
    contributions=terminal.groupby(keys,dropna=False)[['initialization','drift','noise','epsilon','gamma']].first().reset_index()
    fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,3.7),layout='constrained')
    x=np.arange(len(contributions));bottom=np.zeros(len(x))
    for key,label,c in [('initialization','Initialization',color('mode')),('drift','Unmodelled motion',color('iso')),('noise','Measurement noise',color('mean'))]:
        ax.bar(x,contributions[key],bottom=bottom,label=label,color=c,width=.7);bottom+=contributions[key]
    ax.scatter(x,.5*contributions.gamma,marker='_',s=75,color=BLACK,label=r'$\gamma_t/2$',zorder=4)
    labels=[]
    for r in contributions.to_dict('records'):
        text={'static':'Static','smooth':'Smooth','abrupt':'Abrupt'}.get(r['condition'],r['condition'])
        if r.get('nonuniform_prior'):text='Nonuniform prior'
        if r.get('zero_signal'):text='Zero signal'
        if r.get('design')=='concentrated':text='Concentrated inputs'
        labels.append(text+'\n'+r'$\beta='+_beta_label(r['beta']).strip('$')+r'$')
    ax.set(xticks=x,xticklabels=labels,ylabel='Operator perturbation allowance')
    ax.tick_params(axis='x',rotation=35)
    ax.set_yscale('symlog',linthresh=.01)
    ax.legend(frameon=False,ncol=2,fontsize=10.,loc='lower center',bbox_to_anchor=(.5,1.02));clean(ax,'y')
    fig._final_selection={'terminal_cohorts':contributions[keys].to_dict('records'),'quantity':'epsilon = initialization + drift + noise; gamma/2 shown separately'}
    figures.append(fig)
    # Special fixtures never enter the base error distribution.
    special=all_rows.copy();mask=np.zeros(len(special),bool)
    for flag in ('nonuniform_prior','zero_signal'):
        if flag in special:mask|=special[flag].fillna(False).astype(bool).to_numpy()
    if 'design' in special:mask|=special.design.ne('balanced').to_numpy()
    special=special[mask]
    for selection,group in special.groupby(keys,dropna=False,sort=False):
        fig,ax=plt.subplots(figsize=(FIGURE_WIDTH,3.),layout='constrained')
        _tracking_plot(ax,group)
        ax.legend(frameon=False,ncol=2,fontsize=10.,loc='lower center',bbox_to_anchor=(.5,1.02))
        fig._final_selection=dict(zip(keys,selection if isinstance(selection,tuple) else [selection]))
        figures.append(fig)
    return figures


def block_updates(evidence):
    checks = evidence.frame('theory', 'block_checks')
    trade = _normalize(evidence.frame('performance', 'blocks', 'block_metrics', 'block_updates'))
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.8), layout='constrained')
    L = _column(checks, 'L', 'block_length')
    if 'metric' in checks and 'error' in checks:
        for index, (metric, g) in enumerate(checks.groupby('metric', sort=False)):
            if index > 3:
                break
            points = g.groupby(L).error.max()
            axes[0].plot(points.index, np.maximum(points, 1e-16), 'o-', label=str(metric).replace('_', ' '))
    else:
        for name in ('parameter_error', 'mean_error', 'density_error', 'mode_error'):
            if name in checks:
                points = checks.groupby(L)[name].max()
                axes[0].plot(points.index, np.maximum(points, 1e-16), 'o-', label=name.replace('_error', ''))
    axes[0].axhline(1e-10, color=GREY, linestyle='--', linewidth=.8)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Observations per block')
    axes[0].set_ylabel('Aggregation discrepancy')
    axes[0].legend(frameon=False, fontsize=10.); clean(axes[0])
    L = _column(trade, 'L', 'block_length')
    if 'condition' in trade:
        condition = 'sphere' if 'sphere' in set(trade.condition) else trade.condition.iloc[0]
        trade = trade[trade.condition.eq(condition)]
    endpoint = next((name for name in ('heldout_ratio_db', 'output_oracle_db', 'response_mse',
                        'task_loss', 'loss') if name in trade), None)
    if endpoint is None and 'delayed_response_mse' in trade:
        for column, label, c, m in [
                ('delayed_response_mse', 'Delayed action', color('mean'), 'o'),
                ('common_endpoint_response_mse', 'Common endpoint', color('mode'), 's'),
                ('constant_block_response_mse', 'Block likelihood', color('iso'), '^')]:
            points = trade.groupby(L)[column].mean()*1e6
            axes[1].plot(points.index, points, marker=m, color=c, label=label)
        groups = []
    elif endpoint is None:
        endpoint = _column(trade, 'delayed_ratio_db', 'delayed_loss', 'mean_delayed_power_db')
        groups = [('Delayed output', trade)]
    elif 'kind' in trade:
        groups = trade.groupby('kind', sort=False)
    elif 'comparison' in trade:
        groups = trade.groupby('comparison', sort=False)
    elif 'schedule' in trade:
        groups = trade.groupby('schedule', sort=False)
    else:
        groups = [('Delayed output', trade)]
    for index, (name, group) in enumerate(groups):
        points = group.groupby(L)[endpoint].mean()
        axes[1].plot(points.index, points*(1e6 if evidence.example == 'contact' and 'mse' in endpoint else 1.),
                     marker=('o', 's', '^')[index % 3], label={
                        'delayed_exact':'Delayed exact aggregation',
                        'constant_state_window':'Block likelihood'}.get(str(name),str(name).replace('_',' ')))
    axes[1].set_xlabel('Observations per block')
    axes[1].set_ylabel('Output power / oracle power (dB)' if evidence.example == 'beamforming'
                       else r'Response MSE $((\mathrm{mm}/\mathrm{s})^2)$')
    axes[1].legend(frameon=False, fontsize=10.); clean(axes[1])
    delay = next((name for name in ('delay_s', 'mean_delay_s', 'action_delay_s',
                                    'maximum_output_age_s','maximum_delay_s') if name in trade), None)
    cost = next((name for name in ('processing_ms', 'total_ms', 'computation_ms', 'elapsed_ms') if name in trade), None)
    if cost is not None:
        points = trade.groupby(L)[cost].mean()
        axes[2].plot(points.index, points, 'o-', color=color('mean'))
        axes[2].set_xlabel('Observations per block'); axes[2].set_ylabel('Computation time (ms)')
    elif delay is not None:
        points = trade.groupby(L)[delay].mean()
        axes[2].plot(points.index, points, 'o-', color=color('mode'))
        axes[2].set_xlabel('Observations per block'); axes[2].set_ylabel('Action delay (s)')
    else:
        raise KeyError('Block tradeoff needs saved processing cost or physical action delay')
    clean(axes[2])
    for ax, tag in zip(axes, ['(a)', '(b)', '(c)']):
        letter(ax, tag)
    return [fig]


CATALOGUE = {name: globals()[name] for name in CATALOGUE_NAMES}
