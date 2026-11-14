"""Render the five manuscript figures from the frozen paper records."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .plotting import plt
from matplotlib import colors


LABELS = {
    'mean': r'$\overline P_t^+$',
    'mode': r'$P_{\widehat{\mathcal X}_t}$',
    'iso': r'$\overline P_t^{\mathrm{iso}}$',
    'mean_identity': r'$\overline P_t^+,\ G_t=I_3$',
    'great_window': 'GREAT (sliding window)',
    'known': 'True variances',
    'misspecified': 'Fixed variance estimates',
    'learned': 'EM variance estimates',
}
COLORS = {
    'mean': '#0072B2', 'mode': '#D55E00', 'iso': '#CC79A7',
    'mean_identity': '#484848', 'great_window': '#009E73',
    'known': '#111111', 'misspecified': '#999999', 'learned': '#0072B2',
}
FIGURE_SIZES = {
    'contact/contact_comparison.pdf': (504., 199.48),
    'contact/information_rotation.pdf': (1080., 360.),
    'contact/posterior_anisotropy_spheres.pdf': (1080., 360.),
    'beamforming/moving_interference_comparison.pdf': (504., 252.),
    'beamforming/noise_adaptation.pdf': (501.181, 319.47),
}


def _style(large=False):
    label, tick, legend = (25., 23., 22.) if large else (10.5, 10., 10.5)
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.family': 'serif', 'font.size': label,
        'axes.labelsize': label, 'axes.titlesize': label,
        'xtick.labelsize': tick, 'ytick.labelsize': tick,
        'legend.fontsize': legend, 'axes.labelpad': 8. if large else 4.,
        'lines.linewidth': 15./7. if large else 1.6,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': None,
    })


def _border(ax, width=.9):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(width)
    ax.tick_params(axis='both', which='both', direction='in', width=width,
                   top=True, right=True, bottom=True, left=True,
                   labeltop=False, labelright=False)
    ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)


def _figure_size(name):
    return tuple(value / 72. for value in FIGURE_SIZES[name])


def _anisotropy(results):
    with np.load(Path(results) / 'contact/data/anisotropy.npz', allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def contact_comparison(root, results=None):
    results = Path(results) if results is not None else Path(root) / 'results'
    _style()
    saved = _anisotropy(results)
    selected = saved['query_family'] == 0
    theta = saved['query_angles'][selected]
    np.testing.assert_allclose(saved['unit_queries'][selected],
        np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]),
        rtol=0., atol=1e-14)
    fig, axes = plt.subplots(1, 2, figsize=_figure_size('contact/contact_comparison.pdf'),
                             layout='constrained')
    ax = axes[0]
    ax.plot(theta, saved['risks_full'][selected], color=COLORS['mean'], label=LABELS['mean'])
    ax.plot(theta, saved['risks_isotropic'][selected], color=COLORS['iso'], label=LABELS['iso'])
    ax.set(xlabel=r'$\theta$ (rad)', ylabel=r'$r_t(v(\theta))/v_0^2$',
           xticks=np.linspace(0., 2*np.pi, 5),
           xticklabels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
           yticks=[.05, .10, .15])
    frame = pd.read_csv(results / 'contact/data/checkpoint_summary.csv')
    for method in ('mean', 'mode', 'iso', 'mean_identity'):
        rows = frame[frame.method.eq(method)].sort_values('t')
        if len(rows) != 48:
            raise ValueError('The contact figure requires all 48 dense reporting times')
        axes[1].plot(rows.t, rows['mean'], color=COLORS[method],
                     label=LABELS[method], linewidth=1.8)
        if method in ('mean', 'mean_identity'):
            axes[1].fill_between(rows.t, rows.lower, rows.upper,
                                 color=COLORS[method], alpha=.14, linewidth=0, zorder=1)
    axes[1].set(xlabel=r'$t$', ylabel=r'$\mathcal E_t(D_t)$ $((\mathrm{mm}/\mathrm{s})^2)$',
                xlim=(1, 192))
    for ax in axes:
        _border(ax)
    fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center',
               ncol=4, frameon=False, fontsize=10.)
    fig.get_layout_engine().set(h_pad=.04, w_pad=.04)
    return fig


def contact_information_matrix(saved, sigma2):
    inputs, normal = saved['input_vectors'], saved['true_normal']
    if not (inputs.shape == (16, 3) and np.allclose(normal, [0., 0., 1.], atol=1e-12)):
        raise ValueError('Information geometry requires the frozen 16-measurement contact design')
    covariance = inputs.T @ inputs
    fisher = (covariance[:2, :2] + covariance[2, 2]*np.eye(2))/sigma2
    np.testing.assert_allclose(np.linalg.eigvalsh(fisher), saved['fisher_eigenvalues'],
                               rtol=1e-11, atol=1e-11)
    return fisher


def exact_contact_kl(z, fisher):
    z = np.asarray(z, float)
    return np.einsum('...a,ab,...b->...', z, fisher, z)/(2*(1+np.sum(z*z, axis=-1)))


def information_rotation(root, results=None):
    results = Path(results) if results is not None else Path(root) / 'results'
    _style(large=True)
    config = json.loads((Path(root) / 'configs/contact.json').read_text())
    fisher = contact_information_matrix(_anisotropy(results), config['config']['anisotropy_sigma2'])
    angle = np.linspace(.25, 35., 201)
    tangent = np.tan(np.deg2rad(angle))
    marked = np.array([[np.tan(np.deg2rad(5.)), 0.], [0., np.tan(np.deg2rad(30.))]])
    fig, ax = plt.subplots(figsize=_figure_size('contact/information_rotation.pdf'),
                           layout='constrained')
    for ray, label, color, point, degrees in zip(
            [np.column_stack([tangent, np.zeros_like(tangent)]),
             np.column_stack([np.zeros_like(tangent), tangent])],
            [r'$z=(\tan\delta,0)^\top$', r'$z=(0,\tan\delta)^\top$'],
            ['#D55E00', '#0072B2'], marked, [5., 30.]):
        ax.plot(angle, exact_contact_kl(ray, fisher), color=color, label=label)
        ax.plot(degrees, exact_contact_kl(point, fisher), 'o', color=color, markersize=4.)
    divergence = exact_contact_kl(marked, fisher)
    ax.plot([5., 30.], divergence, color='#8A8A8A', linewidth=1., zorder=8)
    for degrees, value in zip([5., 30.], divergence):
        ax.annotate(r'$'+f'{degrees:g}'+r'^\circ$', xy=(degrees, value),
                    xytext=(0, 9), textcoords='offset points',
                    ha='center', va='bottom', fontsize=23., zorder=9)
    ax.set(xlabel=r'$\delta\ ({}^\circ)$', ylabel=r'$D_{\mathrm{KL}}(p_0\,\Vert\,p_Z)$',
           xlim=(0., 35.), yscale='log', yticks=[1e-5, 1e-2, 1e1])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=22., loc='lower right')
    _border(ax)
    fig.get_layout_engine().set(h_pad=.18, w_pad=.16)
    return fig


def bingham_sphere_nodes(parameter, polar=161, azimuth=321):
    parameter = np.asarray(parameter, float)
    if parameter.shape != (3, 3) or not np.allclose(parameter, parameter.T, atol=1e-10):
        raise ValueError('A real symmetric 3 by 3 parameter is required')
    values, vectors = np.linalg.eigh(parameter)
    vectors = vectors[:, ::-1]
    gaps = values[-1]-values[-2::-1]
    base = np.linspace(0., 2*np.pi, azimuth)
    if gaps[-1] < 1e-10:
        theta, phi = np.meshgrid(np.linspace(0., np.pi, polar), base, indexing='ij')
        return np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
    concentration = max(1., .5*np.log1p(gaps[-1]))
    q = np.tanh(concentration*np.linspace(-1., 1., polar))/np.tanh(concentration)
    theta = np.arccos(np.clip(q[::-1], -1., 1.))
    ratio = np.sqrt((1+gaps[0])/(1+gaps[1]))
    phi = np.unwrap(np.arctan2(ratio*np.sin(base), np.cos(base)))
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    local = np.stack([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)], axis=-1)
    return local @ vectors.T


def posterior_anisotropy_spheres(root, results=None):
    results = Path(results) if results is not None else Path(root) / 'results'
    _style(large=True)
    saved = _anisotropy(results)
    parameters = [saved['normal_A'], saved['iso_normal_A']]
    nodes = [bingham_sphere_nodes(parameter) for parameter in parameters]
    log_density = [np.einsum('...i,ij,...j->...', mesh, parameter, mesh)-float(saved[key])
                   for parameter, mesh, key in zip(parameters, nodes, ['normal_logz', 'iso_normal_logz'])]
    cmap = colors.LinearSegmentedColormap.from_list('density_blue_pink_yellow',
        [(0., '#263a8b'), (.28, '#55209a'), (.53, '#b33695'),
         (.72, '#ed78a4'), (.88, '#ffc85a'), (1., '#f4f52b')], N=256)
    cmap.set_under(cmap(0.))
    cmap.set_over(cmap(1.))
    peak = max(float(np.max(values)) for values in log_density)
    norm = colors.LogNorm(vmin=np.exp(peak-4*np.log(10.)), vmax=np.exp(peak), clip=True)
    fig = plt.figure(figsize=_figure_size('contact/posterior_anisotropy_spheres.pdf'),
                     layout='constrained')
    grid = fig.add_gridspec(1, 2)
    axes = [fig.add_subplot(grid[0, index], projection='3d') for index in range(2)]
    elevation, azimuth = np.deg2rad([25., -50.])
    facing = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    for ax, mesh, logq in zip(axes, nodes, log_density):
        ax.computed_zorder = False
        surface = ax.plot_surface(mesh[..., 0], mesh[..., 1], mesh[..., 2],
            facecolors=cmap(norm(np.exp(logq))), shade=False, linewidth=0,
            antialiased=False, rstride=1, cstride=1, zorder=1)
        surface.set_rasterized(True)
        phi = np.linspace(0., 2*np.pi, 241)
        for first, second in [(0, 1), (0, 2), (1, 2)]:
            ring = np.zeros((len(phi), 3))
            ring[:, first], ring[:, second] = 1.006*np.cos(phi), 1.006*np.sin(phi)
            visible = np.where((ring @ facing >= 0.)[:, None], ring, np.nan)
            ax.plot(*visible.T, color='#B7C0CB', linewidth=.5, alpha=.55, zorder=3)
        mode = saved['mode_normal']/np.linalg.norm(saved['mode_normal'])
        ax.scatter(*(mode if mode @ facing >= 0. else -mode), marker='o', s=12,
                   color='#D55E00', depthshade=False, zorder=4)
        ax.set_proj_type('ortho')
        ax.view_init(elev=25., azim=-50.)
        ax.set_box_aspect((1., 1., 1.), zoom=.84)
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), zlim=(-1.12, 1.12),
               xticks=(-1., 0., 1.), yticks=(-1., 0., 1.), zticks=(-1., 0., 1.))
        ax.set_xlabel(r'$x_1$', labelpad=8)
        ax.set_ylabel(r'$x_2$', labelpad=8)
        ax.set_zlabel('')
        ax.text2D(.95, .67, r'$x_3$', transform=ax.transAxes, rotation=90,
                  ha='center', va='center', fontsize=25.)
        ax.tick_params(labelsize=23., pad=3, length=1)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1., 1., 1., 0.))
            axis.pane.set_edgecolor((.65, .68, .72, .35))
            axis.line.set_color('#70777D')
            axis.line.set_linewidth(.65)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label=r'$q([x])$', shrink=.64, pad=.015)
    fig.get_layout_engine().set(h_pad=.18, w_pad=.16)
    return fig


def moving_interference_comparison(root, results=None):
    results = Path(results) if results is not None else Path(root) / 'results'
    _style()
    frame = pd.read_csv(results / 'beamforming/data/moving.csv.gz')
    frame['method'] = frame.method.replace({'isotropic': 'iso'})
    fig, axes = plt.subplots(1, 2,
        figsize=_figure_size('beamforming/moving_interference_comparison.pdf'), layout='constrained')
    for ax, condition in zip(axes, ['smooth', 'abrupt']):
        group = frame[frame.condition.eq(condition)]
        for method in ('mean', 'mode', 'iso', 'great_window'):
            summary = group[group.method.eq(method)].groupby('t').true_ratio_db
            average = summary.mean()
            if len(average) != 32 or (summary.count() != 256).any():
                raise ValueError('Moving figures require 256 trials at 32 reporting times')
            ax.fill_between(average.index, summary.quantile(.25), summary.quantile(.75),
                             color=COLORS[method], alpha=.22, linewidth=0., zorder=1)
            ax.plot(average.index, average.to_numpy(), color=COLORS[method],
                     label=('Posterior mean '+LABELS[method] if method == 'mean' else
                            'Mode '+LABELS[method] if method == 'mode' else
                            'Isotropic mean '+LABELS[method] if method == 'iso' else LABELS[method]),
                     linewidth=1., zorder=10)
        if condition == 'abrupt':
            ax.axvline(129, color='#7B3294', linewidth=1.3, linestyle='--', zorder=20,
                       label='Direction change')
        ax.set(xlabel=r'$t$', ylabel=r'$\ell_t$ (dB)', xlim=(1, 256))
        _border(ax, width=.95)
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False,
               fontsize=10.5, columnspacing=1., handlelength=1.8, handletextpad=.5, labelspacing=.3)
    fig.get_layout_engine().set(h_pad=.04, w_pad=.04)
    return fig


def _interval(values):
    values = np.asarray(values, float)
    average = values.mean()
    error = student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
    return average, average-error, average+error


def noise_adaptation(root, results=None):
    results = Path(results) if results is not None else Path(root) / 'results'
    _style()
    frame = pd.read_csv(results / 'beamforming/data/noise.csv.gz')
    if 'split' in frame:
        frame = frame[frame.split.eq('test')]
    frame = frame[frame.control.eq('mean') & frame.condition.eq('noise_change')]
    time = np.arange(1, 257)
    truth = {'sigma2': np.where(time <= 128, .2, .8), 'tau2': np.ones(256)}
    fig, axes = plt.subplots(2, 2, figsize=_figure_size('beamforming/noise_adaptation.pdf'),
                             layout='constrained')
    for ax, key in zip(axes[0], ['sigma2', 'tau2']):
        for method in ('misspecified', 'learned'):
            group = frame[frame.method.eq(method)].groupby('t')[key+'_estimate']
            median = group.median()
            ax.plot(median.index, median.to_numpy(), color=COLORS[method], label=LABELS[method])
            if method == 'learned':
                ax.fill_between(median.index, group.quantile(.25), group.quantile(.75),
                                 color=COLORS[method], alpha=.13)
        ax.step(time, truth[key], where='post', color=COLORS['known'], linestyle='--',
                 linewidth=1.6, label=LABELS['known'])
        ax.set(xlabel=r'$t$', ylabel=(r'$\sigma_t^2,\ \widehat\sigma_t^2$'
               if key == 'sigma2' else r'$\tau_t^2,\ \widehat\tau_t^2$'))
    for ax, endpoint in zip(axes[1], ['output_oracle_db', 'prequential_score']):
        for method in ('known', 'misspecified', 'learned'):
            rows = frame[frame.method.eq(method)]
            points = np.array([(t, *_interval(group.groupby('episode')[endpoint].mean().dropna()))
                               for t, group in rows.groupby('t', sort=True)])
            ax.plot(points[:, 0], points[:, 1], color=COLORS[method], label=LABELS[method])
            ax.fill_between(points[:, 0], points[:, 2], points[:, 3], color=COLORS[method], alpha=.09)
        ax.set(xlabel=r'$t$', ylabel=r'$\ell_t$ (dB)' if endpoint == 'output_oracle_db'
               else r'$\log\pi_t(u_t)$')
    handles = {}
    for ax in axes.flat:
        ax.axvline(129, color='#7B3294', linewidth=1.3, linestyle='--', zorder=20,
                   label='Noise variance change')
        ax.set_xlim(1, 256)
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
        _border(ax)
    fig.legend(list(handles.values()), list(handles), loc='outside lower center', ncol=2,
               columnspacing=1.2, frameon=False, fontsize=10.5)
    fig.get_layout_engine().set(h_pad=.07, w_pad=.06)
    return fig


def export_figures(root, results=None):
    root = Path(root).resolve()
    results = Path(results).resolve() if results is not None else root / 'results'
    producers = [contact_comparison, information_rotation, posterior_anisotropy_spheres,
                 moving_interference_comparison, noise_adaptation]
    paths = []
    for name, producer in zip(FIGURE_SIZES, producers):
        target = results / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context():
            figure = producer(root, results)
            try:
                figure.savefig(target, dpi=300, bbox_inches=None, metadata={'Title': ''})
            finally:
                plt.close(figure)
        paths.append(target)
    return paths
