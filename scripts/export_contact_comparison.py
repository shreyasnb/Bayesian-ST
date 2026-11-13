"""Export the two contact panels side by side from the existing saved results.

The individual notebook PDFs remain unchanged. This combined layout uses the
prediction-error panel's superset legend once, centered below both panels.
"""
from pathlib import Path
import argparse
import sys


def make_figure(root):
    """Read the same saved arrays and retain the individual plots' coordinates."""
    import numpy as np
    from lib.plotting import plt
    from lib.contact_transport_diagnostics import load_contact_transport_diagnostics

    DPI = 300
    FIGSIZE = (7, 2.8)
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE = 10.5, 10., 10.
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.family': 'serif', 'font.size': LABEL_FONT_SIZE,
        'axes.labelsize': LABEL_FONT_SIZE, 'axes.titlesize': LABEL_FONT_SIZE,
        'xtick.labelsize': TICK_FONT_SIZE, 'ytick.labelsize': TICK_FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE, 'axes.labelpad': 4.,
        'figure.dpi': DPI, 'savefig.dpi': DPI,
    })
    LABELS = {
        'mean': r'$\overline P_t^+$',
        'mode': r'$P_{\widehat{\mathcal X}_t}$',
        'iso': r'$\overline P_t^{\mathrm{iso}}$',
        'mean_identity': r'$\overline P_t^+,\ G_t=I_3$',
    }
    COLORS = {'mean': '#0072B2', 'mode': '#D55E00',
              'iso': '#CC79A7', 'mean_identity': '#484848'}
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, layout='constrained')

    # Same saved unit-query risks as posterior_anisotropy_risk.pdf.
    with np.load(root / 'results/contact/data/paper/performance/anisotropy.npz',
                 allow_pickle=False) as saved:
        query_family = np.asarray(saved['query_family']) == 0
        theta = np.asarray(saved['query_angles'])[query_family].copy()
        unit_queries = np.asarray(saved['unit_queries'])[query_family].copy()
        risk_full = np.asarray(saved['risks_full'])[query_family].copy()
        risk_iso = np.asarray(saved['risks_isotropic'])[query_family].copy()
    np.testing.assert_allclose(unit_queries,
        np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]),
        rtol=0., atol=1e-14)
    ax = axes[0]
    ax.plot(theta, risk_full, color=COLORS['mean'], label=LABELS['mean'],
            linestyle='-', linewidth=1.6)
    ax.plot(theta, risk_iso, color=COLORS['iso'], label=LABELS['iso'],
            linestyle='-', linewidth=1.6)
    ax.set_xlabel(r'$\theta$ (rad)')
    ax.set_ylabel(r'$r_t(v(\theta))/v_0^2$')
    ax.set(xticks=np.linspace(0., 2*np.pi, 5),
           xticklabels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticks([.05, .10, .15])

    # Same 48 action times, four methods, and saved interval as prediction_error_sphere.pdf.
    transport = load_contact_transport_diagnostics(root)
    ax = axes[1]
    for curve in transport['sphere_curves']:
        method = curve['method']
        points = np.asarray(curve['points'], dtype=float)
        ax.plot(points[:, 0], points[:, 1], color=COLORS[method], label=LABELS[method],
                linestyle='-', linewidth=1.8)
        if method in ('mean', 'mean_identity'):
            ax.fill_between(points[:, 0], points[:, 2], points[:, 3],
                            color=COLORS[method], alpha=.14, linewidth=0, zorder=1)
    ax.set(xlabel=r'$t$', ylabel=r'$\mathcal E_t(D_t)$ $((\mathrm{mm}/\mathrm{s})^2)$',
           xlim=(1, int(transport['config']['T'])))

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.9)
        ax.tick_params(axis='both', which='both', direction='in', width=.9,
                       top=True, right=True, bottom=True, left=True,
                       labeltop=False, labelright=False, labelbottom=True, labelleft=True)
        ax.xaxis.set_label_position('bottom')
        ax.yaxis.set_label_position('left')
        ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
    fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center',
               ncol=4, frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig.get_layout_engine().set(h_pad=.04, w_pad=.04)
    return fig


def export(root):
    sys.path.insert(0, str(root))
    fig = make_figure(root)
    target = root / 'results/main/contact/contact_comparison.pdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, bbox_inches='tight', pad_inches=.04)
    fig.savefig(target.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=.04)
    print(target)
    from matplotlib import pyplot as plt
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    export(parser.parse_args().root.resolve())
