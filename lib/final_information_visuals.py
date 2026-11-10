"""Likelihood geometry of the saved contact experiment, without new inference.

The normal graph chart is n(z)=(n_0-X z)/sqrt(1+||z||^2), z=(z_1,z_2)^T.
This module evaluates the Gaussian observation law at other planes; it does
not fit a posterior, simulate observations, or interpret KL contours as regions
of posterior probability.
"""
from __future__ import annotations

import numpy as np
from .plotting import plt
from matplotlib.lines import Line2D


CONTACT_INFORMATION_CAPTION = r'''Directional likelihood information for the original concentrated-input contact fixture at $t=16$. The generating normal is $n_0=e_3$, $X_0=(e_1,e_2)$, and the plane graph chart has $z=(z_1,z_2)^\top$ and normal $n(z)=(n_0-X_0z)/\sqrt{1+\|z\|^2}$. For the same frozen prescribed inputs and true variance, $p_z$ is the joint Gaussian response law with means $P_{\mathcal X(z)}v_j$ and covariance $\sigma^2I_3$. Panel (a) shows equal values of the exact divergence $D_{\mathrm{KL}}(p_0\|p_z)=z^T\mathcal F_{\mathrm{cond}}z/[2(1+\|z\|^2)]$ in native chart coordinates, with equal axis scales. These are likelihood-divergence contours, not credible or confidence regions. Panel (b) evaluates the same divergence along the two native information directions; $\delta$ is the physical angle between $n_0$ and $n(z)$. The marked rotations $5^\circ$ and $30^\circ$ produce nearly equal divergences, approximately $0.2396$ and $0.2308$, despite their different angular sizes. Thus this input design distinguishes rotations much more strongly in one direction. The local limit is $D_{\mathrm{KL}}=\tfrac12z^T\mathcal F_{\mathrm{cond}}z+o(\|z\|^2)$. This is expected observation-law information at the generating plane, distinct from finite-data posterior curvature. No new observations, inference, fitting, or experimental repetitions are introduced.'''


def contact_information_geometry(record, sigma2):
    """Recover native graph coordinates and information from frozen inputs."""
    V = np.asarray(record['input_vectors'], float)
    normal = np.asarray(record['true_normal'], float)
    sigma2 = float(sigma2)
    if (V.ndim != 2 or V.shape[1] != 3 or not np.isfinite(V).all()
            or normal.shape != (3,) or not np.isfinite(normal).all()
            or not np.isclose(normal @ normal, 1., atol=1e-12)
            or not np.isfinite(sigma2) or sigma2 <= 0):
        raise ValueError('Contact information requires saved real inputs, a unit normal and positive variance')
    if int(np.asarray(record['report_time']).item()) != len(V):
        raise ValueError('The saved information inputs must end at the declared checkpoint')
    # Fix the chart by native coordinate axes rather than by a fitted mode.
    candidate = np.eye(3)[np.argmin(np.abs(normal))]
    first = candidate-normal*(normal @ candidate)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    X = np.column_stack([first, second])
    S = V.T @ V
    F = (X.T @ S @ X + (normal @ S @ normal)*np.eye(2))/sigma2
    values = np.linalg.eigvalsh(F)
    if not np.allclose(values, np.asarray(record['fisher_eigenvalues']), rtol=1e-11, atol=1e-11):
        raise ValueError('Input-derived Fisher information disagrees with the frozen result')
    # The current fixed fixture has the two native chart axes as information
    # directions, so their labels need no additional algorithm-like names.
    if not (np.allclose(normal, [0., 0., 1.], atol=1e-12)
            and np.allclose(X, np.eye(3)[:, :2], atol=1e-12)
            and np.allclose(F, np.diag(np.diag(F)), atol=1e-11)):
        raise ValueError('This native-coordinate illustration requires the declared n_0=e_3 fixture')
    return dict(inputs=V, normal=normal, basis=X, S=S, F=F,
                fisher_eigenvalues=values, sigma2=sigma2, t=len(V))


def chart_normals(z, geometry):
    """Native unit normals associated with the manuscript's plane chart."""
    z = np.asarray(z, float)
    if z.shape[-1] != 2 or not np.isfinite(z).all():
        raise ValueError('A contact plane graph chart has two finite coordinates')
    return ((geometry['normal']-np.einsum('...a,ia->...i', z, geometry['basis']))
            / np.sqrt(1+np.sum(z*z, axis=-1))[..., None])


def exact_contact_kl(z, geometry):
    """Exact same-variance Gaussian KL at the saved input design."""
    z = np.asarray(z, float)
    return np.einsum('...a,ab,...b->...', z, geometry['F'], z)/(2*(1+np.sum(z*z, axis=-1)))


def direct_contact_kl(z, geometry):
    """Independent expression using the differences of Gaussian means."""
    n = chart_normals(z, geometry)
    P = np.eye(3)-np.einsum('...i,...j->...ij', n, n)
    P0 = np.eye(3)-np.outer(geometry['normal'], geometry['normal'])
    difference = np.einsum('...ij,tj->...ti', P-P0, geometry['inputs'])
    return np.sum(difference*difference, axis=(-1, -2))/(2*geometry['sigma2'])


def contact_information(evidence):
    """Show which physical normal tilts this saved experiment distinguishes."""
    # Local import allows the parent renderer to call this helper while its
    # shared visual utilities are already initialized.
    from .final_paper_visuals import FIGURE_WIDTH, clean, letter
    record = evidence.array('performance', 'anisotropy')
    sigma2 = evidence.config('performance')['anisotropy_sigma2']
    geometry = contact_information_geometry(record, sigma2)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.8), layout='constrained')

    coordinate = np.linspace(-.78, .78, 501)
    Z1, Z2 = np.meshgrid(coordinate, coordinate)
    grid = np.stack([Z1, Z2], axis=-1)
    kl = exact_contact_kl(grid, geometry)
    levels = [.01, .05, .15, .3]
    axes[0].contourf(Z1, Z2, kl, levels=[0., levels[-1]], colors=['#F4EDF7'])
    contour_styles = [':', '--', '-.', '-']
    axes[0].contour(Z1, Z2, kl, levels=levels, colors=['#7B3294'],
                    linewidths=1.2, linestyles=contour_styles)
    # A small legend in the empty side of the chart avoids stacking inline
    # contour labels over a genuinely very narrow information direction.
    handles = [Line2D([], [], color='#7B3294', linestyle=style,
                      label=r'$D_{\mathrm{KL}}='+f'{level:g}'+r'$')
               for level, style in zip(levels, contour_styles)]
    axes[0].legend(handles=handles, frameon=False, fontsize=12., loc='lower left',
                   handlelength=1.5, labelspacing=.35, borderaxespad=.3)
    axes[0].plot(0., 0., '+', color='#303030', markersize=6.)
    axes[0].set(xlabel=r'$z_1$', ylabel=r'$z_2$', xlim=(-.78, .78), ylim=(-.78, .78),
                xticks=[-.6, 0., .6], yticks=[-.6, 0., .6])
    axes[0].set_xlabel(r'$z_1$', fontsize=12.)
    axes[0].set_ylabel(r'$z_2$', fontsize=12.)
    axes[0].set_aspect('equal', adjustable='box')
    clean(axes[0])

    delta_deg = np.linspace(.25, 35., 201)
    delta = np.deg2rad(delta_deg)
    rays = [np.column_stack([np.tan(delta), np.zeros_like(delta)]),
            np.column_stack([np.zeros_like(delta), np.tan(delta)])]
    labels = [r'$z=(\tan\delta,0)^\top$', r'$z=(0,\tan\delta)^\top$']
    colors = ['#D55E00', '#0072B2']
    styles = ['-', '--']
    examples = [(5., np.array([np.tan(np.deg2rad(5.)), 0.])),
                (30., np.array([0., np.tan(np.deg2rad(30.))]))]
    for ray, label, c, style, (angle, point) in zip(rays, labels, colors, styles, examples):
        axes[1].plot(delta_deg, exact_contact_kl(ray, geometry), style, color=c, label=label)
        value = float(exact_contact_kl(point, geometry))
        axes[0].plot(*point, 'o', color=c, markersize=4.)
        axes[1].plot(angle, value, 'o', color=c, markersize=4.)
    axes[1].set(xlabel=r'$\delta\ ({}^\circ)$',
                ylabel=r'$D_{\mathrm{KL}}(p_0\,\Vert\,p_z)$', xlim=(0., 35.))
    axes[1].set_xlabel(r'$\delta\ ({}^\circ)$', fontsize=12.)
    axes[1].set_ylabel(r'$D_{\mathrm{KL}}(p_0\,\Vert\,p_z)$', fontsize=12.)
    axes[1].set_yscale('log')
    axes[1].set_box_aspect(1.)
    axes[1].legend(frameon=False, fontsize=12., loc='lower right')
    clean(axes[1])
    for ax, tag in zip(axes, ['(a)', '(b)']):
        letter(ax, tag)
        ax.texts[-1].set_fontsize(12.)
        ax.tick_params(axis='both', labelsize=11.)
    fig._final_selection = dict(
        scope='Exact likelihood geometry at the original frozen contact input design; not a new experiment',
        condition='static_concentrated', episode=0, t=geometry['t'],
        source_arrays=['input_vectors', 'true_normal', 'fisher_eigenvalues', 'report_time'],
        variance_source='performance.config.anisotropy_sigma2', sigma2=geometry['sigma2'],
        normal=geometry['normal'].tolist(), chart_basis=geometry['basis'].tolist(),
        fisher_matrix=geometry['F'].tolist(), fisher_eigenvalues=geometry['fisher_eigenvalues'].tolist(),
        formula='D_KL(p_0||p_z)=z^T F_cond z/[2(1+||z||^2)]',
        chart='n(z)=(n_0-X z)/sqrt(1+||z||^2), z=(z_1,z_2)^T, X=(e_1,e_2), n_0=e_3',
        contour_levels=levels, contour_scope='Equal Gaussian KL divergence; not credible or confidence regions',
        angular_curves='Exact Gaussian KL along the two native Fisher directions; delta is the physical normal angle',
        marked_angles_deg=[5., 30.],
        marked_KL=[float(exact_contact_kl(point, geometry)) for _, point in examples],
        interpretation='Expected information at the true plane; distinct from finite-data posterior curvature')
    fig._final_repetition_unit = 'One saved input design; analytic observation-law geometry, no new Monte Carlo repetitions'
    return [fig]
