"""Small physical figures for the contact notebook.

The positions and true normals in these figures are plant/scoring records.
They are never estimator inputs.  Posterior point clouds illustrate a density;
reported intervals must come from the checked continuous-CDF calculations.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .plotting import plt, save
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .contact_posterior import normal_quadrature


BLUE = '#0072B2'
ORANGE = '#D55E00'
GREY = '#6E777D'
BLACK = '#202124'
PALE = '#DEE4E8'


def _finish(fig, path):
    if not plt.rcParams['text.usetex'] or fig.dpi != 200:
        raise AssertionError('Contact figures require TeX and 200 dpi')
    if path is not None:
        save(fig, Path(path))
    return fig


def _basis(normal):
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    seed = np.eye(3)[np.argmin(abs(normal))]
    first = np.cross(normal, seed)
    first /= np.linalg.norm(first)
    return np.column_stack([first, np.cross(normal, first)])


def _normal(projector):
    values, vectors = np.linalg.eigh((projector + projector.T) / 2)
    return vectors[:, np.argmin(values)]


def _angle(n, m):
    return np.degrees(np.arccos(np.clip(abs(n @ m), 0, 1)))


def _plane(ax, centre, normal, size, color, alpha=.2, linewidth=1):
    offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    vertices = np.asarray(centre) + size * offsets @ _basis(normal).T
    surface = Poly3DCollection([vertices], facecolors=color,
        edgecolors=color, alpha=alpha, linewidths=linewidth)
    ax.add_collection3d(surface)
    return vertices


def _axes(ax, centre, radius, units='mm', labels=None):
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c-radius, c+radius)
    if labels is None:
        labels = [rf'${axis}$ ({units})' for axis in ('x', 'y', 'z')]
    ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-52)
    ax.tick_params(labelsize=9, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo['grid'].update(color=(.7, .7, .7, .25), linewidth=.5)
        axis.set_major_locator(plt.MaxNLocator(3))


def _surface(ax, data, index, zoom=False):
    positions = np.asarray(data['positions']) * 1000
    radius = data.get('condition', {}).get('radius')
    if radius is not None:
        radius *= 1000
        u, v = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 25))
        ax.plot_surface(radius*np.cos(u)*np.sin(v), radius*np.sin(u)*np.sin(v),
            radius*np.cos(v), color=PALE, alpha=.16, linewidth=0, shade=False)
    else:
        radius = max(np.max(np.ptp(positions, axis=0)) * .5, 20.)
        _plane(ax, positions.mean(axis=0), data['normals'][index], radius, GREY, .08)
    ax.plot(*positions[:index+1].T, color=BLACK, lw=1.8, alpha=.8)
    ax.scatter(*positions[index], color=BLACK, s=35, depthshade=False)
    _plane(ax, positions[index], data['normals'][index], radius*.25, BLUE, .2)
    if zoom:
        _axes(ax, positions[index], radius*.5)
    else:
        centre = np.zeros(3) if data.get('condition', {}).get('radius') else positions.mean(axis=0)
        _axes(ax, centre, radius*1.14)
    return positions, radius


def contact_scene(data, index=179, path=None):
    """One observed contact: physical path, then its velocity projection.

    The right panel is in velocity space.  Arrows have their physical lengths;
    no arbitrary visual scaling is mixed into their mm/s axes.
    """
    index = int(index)
    if data.get('condition', {}).get('anisotropic', False):
        raise ValueError('The projection illustration assumes isotropic resistance')
    fig = plt.figure(figsize=(13, 5.7), dpi=200, layout='constrained')
    physical = fig.add_subplot(121, projection='3d')
    response = fig.add_subplot(122, projection='3d')
    _surface(physical, data, index)
    physical.set_title('Contact on the surface', pad=12)
    free = np.asarray(data['free_velocities'][index]) * 1000
    actual = np.asarray(data['velocities'][index]) * 1000
    observed = np.asarray(data['observations'][index]) * 1000
    normal = np.asarray(data['normals'][index])
    limit = max(np.linalg.norm(free), np.linalg.norm(observed), .1) * 1.1
    _plane(response, np.zeros(3), normal, limit*.72, GREY, .1)
    for vector, color in ((free, BLUE), (actual, ORANGE)):
        response.quiver(0, 0, 0, *vector, color=color, linewidth=2.3,
            arrow_length_ratio=.13)
    response.plot(*np.stack([actual, free]).T, color=GREY, linestyle='--', lw=1.6)
    response.scatter(*observed, facecolors='white', edgecolors=BLACK, s=45,
        linewidth=1.3, depthshade=False)
    _axes(response, np.zeros(3), limit, units='mm/s',
        labels=[rf'$v_{c}$'+' (mm/s)' for c in ('x', 'y', 'z')])
    response.set_title(r'Force $F$ produces velocity $P(F/d)$', pad=12)
    fig.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.3, label=r'Free velocity $F/d$'),
        Line2D([], [], color=ORANGE, lw=2.3, label=r'Constrained velocity $\dot p$'),
        Line2D([], [], marker='o', markerfacecolor='white', markeredgecolor=BLACK,
            linestyle='', label=r'Observed velocity $u$'),
        Line2D([], [], color=GREY, lw=1.6, linestyle='--', label='Normal component')],
        loc='outside lower center', ncol=2, fontsize=11)
    fig.suptitle(rf'One observation at $t={float(data["times"][index]):.2f}$ s', fontsize=15)
    return _finish(fig, path)


def tracking_scene(data, corrected_projectors, forecast_projectors, index=179,
                   history_indices=None, path=None, baseline_forecasts=None):
    """Physical turning, followed by a magnified angular forecast comparison.

    The last estimate is corrected_projectors[t-1]; the forecast uses only
    observations through t-1. The right panel uses the projective logarithm
    around the current *true* normal for scoring. Its radial distance is the
    actual normal error in degrees, so small errors remain visible.
    If baseline_forecasts is supplied, compare its prior at t with the motion
    prior at t; both use observations through t-1. This shows accumulated
    differences between policies, rather than only the final one-step change.
    """
    index = int(index)
    if index < 1:
        raise ValueError('A previous observation is required')
    if history_indices is None:
        history_indices = np.unique(np.maximum(0, index-np.array([20, 10, 5, 1])))
    history_indices = np.asarray(history_indices, int)
    if np.any(history_indices >= index) or np.any(history_indices < 0):
        raise ValueError('History must precede the prediction time')
    fig = plt.figure(figsize=(13, 5.7), dpi=200, layout='constrained')
    physical = fig.add_subplot(121, projection='3d')
    angular = fig.add_subplot(122)
    true_normal = np.asarray(data['normals'][index])
    centre = np.zeros(3)
    for j, past in enumerate(history_indices):
        alpha = .05 + .15*(j+1)/len(history_indices)
        _plane(physical, centre, data['normals'][past], .65, ORANGE, alpha, .7)
    _plane(physical, centre, true_normal, .75, GREY, .1, 1.8)
    physical.quiver(0, 0, 0, *(.95*true_normal), color=BLACK, linewidth=1.8,
        arrow_length_ratio=.12)
    physical.set_title('The surface turns during memory', pad=12)
    _axes(physical, centre, 1.1, labels=[r'$x$', r'$y$', r'$z$'])
    physical.set_xticks([]); physical.set_yticks([]); physical.set_zticks([])
    reference = (corrected_projectors[index-1] if baseline_forecasts is None
                 else baseline_forecasts[index])
    estimates = [_normal(np.asarray(reference)),
                 _normal(np.asarray(forecast_projectors[index]))]
    labels = (('Last estimate', 'One step ahead') if baseline_forecasts is None
              else ('Identity prediction', 'Causal motion prediction'))
    coordinates = []
    basis = _basis(true_normal)
    for normal in estimates:
        if normal @ true_normal < 0:
            normal = -normal
        angle = np.arccos(np.clip(normal @ true_normal, 0, 1))
        local = normal - true_normal*(normal @ true_normal)
        length = np.linalg.norm(local)
        coordinates.append(np.zeros(2) if length < 1e-14 else
            np.degrees(angle)*(basis.T @ local)/length)
    limit = max(np.linalg.norm(coordinates, axis=1).max()*1.25, .1)
    for xy, color, label in zip(coordinates, (ORANGE, BLUE), labels):
        angular.annotate('', xy=xy, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=2))
        angular.scatter(*xy, color=color, s=65, zorder=4,
            label=label+rf': ${np.linalg.norm(xy):.1f}^\circ$')
    angular.scatter(0, 0, marker='+', color=BLACK, s=140, linewidths=2,
        label='Current true normal', zorder=5)
    angular.axhline(0, color=GREY, lw=.6, alpha=.35)
    angular.axvline(0, color=GREY, lw=.6, alpha=.35)
    angular.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal',
        xlabel=r'Normal rotation $\theta_1$ (deg)',
        ylabel=r'Normal rotation $\theta_2$ (deg)',
        title='Forecast error before the next observation')
    angular.legend(loc='upper left', fontsize=10)
    angular.grid(alpha=.15)
    angular.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Predict the current tangent plane before seeing its velocity', fontsize=15)
    return _finish(fig, path)


def illustrative_samples(A, query, count=256, seed=47013):
    """Weighted quadrature resampling for a picture, never for scored intervals.

    Antipodal signs are aligned only for drawing normal arrows.  The response
    P_n w is sign invariant.  The sampling seed and count are fixed before test.
    """
    rule = normal_quadrature(-np.asarray(A), radial_order=64, angular_order=192)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rule['weights']), size=int(count), p=rule['weights'])
    normals = rule['normal_nodes'][indices].copy()
    modal = np.linalg.eigh(-np.asarray(A))[1][:, -1]
    normals[(normals@modal) < 0] *= -1
    query = np.asarray(query, float)
    responses = query - normals * (normals@query)[:, None]
    return normals, responses


def posterior_scene(data, index, A, query, mean, lower=None, upper=None,
                    path=None, seed=47013, response_components=(0, 2)):
    """Plausible tangent planes mapped to two clean response components.

    ``mean`` is the checked 3-vector E[P]w, not a sampled cloud average.  The
    optional interval arguments are reserved for a matching response-band
    figure; no multivariate coverage region is implied by this cloud.
    """
    del lower, upper
    fig = plt.figure(figsize=(13.2, 5.8), dpi=200, layout='constrained')
    state = fig.add_subplot(121, projection='3d')
    response = fig.add_subplot(122)
    normals, values = illustrative_samples(A, query, seed=seed)
    true_normal = np.asarray(data['normals'][int(index)])
    modal = np.linalg.eigh(-np.asarray(A))[1][:, -1]
    if modal @ true_normal < 0:
        normals *= -1
    _plane(state, np.zeros(3), true_normal, .76, GREY, .08, 1.8)
    for normal in normals[np.linspace(0, len(normals)-1, 9, dtype=int)]:
        _plane(state, np.zeros(3), normal, .7, BLUE, .06, .55)
    state.quiver(0, 0, 0, *(.95*true_normal), color=BLACK, linewidth=2,
        arrow_length_ratio=.12)
    state.scatter(*(normals*.95).T, color=BLUE, alpha=.3, s=7, depthshade=False)
    _axes(state, np.zeros(3), 1.1, labels=[r'$x$', r'$y$', r'$z$'])
    state.set_xticks([]); state.set_yticks([]); state.set_zticks([])
    state.set_title('Uncertain tangent plane', pad=12)
    values = 1000*values
    true_velocity = (np.eye(3)-np.outer(true_normal, true_normal))@np.asarray(query)*1000
    # This figure is used only for matched isotropic resistance.  In the
    # resistance-mismatch stress the physical KKT response must be supplied
    # explicitly rather than pretending P w is the plant velocity.
    if data.get('condition', {}).get('anisotropic', False):
        from .contact_mechanics import solve_contact_velocity
        free = np.asarray(data['free_velocities'])
        d = float(np.sum(np.asarray(data['forces'])*free)/np.sum(free*free))
        true_velocity = 1000*solve_contact_velocity(d*np.asarray(query),
            true_normal, data['resistance'])[0]
    components = np.asarray(response_components, int)
    if components.shape != (2,) or len(set(components)) != 2 or np.any((components < 0) | (components > 2)):
        raise ValueError('Select two different velocity components')
    projected = values[:, components]
    predicted = 1000*np.asarray(mean)[components]
    truth = true_velocity[components]
    response.scatter(*projected.T, color=BLUE, alpha=.3, s=17,
        edgecolors='none', label='Illustrative posterior draws')
    response.scatter(*predicted, color=BLUE, s=95, marker='D',
        edgecolors='white', linewidths=.8, zorder=5,
        label=r'Expected response $\mathrm{E}[P]w$')
    response.scatter(*truth, color=BLACK, s=145, marker='+', linewidths=2.1,
        zorder=6, label='True mechanical response')
    points = np.vstack([projected, predicted, truth])
    low, high = points.min(axis=0), points.max(axis=0)
    span = max(np.ptp(points, axis=0).max(), .01)*1.3
    centre = (low+high)/2
    names = ['x', 'y', 'z']
    response.set(xlim=(centre[0]-span/2, centre[0]+span/2),
        ylim=(centre[1]-span/2, centre[1]+span/2), aspect='equal',
        xlabel=rf'$\dot p_{names[components[0]]}$ (mm/s)',
        ylabel=rf'$\dot p_{names[components[1]]}$ (mm/s)',
        title='Response to the test force')
    response.grid(alpha=.15)
    response.spines[['top', 'right']].set_visible(False)
    response.legend(loc='best', fontsize=10)
    fig.suptitle(r'One fixed force maps plane uncertainty into velocity uncertainty', fontsize=15)
    return _finish(fig, path)


def response_band(times, truth, mean, lower, upper, path=None,
                  component='y', title='A prediction and its uncertainty over time'):
    """One predeclared query/component, with pointwise continuous-CDF intervals."""
    times, truth, mean, lower, upper = [np.asarray(x, float)
        for x in (times, truth, mean, lower, upper)]
    if not (times.shape == truth.shape == mean.shape == lower.shape == upper.shape):
        raise ValueError('Time, truth, prediction and interval arrays must match')
    if np.any(lower > upper):
        raise ValueError('Reversed prediction interval')
    fig, ax = plt.subplots(figsize=(12, 4.7), dpi=200, layout='constrained')
    ax.fill_between(times, 1000*lower, 1000*upper, color=BLUE, alpha=.2,
        linewidth=0, label=r'Pointwise $95\%$ interval')
    ax.plot(times, 1000*mean, color=BLUE, lw=2.2, label='Expected response')
    ax.plot(times, 1000*truth, color=BLACK, lw=1.8, linestyle='--', label='True response')
    ax.set(xlabel='Time (s)', ylabel=rf'$\dot p_{component}$ (mm/s)', title=title)
    ax.legend(loc='best', ncol=3, fontsize=11)
    ax.grid(axis='y', alpha=.18, linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    return _finish(fig, path)


def response_comparison(series, path=None, component='x', center_on_truth=False):
    """Two interval policies for the same predeclared query and trajectory.

    ``series`` maps two short labels to dictionaries containing times, truth,
    mean, lower and upper.  These are checked continuous-CDF pointwise
    intervals, not simultaneous trajectory bands or noisy-sensor intervals.
    """
    if len(series) != 2:
        raise ValueError('Show exactly two interval policies')
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), dpi=200,
        layout='constrained', sharex=True, sharey=True)
    for ax, (label, values) in zip(axes, series.items()):
        arrays = [np.asarray(values[key], float) for key in
                  ('times', 'truth', 'mean', 'lower', 'upper')]
        times, truth, mean, lower, upper = arrays
        if any(a.shape != times.shape for a in arrays):
            raise ValueError('Time, truth, prediction and interval arrays must match')
        if np.any(lower > upper):
            raise ValueError('Reversed prediction interval')
        if center_on_truth:
            mean, lower, upper = mean-truth, lower-truth, upper-truth
            truth = np.zeros_like(truth)
        ax.fill_between(times, 1000*lower, 1000*upper, color=BLUE, alpha=.2,
            linewidth=0, label=r'Pointwise $95\%$ interval')
        ax.plot(times, 1000*mean, color=BLUE, lw=1.8, marker='o', markersize=3,
            label='Expected response error' if center_on_truth else 'Expected response')
        ax.plot(times, 1000*truth, color=BLACK, lw=1.4, linestyle='--',
            label='Zero error' if center_on_truth else 'True response')
        ylabel = (rf'$\dot p_{component}-\dot p^{{\mathrm{{true}}}}_{component}$ (mm/s)'
                  if center_on_truth else rf'$\dot p_{component}$ (mm/s)')
        ax.set(ylabel=ylabel, title=label)
        ax.grid(axis='y', alpha=.18, linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel('Time (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=11)
    return _finish(fig, path)


def paired_prediction_errors(frame, reference, comparisons, condition='sphere_small',
                             time=600, value='clean_mse', reference_value='mode_mse', labels=None, path=None):
    """Each dot is one independent trajectory, paired against a point method.

    The tidy input requires condition, t, trial, method and ``value`` columns.
    Queries have already been averaged within each trajectory, not treated as
    independent replicates.  Values are converted from (m/s)^2 to (mm/s)^2.
    """
    comparisons = list(comparisons)
    labels = {} if labels is None else labels
    if not 1 <= len(comparisons) <= 2:
        raise ValueError('Use one or two comparisons per figure')
    table = frame[(frame.condition == condition) & (frame.t == time)]
    planned = table.trial.nunique()
    paired = table.pivot(index='trial', columns='method', values=value)
    reference_rows = table[table.method == reference].set_index('trial')[reference_value]
    if not len(reference_rows):
        raise ValueError('The reference method is absent')
    fig, axes = plt.subplots(1, len(comparisons), figsize=(6.7*len(comparisons), 5.7),
        dpi=200, layout='constrained', squeeze=False)
    shown = np.concatenate([reference_rows.to_numpy(),
        paired.reindex(columns=comparisons).to_numpy().ravel()])*1e6
    shown = shown[np.isfinite(shown)]
    if not len(shown):
        raise ValueError('No resolved paired losses')
    limit = max(float(shown.max())*1.08, np.finfo(float).eps)
    for ax, method in zip(axes[0], comparisons):
        if method not in paired:
            raise ValueError(f'Missing comparison: {method}')
        import pandas as pd
        subset = pd.concat([reference_rows.rename('reference'),
            paired[method].rename('comparison')], axis=1).dropna()*1e6
        subset = subset[np.isfinite(subset).all(axis=1)]
        if not len(subset):
            raise ValueError('No resolved paired trajectories')
        x, y = subset['reference'].to_numpy(), subset['comparison'].to_numpy()
        ax.plot([0, limit], [0, limit], color=GREY, linestyle='--', lw=1.2)
        ax.scatter(x, y, s=35, color=BLUE, alpha=.6, linewidths=.4,
            edgecolors='white')
        ax.set(xlim=(0, limit), ylim=(0, limit), aspect='equal',
            xlabel=r'Point prediction MSE $\mathrm{(mm/s)^2}$',
            ylabel=r'Mean prediction MSE $\mathrm{(mm/s)^2}$',
            title=labels.get(method, method))
        ax.text(.04, .96, rf'Lower loss in ${np.count_nonzero(y<x)}/{len(x)}$ trajectories',
            transform=ax.transAxes, va='top', fontsize=11)
        if len(x) != planned:
            ax.text(.04, .04, rf'Unresolved: ${planned-len(x)}/{planned}$ trajectories',
                transform=ax.transAxes, va='bottom', fontsize=10, color=ORANGE)
        ax.grid(alpha=.15, linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
    return _finish(fig, path)
