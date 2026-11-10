"""Visible source for numerical accuracy, memory and study overview figures.

These templates read frozen evidence. They do not run inference or simulations.
"""
from scripts.notebook_supplementary_common import plot_cell


_ACCURACY_1 = r'''
small = evidence.frame('theory', 'numerical_accuracy')
large = evidence.frame('computation', 'accuracy')
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, layout='constrained')
for name, label, c, symbol in [
        ('mean_error', 'Mean', color('mean'), 'o'),
        ('log_normalizer_error', 'Log normalizer', color('mode'), 's'),
        ('second_moment_error', 'Second moment', color('iso'), '^'),
        ('quantile_error', 'Quantile', '#009E73', 'D')]:
    if name not in small:
        continue
    selected = np.isfinite(small[name])
    axes[0].scatter(small.loc[selected, 'span'],
        np.maximum(small.loc[selected, name], 1e-17), color=c,
        marker=symbol, s=18, label=label)
axes[0].set(xscale='log', yscale='log', xlabel='Spectral span',
            ylabel='Reference error')
axes[0].legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
for names, label, c, symbol in [
        (('mean_reference_error', 'mean_error', 'selected_mean_error',
          'maximum_selected_mean_error'), 'Selected mean', color('mean'), 'o'),
        (('normalizer_reference_error', 'log_normalizer_error', 'logz_error',
          'logz_absolute_error'), 'Log normalizer', color('mode'), 's')]:
    column = next((name for name in names if name in large), None)
    if column is None:
        continue
    selected = np.isfinite(large[column])
    axes[1].scatter(large.loc[selected, 'n'],
        np.maximum(large.loc[selected, column], 1e-17), color=c,
        marker=symbol, s=18, label=label)
axes[1].set_xscale('log', base=2)
axes[1].set(yscale='log', xlabel='Ambient dimension',
            ylabel='Reference error')
axes[1].legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
for ax, tag in zip(axes, ['(a)', '(b)']):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
    ax.text(.015, .985, tag, transform=ax.transAxes, ha='left', va='top')
'''

_ACCURACY_2 = r'''
small = evidence.frame('theory', 'numerical_accuracy')
flags = [name for name in ('mean_ok', 'normalizer_ok', 'second_moment_ok',
    'derivative_ok', 'reference_ok', 'quantile_ok') if name in small]
if not flags:
    raise ValueError('The saved small-state numerical flags are missing')
passed = np.array([int(small[name].fillna(False).astype(bool).sum()) for name in flags])
total = np.full(len(flags), len(small))
x = np.arange(len(flags))
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
ax.bar(x, passed, color=color('mean'), label='Resolved')
ax.bar(x, total-passed, bottom=passed, color='#B8BEC4', label='Unresolved')
ax.set(xticks=x, xticklabels=[name.replace('_ok', '').replace('_', ' ') for name in flags],
       xlabel='Numerical target', ylabel='Number of cases')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#BFC4C8', alpha=.30, linewidth=.45)
'''

_ACCURACY_3 = r'''
large = evidence.frame('computation', 'accuracy')
targets = [('mean_ok', 'Mean', None), ('normalizer_ok', 'Normalizer', None),
    ('reference_ok', 'Reference', None),
    ('second_moment_ok', 'Second moment', 'second_moment_evaluated'),
    ('tail_ok', 'Response quantiles', 'tail_evaluated'),
    ('state_report_ok', 'State regions', None)]
labels, resolved, unresolved, not_evaluated = [], [], [], []
for flag, label, evaluated in targets:
    if flag not in large:
        continue
    requested = large[flag].notna()
    if evaluated is not None and evaluated in large:
        requested = large[evaluated].eq(True)
    if flag == 'reference_ok' and 'reference_scope' in large:
        requested &= ~large.reference_scope.fillna('').str.contains('not supplied', case=False)
    good = large[flag].eq(True) & requested
    labels.append(label)
    resolved.append(int(good.sum()))
    unresolved.append(int((requested & ~good).sum()))
    not_evaluated.append(int((~requested).sum()))
x = np.arange(len(labels))
first, second = np.array(resolved), np.array(unresolved)
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
ax.bar(x, resolved, color=color('mean'), label='Resolved')
ax.bar(x, unresolved, bottom=first, color=color('mode'), label='Unresolved')
ax.bar(x, not_evaluated, bottom=first+second, color='#B8BEC4', label='Not evaluated')
ax.set(xticks=x, xticklabels=labels, ylabel='Number of cases')
ax.tick_params(axis='x', rotation=20)
ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.02), fontsize=LEGEND_FONT_SIZE)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#BFC4C8', alpha=.30, linewidth=.45)
'''

_MEMORY = r'''
memory = _normalize(evidence.frame('computation', 'memory'))
if 'case' in memory:
    selected = memory[memory.case.eq('data_T64')]
    if not selected.empty:
        memory = selected
if 'method' in memory:
    memory = memory[memory.method.eq('mean')]
if 'report_every' in memory:
    memory = memory[memory.report_every.eq(0)]
if 'action_cadence' in memory:
    intervals = sorted(memory.action_cadence.dropna().unique())
    if len(intervals) != 1:
        raise ValueError('Memory records require one declared task-decision interval')
    memory = memory[memory.action_cadence.eq(intervals[0])]
if memory.empty:
    raise ValueError('The saved posterior-mean memory cohort is missing')
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
for column, label, c, style in [
        ('persistent_array_bytes', 'Stored arrays', color('mean'), '-'),
        ('python_numpy_peak_bytes', 'Tracked allocation', color('mode'), '--')]:
    if column in memory:
        points = memory.groupby('n')[column].median()/1024
        ax.plot(points.index, points, marker='o', linestyle=style, color=c, label=label)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set(xlabel='Number of sensors' if example == 'beamforming' else 'Ambient dimension',
       ylabel='Memory (KiB)')
ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
'''

# A complete ordinary plotting block, inserted into the beamforming overview.
_BEAM_SUMMARY = r'''
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, dpi=DPI, layout='constrained')
frame = _primary(evidence.frame('performance', 'stationary', 'metrics'))
if 'ratio' not in frame:
    horizon = _column(frame, 'L', 't', 'observations')
    frame['ratio'] = frame[horizon]/frame.n
endpoint = 'true_ratio_db'
selected = frame[frame.ratio.eq(1.) & frame.condition.eq('static') & frame.n.isin([32,64,128])]
for method in ('mean', 'mode', 'scm_loaded', 'great_window'):
    rows = []
    for n, group in selected[selected.method.eq(method)].groupby('n', sort=True):
        rows.append((n, *_mean_interval(group.groupby('episode')[endpoint].mean())))
    if not rows:
        continue
    points = np.asarray(rows)
    axes[0, 0].plot(points[:, 0], points[:, 1], marker=marker(method), color=color(method),
                    label=method_label(method, example))
    if method == 'mean':
        axes[0, 0].fill_between(points[:, 0], points[:, 2], points[:, 3], color=color(method), alpha=.10)
axes[0, 0].set(xlabel='Number of sensors', ylabel=r'$\ell_t$' + '\n(dB)')

moving = evidence.frame('performance', 'moving_episodes', 'episode_metrics', 'moving', 'stationary')
endpoint = 'true_ratio_db'
moving = _episode_losses(moving, endpoint)
conditions = list(moving.condition.drop_duplicates())
condition = 'smooth' if 'smooth' in conditions else conditions[0]
moving = moving[moving.condition.eq(condition)]
n = 32 if 32 in set(moving.n) else int(moving.n.iloc[0])
moving = moving[moving.n.eq(n)]
base = moving[moving.method.eq('mean')].set_index('episode')[endpoint]
values, labels, box_colors = [], [], []
for method in ('mode', 'iso', 'isotropic', 'great_window', 'great', 'great_ew'):
    other = moving[moving.method.eq(method)].set_index('episode')[endpoint]
    common = base.index.intersection(other.index)
    if len(common):
        values.append((other.loc[common]-base.loc[common]).to_numpy())
        labels.append(method_label(method, example)); box_colors.append(color(method))
boxes = axes[0, 1].boxplot(values, vert=False, patch_artist=True,
    medianprops={'color': 'white', 'linewidth': 1.2}, flierprops={'markersize': 1.8, 'alpha': .35})
axes[0, 1].set_yticks(np.arange(1, len(labels)+1), labels=labels)
for patch, c in zip(boxes['boxes'], box_colors):
    patch.set_facecolor(c); patch.set_alpha(.85)
axes[0, 1].axvline(0., linestyle='--', color='#222222', linewidth=1.)
axes[0, 1].set_xlabel('Power difference (dB)')

adaptation = _normalize(evidence.frame('prediction', 'prediction_adaptation', 'adaptation'))
if 'split' in adaptation and adaptation.split.eq('test').any():
    adaptation = adaptation[adaptation.split.eq('test')]
if 'control' in adaptation:
    adaptation = adaptation[adaptation.control.eq('mean')]
adaptation = adaptation[adaptation.condition.eq('noise_change')]
if adaptation.empty:
    raise ValueError('Missing saved noise-change simulations')
time = 't'
for method in ('known', 'exact_known', 'known_forgetting', 'misspecified', 'learned'):
    group = adaptation[adaptation.method.eq(method)]
    if group.empty:
        continue
    grouped = group.groupby(time).sigma2_estimate
    points = grouped.median()
    axes[1, 0].plot(points.index, points, color=color(method),
        linestyle='--' if method == 'misspecified' else '-', marker=marker(method), label=method_label(method))
    if method == 'learned':
        axes[1, 0].fill_between(points.index, grouped.quantile(.25), grouped.quantile(.75),
                              color=color(method), alpha=.13)
measurement_indices = np.arange(1, 257)
true_noise_variance = np.where(measurement_indices <= 128, .2, .8)
axes[1, 0].step(measurement_indices, true_noise_variance, where='mid', color='#222222',
                linestyle=':', linewidth=1.35, label=r'$\sigma_t^2$')
axes[1, 0].axvline(128.5, color='#70777D', linestyle=':', linewidth=.7)
axes[1, 0].set(xlabel=r'Measurement index $t$',
               ylabel=r'Noise variance $\widehat\sigma_t^2$')
axes[1, 0].legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc='upper left')

cost = _normalize(evidence.frame('computation', 'summary'))
cost = cost[cost.case.eq('task_stationary')]
stationary = _normalize(evidence.frame('performance', 'stationary'))
horizon = _column(stationary, 'L', 't', 'observations')
stationary = stationary[stationary[horizon].eq(stationary.n) & stationary.episode.eq(0)]
for method in ('mean', 'mode', 'scm_loaded', 'great_window'):
    x, y = [], []
    for n, group in cost[cost.method.eq(method)].groupby('n', sort=True):
        group = group[group.action_cadence.eq(n)]
        target = stationary[stationary.n.eq(n) & stationary.method.eq('mean')]
        if group.empty or target.empty:
            continue
        reference = target.oracle_power.dropna().unique()
        if len(reference) != 1 or reference[0] <= 0:
            raise ValueError('The saved timing simulation requires one positive reference power')
        loss = float(group.task_loss.mean())
        if not np.isfinite(loss) or loss <= 0:
            continue
        x.append(float(group.median_ms.median()))
        y.append(10*np.log10(loss/reference[0]))
    if x:
        axes[1, 1].plot(x, y, marker=marker(method), color=color(method), label=method_label(method, example))
axes[1, 1].set(xscale='log', xlabel='Computation time (ms)',
                ylabel=r'$\ell_t$' + '\n(dB)')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False, fontsize=LEGEND_FONT_SIZE)
for ax, tag in zip(axes.ravel(), ['(a)', '(b)', '(c)', '(d)']):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x' if ax is axes[0, 1] else 'both', color='#BFC4C8', alpha=.30, linewidth=.45)
    ax.text(.015, .985, tag, transform=ax.transAxes, ha='left', va='top')
'''

_CONTACT_SUMMARY = r'''
from lib.distribution_geometry_visuals import bingham_sphere_mesh, real_bingham_log_density
record = evidence.array('performance', 'anisotropy')
parameters = [_saved(record, 'normal_A'), _saved(record, 'iso_normal_A')]
normalizers = [_saved(record, 'normal_logz'), _saved(record, 'iso_normal_logz')]
meshes = [bingham_sphere_mesh(A, 161, 321).nodes for A in parameters]
densities = [real_bingham_log_density(A, float(logz), mesh)
             for A, logz, mesh in zip(parameters, normalizers, meshes)]
peak = max(float(values.max()) for values in densities)
norm = colors.LogNorm(vmin=np.exp(peak-4.*np.log(10.)), vmax=np.exp(peak), clip=True)
cmap = colors.LinearSegmentedColormap.from_list('density_blue_pink_yellow', [
    (0., '#263a8b'), (.28, '#55209a'), (.53, '#b33695'),
    (.72, '#ed78a4'), (.88, '#ffc85a'), (1., '#f4f52b')], N=256)
cmap.set_under(cmap(0.)); cmap.set_over(cmap(1.))
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout='constrained')
grid = fig.add_gridspec(2, 2)
spheres = [fig.add_subplot(grid[0, index], projection='3d') for index in range(2)]
el, az = np.deg2rad([25., -50.])
facing = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
for ax, nodes, log_density in zip(spheres, meshes, densities):
    ax.computed_zorder = False
    surface = ax.plot_surface(nodes[..., 0], nodes[..., 1], nodes[..., 2],
        facecolors=cmap(norm(np.exp(log_density))), shade=False, linewidth=0,
        antialiased=False, rstride=1, cstride=1, zorder=1)
    surface.set_rasterized(True)
    phi = np.linspace(0., 2*np.pi, 241)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        ring = np.zeros((len(phi), 3))
        ring[:, first], ring[:, second] = 1.006*np.cos(phi), 1.006*np.sin(phi)
        visible = np.where((ring@facing >= 0.)[:, None], ring, np.nan)
        ax.plot(*visible.T, color='#B7C0CB', linewidth=.5, alpha=.55, zorder=3)
    mode = np.array(_saved(record, 'mode_normal'), copy=True)
    mode /= np.linalg.norm(mode)
    visible_mode = mode if mode@facing >= 0. else -mode
    ax.scatter(*visible_mode, marker='o', s=12, color='#D55E00', depthshade=False, zorder=4)
    ax.set_proj_type('ortho'); ax.view_init(elev=25., azim=-50.)
    ax.set_box_aspect((1., 1., 1.))
    ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), zlim=(-1.12, 1.12),
           xticks=(-1., 0., 1.), yticks=(-1., 0., 1.), zticks=(-1., 0., 1.))
    ax.set_xlabel(r'$n_1$', labelpad=-2); ax.set_ylabel(r'$n_2$', labelpad=-2)
    ax.set_zlabel('')
    ax.text2D(.95, .67, r'$n_3$', transform=ax.transAxes, rotation=90,
              ha='center', va='center', fontsize=LABEL_FONT_SIZE)
    ax.tick_params(labelsize=TICK_FONT_SIZE, pad=-2, length=1); ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1., 1., 1., 0.))
        axis.pane.set_edgecolor((.65, .68, .72, .35))
        axis.line.set_color('#70777D'); axis.line.set_linewidth(.65)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=spheres,
             label=r'$q([n])$', shrink=.68, pad=.02)

risk_ax = fig.add_subplot(grid[1, 0])
query = _saved(record, 'queries', 'unit_queries')
risk = _saved(record, 'risks_full', 'risk_full')
iso_risk = _saved(record, 'risks_iso', 'risk_iso', 'risks_isotropic')
norm2 = np.sum(abs(query)**2, axis=-1)
angles, family = _optional(record, 'query_angles'), _optional(record, 'query_family')
if family is not None:
    selected = family == 1
    query, risk, iso_risk, norm2 = query[selected], risk[selected], iso_risk[selected], norm2[selected]
    if angles is not None:
        angles = angles[selected]
if angles is None:
    angles = np.linspace(0., 2*np.pi, len(query), endpoint=False)
risk_ax.plot(angles, risk/norm2, color=color('mean'), label=r'$\overline P_t^+$')
risk_ax.plot(angles, iso_risk/norm2, '--', color=color('iso'), label=r'$\overline P_t^{\mathrm{iso}}$')
risk_ax.set_xlabel(r'$\theta$ (rad)')
risk_ax.set_ylabel(r'$r_t(v(\theta))/v_0^2$')
risk_ax.set(xticks=np.linspace(0., 2*np.pi, 5), xticklabels=[r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
risk_ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

paired_ax = fig.add_subplot(grid[1, 1])
frame = evidence.frame('performance', 'episode_metrics', 'metrics')
endpoint = _column(frame, 'response_mse', 'response_loss', 'heldout_response_mse')
frame = _episode_losses(frame, endpoint)
conditions = list(frame.condition.drop_duplicates())
condition = 'sphere' if 'sphere' in conditions else conditions[0]
frame = frame[frame.condition.eq(condition)]
if 'n' in frame:
    n = 32 if 32 in set(frame.n) else int(frame.n.iloc[0])
    frame = frame[frame.n.eq(n)]
base = frame[frame.method.eq('mean')].set_index('episode')[endpoint]
values, labels, box_colors = [], [], []
for method in ('mode', 'iso', 'isotropic', 'great_window', 'great', 'great_ew'):
    other = frame[frame.method.eq(method)].set_index('episode')[endpoint]
    common = base.index.intersection(other.index)
    if len(common):
        values.append((other.loc[common]-base.loc[common]).to_numpy()*1e6)
        labels.append(method_label(method, example)); box_colors.append(color(method))
boxes = paired_ax.boxplot(values, vert=False, patch_artist=True,
    medianprops={'color': 'white', 'linewidth': 1.2}, flierprops={'markersize': 1.8, 'alpha': .35})
paired_ax.set_yticks(np.arange(1, len(labels)+1), labels=labels)
for patch, c in zip(boxes['boxes'], box_colors):
    patch.set_facecolor(c); patch.set_alpha(.85)
paired_ax.axvline(0., linestyle='--', color='#222222', linewidth=1.)
paired_ax.set_xlabel(r'Time-averaged $\mathcal E_t$ difference' + '\n'
                      + r'$((\mathrm{mm}/\mathrm{s})^2)$')
for ax in (risk_ax, paired_ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x' if ax is paired_ax else 'both', color='#BFC4C8', alpha=.30, linewidth=.45)
for ax, tag in zip(spheres+[risk_ax, paired_ax], ['(a)', '(b)', '(c)', '(d)']):
    if ax in spheres:
        ax.text2D(.02, .97, tag, transform=ax.transAxes, ha='left', va='top')
    else:
        ax.text(.015, .985, tag, transform=ax.transAxes, ha='left', va='top')
'''


def build_cells(example):
    if example not in ('beamforming', 'contact'):
        raise ValueError('Unknown application: '+str(example))
    return [
        plot_cell(example, 'computation_time', 2, _MEMORY,
            title='Memory', note='Stored NumPy arrays and peak tracked Python/NumPy allocation for the saved 64-observation workload. Values are medians of warmed timing repetitions; native workspaces and process RSS are not measured.'),
        plot_cell(example, 'numerical_accuracy', 1, _ACCURACY_1,
            title='Numerical discrepancies', note='Spectral span is the largest minus smallest eigenvalue of the posterior parameter. Each target uses its independently checked reference. The 10^{-17} display floor keeps numerical zeros visible on logarithmic axes; target flags are reported separately.'),
        plot_cell(example, 'numerical_accuracy', 2, _ACCURACY_2,
            title='Small-state numerical checks', note='Resolved and unresolved counts retain every requested numerical target. A resolved first moment does not replace a failed second moment, derivative, reference or quantile check.'),
        plot_cell(example, 'numerical_accuracy', 3, _ACCURACY_3,
            title='Numerical checks across dimensions', note='Not-evaluated targets remain separate from unresolved targets and are never counted as resolved. Reference, second-moment, quantile and state-report checks have distinct applicability flags.'),
        plot_cell(example, 'paper_summary', 1, _BEAM_SUMMARY if example == 'beamforming' else _CONTACT_SUMMARY,
            title='Study overview', note=(r'These panels reuse the same saved arrays and selected cohorts as the individual figures. The cost–power panel uses one preassigned timing simulation; clock repetitions are execution measurements, not independent Monte Carlo simulations. Variance estimates are the values used before each recorded measurement; connecting segments do not reconstruct intervening estimates. Exact stationary power uses true_ratio_db, and moving comparisons use exact covariance ratios averaged within each Monte Carlo simulation. The power ordinate is $\ell_t=10\log_{10}[(w_t^H R_t w_t)/p_t^\star]$, with $p_t^\star=(a^H R_t^{-1}a)^{-1}$.' if example == 'beamforming' else 'These panels reuse the saved concentrated-input normal law, its matched isotropic comparison, and paired moving-sphere Monte Carlo simulations. The two risk curves evaluate different maintained beliefs; the paired error panel uses common Monte Carlo simulations. No new inference or Monte Carlo simulations are introduced.')),
    ]
