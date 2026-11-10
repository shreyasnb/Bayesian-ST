"""Source templates for the remaining variance, belief and paired-task figures.

Only ordinary plotting source is returned. Numerical data remain the verified
saved study and mathematical reference cohorts; these templates do not fit or
simulate a model.
"""
from textwrap import dedent, indent

from scripts.notebook_supplementary_common import plot_cell


_ADAPTATION = r'''
frame = _normalize(evidence.frame('prediction', 'prediction_adaptation', 'adaptation'))
if 'split' in frame and frame.split.eq('test').any():
    frame = frame[frame.split.eq('test')]
if 'control' in frame:
    frame = frame[frame.control.eq('mean')]
frame = frame[frame.condition.eq(CONDITION)]
time = 't'
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout='constrained')
if example == 'beamforming':
    grid = fig.add_gridspec(2,2)
    axes = np.asarray([[fig.add_subplot(grid[row,col]) for col in range(2)] for row in range(2)])
    variance_axes = [(axes[0,0], 'sigma2'), (axes[0,1], 'tau2')]
    future_axes = [(axes[1,0], False), (axes[1,1], True)]
    plotted_axes = list(axes.ravel())
else:
    axes = [fig.add_subplot(1,3,index+1) for index in range(3)]
    variance_axes = [(axes[0], 'sigma2')]
    future_axes = [(axes[1], False), (axes[2], True)]
    plotted_axes = list(axes)
for ax, key in variance_axes:
    scale = 1e6 if example == 'contact' and key == 'sigma2' else 1.
    for method in ('known', 'exact_known', 'known_forgetting', 'misspecified', 'learned'):
        group = frame[frame.method.eq(method)]
        if group.empty:
            continue
        grouped = group.groupby(time)[key+'_estimate']
        median = grouped.median()
        ax.plot(median.index, median*scale, color=color(method),
            linestyle='--' if method == 'misspecified' else '-', marker=marker(method),
            label=method_label(method))
        if method == 'learned':
            ax.fill_between(median.index, grouped.quantile(.25)*scale,
                grouped.quantile(.75)*scale, color=color(method), alpha=.13)
    truth = frame.groupby(time)[key+'_true'].first()
    total = int(evidence.config('prediction')[
        'primary_beam_T' if example == 'beamforming' else 'primary_contact_T'])
    measurement_index = np.arange(1, total+1)
    initial_truth, final_truth = float(truth.iloc[0]), float(truth.iloc[-1])
    true_schedule = np.where(measurement_index <= total//2,
                             initial_truth, final_truth)
    ax.step(measurement_index, true_schedule*scale, where='mid', color='#222222', linestyle=':',
        linewidth=1.35, label=r'$\tau_t^2$' if key == 'tau2' else r'$\sigma_t^2$')
    if not np.isclose(initial_truth, final_truth):
        ax.axvline(total//2+.5, color='#70777D', linestyle=':', linewidth=.7)
    ax.set_xlabel(r'Measurement index $t$')
    ax.set_ylabel(r'$\tau_t^2,\ \widehat\tau_t^2$' if key == 'tau2' else
        r'$\sigma_t^2,\ \widehat\sigma_t^2$'+'\n'+r'$((\mathrm{mm}/\mathrm{s})^2)$' if example == 'contact'
        else r'$\sigma_t^2,\ \widehat\sigma_t^2$')
for ax, scores in future_axes:
    endpoint = ('prequential_score' if scores else
        'output_oracle_db' if example == 'beamforming'
        else _finite_column(frame, 'future_task_loss', 'observed_response_mse'))
    if endpoint not in frame:
        raise KeyError('Missing saved numerical quantity: '+endpoint)
    scale = 1e6 if example == 'contact' and not scores else 1.
    for method in ('known', 'exact_known', 'known_forgetting', 'misspecified', 'learned'):
        rows = []
        for t, group in frame[frame.method.eq(method)].groupby(time, sort=True):
            values = (group.groupby('episode')[endpoint].mean()*scale).dropna()
            if len(values):
                rows.append((t, *_mean_interval(values)))
        if not rows:
            continue
        points = np.asarray(rows)
        ax.plot(points[:,0], points[:,1], marker=marker(method), color=color(method),
            linestyle='--' if method == 'misspecified' else '-', label=method_label(method))
        ax.fill_between(points[:,0], points[:,2], points[:,3], color=color(method), alpha=.09)
    ax.set_xlabel(r'Measurement index $t$')
    ax.set_ylabel(r'$\log\pi_t(u_t)$' if scores else
        r'$\ell_t$ (dB)' if example == 'beamforming' else
        'Noisy velocity error\n'+r'$((\mathrm{mm}/\mathrm{s})^2)$')
handles = {}
for ax, tag in zip(plotted_axes, ['(a)', '(b)', '(c)', '(d)']):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(color='#BFC4C8', alpha=.30, linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha='left',va='top')
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(label, handle)
fig.legend(list(handles.values()), list(handles), loc='outside lower center',
    ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE)
'''

_VARIANCE_ESTIMATION = r'''
iterations = _reference_frame(evidence, 'em', 'iterations')
fits = _reference_frame(evidence, 'em', 'fits')
model = 'complex_marginal_rank_one' if example == 'beamforming' else 'conditional_contact_plane'
iterations = iterations[iterations.model.eq(model)]
fits = fits[fits.model.eq(model)]
trial = int(iterations.trial.min())
story = iterations[iterations.trial.eq(trial)]
fig, axes = plt.subplots(1, 2 if example == 'beamforming' else 3,
    figsize=FIGSIZE, dpi=DPI, layout='constrained')
starts = list(story.start.drop_duplicates())
palette = [color('mean'), color('iso'), color('mode'), '#70777D']
for index, (start, group) in enumerate(story.groupby('start', sort=False)):
    group = group.sort_values('iteration')
    initial = group.iloc[0]
    label = (r'$\widehat\sigma_0^2='+f'{initial.s:g}'+r',\ \widehat\tau_0^2='+f'{initial.tau2:g}'+r'$'
        if example == 'beamforming' else r'$\widehat\sigma_0^2='+f'{initial.s:g}'+r'$')
    axes[0].plot(group.iteration, group.log_evidence-group.log_evidence.iloc[0],
        'o-', color=palette[index % len(palette)], markersize=2.5, label=label)
axes[0].set(xlabel='EM iteration', ylabel='Log-evidence increase')
axes[0].legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
values, positions, ticks, box_colors = [], [], [], []
for index, start in enumerate(starts):
    group = fits[fits.start.eq(start)]
    values.append((group.s/group.true_s).to_numpy())
    positions.append(index-.12 if example == 'beamforming' else index)
    ticks.append(index)
    box_colors.append(color('mean'))
    if example == 'beamforming':
        values.append((group.tau2/group.true_tau2).to_numpy())
        positions.append(index+.12)
        box_colors.append(color('iso'))
boxes = axes[1].boxplot(values, positions=positions, widths=.20, patch_artist=True,
    manage_ticks=False, medianprops={'color':'white'}, flierprops={'markersize':1.8,'alpha':.4})
for box, shade in zip(boxes['boxes'], box_colors):
    box.set_facecolor(shade)
axes[1].axhline(1., linestyle='--', color='#222222', linewidth=1.)
axes[1].set(xticks=ticks, xticklabels=[str(i+1) for i in ticks],
    xlabel='Initialization', ylabel='Estimate / truth')
legend_handles = [Patch(facecolor=color('mean'), label=r'$\widehat\sigma^2/\sigma^2$')]
if example == 'beamforming':
    legend_handles.append(Patch(facecolor=color('iso'), label=r'$\widehat\tau^2/\tau^2$'))
axes[1].legend(handles=legend_handles, frameon=False, fontsize=LEGEND_FONT_SIZE)
if example == 'contact':
    residual = _reference_frame(evidence, 'em', 'residuals')
    residual = residual[residual.model.eq(model)]
    lookup = fits[['trial','start','true_s']].drop_duplicates()
    residual = residual.merge(lookup, on=['trial','start'], how='left')
    x = residual.mean_only_update/residual.true_s
    y = residual.expected_update/residual.true_s
    axes[2].scatter(x, y, color=color('mean'), s=10, alpha=.45)
    identity_values = np.r_[np.asarray(x,float),np.asarray(y,float)]
    identity_values = identity_values[np.isfinite(identity_values)]
    low, high = identity_values.min(), identity_values.max()
    padding = max(.06*(high-low),.01*max(abs(low),abs(high)),1e-10)
    lower = max(low-padding,.8*low) if low>0 else low-padding
    limits = [lower,high+padding]
    axes[2].plot(limits, limits, '--', color='#70777D', linewidth=1.)
    axes[2].set(xlim=limits,ylim=limits)
    axes[2].set(xlabel=r'Mean-response residual / $\sigma^2$',
        ylabel=r'Expected residual / $\sigma^2$')
for ax, tag in zip(axes, ['(a)','(b)','(c)']):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y' if ax is axes[1] else 'both', color='#BFC4C8', alpha=.30, linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha='left',va='top')
'''

# Expanded into each figure's source; no runtime rendering helper is emitted.
_SPHERE_ARTISTS = r'''
ax.computed_zorder = False
surface = ax.plot_surface(nodes[...,0], nodes[...,1], nodes[...,2],
    facecolors=cmap(norm(np.exp(log_density))), shade=False, linewidth=0,
    antialiased=False, rstride=1, cstride=1, zorder=1)
surface.set_rasterized(True)
phi = np.linspace(0.,2*np.pi,241)
el, az = np.deg2rad([25.,-50.])
facing = np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
for plane in ((0,1),(0,2),(1,2)):
    ring = np.zeros((len(phi),3))
    ring[:,plane[0]], ring[:,plane[1]] = 1.006*np.cos(phi),1.006*np.sin(phi)
    visible = np.where((ring@facing >= 0.)[:,None],ring,np.nan)
    ax.plot(*visible.T,color='#B7C0CB',linewidth=.5,alpha=.55,zorder=3)
if mode is not None:
    unit_mode = mode/np.linalg.norm(mode)
    visible_mode = unit_mode if unit_mode@facing>=0. else -unit_mode
    ax.scatter(*visible_mode,marker='o',s=12,color='#D55E00',depthshade=False,zorder=4)
ax.set_proj_type('ortho')
ax.view_init(elev=25., azim=-50.)
ax.set_box_aspect((1.,1.,1.))
ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),zlim=(-1.12,1.12),
    xticks=(-1.,0.,1.),yticks=(-1.,0.,1.),zticks=(-1.,0.,1.))
ax.set_xlabel(r'$n_1$',labelpad=-2)
ax.set_ylabel(r'$n_2$',labelpad=-2)
ax.set_zlabel('')
ax.text2D(.95,.67,r'$n_3$',transform=ax.transAxes,rotation=90,ha='center',va='center',fontsize=LABEL_FONT_SIZE)
ax.tick_params(labelsize=TICK_FONT_SIZE,pad=-2,length=1)
ax.grid(False)
for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
    axis.pane.set_facecolor((1.,1.,1.,0.))
    axis.pane.set_edgecolor((.65,.68,.72,.35))
    axis.line.set_color('#70777D')
    axis.line.set_linewidth(.65)
ax.text2D(.02,.97,tag,transform=ax.transAxes,ha='left',va='top')
'''

_DENSITY_STYLE = r'''
from lib.notebook_supplementary_data import bingham_sphere_mesh, real_bingham_log_density
cmap = colors.LinearSegmentedColormap.from_list('density_blue_pink_yellow', [
    (0.,'#263a8b'),(.28,'#55209a'),(.53,'#b33695'),(.72,'#ed78a4'),
    (.88,'#ffc85a'),(1.,'#f4f52b')], N=256)
cmap.set_under(cmap(0.)); cmap.set_over(cmap(1.))
'''

_ADAPTIVE_DENSITY_DATA = r'''
story = evidence.array('prediction', 'stories')
prefix = 'contact_noise_change_'
checkpoints = (32,96,192)
meshes, densities, risks = [], [], []
for t in checkpoints:
    key = prefix+'t'+str(t)+'_'
    assert bool(np.asarray(_saved(story,key+'mean_numerical_ok')).item())
    plane_A = _saved(story,key+'A')
    normal_A = -plane_A
    normal_logz = float(_saved(story,key+'log_normalizer'))-float(np.trace(plane_A))
    nodes = bingham_sphere_mesh(normal_A,161,321).nodes
    meshes.append(nodes)
    densities.append(real_bingham_log_density(normal_A,normal_logz,nodes))
    mean = _saved(story,key+'M')
    query = _saved(story,prefix+'V')[t-1]
    risks.append(float(query@(mean-mean@mean)@query)*1e6)
peak = max(float(q.max()) for q in densities)
norm = colors.LogNorm(vmin=np.exp(peak-4*np.log(10.)),vmax=np.exp(peak),clip=True)
'''

_ADAPTIVE_TWO_SPHERES = _DENSITY_STYLE + _ADAPTIVE_DENSITY_DATA + r'''
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout='constrained')
spheres = [fig.add_subplot(1,2,i+1,projection='3d') for i in range(2)]
for ax,nodes,log_density,tag in zip(spheres,meshes[:2],densities[:2],['(a)','(b)']):
    mode = None
''' + indent(dedent(_SPHERE_ARTISTS).strip(), '    ') + r'''
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=spheres,
    label=r'$q_t^-([n])$',shrink=.7,pad=.07)
'''

_ADAPTIVE_FINAL = _DENSITY_STYLE + _ADAPTIVE_DENSITY_DATA + r'''
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout='constrained')
grid = fig.add_gridspec(1,2,width_ratios=(1.15,1.))
ax = fig.add_subplot(grid[0,0],projection='3d')
nodes, log_density, mode, tag = meshes[2], densities[2], None, '(a)'
''' + dedent(_SPHERE_ARTISTS) + r'''
sphere = ax
ax = fig.add_subplot(grid[0,1])
ax.plot(np.asarray(checkpoints),risks,'o-',color=color('mean'))
ax.set(xlabel=r'Measurement index $t$',ylabel=r'$r_t(v_t)$ $((\mathrm{mm}/\mathrm{s})^2)$')
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax.spines[['top','right']].set_visible(False)
ax.grid(color='#BFC4C8',alpha=.30,linewidth=.45)
ax.text(.015,.985,'(b)',transform=ax.transAxes,ha='left',va='top')
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=[sphere,ax],
    label=r'$q_t^-([n])$',orientation='horizontal',shrink=.65,fraction=.06,pad=.08)
'''

_CONSTRUCTED = _DENSITY_STYLE + r'''
from lib.notebook_supplementary_data import load_constructed_geometry
record = load_constructed_geometry(ROOT)
parameters, logz, modes = record['parameters'], record['log_normalizers'], record['modes']
directional, cdf = record['directional'], record['distance_cdf']
meshes = [bingham_sphere_mesh(A,161,321).nodes for A in parameters]
densities = [real_bingham_log_density(A,z,nodes) for A,z,nodes in zip(parameters,logz,meshes)]
peak = max(float(q.max()) for q in densities)
norm = colors.LogNorm(vmin=np.exp(peak-4*np.log(10.)),vmax=np.exp(peak),clip=True)
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout='constrained')
grid = fig.add_gridspec(1,4,width_ratios=(1.,1.,1.1,1.1))
spheres = [fig.add_subplot(grid[0,index],projection='3d') for index in range(2)]
for ax,nodes,log_density,mode,tag in zip(spheres,meshes,densities,modes,['(a)','(b)']):
''' + indent(dedent(_SPHERE_ARTISTS).strip(), '    ') + r'''
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=spheres,
    label=r'$q([n])$',shrink=.7,pad=.015)
lower = [fig.add_subplot(grid[0,index]) for index in (2,3)]
for case, shade, style in [(0,color('mean'),'-'),(1,color('iso'),'--')]:
    group = cdf[cdf.case.eq(case)]
    label = r'$q_{A^{('+str(case+1)+r')}}$'
    lower[0].plot(group.radius,group.cdf,color=shade,linestyle=style,label=label)
    group = directional[directional.case.eq(case)&directional.angle.notna()]
    lower[1].plot(group.angle,group.risk,color=shade,linestyle=style,label=label)
lower[0].set(xlabel=r'$d_c$',ylabel='Cumulative probability')
lower[1].set(xlabel=r'$\theta$ (rad)',ylabel=r'$v^\top(\overline P-\overline P^2)v$')
for ax,tag in zip(lower,['(c)','(d)']):
    ax.legend(frameon=False,fontsize=LEGEND_FONT_SIZE)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(color='#BFC4C8',alpha=.30,linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha='left',va='top')
'''

_CONTACT_PAIRED = r'''
frame = evidence.frame('performance', 'episode_metrics', 'metrics')
endpoint = _column(frame, 'response_mse', 'response_loss', 'heldout_response_mse')
frame = _episode_losses(frame, endpoint)
frame = frame[frame.condition.eq('sphere')]
if 'n' in frame:
    frame = frame[frame.n.eq(32 if 32 in set(frame.n) else int(frame.n.iloc[0]))]
base = frame[frame.method.eq('mean')].set_index('episode')[endpoint]
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout='constrained')
labels = []
for method in ('mode','iso','isotropic','regularized'):
    other = frame[frame.method.eq(method)].set_index('episode')[endpoint]
    indices = base.index.intersection(other.index)
    if not len(indices):
        continue
    values = (other.loc[indices]-base.loc[indices]).to_numpy()*1e6
    mean, low, high = _mean_interval(values)
    position = len(labels)+1
    labels.append(method_label(method, example))
    ax.errorbar(mean,position,xerr=[[mean-low],[high-mean]],fmt='o',
        color=color(method),capsize=3.,markersize=4.)
ax.set(yticks=np.arange(1,len(labels)+1),yticklabels=labels,ylim=(.5,len(labels)+.5))
ax.axvline(0.,linestyle='--',color='#222222',linewidth=1.)
ax.set_xlabel(r'$\overline{\mathcal E}(D)-\overline{\mathcal E}(\overline P_t^+)$'+'\n'+r'$((\mathrm{mm}/\mathrm{s})^2)$')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='x',color='#BFC4C8',alpha=.30,linewidth=.45)
ax.text(.015,.985,'(a)',transform=ax.transAxes,ha='left',va='top')
'''


def build_cells(example):
    cells = []
    conditions = [('smooth',2),('power_change',3)] if example == 'beamforming' else [('fast_sphere',2)]
    for condition,page in conditions:
        cells.append(plot_cell(example, 'noise_adaptation', page,
            'CONDITION = '+repr(condition)+'\n'+dedent(_ADAPTATION),
            title='Variance learning: '+('moving interference' if condition=='smooth' else
                'signal-power change' if condition=='power_change' else 'faster contact motion'),
            note=r'Variance curves show medians and 25th–75th percentiles across Monte Carlo simulations. Markers are estimates used before receiving $u_t$, at the stored integer reporting index $t$; connecting lines do not reconstruct estimates between reports. True schedules change between measurements $T/2$ and $T/2+1$, where applicable. Power and log-density bands are pointwise 95% Student confidence intervals for Monte Carlo means. For the array, $\ell_t=10\log_{10}[(w_t^H R_t w_t)/p_t^\star]$ uses saved exact covariance power, and $p_t^\star=(a^H R_t^{-1}a)^{-1}$. The density $\pi_t$ integrates the measurement law under the prior before assimilation. The contact error panel retains prediction errors against independent noisy velocities, including their measurement noise; this is distinct from the clean-velocity error $\mathcal E_t$ in the main comparison.'))
    cells.append(plot_cell(example, 'variance_estimation', 1, dedent(_VARIANCE_ESTIMATION),
        title='Fixed-block variance estimation',
        note=r'These saved stationary-block EM references hold the boundary prior fixed. The evidence curve for each initialization is shifted by its own initial value, so heights do not compare different starts. The Estimate / truth boxes summarize $\widehat\sigma^2/\sigma^2$ and, for the array, $\widehat\tau^2/\tau^2$ across the original Monte Carlo simulations. For contact, the mean-response residual omits posterior response uncertainty; the expected residual includes it.'))
    if example == 'contact':
        cells.append(plot_cell(example,'noise_adaptation',3,_ADAPTIVE_TWO_SPHERES,
            title='Predictive contact beliefs at two checkpoints',
            note='The two saved pre-correction normal densities are from one preassigned noise-change Monte Carlo simulation at observations 32 and 96. They share the density scale of observations 32, 96 and 192. Antipodal normals represent the same plane; these are individual beliefs, not confidence regions across Monte Carlo simulations.'))
        cells.append(plot_cell(example,'noise_adaptation',4,_ADAPTIVE_FINAL,
            title='Predictive contact belief and response risk',
            note=r'The sphere is the saved pre-correction density at $t=192$. Here $\overline P_t^-$ is the predictive mean plane projector before receiving $u_t$. The curve evaluates $r_t(v_t)=v_t^\top[\overline P_t^--(\overline P_t^-)^2]v_t$ at the physical applied input for all three saved measurement indices, in squared millimetres per second. The normal parameter is $-A_t^-$ and its log normalizer is the saved plane log normalizer minus $\operatorname{tr}(A_t^-)$.'))
        cells.append(plot_cell(example,'posterior_anisotropy',2,_CONSTRUCTED,
            title='Equal distance distributions and different directional risks',
            note=r'This constructed identity uses $A^{(1)}=\operatorname{diag}(0,-8,-80)$ and $A^{(2)}=\operatorname{diag}(0,-80,-8)$. Their mode and the entire chordal-distance distribution around it coincide. The directional response risks differ because the two transverse eigenvalues are interchanged. These deterministic reference calculations are not extra task-performance Monte Carlo simulations.'))
        cells.append(plot_cell(example,'task_performance',2,dedent(_CONTACT_PAIRED),
            title='Paired contact response errors',
            note=r'Each point is the mean same-simulation response-MSE difference relative to $\overline P_t^+$ for the moving-sphere condition; horizontal bars are pointwise 95% Student intervals. Positive differences favor the posterior mean. Its self-difference is identically zero, so no artificial self confidence interval is shown.'))
    return cells
