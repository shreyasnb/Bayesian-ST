"""Render the guide checks from verified, already-completed studies.

This script evaluates no likelihood, posterior moment, EM update or simulation.
Main-panel aliases retain their original meanings and are explicitly identified
as aliases, rather than additional independent evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from lib.plotting import plt, save, DPI
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

COMPONENTS = ('information', 'information_counts', 'tracking', 'em', 'reliability')
SOURCES = ('scripts/plot_implementation_audit.py', 'lib/plotting.py',
           'lib/distribution_geometry_visuals.py', 'lib/simulation_story_visuals.py')
BLUE, PINK, ORANGE, GREY, BLACK = '#245FA8', '#D865A7', '#D55E00', '#69737D', '#252525'
CHANNELS = {'marginal_real': 'Real marginal', 'marginal_complex': 'Complex marginal',
            'conditional_real': 'Real conditional', 'conditional_complex': 'Complex conditional',
            'real_marginal': 'Real marginal', 'complex_marginal': 'Complex marginal',
            'real_conditional': 'Real conditional', 'complex_conditional': 'Complex conditional'}
METHODS = {'mean': r'$\overline P_t^+$', 'mode': r'$P_{\widehat{\mathcal X}_t}$',
           'iso': r'$\overline P(\kappa P_{\widehat{\mathcal X}_t})$',
           'isotropic': r'$\overline P(\kappa P_{\widehat{\mathcal X}_t})$',
           'scm': 'Loaded covariance', 'lw': 'Ledoit--Wolf', 'grouse': 'GROUSE',
           'known': 'Known scales', 'misspecified': 'Misspecified scales',
           'block_em': 'Block EM', 'residual': 'Residual fit'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_label(value):
    labels = {'beamforming': 'Beamforming', 'contact': 'Contact',
        'prior_predictive': 'Prior predictive', 'fixed_state': 'Fixed state',
        'drifting': 'Drifting state', 'noise_only': 'Noise changes',
        'motion_only': 'Motion changes', 'matched': 'Matched model',
        'desired_contamination': 'Desired signal in training',
        'unequal_rank_two_powers': 'Unequal signal powers',
        'anisotropic_damping': 'Anisotropic damping', 'friction': 'Friction',
        'contact_loss': 'Contact loss',
        'real_marginal_rank_one': r'Real marginal, $k=1$',
        'real_marginal_rank_two': r'Real marginal, $k=2$',
        'complex_marginal_rank_one': r'Complex marginal, $k=1$',
        'conditional_contact_plane': 'Conditional contact plane'}
    return labels.get(str(value), str(value).replace('_', r'\_'))


def verify(mapping, directory=ROOT):
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        path = directory / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Changed or missing figure evidence: ' + str(path))


def load(profile):
    from lib.implementation_validation import require_gate, verify_component
    gate = require_gate(ROOT)
    base = ROOT / 'results/implementation_audit' / profile
    data, inputs = {}, {}
    for component in COMPONENTS:
        meta = verify_component(ROOT, profile, component)
        folder = base / component
        data[component] = {'meta': meta, 'folder': folder}
        for path in sorted(folder.glob('*.csv')):
            data[component][path.stem] = pd.read_csv(path)
        for path in sorted(folder.glob('*.npz')):
            with np.load(path, allow_pickle=False) as record:
                data[component][path.stem] = {key: record[key] for key in record.files}
        path = folder / 'manifest.json'
        inputs[str(path.relative_to(ROOT))] = sha(path)
    return base, data, gate, inputs


def clean(ax, axis='both'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=axis, alpha=.13, linewidth=.6)


def finish(fig):
    if (plt.rcParams['text.usetex'] is not True or fig.dpi != DPI
            or plt.rcParams['savefig.dpi'] != DPI):
        raise RuntimeError('The guide plots require actual LaTeX and 300 dpi')
    return fig


def identity(ax, x, y):
    values = np.r_[np.asarray(x, float), np.asarray(y, float)]
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi-lo)*.06, abs(hi)*.02, 1e-12)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--', color=GREY, lw=1.2)
    ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad))


def fisher(data):
    frame, fd, grid = data['fisher_matrices'], data['finite_differences'], data['kl_grid']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    all_x, all_y = [], []
    for j, ((case, channel), group) in enumerate(frame.groupby(['case', 'channel'], sort=False)):
        expected = group.pivot(index='i', columns='j', values='expected').to_numpy()
        empirical = group.pivot(index='i', columns='j', values='score_covariance').to_numpy()
        x, y = np.linalg.eigvalsh(expected), np.linalg.eigvalsh(empirical)
        label = ('Complex ' if bool(group.complex_data.iloc[0]) else 'Real ')+str(channel)
        color_index = (2 if bool(group.complex_data.iloc[0]) else 0)+(1 if str(channel)=='conditional' else 0)
        used = axes[0].get_legend_handles_labels()[1]
        axes[0].scatter(x, y, s=20, alpha=.65, color=(BLUE, PINK, ORANGE, GREY)[color_index],
                        label=label if label not in used else None)
        all_x.extend(x); all_y.extend(y)
    if np.min(all_x) > 0 and np.min(all_y) > 0:
        lo, hi = min(all_x+all_y)*.75, max(all_x+all_y)*1.3
        axes[0].plot([lo,hi],[lo,hi],'--',color=GREY,lw=1.2)
        axes[0].set(xlim=(lo,hi),ylim=(lo,hi),xscale='log',yscale='log')
    else:
        identity(axes[0],all_x,all_y)
        axes[0].set_xscale('symlog',linthresh=1e-3)
        axes[0].set_yscale('symlog',linthresh=1e-3)
    axes[0].set(xlabel='Analytical Fisher eigenvalue', ylabel='Score-covariance eigenvalue',
                title='Expected information')
    axes[0].legend(frameon=False, fontsize=8)
    errors = []
    for (case, channel, step), g in fd.groupby(['case', 'channel', 'step'], sort=False):
        f = g.fisher.to_numpy()
        norm = max(np.linalg.norm(f), np.finfo(float).eps)
        for column, label in [('negative_expected_hessian', 'Expected Hessian'), ('kl_hessian', 'KL curvature')]:
            errors.append((step, label, np.linalg.norm(g[column].to_numpy()-f)/norm))
    errors = pd.DataFrame(errors, columns=['step', 'method', 'error'])
    for color, (method, group) in zip((BLUE, PINK), errors.groupby('method', sort=False)):
        median = group.groupby('step').error.median()
        lower = group.groupby('step').error.min()
        upper = group.groupby('step').error.max()
        axes[1].loglog(median.index, np.maximum(median, 1e-16), 'o-', color=color, label=method)
        axes[1].fill_between(median.index, np.maximum(lower, 1e-16), np.maximum(upper, 1e-16),
                             color=color, alpha=.12)
    axes[1].set(xlabel='Finite-difference step', ylabel='Relative matrix error',
                title='Several step sizes')
    axes[1].legend(frameon=False)
    names = list(map(str, grid['names']))
    selected = names.index('real_conditional') if 'real_conditional' in names else 0
    z = np.asarray(grid['kl'][selected]); q = np.asarray(grid['fisher_quadratic'][selected])
    levels = [.001, .003]
    exact = axes[2].contour(grid['x'], grid['y'], z, levels=levels, colors=BLUE)
    axes[2].contour(grid['x'], grid['y'], q, levels=levels, colors=BLACK, linestyles='--')
    axes[2].clabel(exact, fmt='%.2g', fontsize=8)
    axes[2].set(xlabel=r'$z_1$', ylabel=r'$z_2$', title='Gaussian KL and Fisher quadratic')
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].legend(handles=[Line2D([0], [0], color=BLUE, label='Exact KL'),
                            Line2D([0], [0], color=BLACK, ls='--', label='Fisher quadratic')],
                    frameon=False, fontsize=8)
    for ax in axes: clean(ax)
    return finish(fig)


def count_errors(data):
    from matplotlib.colors import LogNorm
    from lib.distribution_geometry_visuals import DENSITY_CMAP
    frame=data['summary']
    positive=frame.relative_frobenius_error.to_numpy()
    positive=positive[np.isfinite(positive)&(positive>0)]
    norm=LogNorm(vmin=float(positive.min()),vmax=float(positive.max()))
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,complex_data,title in zip(axes,[False,True],['Real marginal','Proper-complex marginal']):
        group=frame[frame.complex_data.eq(complex_data)]
        matrix=group.pivot(index='snr',columns='observations',values='relative_frobenius_error')
        im=ax.imshow(matrix,origin='lower',aspect='auto',cmap=DENSITY_CMAP,norm=norm)
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                value=float(matrix.iloc[i,j])
                ax.text(j,i,rf'{100*value:.1f}\%',ha='center',va='center',fontsize=10,
                    color=BLACK if norm(value)>.55 else 'white')
        ax.set(xticks=np.arange(len(matrix.columns)),xticklabels=[str(v) for v in matrix.columns],
            yticks=np.arange(len(matrix.index)),yticklabels=[f'{v:g}' for v in matrix.index],
            xlabel=r'Observations $L$',ylabel=r'$\tau^2/s$',title=title)
    fig.colorbar(im,ax=axes,shrink=.75,label='Relative Fisher matrix error')
    return finish(fig)


def curvature(data):
    frame = data['curvature']
    settings = [(False, 1, 'Real, rank one'), (False, 2, 'Real, rank two'),
                (True, 1, 'Complex, rank one')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained', sharey=True)
    for ax, (complex_data, k, title) in zip(axes, settings):
        group = frame[frame.n.eq(3) & frame.k.eq(k) & frame.complex_data.eq(complex_data)]
        group = group[np.isclose(group.s, group.s.min())]
        times = list(sorted(group.t.unique()))
        arrays = []
        for time in times:
            g = group[group.t.eq(time)]
            ratio = g.realized_curvature.to_numpy()/g.expected_information.to_numpy()
            arrays.append(ratio[np.isfinite(ratio)])
        artists = ax.boxplot(arrays, tick_labels=[str(t) for t in times], patch_artist=True,
            widths=.55, medianprops={'color': 'white', 'linewidth': 1.4},
            flierprops={'markersize': 2.5, 'alpha': .35})
        for j, box in enumerate(artists['boxes']): box.set_facecolor((BLUE, PINK, ORANGE)[j % 3])
        ax.axhline(1, color=BLACK, ls='--', lw=1.3)
        ax.set(xlabel='Observations', ylabel='Realized curvature / expected information', title=title)
        clean(ax, 'y')
    return finish(fig)


def bound_envelopes(data):
    frame = data['bounds']
    types = [t for t in ['marginal_real', 'marginal_complex', 'conditional_real', 'conditional_complex']
             if t in set(frame.observation_type)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained', sharey=True)
    for ax, kind in zip(axes.ravel(), types):
        g = frame[frame.observation_type.eq(kind) & frame.condition.eq('static')]
        g = g[np.isclose(g.beta, 1.)]
        if g.empty: raise RuntimeError('Missing declared static beta=1 reference for '+kind)
        g = g[g.n.eq(g.n.min())]
        fixed = g[['s', 'tau2']].drop_duplicates().iloc[0]
        g = g[np.isclose(g.s, fixed.s) & (g.tau2.isna() | np.isclose(g.tau2, fixed.tau2, equal_nan=True))]
        summary = g.groupby('t').agg(error=('modal_error', 'median'),
            lower=('modal_error', lambda x: x.quantile(.25)), upper=('modal_error', lambda x: x.quantile(.75)),
            bound=('bound', 'median'), valid=('valid', 'mean'), ceiling=('ceiling', 'first'))
        ax.plot(summary.index, summary.error, 'o-', color=BLUE, label='Modal error')
        ax.fill_between(summary.index, summary.lower, summary.upper, color=BLUE, alpha=.16)
        ax.plot(summary.index, summary.bound, 's-', color=PINK, label='Theorem envelope')
        ax.plot(summary.index, summary.ceiling, '--', color=BLACK, label='Distance ceiling')
        failed = summary[summary.valid.lt(1)]
        for time in failed.index:
            ax.axvspan(time/1.12, time*1.12, color='#D9DDE2', alpha=.55, zorder=0)
        ax.scatter(failed.index, failed.ceiling, marker='x', color=GREY, s=45,
                    label=r'$\epsilon_t\ge\gamma_t/2$')
        ax.set(xscale='log', yscale='log', xlabel='Observations', ylabel='Chordal distance',
               title=CHANNELS.get(kind, text_label(kind)))
        clean(ax)
    for ax in axes.ravel()[len(types):]: ax.set_visible(False)
    axes.ravel()[0].legend(frameon=False, fontsize=9)
    return finish(fig)


def bound_phase(data):
    frame = data['phase']
    types = list(frame.observation_type.drop_duplicates())
    fig, axes = plt.subplots(1, len(types)+1, figsize=(17, 5), layout='constrained', squeeze=False)
    cmap = ListedColormap(['#DBDDE1', '#6E8DC2', '#E579AF'])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], 3)
    for ax, kind in zip(axes.ravel(), types):
        g = frame[frame.observation_type.eq(kind)]
        g = g[g.t.eq(g.t.max()) & g.n.eq(g.n.min())].copy()
        g['display'] = np.where(g.informative, 2, np.where(g.valid, 1, 0))
        matrix = g.pivot_table(index='drift_rate', columns='beta', values='display', aggfunc='min')
        im = ax.imshow(matrix, origin='lower', aspect='auto', cmap=cmap, norm=norm)
        ax.set(xticks=np.arange(len(matrix.columns)), xticklabels=[f'{v:g}' for v in matrix.columns],
               yticks=np.arange(len(matrix.index)), yticklabels=[f'{v:g}' for v in matrix.index],
               xlabel=r'$\beta$', ylabel='Subspace speed', title=CHANNELS.get(kind, text_label(kind)))
        ax.tick_params(axis='x', labelrotation=35)
    colorbar = fig.colorbar(im, ax=list(axes.ravel()[:-1]), ticks=[0, 1, 2], shrink=.65)
    colorbar.ax.set_yticklabels([r'$\epsilon\ge\gamma/2$', r'$\epsilon<\gamma/2$, above ceiling', 'Below ceiling'])
    actual = data['bounds']; ax = axes.ravel()[-1]
    for kind, group in actual.groupby('observation_type', sort=False):
        ax.scatter(group.epsilon, group.actual_perturbation, s=7, alpha=.2,
                   label=CHANNELS.get(kind, text_label(kind)))
    x = actual.epsilon.to_numpy(); y = actual.actual_perturbation.to_numpy()
    positive=np.r_[x,y];positive=positive[np.isfinite(positive)&(positive>0)]
    lo,hi=float(positive.min()*.7),float(positive.max()*1.3)
    ax.plot([lo,hi],[lo,hi],'--',color=GREY,lw=1.2)
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xscale='log', yscale='log', xlabel=r'$\epsilon_t$', ylabel='Actual operator perturbation',
           title='Allowance and realized perturbation')
    ax.legend(frameon=False, fontsize=8); clean(ax)
    return finish(fig)


def heldout(main):
    metrics = main['beamforming']['metrics']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, n in zip(axes, [8, 16]):
        f = metrics[metrics.n.eq(n) & metrics.law.eq('haar') & metrics.t.eq(16) & np.isclose(metrics.s, .2)]
        g = f.pivot(index='trial', columns='method', values='heldout_power')
        ax.scatter(g.lw, g['mean'], s=17, alpha=.55, color=BLUE)
        identity(ax, g.lw, g['mean'])
        ax.set(xlabel='Ledoit--Wolf held-out power', ylabel=r'$\overline P_t^+$ held-out power',
               title=rf'$N={n}$, independent training episodes')
        clean(ax)
    return finish(fig)


def tracking_decisions(data):
    frame=data['beamforming']
    frame=frame[frame.t.eq(frame.t.max())]
    names=[name for name in ['static','drift','abrupt','noise','power'] if name in set(frame.condition)]
    labels={'static':'Static','drift':'Smooth drift','abrupt':'Abrupt change',
            'noise':'Noise change','power':'Power change'}
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained',sharey=True)
    for ax,reference,own,title,color in zip(axes,['scm','mode_beta1'],['mean','mode'],
            ['Loaded covariance comparison','Effect of memory'],[BLUE,PINK]):
        arrays=[]
        for condition in names:
            rows=frame[frame.condition.eq(condition)]
            pair=rows.pivot(index='trial',columns='method',values='heldout_power')
            arrays.append((pair[reference]-pair[own]).to_numpy())
        artists=ax.boxplot(arrays,tick_labels=[labels[name] for name in names],patch_artist=True,
            medianprops={'color':'white','linewidth':1.4},flierprops={'markersize':2.5,'alpha':.4})
        for box in artists['boxes']:box.set_facecolor(color)
        for position,values in enumerate(arrays,1):
            if len(values) and np.ptp(values)==0:
                ax.scatter(position,values[0],marker='D',s=30,color=color,edgecolors=BLACK,lw=.6,zorder=5)
        ax.axhline(0,color=BLACK,ls='--',lw=1.2)
        ax.set(ylabel='Paired held-out power difference',title=title)
        ax.tick_params(axis='x',labelrotation=20,labelsize=9);clean(ax,'y')
    return finish(fig)


def calibration(data, model):
    raw, scores = data['calibration'], data['scores']
    raw = raw[raw.model.eq(model)]
    if 'stage' in raw: raw = raw[raw.stage.eq('test')]
    scores = scores[scores.model.eq(model)]
    if 'stage' in scores: scores = scores[scores.stage.eq('test')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    coverage = data['calibration_summary']
    coverage = coverage[coverage.model.eq(model) & coverage.stage.eq('test')
                        & coverage.kind.eq('state_ball') & coverage.method.eq('posterior')]
    for color, (regime, group) in zip((BLUE, PINK, ORANGE), coverage.groupby('regime', sort=False)):
        group = group.sort_values('nominal')
        axes[0].plot(group.nominal, group.coverage, 'o-', color=color, label=text_label(regime))
        axes[0].fill_between(group.nominal, group.lower, group.upper, color=color, alpha=.12)
    axes[0].plot([0, 1], [0, 1], '--', color=GREY)
    axes[0].set(xlabel='Nominal state-region mass', ylabel='Empirical inclusion',
                xlim=(.4, 1), ylim=(-.02, 1.02), title='Different calibration experiments')
    axes[0].legend(frameon=False, fontsize=8)
    all_x, all_y = [], []
    for color, (regime, group) in zip((BLUE, PINK, ORANGE), scores.groupby('regime', sort=False)):
        episode = group.groupby('episode')[['mixture_log_score', 'gaussian_log_score']].mean()
        axes[1].scatter(episode.gaussian_log_score, episode.mixture_log_score, s=18, alpha=.5,
                        color=color, label=text_label(regime))
        all_x.extend(episode.gaussian_log_score); all_y.extend(episode.mixture_log_score)
    identity(axes[1], all_x, all_y)
    axes[1].set(xlabel='Gaussian predictive log score', ylabel='Mixture predictive log score',
                title='Score before assimilation')
    prior = scores[scores.regime.eq('prior_predictive')]
    if prior.empty: raise RuntimeError('Missing prior-predictive risk reference')
    episode = prior.groupby('episode')[['risk_predicted', 'risk_realized']].mean()
    episode['bin'] = pd.qcut(episode.risk_predicted, q=min(5, len(episode)), labels=False, duplicates='drop')
    points = episode.groupby('bin')[['risk_predicted', 'risk_realized']].mean()
    axes[2].scatter(points.risk_predicted, points.risk_realized, color=BLUE, s=50)
    identity(axes[2], points.risk_predicted, points.risk_realized)
    axes[2].set(xlabel='Posterior response risk', ylabel='Realized clean-response loss',
                title='Prior-predictive risk calibration')
    for ax in axes: clean(ax)
    return finish(fig)


def block_evidence(data):
    frame = data['iterations']
    models = list(frame.model.drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained', squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        group = frame[frame.model.eq(model)]
        if 'trial' in group: group = group[group.trial.eq(group.trial.min())]
        for j, (start, g) in enumerate(group.groupby('start', sort=False)):
            g = g.sort_values('iteration')
            ax.plot(g.iteration, g.log_evidence-g.log_evidence.iloc[0], 'o-', ms=3,
                    color=(BLUE, PINK, ORANGE, GREY)[j % 4],
                    label=rf'$s_0={g.s.iloc[0]:.2g}$'+(rf'$\,,\ \tau_0^2={g.tau2.iloc[0]:.2g}$' if np.isfinite(g.tau2.iloc[0]) else ''))
        ax.set(xlabel='EM iteration', ylabel='Normalized log-evidence increase',
               title=text_label(model))
        ax.legend(frameon=False, fontsize=8); clean(ax)
    for ax in axes.ravel()[len(models):]: ax.set_visible(False)
    return finish(fig)


def adaptation(data):
    frame = data['adaptation']; models = list(frame.model.drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(14, 5), layout='constrained', squeeze=False, sharey=True)
    colors = {'known': BLACK, 'misspecified': GREY, 'block_em': PINK, 'residual': ORANGE}
    for ax, model in zip(axes.ravel(), models):
        g = frame[frame.model.eq(model)]
        g = g[g.block.eq(g.block.max())]
        methods = [m for m in ['known', 'misspecified', 'block_em', 'residual'] if m in set(g.method)]
        changes = list(g.change.drop_duplicates())
        for j, method in enumerate(methods):
            arrays = []
            for change in changes:
                h = g[g.method.eq(method) & g.change.eq(change)]
                arrays.append((h.fitted_s/h.true_s).to_numpy())
            positions = np.arange(len(changes))+(j-(len(methods)-1)/2)*.18
            boxes = ax.boxplot(arrays, positions=positions, widths=.15, patch_artist=True,
                manage_ticks=False, medianprops={'color': 'white'},
                flierprops={'markersize': 2.5, 'alpha': .4})
            for box in boxes['boxes']: box.set_facecolor(colors[method])
            for position,values in zip(positions,arrays):
                if len(values) and np.ptp(values)==0:
                    ax.scatter(position,values[0],marker='D',s=30,color=colors[method],edgecolors=BLACK,lw=.6,zorder=5)
        ax.axhline(1, color=BLACK, ls='--', lw=1.2)
        ax.set(xticks=np.arange(len(changes)), xticklabels=[text_label(v) for v in changes],
               ylabel=r'$\widehat s/s$', title=text_label(model))
        ax.legend(handles=[Line2D([0], [0], color=colors[m], lw=5, label=METHODS[m]) for m in methods],
                   frameon=False, fontsize=8); clean(ax, 'y')
    return finish(fig)


def mismatch(data):
    frame = data['mismatch']; models = list(frame.model.drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5), layout='constrained', squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        group = frame[frame.model.eq(model)]
        names = list(group.stress.drop_duplicates())
        arrays = [group[group.stress.eq(name)].groupby('episode').response_loss.mean().to_numpy() for name in names]
        boxes = ax.boxplot(arrays, tick_labels=[text_label(name) for name in names], patch_artist=True,
            medianprops={'color': 'white'}, flierprops={'markersize': 2.5, 'alpha': .4})
        for j, box in enumerate(boxes['boxes']): box.set_facecolor((BLUE, PINK, ORANGE, GREY)[j % 4])
        ax.set(ylabel='Clean-response loss', title=text_label(model)); ax.tick_params(axis='x', labelrotation=25, labelsize=8)
        clean(ax, 'y')
    return finish(fig)


CAPTIONS = {
    'B2a_C2_information': 'Fisher eigenvalues use aligned tangent coordinates and directly recorded inputs for the conditional law. Finite-difference bands show the range across the declared cases, not confidence intervals. Exact Gaussian KL contours and the Fisher quadratic use the same displayed levels and tangent coordinates.',
    'B2a_count_error': 'The observation-count and signal-to-noise axes correspond to a separate declared Fisher experiment. Score sums are formed inside each independent episode before estimating covariance across episodes. Common within-episode prefixes pair counts; coordinates are not independent repetitions. Cells report relative Frobenius Monte Carlo error of score information, with the independent episode count fixed across observation counts. The relative Monte Carlo error need not decrease as one over the square root of the observation count. Product-level checks remain in the tables.',
    'B2b_posterior_curvature': 'Boxes retain the distribution of realized curvature eigenvalues across independently generated datasets. Expected likelihood information is evaluated at the generating state; posterior curvature is evaluated at its own mode. A spectral comparison does not assert equality of those two tangent frames.',
    'B3_C4_modal_bounds': 'The fixed deterministic state path is repeated with independent signal and sensor noise. Blue bands are the interquartile spread of modal errors, not credible or confidence intervals. Gray bands and crosses mark reported checkpoints with insufficient gap conditions; the shading makes no claim about unreported times. The theorem envelope and geometric ceiling are different quantities; bounds use true fixed scales for evaluation.',
    'B3_C4_validity': 'The phase diagram uses the predeclared largest horizon and smallest dimension. A sufficient gap condition can hold while the envelope remains above the geometric ceiling. Actual perturbations and theoretical allowances retain all trials. These truth-dependent diagnostics are not online confidence radii.',
    'B5_heldout_decisions': 'Each dot is one independent training episode at the preassigned Haar-signature setting, 16 snapshots and sensor variance 0.2. Both decisions are scored on exactly the same independent held-out snapshots. Below the diagonal favors the posterior mean. These observations complement the complete true-covariance and held-out paired tables; this display is not a universal ranking.',
    'B5_tracking_decisions': 'All five declared conditions are retained at the final reporting time. Each box contains paired independent test episodes. The left difference is loaded-covariance power minus posterior-mean power. The right difference is beta=1 modal power minus modal power with independently validation-selected fixed memory; the decision rule is held fixed there. Positive differences favor the second decision. No hidden input or oracle motion enters the tracker. Numerical flags and pointwise paired intervals remain in the tables.',
    'B6_beam_calibration': 'State inclusion uses independent episodes, with the saved pointwise binomial 95% intervals shaded. Prior-predictive, fixed-state and drifting experiments have different probability interpretations. Scores are computed before fitting or assimilation; the full normalized mixture and Gaussian moment surrogate are separate densities. Risk points average independent episodes in bins, and do not turn repeated time samples into extra trials.',
    'C5_contact_calibration': 'State inclusion uses independent episodes, with the saved pointwise binomial 95% intervals shaded. Clean-response intervals and noisy predictive intervals are distinct in the tables. Prior-predictive calibration does not imply fixed-state or drifting-state coverage. Mixture scores are evaluated before assimilation, and risk bins use episode averages.',
    'B7a_C6a_block_evidence': 'The displayed episode and starts are fixed before the test comparisons. Each E-step reconstructs the entire stationary block from one boundary prior excluding that block. Curves show the full normalized block evidence relative to its own initial value. A boundary signal-variance start can remain absorbing. Rejected updates and independent reference discrepancies remain in the tables.',
    'B7b_C6b_scale_adaptation': 'Separate noise-only and motion-only interventions use the final declared block. Box distributions contain independent episodes. The horizontal line is the true sensor variance. Motion-induced variance inflation is evidence of model mismatch, not recovery of a physical sensor parameter. Plug-in scale decisions omit scale uncertainty and do not inherit the fixed-scale tracking theorem.',
    'C5_model_mismatch': 'Each box contains episode-averaged clean-response loss. Friction, contact loss, anisotropic damping or correlated velocity errors are explicitly identified mismatch cases. They do not validate the isotropic independent-noise likelihood under changed physics. Correct scaled-coordinate mechanics and sensor covariance checks are recorded separately.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'paper'], default='paper')
    args = parser.parse_args()
    base, studies, gate, inputs = load(args.profile)
    out = base/'figures'; out.mkdir(parents=True, exist_ok=True)
    sources = {name: sha(ROOT/name) for name in SOURCES}
    from lib.simulation_story_visuals import load_studies
    legacy_profile = 'quick' if args.profile == 'smoke' else 'paper'
    main_data = load_studies(ROOT, legacy_profile)
    old_folder = main_data['base']/'figures'
    old_path = old_folder/'manifest.json'; old = json.loads(old_path.read_text())
    verify(old['sources']); verify(old['input_manifests']); verify(old['files'], old_folder)
    inputs[str(old_path.relative_to(ROOT))] = sha(old_path)
    figures = [
        ('B2a_C2_information', ['B2a', 'C2'], lambda: fisher(studies['information'])),
        ('B2a_count_error', ['B2a'], lambda: count_errors(studies['information_counts'])),
        ('B2b_posterior_curvature', ['B2b'], lambda: curvature(studies['information'])),
        ('B3_C4_modal_bounds', ['B3', 'C4'], lambda: bound_envelopes(studies['tracking'])),
        ('B3_C4_validity', ['B3', 'C4'], lambda: bound_phase(studies['tracking'])),
        ('B5_heldout_decisions', ['B5'], lambda: heldout(main_data)),
        ('B5_tracking_decisions', ['B5'], lambda: tracking_decisions(studies['tracking'])),
        ('B6_beam_calibration', ['B6'], lambda: calibration(studies['reliability'], 'beamforming')),
        ('C5_contact_calibration', ['C5'], lambda: calibration(studies['reliability'], 'contact')),
        ('B7a_C6a_block_evidence', ['B7a', 'C6a'], lambda: block_evidence(studies['em'])),
        ('B7b_C6b_scale_adaptation', ['B7b', 'C6b'], lambda: adaptation(studies['reliability'])),
        ('C5_model_mismatch', ['C5'], lambda: mismatch(studies['reliability'])),
    ]
    ids, aliases, captions = {}, {}, dict(CAPTIONS)
    for name, figure_ids, function in figures:
        fig = function(); save(fig, out/name); fig.savefig(out/(name+'.svg'), dpi=DPI); plt.close(fig)
        for identifier in figure_ids: ids.setdefault(identifier, []).append(name)
        print('Rendered', name, flush=True)
    mapping = {
        'B1_array_measurements': ('beamforming_01_setup', ['B1']),
        'B4_posterior_section': ('beamforming_02_posterior', ['B4']),
        'B4_directional_decisions': ('beamforming_03_directional', ['B4']),
        'B4_power_uncertainty': ('beamforming_05_responses', ['B4']),
        'B5_covariance_decisions': ('beamforming_04_benchmarks', ['B5']),
        'C1_contact_mechanics': ('contact_01_mechanics', ['C1']),
        'C1_current_contact_predictions': ('contact_05_tracking', ['C1', 'C3']),
        'C2_directional_contact_information': ('contact_02_posterior', ['C2']),
        'C3_expected_contact_commands': ('contact_03_directional', ['C3']),
        'C3_stationary_contact_costs': ('contact_04_benchmarks', ['C3']),
        'S1_native_scaling': ('beamforming_06_scaling', ['S1']),
        'S1_generalized_constraints': ('contact_06_higher_dimension', ['S1']),
    }
    for name, (original, figure_ids) in mapping.items():
        records = {}
        for suffix in ['.png', '.pdf']:
            source = old_folder/(original+suffix)
            shutil.copyfile(source, out/(name+suffix)); records[str(source.relative_to(ROOT))] = sha(source)
        aliases[name] = dict(source_files=records, source_figure=original,
            scope='Filename alias of the existing verified main panel; no additional independent evidence')
        captions[name] = old['captions'][original]
        for identifier in figure_ids: ids.setdefault(identifier, []).append(name)
    verify(sources); verify(inputs)
    files = {path.name: sha(path) for path in sorted(out.iterdir()) if path.suffix in ['.png', '.pdf', '.svg']}
    manifest = dict(profile=args.profile, status='completed', sources=sources, input_manifests=inputs,
        files=files, figure_ids=ids, aliases=aliases, captions=captions,
        validation_sha256=sha(ROOT/'results/implementation_audit/validation/validation.json'),
        text_usetex=True, figure_dpi=DPI, savefig_dpi=DPI,
        density_colormap='continuous blue-purple-pink, blue floor and pink maximum',
        svg_scope='New Matplotlib panels export SVG. Exact main-panel aliases retain their verified PDF and PNG forms.',
        scope='Plotting reads verified scientific artifacts only; main aliases do not add independent evidence.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Verified guide exports:', len(files), 'files;', len(ids), 'figure IDs', flush=True)


if __name__ == '__main__':
    main()
