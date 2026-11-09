"""Question-led visual summaries of paired simulation results.

The effect intervals use paired trajectory differences and Student's t critical
value. Percentage differences divide every paired difference by the same
observed mean reference loss, avoiding unstable per-trial ratios. These are
pilot intervals for the paired difference on that scale, not simultaneous
confidence bands or uncertainty intervals for a random denominator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .plotting import plt, save
from matplotlib.colors import TwoSlopeNorm, LogNorm
from .reporting import coverage_table

COLORS = {
    'Posterior mean': '#0072B2', 'Mode': '#D55E00',
    'GROUSE': '#009E73', 'EW PCA': '#CC79A7',
    'Rolling PCA': '#E69F00', 'Loaded covariance': '#009E73',
    'Oracle': '#666666', 'Ball': '#0072B2', 'HPD': '#D55E00',
}
ARRAY_SCENARIOS = ['Static','Near interference','Far from interference','Drift',
                   'Unequal powers','Correlation','Contamination']
FIELD_SCENARIOS = ['Matched','Unequal powers','Sparse corruption']


def _finish(fig, path=None):
    fig.set_dpi(200)
    if path is not None:
        save(fig, path)
    return fig


def _ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    half = float(student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
    return mean, mean-half, mean+half


def _terminal(frame):
    return frame[frame.t == frame.t.max()].copy()


def _pairs(frame, value, reference, method='Posterior mean'):
    """Exact scenario/repetition/time pairing; no independent-error shortcut."""
    p = frame.pivot(index=['scenario','repeat','t'], columns='method', values=value)
    return p[[reference,method]].dropna()


def _gain_cells(frame, value, reference, scenarios, percent=False):
    pairs = _pairs(frame, value, reference)
    times = sorted(frame.t.unique())
    means = np.full((len(scenarios),len(times)), np.nan)
    low = means.copy(); high = means.copy()
    for i, scenario in enumerate(scenarios):
        for j, time in enumerate(times):
            try:
                block = pairs.xs((scenario,time), level=('scenario','t'))
            except KeyError:
                continue
            difference = (block[reference]-block['Posterior mean']).to_numpy() if percent else (block['Posterior mean']-block[reference]).to_numpy()
            if percent:
                difference = 100*difference/block[reference].mean()
            means[i,j], low[i,j], high[i,j] = _ci(difference)
    return times, means, low, high


def _effect_map(ax, means, low, high, rows, columns, title, unit, extent=None):
    finite = np.abs(means.ravel())
    finite = finite[np.isfinite(finite)]
    bound = max(float(finite.max()) if finite.size else 1., 1e-6) if extent is None else extent
    image = ax.imshow(means, cmap='RdBu', norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound), aspect='auto')
    ax.set_xticks(np.arange(len(columns)),columns)
    ax.set_yticks(np.arange(len(rows)),rows)
    ax.set_title(title,pad=12)
    for i in range(len(rows)):
        for j in range(len(columns)):
            if not np.isfinite(means[i,j]): continue
            color = 'white' if abs(means[i,j])>.55*bound else '#222222'
            decimals = 2 if unit=='dB' else 1
            label = f'{means[i,j]:+.{decimals}f}\n[{low[i,j]:+.{decimals}f}, {high[i,j]:+.{decimals}f}]'
            ax.text(j,i,label,ha='center',va='center',fontsize=10,color=color,linespacing=1.35)
            if low[i,j] <= 0 <= high[i,j]:
                ax.scatter(j+.38,i-.33,s=25,facecolors='white',edgecolors='#444444',linewidths=.8,zorder=4)
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5)
    ax.tick_params(which='minor',bottom=False,left=False)
    ax.set_xlabel('Training snapshots' if unit=='dB' else 'Observations')
    for spine in ax.spines.values(): spine.set_visible(False)
    return image


def array_overview(result, path=None):
    """Where does the mean decision help? Paired SINR effects versus two controls."""
    frame=result['metrics']; scenarios=[s for s in ARRAY_SCENARIOS if s in frame.scenario.unique()]
    fig,axs=plt.subplots(1,2,figsize=(14,8),dpi=200,layout='constrained',sharey=True)
    cells=[_gain_cells(frame,'sinr_db',ref,scenarios) for ref in ['Mode','Loaded covariance']]
    # Separate color scales retain visibility of small mean/mode effects.
    for ax,ref,(times,means,low,high) in zip(axs,['Mode','Loaded covariance'],cells):
        im=_effect_map(ax,means,low,high,scenarios,times,f'Versus {ref.lower()}','dB')
        fig.colorbar(im,ax=ax,shrink=.8,label='SINR gain (dB)')
    fig.suptitle('Blue favors the posterior mean; red favors the reference',fontsize=14)
    fig.supxlabel(rf'Cell values: paired mean [95\% CI]; {frame.repeat.nunique()} independent trajectories. Open circles: CI includes zero.',fontsize=11)
    return _finish(fig,path)


def field_overview(result, path=None):
    """Separate tracker differences, posterior averaging, and a competitive reference."""
    frame=result['metrics'];scenarios=[s for s in FIELD_SCENARIOS if s in frame.scenario.unique()]
    references=['GROUSE','Mode','Rolling PCA']
    fig,axs=plt.subplots(1,3,figsize=(17,5.3),dpi=200,layout='constrained',sharey=True)
    for ax,ref in zip(axs,references):
        times,mean,low,high=_gain_cells(frame,'field_squared_error',ref,scenarios,percent=True)
        im=_effect_map(ax,mean,low,high,scenarios,times,f'Versus {ref}',r'\%')
        fig.colorbar(im,ax=ax,shrink=.8,label=r'MSE reduction (\% of reference mean)')
    fig.suptitle('Blue favors the posterior mean; red favors the reference',fontsize=14)
    fig.supxlabel(r'Paired mean difference [95\% CI], scaled by the mean reference loss. Mode and matching EW PCA coincide. Open circles: CI includes zero.',fontsize=11)
    return _finish(fig,path)


def paired_field(result, baseline='GROUSE', path=None):
    """Each point is one paired trajectory; the diagonal marks equal loss."""
    frame=_terminal(result['metrics']); pairs=_pairs(frame,'field_squared_error',baseline)
    scenarios=[s for s in FIELD_SCENARIOS if s in frame.scenario.unique()]
    fig,axs=plt.subplots(1,len(scenarios),figsize=(14,4.8),dpi=200,layout='constrained',squeeze=False)
    for ax,scenario in zip(axs.flat,scenarios):
        block=pairs.xs(scenario,level='scenario')
        x=block[baseline].to_numpy();y=block['Posterior mean'].to_numpy()
        lim=max(np.max(x),np.max(y))*1.12
        ax.fill_between([0,lim],[0,0],[0,lim],color=COLORS['Posterior mean'],alpha=.06)
        ax.plot([0,lim],[0,lim],color='#777777',ls='--',lw=1)
        ax.scatter(x,y,s=65,color=COLORS['Posterior mean'],edgecolor='white',lw=.8,zorder=3)
        ax.scatter([x.mean()],[y.mean()],marker='D',s=95,color='#222222',zorder=4,label='Mean')
        ax.set(xlim=(0,lim),ylim=(0,lim),xlabel=f'{baseline} squared field error',ylabel='Posterior mean squared field error',title=scenario)
        ax.set_aspect('equal',adjustable='box')
        ax.text(.04,.96,f'Lower loss in {int(np.sum(y<x))}/{len(x)} trials',transform=ax.transAxes,va='top',fontsize=11)
    fig.suptitle('Points below the diagonal favor the posterior mean',fontsize=14)
    return _finish(fig,path)


def coverage_regions(result, array=False, path=None):
    """Coverage intervals and directional bound ratios; no nominal coverage assumption."""
    frame=_terminal(result['metrics']);p=frame[frame.method=='Posterior mean']
    scenarios=[s for s in (ARRAY_SCENARIOS if array else FIELD_SCENARIOS) if s in p.scenario.unique()]
    table=coverage_table(p)
    fig,axs=plt.subplots(1,2 if array else 1,figsize=(14,7) if array else (11,5.5),dpi=200,layout='constrained',squeeze=False)
    ax=axs[0,0]; positions=np.arange(len(scenarios))
    for region,offset in [('Ball',-.13),('HPD',.13)]:
        g=table[table.region==region].set_index('scenario').reindex(scenarios)
        ax.errorbar(g.coverage,positions+offset,xerr=np.maximum(0,[g.coverage-g.low,g.high-g.coverage]),fmt='o',ms=7,capsize=3,color=COLORS[region],label=region)
    ax.axvline(.95,color='#555555',ls='--',lw=1.2,label=r'Nominal 95\%')
    ax.set(yticks=positions,yticklabels=scenarios,xlim=(-.03,1.04),xlabel='Repeated-trial coverage',title='Empirical terminal coverage')
    ax.invert_yaxis();ax.legend(loc='lower left',fontsize=10)
    if array:
        r=_terminal(result['regions']);pair=r.pivot(index=['scenario','repeat'],columns='region',values='upper')
        ax=axs[0,1]
        for i,scenario in enumerate(scenarios):
            block=pair.xs(scenario,level='scenario');ratios=block.HPD/block.Ball
            offsets=np.linspace(-.09,.09,len(ratios))
            ax.scatter(ratios,i+offsets,s=34,color=COLORS['Posterior mean'],alpha=.65)
            ax.scatter([ratios.median()],[i],s=85,marker='D',color='#222222',zorder=3)
        ax.axvline(1,color='#555555',ls='--',lw=1.2)
        ax.set(yticks=positions,yticklabels=scenarios,xlabel='HPD / ball leakage bound',title='Directional leakage bounds')
        ax.invert_yaxis()
        ax.text(.02,.01,'Each dot is one trial; diamonds are medians',transform=ax.transAxes,fontsize=10)
    fig.supxlabel(rf'Wilson 95\% intervals use {p.repeat.nunique()} trials; region thresholds and optimization have separate numerical errors.',fontsize=11)
    return _finish(fig,path)


def cost_tradeoff(result, field=False, path=None):
    """Task performance versus measured cumulative work, excluding oracle cost."""
    frame=_terminal(result['metrics'])
    scenarios=FIELD_SCENARIOS if field else ['Static','Drift','Contamination']
    value='field_squared_error' if field else 'sinr_db'
    fig,axs=plt.subplots(1,3,figsize=(16,5.3),dpi=200,layout='constrained')
    methods=['Posterior mean','Mode','GROUSE','Rolling PCA'] if field else ['Posterior mean','Mode','Loaded covariance']
    for ax,scenario in zip(axs,scenarios):
        block=frame[frame.scenario==scenario]
        for method in methods:
            g=block[block.method==method]
            if g.empty:continue
            center,low,high=_ci(g[value])
            x=float(g.decision_seconds.mean())
            ax.errorbar(x,center,yerr=[[max(0,center-low)],[max(0,high-center)]],fmt='o',ms=9,capsize=3,color=COLORS[method],label=method)
        oracle=block[block.method=='Oracle'][value]
        if len(oracle):ax.axhline(oracle.mean(),color=COLORS['Oracle'],ls=':',lw=1.4,label='Oracle performance')
        ax.set(xscale='log',xlabel='Cumulative update and moment time (s)',ylabel='Squared field error' if field else 'SINR (dB)',title=scenario)
        ax.grid(axis='x',which='major',alpha=.15)
    axs[0].legend(loc='best',fontsize=10)
    suffix='lower error is better' if field else 'higher SINR is better'
    fig.suptitle(f'The posterior decision has a computation cost; {suffix}',fontsize=14)
    fig.supxlabel('Oracle performance is a truth-based reference, not an available computation-cost competitor.',fontsize=11)
    return _finish(fig,path)


def field_maps(result, path=None):
    """A common amplitude scale and a separate common residual scale."""
    f=result['fields'].sort_values(['row','column'])
    fields={name:np.asarray(f[name]).reshape(8,8) for name in ['clean','observed','posterior_mean','mode']}
    residuals={name:fields[name]-fields['clean'] for name in ['posterior_mean','mode']}
    amplitude=max(float(np.max(abs(v))) for v in fields.values())
    error=max(float(np.max(abs(v))) for v in residuals.values())
    fig=plt.figure(figsize=(14,8.5),dpi=200,layout='constrained')
    grid=fig.add_gridspec(2,4,width_ratios=[1,1,1,.055])
    axs=np.array([[fig.add_subplot(grid[i,j]) for j in range(3)] for i in range(2)])
    amplitude_bar=fig.add_subplot(grid[0,3]);error_bar=fig.add_subplot(grid[1,3])
    items=[('clean','Clean'),('posterior_mean','Posterior mean'),('mode','Mode'),('observed','Observed')]
    field_axes=[axs[0,0],axs[0,1],axs[0,2],axs[1,0]]
    for ax,(key,title) in zip(field_axes,items):
        im=ax.imshow(fields[key],cmap='RdBu_r',vmin=-amplitude,vmax=amplitude,interpolation='nearest')
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        if key!='clean':
            rmse=np.sqrt(np.mean((fields[key]-fields['clean'])**2))
            ax.set_xlabel(rf'RMSE $={rmse:.3f}$')
    for ax,key,title in [(axs[1,1],'posterior_mean','Mean error'),(axs[1,2],'mode','Mode error')]:
        err=ax.imshow(residuals[key],cmap='RdBu_r',vmin=-error,vmax=error,interpolation='nearest')
        ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,cax=amplitude_bar,label='Field amplitude')
    fig.colorbar(err,cax=error_bar,label='Estimate minus clean field')
    fig.suptitle('Spatial reconstruction and residual structure',fontsize=14)
    fig.supxlabel('Fields share the amplitude scale; the two residuals share a separate error scale.',fontsize=11)
    return _finish(fig,path)


def calibration_overview(result, path=None):
    """Empirical paired benefits versus the posterior risk identity."""
    c=result['calibration'];ranks=sorted(c.input_rank.unique());times=sorted(c.t.unique())
    gains=np.empty((len(ranks),len(times)));low=gains.copy();high=gains.copy();excess=gains.copy()
    for i,rank in enumerate(ranks):
        for j,time in enumerate(times):
            g=c[(c.input_rank==rank)&(c.t==time)]
            d=100*(g.modal_squared_error-g.mean_squared_error)/g.modal_squared_error.mean()
            gains[i,j],low[i,j],high[i,j]=_ci(d)
            excess[i,j]=g.modal_excess_risk.mean()
    fig,axs=plt.subplots(1,2,figsize=(13,5),dpi=200,layout='constrained')
    im=_effect_map(axs[0],gains,low,high,[f'Rank {r}' for r in ranks],times,'Realized response error reduction',r'\%')
    fig.colorbar(im,ax=axs[0],shrink=.8,label=r'MSE reduction versus mode (\%)')
    im=axs[1].imshow(excess,cmap='Blues',aspect='auto')
    axs[1].set(xticks=range(len(times)),xticklabels=times,yticks=range(len(ranks)),yticklabels=[f'Rank {r}' for r in ranks],xlabel='Observations',title='Posterior modal excess risk')
    for i in range(len(ranks)):
        for j in range(len(times)):
            axs[1].text(j,i,f'{excess[i,j]:.4f}',ha='center',va='center',fontsize=12,color='white' if excess[i,j]>.6*excess.max() else '#222222')
    fig.colorbar(im,ax=axs[1],shrink=.8,label='Expected extra loss from using the mode')
    fig.supxlabel(r'Left: paired mean [95\% CI] across trials. Right: Eq. (30). Open circles on the left: CI includes zero.',fontsize=11)
    return _finish(fig,path)


def noise_overview(result, path=None):
    """Separate noise-estimation outcomes from the algebraic uncertainty correction."""
    frame=result['noise'];times=sorted(frame.t.unique());x=np.arange(len(times))
    fig,axs=plt.subplots(1,2,figsize=(13,5),dpi=200,layout='constrained')
    for protocol,offset,color in [('Passive',-.18,COLORS['Posterior mean']),('Calibration',.18,COLORS['Mode'])]:
        g=frame[frame.protocol==protocol].set_index('t').reindex(times)
        axs[0].bar(x+offset,g.s,width=.32,color=color,label=protocol,alpha=.9)
    axs[0].axhline(frame.true_s.iloc[0],color='#222222',ls='--',lw=1.3,label='True variance')
    axs[0].set(xticks=x,xticklabels=times,xlabel='Fitting time',ylabel='Estimated noise variance',title='Noise fitting is a separate experiment')
    axs[0].legend(fontsize=10)
    g=frame[frame.protocol=='Calibration'].set_index('t').reindex(times)
    axs[1].bar(x,g.naive_residual_variance,color='#999999',label='Mean residual')
    axs[1].bar(x,g.uncertainty_variance,bottom=g.naive_residual_variance,color=COLORS['Posterior mean'],label='Subspace uncertainty')
    for j,value in enumerate(g.expected_residual_variance):axs[1].text(j,value+.005,f'{value:.3f}',ha='center',fontsize=11)
    axs[1].set(xticks=x,xticklabels=times,xlabel='Fitting time',ylabel='Expected residual variance',title='Uncertainty adds a positive residual term')
    axs[1].legend(fontsize=10)
    fig.supxlabel('Residual components use the same final-parameter E-step target; sampled EM ascent is unverified.',fontsize=11)
    return _finish(fig,path)


def sampler_overview(result, array=False, path=None):
    """Expose where the original numerical budget is insufficient."""
    p=result['metrics'];p=p[p.method=='Posterior mean']
    scenarios=[s for s in (ARRAY_SCENARIOS if array else FIELD_SCENARIOS) if s in p.scenario.unique()]
    times=sorted(p.t.unique());values=p.groupby(['scenario','t']).min_ess.min().unstack('t').reindex(index=scenarios,columns=times)
    fig,axs=plt.subplots(1,2,figsize=(13,6 if array else 4.8),dpi=200,layout='constrained')
    v=np.asarray(values);im=axs[0].imshow(v,cmap='viridis',norm=LogNorm(vmin=max(1,v.min()),vmax=max(51,v.max())),aspect='auto')
    for i in range(len(scenarios)):
        for j in range(len(times)):
            axs[0].text(j,i,f'{v[i,j]:.0f}',ha='center',va='center',color='white' if v[i,j]<np.sqrt(v.min()*v.max()) else '#111111',fontsize=12)
            if v[i,j]<50:
                from matplotlib.patches import Rectangle
                axs[0].add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor='#D55E00',lw=2))
    axs[0].set(xticks=range(len(times)),xticklabels=times,yticks=range(len(scenarios)),yticklabels=scenarios,xlabel='Observations',title='Worst ESS across trials')
    fig.colorbar(im,ax=axs[0],shrink=.8,label='Minimum ESS')
    for passed,color,label in [(True,COLORS['Posterior mean'],'Passed'),(False,COLORS['Mode'],'Capped failure')]:
        g=p[p.sampling_ok.astype(bool)==passed]
        if len(g):axs[1].hist(g.min_ess,bins=np.geomspace(max(1,p.min_ess.min()*.9),p.min_ess.max()*1.1,14),alpha=.65,color=color,label=label)
    axs[1].axvline(50,color='#222222',ls='--',lw=1.2)
    axs[1].set(xscale='log',xlabel='Minimum ESS',ylabel='Posterior evaluations',title='Original sampling budget')
    axs[1].legend(fontsize=10)
    fig.supxlabel('Orange outlines mark ESS below 50; independent-chain agreement and MCSE are separate diagnostics.',fontsize=11)
    return _finish(fig,path)


def predictive_scores(result, path=None):
    """Predictive data fit and both distinct effective-sample-size diagnostics.

    Scores are computed before fitting/correction at different observations.
    Their progression is not an EM-ascent test. Importance-weight ESS does not
    account for autocorrelation in the prior sampler.
    """
    frame=result['noise']; protocols=['Passive','Calibration'];times=sorted(frame.t.unique())
    fig,axs=plt.subplots(1,3,figsize=(14,4.8),dpi=200,layout='constrained')
    specification=[('predictive_log_score','Pre-fit log score','Predictive log density',False),
                   ('score_sampler_ess','Sampler ESS','Minimum chain ESS',True),
                   ('score_importance_ess','Importance ESS','Importance-weight ESS',True)]
    for ax,(column,title,label,is_count) in zip(axs,specification):
        values=frame.pivot(index='protocol',columns='t',values=column).reindex(index=protocols,columns=times)
        array=np.asarray(values,dtype=float)
        image=ax.imshow(array,cmap='cividis',aspect='auto')
        ax.set(xticks=range(len(times)),xticklabels=times,yticks=range(len(protocols)),yticklabels=protocols,xlabel='Fitting time',title=title)
        lo,hi=float(array.min()),float(array.max())
        for i in range(len(protocols)):
            for j in range(len(times)):
                color='white' if array[i,j]<lo+.5*(hi-lo) else '#111111'
                text=f'{array[i,j]:.0f}' if is_count else f'{array[i,j]:.2f}'
                ax.text(j,i,text,ha='center',va='center',fontsize=13,color=color)
                if column=='score_sampler_ess' and array[i,j]<50:
                    from matplotlib.patches import Rectangle
                    ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor='#D55E00',lw=2))
        fig.colorbar(image,ax=ax,shrink=.8,label=label)
    fig.supxlabel('Scores concern different observations. Importance ESS excludes chain dependence; orange outlines mark sampler ESS below 50.',fontsize=11)
    return _finish(fig,path)
