"""Figures for the isolated memory, motion, and process-prediction studies.

Trajectory uncertainty, maintained-posterior uncertainty, and numerical error
have separate labels. Two-dimensional displays do not reduce inference rank.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .plotting import plt, save
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MOTION_LABELS={
    'identity_1':'Identity', 'forgetting_097':r'Fixed $\beta=0.97$',
    'tuned_forgetting':'Tuned forgetting', 'rolling_window':'Rolling window',
    'causal_velocity':'Causal velocity', 'partial_motion':'Half motion control',
    'exact_motion':'Exact motion control'}
MOTION_COLORS={
    'identity_1':'#777777','forgetting_097':'#CC79A7',
    'tuned_forgetting':'#009E73','rolling_window':'#D55E00',
    'causal_velocity':'#0072B2','partial_motion':'#E69F00','exact_motion':'#222222'}
PROCESS_COLORS={'Projected transition':'#0072B2','Full-density reference':'#222222',
                'Identity':'#777777','Power forgetting':'#009E73'}
LAW_LABELS={'slow':'Slow drift','fast':'Fast drift','acceleration':'Acceleration',
            'turn':'Abrupt turn','weak_excitation':'Weak excitation',
            'low_snr':'Lower SNR','corruption':'Corruption'}


def _finish(fig,path):
    assert plt.rcParams['text.usetex'] and fig.dpi==200
    if path is not None:save(fig,Path(path))
    return fig


def _student_radius(values):
    values=np.asarray(values,dtype=float)
    return (float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
            if len(values)>1 else 0.)


def motion_overview(frames,path=None,law='fast',input_frame='sensor'):
    """Report-time estimates, before/after errors, and a terminal credible pilot."""
    metrics=frames['metrics'];cost=frames['cost'];posterior=frames['posterior']
    metrics=metrics[(metrics.law==law)&(metrics.input_frame==input_frame)]
    cost=cost[(cost.law==law)&(cost.input_frame==input_frame)]
    if len(posterior):posterior=posterior[(posterior.law==law)&(posterior.input_frame==input_frame)]
    selected=['tuned_forgetting','causal_velocity','exact_motion']
    methods=['identity_1','tuned_forgetting','rolling_window','causal_velocity','exact_motion']
    fig,axs=plt.subplots(2,2,figsize=(14,9),dpi=200,layout='constrained')
    ax=axs[0,0]
    for method in selected:
        part=metrics[metrics.method==method]
        grouped=part.groupby('t').corrected_error
        mean=grouped.mean();radius=grouped.apply(_student_radius)
        ax.plot(mean.index,mean,'o-',ms=4,color=MOTION_COLORS[method],label=MOTION_LABELS[method])
        ax.fill_between(mean.index,mean-radius,mean+radius,color=MOTION_COLORS[method],alpha=.13)
    ax.set(title='Tracking over time',xlabel='Time',ylabel='Mean corrected error')
    ax.legend(fontsize=9);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.12)
    ax=axs[0,1]
    upper=0.
    for method in selected:
        part=cost[cost.method==method]
        ax.scatter(part.terminal_forecast_error,part.terminal_error,s=26,alpha=.6,
                   color=MOTION_COLORS[method],label=MOTION_LABELS[method],edgecolors='none')
        upper=max(upper,part.terminal_forecast_error.max(),part.terminal_error.max())
    upper*=1.08;ax.plot([0,upper],[0,upper],'--',color='#777777',lw=1)
    ax.set(title='Before and after correction',xlabel='One-step forecast error',
           ylabel='Corrected error',xlim=(0,upper),ylim=(0,upper));ax.grid(alpha=.12)
    ax=axs[1,0]
    values=[cost[cost.method==method].terminal_error.to_numpy() for method in methods]
    violin=ax.violinplot(values,positions=range(len(methods)),vert=False,
                         showmedians=True,showextrema=False)
    for body,method in zip(violin['bodies'],methods):
        body.set_facecolor(MOTION_COLORS[method]);body.set_alpha(.4)
    ax.set(yticks=range(len(methods)),yticklabels=[MOTION_LABELS[m] for m in methods],
           xlabel='Terminal chordal error',title='Repeated-data error')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    ax=axs[1,1];upper=0.
    for method in methods:
        if not len(posterior):break
        part=posterior[posterior.method==method]
        if not len(part):continue
        actual=cost[cost.method==method].set_index('trial').terminal_error.reindex(part.trial).to_numpy()
        ok=part.sampling_ok.astype(bool).to_numpy()
        ax.scatter(part.posterior_radius[ok],actual[ok],s=36,alpha=.75,
                   color=MOTION_COLORS[method],edgecolors='none')
        if (~ok).any():
            ax.scatter(part.posterior_radius[~ok],actual[~ok],s=43,facecolors='none',
                       edgecolors=MOTION_COLORS[method],linewidths=1.4)
        upper=max(upper,part.posterior_radius.max(),actual.max())
    if upper:
        ax.plot([0,upper*1.08],[0,upper*1.08],'--',color='#777777',lw=1)
        ax.set_xlim(0,upper*1.08);ax.set_ylim(0,upper*1.08)
    ax.set(title='State uncertainty',xlabel=r'Nominal 95\% posterior-ball radius',
           ylabel='Actual modal error');ax.grid(alpha=.12)
    if not len(posterior):ax.text(.5,.5,'Posterior pilot not run',ha='center',va='center',transform=ax.transAxes)
    trials=cost.trial.nunique();pilot=posterior.trial.nunique() if len(posterior) else 0
    frame_label='sensor-fixed inputs' if input_frame=='sensor' else 'co-rotating inputs'
    fig.supxlabel(f'{LAW_LABELS.get(law,law)}, {frame_label}, n=8, rank 3. '
                  rf'{trials} independent trajectories; shaded bands are pointwise 95\% intervals for mean error. '
                  f'Credible pilot: {pilot} trajectories; above diagonal means missed state. Open points flag failed sampling checks.',fontsize=9)
    return _finish(fig,path)


def application_gains(paired,path=None,all_scenarios=False,tuning=None):
    """Paired application loss differences, retaining uncertainty and failures."""
    if all_scenarios:
        applications=['array','field'];nrows=max(paired.groupby('application').scenario.nunique())
        fig,axs=plt.subplots(nrows,2,figsize=(14,3.3*nrows),dpi=200,layout='constrained',squeeze=False)
        groups=[]
        for col,application in enumerate(applications):
            for row,scenario in enumerate(paired[paired.application==application].scenario.drop_duplicates()):
                groups.append((axs[row,col],application,scenario))
            for row in range(paired[paired.application==application].scenario.nunique(),nrows):axs[row,col].remove()
    else:
        fig,axs=plt.subplots(1,2,figsize=(13,5),dpi=200,layout='constrained')
        groups=[(axs[0],'array','Drift'),(axs[1],'field','Matched')]
    for ax,application,scenario in groups:
        part=paired[(paired.application==application)&(paired.scenario==scenario)]
        selected_failures=0
        if tuning is not None:
            selected=tuning[(tuning.application==application)&(tuning.method=='Posterior mean')&tuning.selected.astype(bool)]
            selected_failures=int(selected.sampling_failures.sum())
        order=['Mode','Shrunk PCA','Loaded covariance','EW PCA','Rolling PCA','GROUSE']
        part=part.set_index('reference').reindex([m for m in order if m in part.reference.to_numpy()]).reset_index()
        ax.axvline(0,color='#666666',ls='--',lw=1)
        for i,r in enumerate(part.itertuples()):
            color='#0072B2' if r.gain_low>0 else '#D55E00' if r.gain_high<0 else '#777777'
            failed=r.posterior_sampling_failures>0 or selected_failures>0
            boundary=bool(getattr(r,'unresolved_tuning_boundary',False))
            ax.errorbar(r.gain,i,xerr=[[r.gain-r.gain_low],[r.gain_high-r.gain]],
                        fmt='^' if boundary else 'o',capsize=4,color=color,markerfacecolor='white' if failed else color,ms=7)
        ax.set(yticks=range(len(part)),yticklabels=part.reference,
               xlabel='Reduction in held-out power' if application=='array' else 'Reduction in reconstruction error',
               title=f'{"Array" if application=="array" else "Field"}: {scenario}'+
                     ('\nUnresolved numerical accuracy' if selected_failures else ''))
        ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    fig.supxlabel('Reference loss minus posterior-mean loss. Positive values favor the mean decision. '
                  r'Pointwise paired 95\% Student intervals across trajectories; exploratory. '
                  'Open markers: unresolved test or selected-validation sampling. Triangles: unresolved tuning boundary.',fontsize=9)
    return _finish(fig,path)


def process_coverage_width(summary,path=None,law='Matched transition',include_controls=False):
    """Four different coverage/size quantities use four separate axes."""
    data=summary[summary.law==law]
    kinds=[('ball','Subspace ball','Mean ball arc length (radians)'),
           ('hpd','State HPD region','Mean HPD arc length (radians)'),
           ('clean','Clean response','Mean response-interval width'),
           ('noisy','Noisy observation','Mean observation-interval width')]
    fig,axs=plt.subplots(2,2,figsize=(12,6.8),dpi=200,layout='constrained')
    methods=['Projected transition','Full-density reference']
    if include_controls:methods+=['Identity','Power forgetting']
    data=data[data.method.isin(methods)]
    kappas=sorted(data.kappa.unique());markers=['o','D','s','^']
    for ax,(kind,title,label) in zip(axs.flat,kinds):
        ax.axhline(.95,ls='--',color='#555555',lw=1)
        for r in data.itertuples():
            if r.method not in methods:continue
            coverage=getattr(r,kind+'_coverage');lo=getattr(r,kind+'_wilson_low');hi=getattr(r,kind+'_wilson_high')
            ax.errorbar(getattr(r,kind+'_mean_width'),coverage,yerr=[[coverage-lo],[hi-coverage]],
                        fmt=markers[kappas.index(r.kappa)%len(markers)],capsize=3,
                        color=PROCESS_COLORS[r.method],ms=7,alpha=.85,
                        markerfacecolor='white' if r.method=='Full-density reference' else PROCESS_COLORS[r.method])
        lower=-.02 if include_controls else max(0.,np.floor((data[kind+'_wilson_low'].min()-.02)*20)/20)
        ax.set(title=title,xlabel=label,ylabel='Coverage',ylim=(lower,1.005));ax.grid(alpha=.12)
    legend=[Line2D([],[],color=PROCESS_COLORS[m],marker='o',ls='',label=m,
                   markerfacecolor='white' if m=='Full-density reference' else PROCESS_COLORS[m]) for m in methods]
    legend.extend(Line2D([],[],color='#555555',marker=markers[i%len(markers)],ls='',label=rf'$\kappa={k:g}$') for i,k in enumerate(kappas))
    fig.legend(handles=legend,loc='outside upper center',ncol=3,fontsize=10)
    trials='/'.join(str(int(x)) for x in sorted(data.trials.unique()))
    control_note=r' Fixed $\beta=0.85$ is a diagnostic control, not a tuned comparator.' if include_controls else ''
    fig.supxlabel(f'{law}: real Gr(1,2), {trials} independent trajectories per regime; '+r'$\kappa$ is supplied from simulation. '
                  r'Bars are pointwise 95\% Wilson intervals.'+control_note,fontsize=10)
    return _finish(fig,path)


def motion_coverage_width(coverage,path=None):
    """HPD natural-parameter thresholds are not geometric radii or widths."""
    kinds=[('ball_covered','Subspace ball','Mean posterior-ball radius'),
           ('hpd_covered','State HPD region','Mean HPD loss threshold'),
           ('clean_response_covered','Clean response','Mean response-interval width'),
           ('predictive_covered','Noisy observation','Mean observation-interval width')]
    methods=['identity_1','tuned_forgetting','rolling_window','causal_velocity','exact_motion']
    fig,axs=plt.subplots(2,2,figsize=(12,8),dpi=200,layout='constrained')
    for ax,(kind,title,label) in zip(axs.flat,kinds):
        ax.axhline(.95,ls='--',color='#555555',lw=1)
        for r in coverage[coverage.region==kind].itertuples():
            if r.method not in methods:continue
            ax.errorbar(r.mean_width,r.coverage,yerr=[[r.coverage-r.low],[r.high-r.coverage]],
                        fmt='o' if r.law=='slow' else 'D',color=MOTION_COLORS[r.method],
                        capsize=3,ms=7,alpha=.8,
                        markerfacecolor='white' if r.sampling_passes<r.trials else MOTION_COLORS[r.method])
        ax.set(title=title,xlabel=label,ylabel='Coverage',ylim=(-.02,1.04));ax.grid(alpha=.12)
    handles=[Line2D([],[],marker='o',color=MOTION_COLORS[m],ls='',label=MOTION_LABELS[m]) for m in methods]
    handles.extend([Line2D([],[],marker='o',color='#555555',ls='',label='Slow drift'),
                    Line2D([],[],marker='D',color='#555555',ls='',label='Fast drift')])
    fig.legend(handles=handles,loc='outside upper center',ncol=4,fontsize=9)
    counts='/'.join(str(int(n)) for n in sorted(coverage.trials.unique()))
    fig.supxlabel(f'Conditional n=8, rank 3; {counts}-trajectory terminal pilot. Pointwise Wilson intervals. '
                  'Open markers flag capped posterior failures. HPD loss thresholds have different units from geometric region size.',fontsize=10)
    return _finish(fig,path)


def memory_tuning(tuning,selected,path=None):
    """Average each validation trajectory over its law/frame mixture first."""
    fig,axs=plt.subplots(1,3,figsize=(15,4.6),dpi=200,layout='constrained')
    definitions=[('forgetting','beta','Exponential memory'),('window','window','Likelihood window')]
    for ax,(family,column,title) in zip(axs[:2],definitions):
        part=tuning[tuning.family==family]
        trial=part.groupby([column,'trial']).terminal_error.mean().reset_index()
        grouped=trial.groupby(column).terminal_error;mean=grouped.mean();radius=grouped.apply(_student_radius)
        ax.errorbar(mean.index,mean,yerr=radius,fmt='o-',color='#0072B2',capsize=3)
        value=selected['beta' if family=='forgetting' else 'window']
        ax.axvline(value,color='#D55E00',ls='--',lw=1)
        ax.set(title=title,xlabel=r'$\beta$' if family=='forgetting' else 'Window length',ylabel='Validation chordal error')
        ax.grid(axis='y',alpha=.12)
    ax=axs[2];part=tuning[tuning.family=='velocity']
    for lag in sorted(part.lag.unique()):
        trial=part[part.lag==lag].groupby(['damping','trial']).terminal_error.mean().reset_index()
        grouped=trial.groupby('damping').terminal_error;mean=grouped.mean();radius=grouped.apply(_student_radius)
        ax.errorbar(mean.index,mean,yerr=radius,fmt='o-',capsize=3,label=f'Lag {int(lag)}')
        if int(lag)==selected['lag']:
            ax.scatter(selected['damping'],mean.loc[selected['damping']],marker='*',s=160,
                       color='#222222',edgecolor='white',linewidth=.6,zorder=5)
    ax.axvline(selected['damping'],color='#777777',ls='--',lw=1)
    ax.set(title='Causal velocity',xlabel='Damping',ylabel='Validation chordal error');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.12)
    fig.supxlabel('Independent validation trajectories, equally weighted across declared laws and input frames. '
                  'Pointwise Student intervals use trajectory averages; dashed lines mark frozen selections.',fontsize=10)
    return _finish(fig,path)


def motion_stress(summary,path=None):
    methods=['identity_1','forgetting_097','tuned_forgetting','rolling_window','causal_velocity','exact_motion']
    laws=[x for x in LAW_LABELS if x in summary.law.unique()]
    fig,axs=plt.subplots(1,2,figsize=(14,7),dpi=200,layout='constrained',sharey=True)
    offsets=np.linspace(-.28,.28,len(methods))
    for ax,frame,title in zip(axs,['sensor','co_rotating'],['Sensor-fixed inputs','Co-rotating inputs']):
        part=summary[summary.input_frame==frame]
        for j,method in enumerate(methods):
            g=part[part.method==method].set_index('law').reindex(laws)
            half=student_t.ppf(.975,g.trials.to_numpy()-1)*g.se.to_numpy()
            ax.errorbar(g.mean_error,np.arange(len(laws))+offsets[j],xerr=half,fmt='o',
                        color=MOTION_COLORS[method],capsize=2,ms=5,label=MOTION_LABELS[method])
        ax.set(title=title,xlabel='Terminal chordal error',yticks=range(len(laws)),
               yticklabels=[LAW_LABELS[x] for x in laws]);ax.grid(axis='x',alpha=.12)
    axs[0].invert_yaxis()
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=10)
    fig.supxlabel(r'One frozen unknown-motion policy for every test regime. Pointwise 95\% Student intervals across independent trajectories. '
                  'Stress comparisons are exploratory; synthetic exact motion is a separate information control.',fontsize=10)
    return _finish(fig,path)


def application_tuning(tuning,application,path=None):
    part=tuning[tuning.application==application]
    methods=[m for m in ['Posterior mean','EW PCA','Shrunk PCA','Loaded covariance','Rolling PCA','GROUSE'] if m in part.method.unique()]
    cols=2 if len(methods)==4 else 3;rows=int(np.ceil(len(methods)/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(15,4.2*rows),dpi=200,layout='constrained',squeeze=False)
    for ax,method in zip(axs.flat,methods):
        g=part[part.method==method]
        if method in ('Loaded covariance','Shrunk PCA'):
            other='loading' if method=='Loaded covariance' else 'rho'
            table=g.pivot(index=other,columns='beta',values='validation_loss').sort_index()
            color=ax.imshow(table.to_numpy(),origin='lower',aspect='auto',cmap='viridis')
            ax.set(xticks=range(len(table.columns)),xticklabels=[f'{x:g}' for x in table.columns],
                   yticks=range(len(table.index)),yticklabels=[f'{x:g}' for x in table.index],
                   xlabel=r'$\beta$',ylabel='Loading' if other=='loading' else 'Shrinkage weight')
            for r in g[g.selected.astype(bool)].itertuples():
                y=getattr(r,other);ax.scatter(list(table.columns).index(r.beta),list(table.index).index(y),marker='*',s=180,
                                             color='white',edgecolor='black',linewidth=1)
            fig.colorbar(color,ax=ax,label='Validation loss',shrink=.8)
        else:
            column='window' if method=='Rolling PCA' else 'step' if method=='GROUSE' else 'beta'
            g=g.sort_values(column);ax.plot(g[column],g.validation_loss,'o-',color='#0072B2')
            chosen=g[g.selected.astype(bool)];ax.scatter(chosen[column],chosen.validation_loss,s=130,marker='*',color='#D55E00',zorder=4)
            ax.set(xlabel='Window length' if column=='window' else 'Step size' if column=='step' else r'$\beta$',ylabel='Validation loss')
            if column in ('window','step'):ax.set_xscale('log')
            ax.grid(axis='y',alpha=.12)
        boundary=bool(g.loc[g.selected.astype(bool),'boundary'].any())
        ax.set_title(method+(' (boundary)' if boundary else ''))
        failures=int(g.sampling_failures.sum()) if 'sampling_failures' in g else 0
        if failures:ax.text(.02,.02,f'Failed diagnostics: {failures}',transform=ax.transAxes,color='#AA3377',fontsize=9)
    for ax in list(axs.flat)[len(methods):]:ax.remove()
    fig.supxlabel(f'{"Array" if application=="array" else "Field"} validation: full frozen candidate families. '
                  'Stars mark selected settings; boundary flags remain visible. These are validation curves, not test performance.',fontsize=10)
    return _finish(fig,path)


def application_cost(metrics,path=None):
    """Separate per-step filtering cost and per-report posterior/decision cost."""
    fig,axs=plt.subplots(2,2,figsize=(14,9),dpi=200,layout='constrained')
    order=['Posterior mean','Mode','Shrunk PCA','Loaded covariance','EW PCA','Rolling PCA','GROUSE']
    for row,(application,scenario) in enumerate([('array','Drift'),('field','Matched')]):
        part=metrics[(metrics.application==application)&(metrics.scenario==scenario)]
        methods=[m for m in order if m in part.method.unique()]
        g=part.groupby('method').mean(numeric_only=True).reindex(methods)
        positions=np.arange(len(methods));step=g.t.to_numpy()
        ax=axs[row,0]
        prediction=1000*g.prediction_seconds.to_numpy()/step;update=1000*g.update_seconds.to_numpy()/step
        ax.barh(positions,prediction,color='#E69F00',label='Prediction')
        ax.barh(positions,update,left=prediction,color='#777777',label='Correction/update')
        ax.set(yticks=positions,yticklabels=methods,xlabel='Mean time per step (ms)',title=f'{application.capitalize()}: filtering')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
        ax=axs[row,1];moment=g.moment_seconds.to_numpy();decision=g.decision_seconds.to_numpy()
        ax.barh(positions,moment,color='#0072B2',label='Posterior moments')
        ax.barh(positions,decision,left=moment,color='#009E73',label='Decision')
        ax.set(yticks=positions,yticklabels=methods,xlabel='Mean time per terminal report (s)',title=f'{application.capitalize()}: reporting')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#E69F00','Prediction'),('#777777','Correction/update'),('#0072B2','Posterior moments'),('#009E73','Decision')]],loc='outside upper center',ncol=4)
    fig.supxlabel('Measured means on the recorded hardware, with terminal-only posterior reporting. '
                  'No regional optimization is used in this pilot. Timings include concurrent-worker variability.',fontsize=10)
    return _finish(fig,path)


def process_prediction_fans(paths,trials,path=None,law='Matched transition',kappa=4.):
    """One preassigned circular-process trajectory with full-density reference."""
    p=paths[(paths.law==law)&(paths.kappa==kappa)]
    q=trials[(trials.law==law)&(trials.kappa==kappa)&(trials.trial==0)]
    methods=['Projected transition','Identity','Power forgetting']
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),dpi=200,layout='constrained',sharex=True,sharey=True)
    for ax,method in zip(axs,methods):
        g=p[p.method==method].sort_values('t');r=q[q.method==method].sort_values('t')
        ref=q[q.method=='Full-density reference'].sort_values('t')
        ax.fill_between(r.t,r.clean_lower,r.clean_upper,color=PROCESS_COLORS[method],alpha=.18,label=r'95\% posterior interval')
        ax.plot(g.t,g.true_response,color='#222222',lw=1.6,label='True response')
        ax.plot(g.t,g.mean_response,'o-',color=PROCESS_COLORS[method],ms=3,label='Posterior mean')
        ax.plot(ref.t,ref.clean_lower,':',color='#AA3377',lw=1.5,label=r'Reference 95\% interval')
        ax.plot(ref.t,ref.clean_upper,':',color='#AA3377',lw=1.5)
        ax.set(title=method,xlabel='Time');ax.grid(alpha=.12)
    axs[0].set_ylabel(r'$e_2^\top P_t e_1$')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,fontsize=10)
    fig.supxlabel(r'One preassigned real Gr(1,2) trajectory, $\kappa='+f'{kappa:g}'+r'$. '
                  'Posterior quantiles are evaluated at reporting times and joined for display; no simultaneous path guarantee.',fontsize=10)
    return _finish(fig,path)
