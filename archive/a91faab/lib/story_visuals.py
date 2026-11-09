"""Main-paper figures: predictions, projected posterior draws, and drift guarantees.

Inference uses all n dimensions. Coordinate projections are visual summaries,
not a claim that inference has been reduced to two or three dimensions.
"""
from pathlib import Path
import hashlib,json,inspect
import numpy as np
import pandas as pd
from .plotting import plt,save
from matplotlib.lines import Line2D
from .visuals import COLORS
from .validation import require_gate

PROTOCOLS={'static':'Static','known_slow':'Known motion','identity_slow':r'$G=I$',
           'forget_slow':r'$G=I$, forgetting','known_fast':'Known fast motion',
           'identity_fast':r'$G=I$, fast drift','forget_fast':r'$G=I$, fast drift, forgetting'}
PALETTE={'static':'#0072B2','known_slow':'#0072B2','known_fast':'#0072B2',
         'identity_slow':'#D55E00','forget_slow':'#009E73','identity_fast':'#CC79A7','forget_fast':'#E69F00'}


def load_story(root,name):
    root=Path(root);meta=json.loads((root/'results'/f'{name}_config.json').read_text())
    assert meta['core_hash']==require_gate(root)['core_hash']
    sources=meta.get('files',meta.get('source',{}).get('files',{}))
    for path,digest in sources.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
    from . import moments
    functions=meta.get('functions',meta.get('source',{}).get('sampling_functions',{}))
    for function,digest in functions.items():assert hashlib.sha256(inspect.getsource(getattr(moments,function)).encode()).hexdigest()==digest,function
    frames={}
    for table,digest in meta['tables'].items():
        path=root/'results'/f'{name}_{table}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
        frames[table]=pd.read_csv(path)
    return dict(**frames,config=meta['config'],metadata=meta)


def _finish(fig,path):
    assert fig.dpi==200
    if path is not None:save(fig,path)
    return fig


def array_prediction_fans(story,envelope=None,path=None):
    f=story['spectra'];times=sorted(f.t.unique());scenarios=['Static','Drift']
    fig,axs=plt.subplots(2,len(times),figsize=(15,8),dpi=200,layout='constrained',sharex=True,sharey=True)
    for row,scenario in enumerate(scenarios):
        for col,time in enumerate(times):
            ax=axs[row,col];g=f[(f.scenario==scenario)&(f.t==time)].sort_values('angle')
            if envelope is None:
                ax.fill_between(g.angle,g.q025,g.q975,color=COLORS['Posterior mean'],alpha=.15,label=r'95\% response quantiles')
            else:
                e=envelope[(envelope.scenario==scenario)&(envelope.t==time)].sort_values('angle')
                ax.fill_between(e.angle,e.lower,e.upper,color=COLORS['Posterior mean'],alpha=.15,label='HPD response envelope')
            ax.fill_between(g.angle,g.q25,g.q75,color=COLORS['Posterior mean'],alpha=.26,label=r'50\% posterior responses')
            ax.plot(g.angle,g['mean'],color=COLORS['Posterior mean'],lw=1.8,label='Posterior mean')
            ax.plot(g.angle,g['mode'],color=COLORS['Mode'],ls='--',lw=1.2,label='Mode')
            ax.plot(g.angle,g.truth,color='#222222',lw=1.4,label='True response')
            ax.set_title(f'{scenario}: {time} snapshots');ax.set_ylim(.25,2.5)
            ax.set_xticks([-60,-30,0,30,60]);ax.grid(axis='y',alpha=.12)
            if col==0:ax.set_ylabel('Interference-plus-noise power')
            if row==1:ax.set_xlabel('Steering angle (degrees)')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=5,fontsize=10)
    caption='Native complex inference: 16 channels, rank 3. '+('Central response quantiles are pointwise, not simultaneous confidence bands.' if envelope is None else 'Estimated HPD envelopes bound evaluated responses for included states; drift prediction is approximate.')
    fig.supxlabel(caption,fontsize=10)
    return _finish(fig,path)


def array_power_distributions(story,path=None):
    f=story['power'];times=[int(f.t.min()),int(f.t.max())];methods=['Posterior mean','Mode','Loaded covariance']
    fig,axs=plt.subplots(2,2,figsize=(13,7.5),dpi=200,layout='constrained')
    for row,scenario in enumerate(['Static','Drift']):
        for col,time in enumerate(times):
            ax=axs[row,col];g=f[(f.scenario==scenario)&(f.t==time)]
            arrays=[g[g.method==method].power.to_numpy() for method in methods]
            violin=ax.violinplot(arrays,positions=np.arange(3),vert=False,showmeans=False,showextrema=False,quantiles=[[.05,.5,.95]]*3)
            for body,method in zip(violin['bodies'],methods):body.set_facecolor(COLORS[method]);body.set_alpha(.4)
            for i,method in enumerate(methods):
                part=g[g.method==method]
                ax.scatter(part.posterior_expected_power.iloc[0],i,s=45,color=COLORS[method],zorder=4)
                ax.scatter(part.true_power.iloc[0],i,marker='D',s=45,color='#222222',zorder=4)
            ax.set(yticks=range(3),yticklabels=['Mean weights','Modal weights','Loaded weights'],xlabel='Beamformer output power',title=f'{scenario}: {time} snapshots')
            ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#666666',label='Posterior expected power'),Line2D([],[],marker='D',ls='',color='#222222',label='Power at the true state')],loc='outside upper center',ncol=2)
    fig.supxlabel('Distributions use one maintained posterior and different weight choices. They are not repeated-training performance distributions.',fontsize=10)
    return _finish(fig,path)


def prediction_fans(story,fast=False,path=None):
    keys=['known_fast','identity_fast','forget_fast'] if fast else ['known_slow','identity_slow','forget_slow']
    posterior=story['posterior'];modal=story['modal'];bounds=story['bounds']
    fig,axs=plt.subplots(2,3,figsize=(14,8),dpi=200,layout='constrained',sharex=True)
    for col,key in enumerate(keys):
        g=posterior[(posterior.trial==0)&(posterior.protocol==key)].sort_values('t')
        truth=modal[(modal.trial==0)&(modal.protocol==key)&(modal.t>=g.t.min())].sort_values('t')
        ax=axs[0,col]
        ax.fill_between(g.t,g.q025,g.q975,color='#0072B2',alpha=.16,label=r'95\% posterior interval')
        ax.fill_between(g.t,g.q25,g.q75,color='#0072B2',alpha=.3,label=r'50\% posterior interval')
        ax.plot(truth.t,truth.true_response,color='#222222',lw=1.6,label='True response')
        ax.plot(g.t,g.mean_response,'o-',color='#0072B2',ms=4,label='Posterior mean')
        ax.plot(truth.t,truth.mode_response,ls='--',color='#D55E00',lw=1,label='Mode')
        bad=~g.clean_response_covered.astype(bool)
        ax.scatter(g.loc[bad,'t'],g.loc[bad,'true_response'],marker='x',s=45,color='#AA3377',zorder=5)
        ax.set_title(PROTOCOLS[key]);ax.set_ylabel(r'$e_2^\top P_t e_1$')
        # A distinct panel shows subspace uncertainty; this is not the theorem bound.
        ax=axs[1,col];b=bounds[(bounds.trial==0)&(bounds.protocol==key)].sort_values('t')
        ax.fill_between(g.t,0,g.posterior_radius,color='#0072B2',alpha=.18,label=r'95\% posterior ball radius')
        ax.plot(g.t,g.posterior_radius,color='#0072B2',lw=1.4)
        ax.plot(truth.t,truth.chordal_error,color='#222222',lw=1.5,label='Actual modal error')
        misses=~g.ball_covered.astype(bool)
        err=b.set_index('t').chordal_error.reindex(g.loc[misses,'t'])
        ax.scatter(g.loc[misses,'t'],err,marker='x',s=40,color='#AA3377',zorder=4)
        ax.set(xlabel='Time',ylabel=r'Chordal distance $d_c$');ax.set_ylim(bottom=0)
        for sub in axs[:,col]:sub.grid(axis='y',alpha=.12)
        if not g.sampling_ok.astype(bool).all():axs[0,col].text(.02,.02,'Some capped sampler diagnostics failed',transform=axs[0,col].transAxes,color='#AA3377',fontsize=9)
    axs[0,0].legend(fontsize=9,loc='best');axs[1,0].legend(fontsize=9,loc='best')
    fig.supxlabel(r'Inference: $n=8,k=3$. One preassigned trajectory; crosses mark missed truths. Plots begin at the first posterior report; quantiles are joined for display.',fontsize=10)
    return _finish(fig,path)


def projected_clouds(story,fast=False,path=None):
    keys=['known_fast','identity_fast','forget_fast'] if fast else ['known_slow','identity_slow','forget_slow']
    data=story['cloud'];times=[16,64,128]
    selected=data[data.protocol.isin(keys)]
    xmin=min(selected.x.min(),selected.true_x.min())-.015;xmax=max(selected.x.max(),selected.true_x.max())+.015
    ymin=min(selected.y.min(),selected.true_y.min())-.015;ymax=max(selected.y.max(),selected.true_y.max())+.015
    fig,axs=plt.subplots(3,3,figsize=(12,10),dpi=200,layout='constrained',sharex=True,sharey=True)
    for row,key in enumerate(keys):
        for col,time in enumerate(times):
            g=data[(data.protocol==key)&(data.t==time)];ax=axs[row,col];r=g.iloc[0]
            ax.scatter(g.x,g.y,s=9,color='#0072B2',alpha=.25,edgecolors='none')
            ax.scatter(r.mean_x,r.mean_y,s=65,marker='D',color='#0072B2',edgecolor='white',zorder=4)
            ax.scatter(r.mode_x,r.mode_y,s=55,marker='o',color='#D55E00',edgecolor='white',zorder=4)
            ax.scatter(r.true_x,r.true_y,s=115,marker='*',color='#222222',zorder=5)
            ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),title=f'{PROTOCOLS[key]}, t={time}')
            if row==2:ax.set_xlabel(r'$e_2^\top P_t e_1$')
            if col==0:ax.set_ylabel(r'$e_3^\top P_t e_1$')
            ax.grid(alpha=.12)
    fig.legend(handles=[Line2D([],[],marker='.',ls='',color='#0072B2',label='Posterior draw'),Line2D([],[],marker='D',ls='',color='#0072B2',label='Mean prediction'),Line2D([],[],marker='o',ls='',color='#D55E00',label='Modal prediction'),Line2D([],[],marker='*',ls='',color='#222222',label='Truth')],loc='outside upper center',ncol=4,fontsize=10)
    fig.supxlabel('Two fixed coordinates of an eight-dimensional predicted response. All eight dimensions and rank three are used in inference.',fontsize=11)
    return _finish(fig,path)


def drift_certificate(story,path=None):
    b=story['bounds'];g=b[b.t==b.t.max()]
    keys=['static','identity_slow','forget_slow','identity_fast','forget_fast']
    g=g[g.protocol.isin(keys)]
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),dpi=200,layout='constrained')
    ax=axs[0]
    ax.fill([0,.5,0],[0,0,.5],color='#009E73',alpha=.12)
    ax.plot([0,.5],[.5,0],ls='--',color='#555555',lw=1.2)
    for key in keys:
        h=g[g.protocol==key]
        ax.scatter(h.noise_term/h.gamma,h.drift_term/h.gamma,s=22,alpha=.55,color=PALETTE[key],label=PROTOCOLS[key])
    for a,c in [('identity_slow','forget_slow'),('identity_fast','forget_fast')]:
        u=g[g.protocol==a];v=g[g.protocol==c]
        start=(np.median(u.noise_term/u.gamma),np.median(u.drift_term/u.gamma))
        end=(np.median(v.noise_term/v.gamma),np.median(v.drift_term/v.gamma))
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color=PALETTE[a],lw=1.5))
    ax.set(xlabel=r'Noise term / gap $\gamma_t$',ylabel=r'Drift term $d_t/\gamma_t$',title=r'The condition is $\varepsilon_t/\gamma_t<1/2$')
    ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.legend(fontsize=9)
    ax=axs[1];valid=g[g.valid.astype(bool)]
    for key in keys:
        h=valid[valid.protocol==key]
        if len(h):ax.scatter(h.bound,h.chordal_error,s=25,alpha=.65,color=PALETTE[key],label=PROTOCOLS[key])
    ceiling=float(g.ceiling.iloc[0]);ax.plot([0,ceiling],[0,ceiling],'--',color='#777777',lw=1)
    ax.axvline(ceiling,ls=':',color='#999999',lw=1)
    ax.set(xlabel='Theorem error bound',ylabel='Actual modal error',title='Only valid certificates are plotted',xlim=(0,ceiling*1.03),ylim=(0,ceiling*1.03))
    ax.legend(fontsize=9)
    invalid=int((~g.valid.astype(bool)).sum());violations=int(valid.bound_violated.astype(bool).sum())
    fig.supxlabel(f'Fixed terminal time, {story["config"]["modal_trials"]} trajectories, zero initial parameter. Invalid conditions: {invalid}/{len(g)}; valid-certificate violations: {violations}/{len(valid)}. Arrows join protocol medians.',fontsize=10)
    return _finish(fig,path)


def monte_carlo_uncertainty(story,path=None):
    b=story['bounds'];g=b[b.t==b.t.max()];p=story['posterior'];p=p[p.t==p.t.max()]
    keys=['static','identity_slow','forget_slow','identity_fast','forget_fast']
    labels=['Exact frame','Slow drift','Slow + forgetting','Fast drift','Fast + forgetting']
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),dpi=200,layout='constrained')
    values=[g[g.protocol==key].chordal_error.to_numpy() for key in keys]
    violin=axs[0].violinplot(values,positions=range(5),showmeans=False,showmedians=True,showextrema=False)
    for body,key in zip(violin['bodies'],keys):body.set_facecolor(PALETTE[key]);body.set_alpha(.5)
    axs[0].set(xticks=range(5),xticklabels=labels,ylabel=r'Modal error $d_c$',title='Repeated-data error distribution')
    axs[0].tick_params(axis='x',rotation=25)
    upper=0
    for key,label in zip(keys,labels):
        h=p[p.protocol==key];r=g[g.protocol==key].set_index('trial').chordal_error.reindex(h.trial).to_numpy()
        axs[1].scatter(h.posterior_radius,r,s=40,alpha=.75,color=PALETTE[key],label=label)
        upper=max(upper,h.posterior_radius.max(),max(r))
    axs[1].plot([0,upper*1.08],[0,upper*1.08],'--',color='#777777',lw=1.2)
    axs[1].set(xlabel=r'Posterior 95\% ball radius',ylabel='Actual modal error',title='Above the diagonal: missed state')
    axs[1].legend(fontsize=9)
    fig.supxlabel(f'Left: {story["config"]["modal_trials"]} independent trajectories. Right: {story["config"]["posterior_trials"]} posterior-evaluated trajectories; not all have identical interval accuracy.',fontsize=10)
    return _finish(fig,path)
