"""Visual interpretation of the controlled, shared-data decision experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .plotting import plt, save, DPI
from .visuals import _effect_map, COLORS
from .reporting import wilson
from .validation import require_gate


def load_study(root):
    root=Path(root);meta=json.loads((root/'results/decision_study_config.json').read_text())
    assert meta['core_hash']==require_gate(root)['core_hash']
    for name,digest in meta['sources'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    frames={}
    for name,digest in meta['tables'].items():
        path=root/'results'/f'decision_study_{name}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,path
        frames[name]=pd.read_csv(path)
    return dict(**frames,config=meta['config'],checks=meta['checks'])


def gain_maps(study,task='Projection',path=None):
    table=study['paired'];table=table[table.task==task]
    references=['Mode','GROUSE','Shrunk PCA','Loaded covariance' if task=='Beamforming' else 'Prior mean']
    noises=study['config']['noise_levels'];times=study['config']['checkpoints']
    fig,axs=plt.subplots(2,2,figsize=(14,9),dpi=DPI,layout='constrained')
    for ax,reference in zip(axs.flat,references):
        mean=np.full((len(noises),len(times)),np.nan);low=mean.copy();high=mean.copy()
        for i,_ in enumerate(noises):
            for j,time in enumerate(times):
                row=table[(table.regime==i)&(table.t==time)&(table.reference==reference)]
                if row.empty:continue
                row=row.iloc[0];scale=100/row.baseline_loss
                mean[i,j]=row.paired_gain*scale;low[i,j]=row.gain_low*scale;high[i,j]=row.gain_high*scale
        im=_effect_map(ax,mean,low,high,[rf'$s={s:g}$' for s in noises],times,f'Versus {reference}',r'\%')
        fig.colorbar(im,ax=ax,shrink=.8,label=r'Loss reduction (\% of reference mean)')
    title={'Projection':'Fresh-response prediction','Reconstruction':'Current-vector reconstruction','Beamforming':'Interference power'}[task]
    fig.suptitle(title+': same observations, different decisions',fontsize=15)
    fig.supxlabel(r'Blue favors the posterior mean. Cells: paired gain [95\% CI], scaled by mean reference loss. Open circles: CI includes zero.',fontsize=10)
    if path is not None:save(fig,path)
    return fig


def decision_calibration(study,path=None):
    coverage=study['coverage'];metrics=study['metrics']
    noises=study['config']['noise_levels']
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),dpi=DPI,layout='constrained')
    positions=[];labels=[];position=0
    for regime,s in enumerate(noises):
        g=coverage[coverage.regime==regime]
        for region,column in [('Ball','ball_covered'),('HPD','hpd_covered')]:
            n=len(g);p=g[column].astype(bool).mean();low,high=wilson(int(g[column].astype(bool).sum()),n)
            axs[0].errorbar(p,position,xerr=[[max(0,p-low)],[max(0,high-p)]],fmt='o',ms=8,capsize=3,color=COLORS[region])
            positions.append(position);labels.append(rf'$s={s:g}$, {region}');position+=1
    axs[0].axvline(.95,color='#555555',ls='--',lw=1.2)
    axs[0].set(yticks=positions,yticklabels=labels,xlim=(.75,1.01),xlabel='Coverage with Wilson interval',title='Correctly specified stationary model')
    axs[0].invert_yaxis()
    p=metrics[(metrics.method=='Posterior mean')&(metrics.t==metrics.t.max())]
    upper=0.
    for regime,s in enumerate(noises):
        g=p[p.regime==regime].copy();g['bin']=pd.qcut(g.posterior_risk,5,labels=False,duplicates='drop')
        rows=[]
        for _,b in g.groupby('bin'):
            x=b.posterior_risk.mean();y=b.projection_loss.mean()
            ci=student_t.ppf(.975,len(b)-1)*b.projection_loss.std(ddof=1)/np.sqrt(len(b))
            rows.append((x,y,ci));upper=max(upper,x,y+ci)
        a=np.asarray(rows)
        axs[1].errorbar(a[:,0],a[:,1],yerr=a[:,2],fmt='o',ms=7,capsize=3,label=rf'$s={s:g}$')
    axs[1].plot([0,upper*1.08],[0,upper*1.08],'--',color='#777777',lw=1)
    axs[1].set(xlabel='Posterior response risk',ylabel='Observed mean squared loss',title='Risk calibration in five bins')
    axs[1].legend()
    counts=coverage.groupby('regime').size()
    fig.supxlabel(f'Coverage uses {int(counts.min())} predetermined trajectories per noise level. Risk bins use one terminal observation per trajectory.',fontsize=10)
    if path is not None:save(fig,path)
    return fig
