"""TeX figures separating process modeling, posterior shape, and computation.

All intervals concern independent trajectories. Plotting an unresolved value
does not turn it into evidence for superiority; open markers retain that flag.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .plotting import plt,save, DPI
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLORS={'Process Bingham':'#0072B2','Scaled Bingham':'#009E73',
        'Recursive Gaussian':'#D55E00','Process Laplace':'#56B4E9',
        'Scaled Laplace':'#E69F00','Same-parameter Laplace':'#56B4E9',
        'Full-density reference':'#222222','Fixed forgetting diagnostic':'#777777',
        'Point mode':'#777777','Point mean':'#CC79A7'}
LABELS={'Process Bingham':'Process Bingham','Scaled Bingham':'Scaled Bingham',
        'Recursive Gaussian':'Recursive Gaussian','Process Laplace':'Same-A Laplace',
        'Scaled Laplace':'Scaled Laplace','Same-parameter Laplace':'Same-A Laplace',
        'Full-density reference':'Full-density reference'}
CONDITIONS={'plane':'Plane','sphere_large':r'$R=0.20\,\mathrm{m}$',
            'sphere_small':r'$R=0.05\,\mathrm{m}$','weak':'Weak inputs',
            'anisotropic':'Anisotropic resistance'}
MAIN=['Process Bingham','Scaled Bingham','Recursive Gaussian','Process Laplace']


def _finish(fig,path):
    assert plt.rcParams['text.usetex'] and fig.dpi==DPI
    if path is not None:save(fig,Path(path))
    return fig


def _terminal(frame):
    return frame[frame.t==frame.t.max()].copy() if len(frame) and 't' in frame else frame.copy()


def _ci(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if not len(values):return np.nan,np.nan
    half=float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.
    return float(values.mean()),half


def _ok(frame,fields=('interval_ok','interval_numerical_ok','numerical_ok')):
    for name in fields:
        if name in frame:return frame[name].fillna(False).astype(bool).to_numpy()
    return np.ones(len(frame),dtype=bool)


def _method_marker(method):
    return '^' if method=='Recursive Gaussian' else 'D' if 'Laplace' in method else 'o'


def _coverage_panel(ax,data,scale=1.,matched=False):
    keys=['kappa','method'] if matched else ['method']
    methods=['Process Bingham','Recursive Gaussian','Same-parameter Laplace','Full-density reference'] if matched else MAIN
    ax.axhline(.95,color='#777777',ls='--',lw=1)
    for key,g in data[data.method.isin(methods)].groupby(keys,sort=False):
        if matched:kappa,method=key
        else:method=key[0] if isinstance(key,tuple) else key;kappa=None
        width,xerr=_ci(g.clean_width/scale);coverage,yerr=_ci(g.clean_coverage)
        if not np.isfinite(width+coverage):continue
        ok=_ok(g).all();color=COLORS.get(method,'#777777')
        marker=('o' if kappa==4 else 's') if matched else _method_marker(method)
        ax.errorbar(width,coverage,xerr=xerr,yerr=yerr,fmt=marker,capsize=3,
                    color=color,ms=7,markerfacecolor=color if ok and method!='Full-density reference' else 'white')
    ax.set(ylabel='Clean component coverage',ylim=(-.02,1.03),xlabel=r'Mean interval width / $v_0$' if not matched else 'Mean interval width')
    ax.grid(alpha=.12)


def _score_panel(ax,data,scale=1.,matched=False):
    methods=['Process Bingham','Recursive Gaussian','Same-parameter Laplace','Full-density reference'] if matched else MAIN
    if matched:
        pairs=[(k,m) for k in sorted(data.kappa.unique()) for m in methods if ((data.kappa==k)&(data.method==m)).any()]
    else:pairs=[(None,m) for m in methods if (data.method==m).any()]
    positions=[];labels=[]
    for i,(kappa,method) in enumerate(pairs):
        g=data[data.method==method]
        if kappa is not None:g=g[g.kappa==kappa]
        value,half=_ci(g.interval_score/scale);ok=_ok(g).all()
        color=COLORS.get(method,'#777777');positions.append(i)
        labels.append(LABELS.get(method,method)+(rf', $\kappa={kappa:g}$' if kappa is not None else ''))
        if np.isfinite(value):
            ax.errorbar(value,i,xerr=half,fmt=_method_marker(method),capsize=3,color=color,
                markerfacecolor=color if ok and method!='Full-density reference' else 'white',ms=6)
        else:ax.text(.02,i,'unresolved',transform=ax.get_yaxis_transform(),fontsize=9)
    ax.set(yticks=positions,yticklabels=labels,xlabel=r'Mean interval score / $v_0$' if not matched else 'Mean interval score')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.12)


def coverage_score(physical,matched=None,path=None,condition='sphere_small'):
    """Primary mechanical endpoint and a separately labeled matched-process control."""
    data=_terminal(physical['metrics']);data=data[data.condition==condition]
    scale=float(physical['config'].get('v0',.02));rows=2 if matched is not None else 1
    fig,axs=plt.subplots(rows,2,figsize=(15,6.2*rows),dpi=DPI,layout='constrained',squeeze=False)
    _coverage_panel(axs[0,0],data,scale);_score_panel(axs[0,1],data,scale)
    axs[0,0].set_title('Mechanical contact: coverage and width')
    axs[0,1].set_title('Mechanical contact: interval score')
    if matched is not None:
        values=_terminal(matched['metrics'])
        _coverage_panel(axs[1,0],values,matched=True);_score_panel(axs[1,1],values,matched=True)
        axs[1,0].set_title('Matched process: coverage and width')
        axs[1,1].set_title('Matched process: interval score')
    handles=[Line2D([],[],color=COLORS[m],marker=_method_marker(m),ls='',label=LABELS.get(m,m)) for m in MAIN]
    if matched is not None:handles.append(Line2D([],[],color='#222222',marker='o',ls='',markerfacecolor='white',label='Full-density reference'))
    fig.legend(handles=handles,loc='outside upper center',ncol=3,fontsize=10)
    fig.supxlabel(r'Fixed terminal time. Bars are pointwise 95\% Student intervals across trajectory averages. '
                  'Queries and components are not independent trials. Open markers flag unresolved numerical checks; reference counts are separate.',fontsize=9)
    return _finish(fig,path)


def paired_effects_cost(physical,path=None,condition='sphere_small'):
    """Primary paired scores beside actual cumulative and per-report costs."""
    pairs=physical['paired'];pairs=pairs[pairs.condition==condition]
    costs=physical['costs'];costs=costs[costs.condition==condition]
    v0=float(physical['config'].get('v0',.02))
    fig,axs=plt.subplots(1,3,figsize=(17,5.8),dpi=DPI,layout='constrained')
    ax=axs[0];references=['Scaled Bingham','Recursive Gaussian','Process Laplace','Scaled Laplace']
    pairs=pairs.set_index('reference').reindex([m for m in references if m in pairs.reference.to_numpy()]).reset_index()
    ax.axvline(0,color='#777777',ls='--',lw=1)
    for i,r in enumerate(pairs.itertuples()):
        resolved=bool(r.resolved);color=COLORS.get(r.reference,'#777777') if resolved else '#777777'
        if np.isfinite(r.gain):ax.errorbar(r.gain/v0,i,xerr=[[max(0.,r.gain-r.low)/v0],[max(0.,r.high-r.gain)/v0]],
            fmt='o',capsize=4,color=color,markerfacecolor=color if resolved else 'white',ms=7)
        else:ax.text(.02,i,'unresolved',transform=ax.get_yaxis_transform(),fontsize=9)
    ax.set(yticks=range(len(pairs)),yticklabels=[LABELS.get(m,m) for m in pairs.reference],
           xlabel=r'Interval-score reduction / $v_0$',title='Process Bingham versus controls')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    methods=[m for m in ['Process Bingham','Scaled Bingham','Recursive Gaussian','Point mode','Point mean'] if (costs.method==m).any()]
    c=costs.set_index('method').reindex(methods)
    ax=axs[1];left=np.zeros(len(c))
    stages=[('prediction_seconds','Prediction','#0072B2'),('update_seconds','Correction','#009E73'),
            ('score_seconds','Observation score','#E69F00'),('report_seconds','Reports','#D55E00')]
    for name,label,color in stages:
        values=c[name].fillna(0.).to_numpy() if name in c else np.zeros(len(c))
        ax.barh(range(len(c)),values,left=left,color=color,alpha=.8,label=label);left+=values
    ax.set(yticks=range(len(c)),yticklabels=[LABELS.get(m,m) for m in methods],
           xlabel='Cumulative time per trajectory (s)',title='Corrections and three reports')
    ax.invert_yaxis();ax.legend(fontsize=8,loc='best');ax.grid(axis='x',alpha=.12)
    ax=axs[2];metrics=physical['metrics'];metrics=metrics[metrics.condition==condition]
    reported=['Process Bingham','Process Laplace','Scaled Bingham','Scaled Laplace','Recursive Gaussian']
    names=[];averages=[];halves=[]
    for method in reported:
        g=metrics[metrics.method==method]
        if not len(g):continue
        # First average reports within each independent trajectory.
        independent=g.groupby('trial').report_seconds.mean()*1000
        value,half=_ci(independent);names.append(method);averages.append(value);halves.append(half)
    ax.barh(range(len(names)),averages,color=[COLORS[n] for n in names],alpha=.7)
    ax.errorbar(averages,range(len(names)),xerr=halves,fmt='none',color='#444444',capsize=3)
    ax.axvline(50,color='#777777',ls='--',lw=1)
    ax.set(yticks=range(len(names)),yticklabels=[LABELS.get(m,m) for m in names],
           xlabel='Time per full report (ms)',title='Reporting cost');ax.invert_yaxis();ax.grid(axis='x',alpha=.12)
    fig.supxlabel(r'Left: reference score minus process score; paired 95\% intervals across independent trajectories. '
                  'Inverse time is nested in prediction; mean/region/interval clocks are nested in reports. '
                  'Dashed cost line is the 50 ms observation interval, not a real-time guarantee.',fontsize=9)
    return _finish(fig,path)


def directional_calibration(physical,path=None,condition='sphere_small'):
    """Show signed lag bias and dispersion separately, in a scoring-only frame."""
    data=_terminal(physical['directional']);data=data[data.condition==condition]
    methods=[m for m in ['Process Bingham','Scaled Bingham','Recursive Gaussian'] if (data.method==m).any()]
    fig,axs=plt.subplots(1,2,figsize=(14,6),dpi=DPI,layout='constrained')
    ax=axs[0];positions=[];labels=[];all_groups=[]
    for method in methods:
        for direction,g in data[data.method==method].groupby('direction',sort=False):
            all_groups.append((method,direction,g));labels.append(LABELS[method]+('\nMotion' if 'Cross' not in direction else '\nCross-motion'))
    for i,(method,direction,g) in enumerate(all_groups):
        errors=g.tangent_error.to_numpy()*1000;mean,half=_ci(errors)
        expected=float(np.sqrt(np.mean(g.posterior_sd**2)))*1000
        ax.plot([i,i],[-1.96*expected,1.96*expected],color=COLORS[method],lw=10,alpha=.16)
        jitter=np.linspace(-.12,.12,len(g));ax.scatter(i+jitter,errors,s=13,color=COLORS[method],alpha=.25,edgecolors='none')
        ax.errorbar(i,mean,yerr=half,fmt='o',color=COLORS[method],capsize=4,ms=6,zorder=5)
    ax.axhline(0,color='#777777',ls='--',lw=1)
    ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Signed tangent error (mrad)',title='Bias and predicted spread')
    ax.tick_params(axis='x',labelsize=9);ax.grid(axis='y',alpha=.12)
    ax=axs[1];ax.axvline(0,color='#777777',ls='--',lw=1);ax.axhline(1,color='#777777',ls='--',lw=1)
    for method,direction,g in all_groups:
        predicted=float(np.sqrt(np.mean(g.posterior_sd**2)))
        if predicted<=0 or not np.isfinite(predicted):continue
        bias=float(g.tangent_error.mean()/predicted);dispersion=float(g.tangent_error.std(ddof=1)/predicted)
        marker='D' if 'Cross' in direction else 'o'
        ax.scatter(bias,dispersion,s=75,marker=marker,color=COLORS[method],alpha=.85)
    ax.set(xlabel='Mean error / predicted SD',ylabel='Empirical SD / predicted SD',title='Bias is separate from dispersion')
    ax.grid(alpha=.12)
    handles=[Line2D([],[],color=COLORS[m],marker='o',ls='',label=LABELS[m]) for m in methods]
    handles.extend([Line2D([],[],color='#555555',marker='o',ls='',label='Motion direction'),
                    Line2D([],[],color='#555555',marker='D',ls='',label='Cross-motion direction')])
    fig.legend(handles=handles,loc='outside upper center',ncol=3,fontsize=10)
    fig.supxlabel(r'Geometry defines scoring axes only; filters receive no true normal. Dots are trajectories; bars are 95\% intervals for mean error. '
                  'Faint bands are predicted zero-centered 1.96 SD widths, not empirical confidence intervals.',fontsize=9)
    return _finish(fig,path)


def validation_scores(physical,path=None):
    tuning=physical['tuning'];curved=tuning[tuning.scope=='curved']
    fig,axs=plt.subplots(1,3,figsize=(16,5.2),dpi=DPI,layout='constrained')
    for method in ['Scaled Bingham','Process Bingham','Recursive Gaussian']:
        g=curved[curved.method==method]
        if not len(g):continue
        x='beta' if method=='Scaled Bingham' else 'process_rms'
        ax=axs[0] if method=='Scaled Bingham' else axs[1]
        hue='eta' if method=='Scaled Bingham' else None
        groups=g.groupby(hue,sort=True) if hue else [(None,g)]
        for value,h in groups:
            h=h.sort_values(x);label=rf'$\eta={value:g}$' if hue else LABELS[method]
            y='selection_value' if 'selection_value' in h else 'predictive_nll'
            ax.plot(h[x],h[y],'.-',label=label,color=None if hue else COLORS[method],lw=1.3)
            if 'selected' in h:
                s=h[h.selected.astype(bool)];ax.scatter(s[x],s[y],marker='*',s=100,color='#222222',zorder=5)
        # Candidate interval score is retained independently of the NLL choice.
        score_column=next((c for c in ['interval_score','validation_interval_score','mean_interval_score'] if c in g),None)
        nll_column='selection_value' if 'selection_value' in g else 'predictive_nll'
        if score_column:axs[2].scatter(g[nll_column],g[score_column],s=24,alpha=.6,color=COLORS[method],label=LABELS[method])
    axs[0].set(xlabel=r'Memory $\beta$',ylabel='Validation negative log score',title='Memory and likelihood scale')
    axs[1].set(xlabel='Process chordal RMS',ylabel='Validation negative log score',title='Process dispersion')
    axs[2].set(xlabel='Validation negative log score',ylabel='Validation clean interval score',title='Selection losses can disagree')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.12)
    fig.supxlabel('Only independent validation trajectories choose policies. Stars mark the frozen observation-score selections. '
                  'Terminal clean interval scores are retained as a different validation criterion.',fontsize=9)
    return _finish(fig,path)


def shape_concentration(physical,path=None):
    shapes=physical['shape_comparison'];directional=physical['directional']
    gaps=directional.groupby(['condition','trial','t','method'],sort=False).minimum_gap.first().reset_index()
    data=shapes.merge(gaps,on=['condition','trial','t','method'],how='left')
    fig,axs=plt.subplots(1,2,figsize=(13,5.4),dpi=DPI,layout='constrained')
    for method,g in data.groupby('method',sort=False):
        ok=g.full_mean_ok.astype(bool)&g.local_mean_ok.astype(bool)&g.full_interval_ok.astype(bool)&g.local_interval_ok.astype(bool)
        for mask,face in [(ok,None),(~ok,'none')]:
            h=g[mask];color=COLORS.get(method,'#777777')
            axs[0].scatter(h.minimum_gap,h.mean_map_difference,s=13,alpha=.45,color=color if face is None else None,facecolors=face,edgecolors=color if face else 'none',label=LABELS.get(method,method) if face is None else None)
            axs[1].scatter(h.minimum_gap,h.interval_endpoint_difference_over_v0,s=13,alpha=.45,color=color if face is None else None,facecolors=face,edgecolors=color if face else 'none')
    for ax in axs:ax.set_xscale('log');ax.set_yscale('log');ax.grid(alpha=.12);ax.set_xlabel('Smallest modal eigenvalue gap')
    axs[0].set(ylabel='Mean-map difference',title='Same parameter, different shape');axs[0].legend(fontsize=9)
    axs[1].set(ylabel=r'Max endpoint difference / $v_0$',title='Response-tail difference')
    fig.supxlabel('Full Bingham and modal-tangent Gaussian use the same parameter. Small gaps indicate broader or ambiguous laws. '
                  'Open points flag unresolved calculations; this does not compare different process policies.',fontsize=9)
    return _finish(fig,path)


def physical_stress(physical,path=None):
    metrics=_terminal(physical['metrics']);methods=['Process Bingham','Scaled Bingham','Recursive Gaussian','Process Laplace']
    conditions=[c for c in ['plane','sphere_large','sphere_small','weak','anisotropic'] if (metrics.condition==c).any()]
    fig,axs=plt.subplots(1,2,figsize=(14,6),dpi=DPI,layout='constrained')
    for method in methods:
        part=metrics[metrics.method==method];means=[];errors=[]
        for condition in conditions:
            value,half=_ci(part[part.condition==condition].interval_score);means.append(value);errors.append(half)
        axs[0].errorbar(range(len(conditions)),means,yerr=errors,fmt='o-',capsize=3,color=COLORS[method],label=LABELS.get(method,method))
        coverage=[];cover_err=[]
        for condition in conditions:
            value,half=_ci(part[part.condition==condition].clean_coverage);coverage.append(value);cover_err.append(half)
        axs[1].errorbar(range(len(conditions)),coverage,yerr=cover_err,fmt='o-',capsize=3,color=COLORS[method])
    for ax in axs:ax.set_xticks(range(len(conditions)),[CONDITIONS[c] for c in conditions],rotation=25);ax.grid(alpha=.12)
    axs[0].set(ylabel='Clean interval score',title='All retained conditions');axs[0].legend(fontsize=9)
    axs[1].axhline(.95,color='#777777',ls='--',lw=1);axs[1].set(ylabel='Clean component coverage',title='Calibration under physical mismatch',ylim=(-.02,1.03))
    fig.supxlabel(r'Fixed terminal time, pointwise 95\% intervals across trajectory means. The mechanical paths are not the isotropic process model. '
                  'Anisotropic resistance also violates the maintained observation law.',fontsize=9)
    return _finish(fig,path)
