"""Figures for the native-complex passive beamforming study.

Plotting reads frozen scientific outputs. It neither resimulates observations
nor draws a new posterior sample. A two-coordinate projection is always labelled
as such; angular covariance scans are distinct from weight beampatterns.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .plotting import plt, save, DPI
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import t as student_t

BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
PURPLE = '#A55AA3'
GOLD = '#C89721'
GREY = '#68737A'
BLACK = '#222222'
PALE = '#E9EEF1'
COLORS = dict(mean=BLUE, mode=ORANGE, pca=ORANGE, isotropic=PURPLE,
              scalar=GOLD, loaded=GREEN, grouse='#805D45', oracle=GREY)
LABELS = dict(mean='Posterior mean', mode='Mode', pca='Weighted PCA',
              isotropic='Isotropized mean', scalar='Regularized PCA',
              loaded='Loaded covariance', grouse='GROUSE', oracle='Oracle')
CONDITIONS = dict(haar='Stationary complex line', ula_static='Stationary ULA',
                  ula_moving='Moving ULA')
COMPONENTS = ('information', 'decisions', 'tracking', 'process', 'inference', 'numerics')
FIGURE_CAPTIONS = {
    '01_setup': 'Eight half-wavelength-spaced sensors observe zero-mean complex snapshots. The hidden spatial line and its posterior are inferred from passive covariance information; the desired steering vector enters only the beamforming decision.',
    '02_information': 'One-snapshot Fisher information is isotropic in the 14 real native-complex tangent coordinates. Small score-covariance residuals are Monte Carlo fluctuations, not population anisotropy. Physical arrival-angle information is a separate pullback along the ULA steering curve, in inverse radians squared.',
    '03_tracking': 'The physical moving-ULA record is preassigned trial zero at s=0.2. The scan uses modal covariance, not the actual beamforming weights. Noise and truth-dependent drift budgets are evaluated at predeclared fixed times; drift is unavailable to the estimator. Only valid certificates are compared with actual modal error; the valid long stationary cohort has n=4.',
    '04_decisions': 'Each scatter point is an independent training stream at N=8. Posterior expected-power optimality does not imply a win at every true covariance. The complete horizon/noise map retains losses; open circles mark paired pointwise 95% intervals containing zero. Purple boxes mark exact one-snapshot isotropization ties. The final panel is the actual weight beampattern, not covariance scanning power.',
    '05_uncertainty': 'The preassigned stationary-ULA record at N=8 shows scalar central 50/95% power quantiles and, separately, the image of one common 95% state HPD region. The scalar inclusion experiment uses three fixed angular queries, averaged within each trajectory before uncertainty across independent trajectories. At an exact interferer query the truth s+q is a support endpoint and finite continuous-posterior equal-tail intervals exclude it. These fixed-state checks are distinct from prior-predictive Bayesian calibration.',
    '05_state_coverage': 'Fixed stationary states, known powers, n=8 and independent observation streams. Markers show the fraction of true states contained in the computed 95% HPD region or posterior ball, with pointwise 95% Wilson binomial intervals. The horizontal scale is the full 0 to 1 coverage range. Eight cells are displayed without a simultaneous or all-cells calibration claim. State-region numerical flags are separate from scalar-tail flags; unresolved state calculations are marked rather than excluded.',
    '06_prediction_and_risk': 'The cloud is only the two-coordinate projection Re/Im(e2^H P v), not a full-state credible region. The directional risk curve is derived from the same saved posterior mean M using v^H(M-M^2)v; these queries are not known training inputs. Binned risk checks use independent random-query trajectories, averaging queries within each trajectory, with pointwise Student intervals across independent trajectories.',
    '07_parameter_inference': 'The joint posterior of s=b and q=a-b and data-amount intervals use preassigned stationary trial zero. The last panel averages independent streams at N=32 and scores the next observation and next true covariance before assimilation. Prospective log density and output power are different losses. A filled joint point requires parameter and joint covariance-mean diagnostics; the full-state tail diagnostic is reported separately. Horizontal bars show only 1.96 times the recorded score Monte Carlo standard error of the mean, not variability across data sets or a convergence certificate for the full mixture density.',
    '08_process_and_cost': 'The native eight-sensor complex predictive moment law is checked against independent transition draws, including real and imaginary entries. Exact first-moment matching does not imply exact predictive density: the separate two-channel CP1 reference reports density total variation against inverse-moment residual. Timings cover native numerical operations, not end-to-end comparisons against tracking algorithms. The full report includes its nested sampling and moment work and should not be added to those rows.'
}


def _finish(fig, path=None):
    if not plt.rcParams['text.usetex'] or fig.dpi != DPI or plt.rcParams['savefig.dpi'] != DPI:
        raise AssertionError('Beamforming figures require LaTeX and 300 dpi')
    if path is not None:
        save(fig, Path(path))
    return fig


def _clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.14, linewidth=.6)


def _frame(data, name, required=()):
    frame = data[name] if isinstance(data, dict) else data
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    missing = set(required)-set(frame)
    if missing:
        raise ValueError(f'{name}: missing columns {sorted(missing)}')
    return frame


def _column(frame, *names):
    for name in names:
        if name in frame:
            return name
    raise ValueError(f'Expected one of these columns: {names}')


def _slice(frame, **selectors):
    out = frame
    for key, value in selectors.items():
        if key not in out:
            continue
        out = out[np.isclose(out[key].astype(float), value) if isinstance(value, (float, int))
                  else out[key].eq(value)]
    if out.empty:
        raise ValueError(f'No preassigned rows for {selectors}')
    return out


def _category(frame, column, *candidates):
    for candidate in candidates:
        if frame[column].astype(str).eq(candidate).any():
            return candidate
    raise ValueError(f'{column}: expected one of {candidates}')


def _note(ax, value, color=GREY, y=.02):
    ax.text(.03, y, value, transform=ax.transAxes, fontsize=9, color=color,
            ha='left', va='bottom')


def _angles(frame):
    """Convert the explicitly stored radian scan coordinate for display only."""
    out=frame.copy()
    if 'angle_deg' not in out:
        if 'angle' not in out:raise ValueError('No saved scan angle')
        out['angle_deg']=np.rad2deg(out.angle)
    for target,source in [('q025','lower'),('q975','upper'),('gain','response')]:
        if target not in out and source in out:out[target]=out[source]
    return out


def _ok(series):
    """CSV flags may be bool or strings when controls have missing entries."""
    return series.astype(str).str.lower().isin(['true','1','1.0'])


def load_unified(root, profile='paper', verify=True):
    """Load one explicit profile and verify its component manifests if requested."""
    root = Path(root).resolve()
    base = root/'results/beamforming_unified'/profile
    if profile not in ('quick', 'paper'):
        raise ValueError('Profile must be quick or paper')
    data = {'root': root, 'base': base, 'profile': profile}
    if verify:
        from .beamforming_validation import require_beamforming_gate, verify_component
        data['gate'] = require_beamforming_gate(root)
    for component in COMPONENTS:
        folder = base/component
        if not folder.exists():
            raise FileNotFoundError(f'Complete {profile}/{component} before plotting')
        frame = {'folder': folder}
        if verify:
            frame['metadata'] = verify_component(root, profile, component)
        for path in sorted(folder.glob('*.csv')):
            frame[path.stem] = pd.read_csv(path)
        for path in sorted(folder.glob('*.json')):
            frame[path.stem] = json.loads(path.read_text())
        for path in sorted(folder.glob('*.npz')):
            with np.load(path, allow_pickle=False) as record:
                frame[path.stem] = {key: record[key] for key in record.files}
        if component=='tracking':
            document=frame.get('manifest',frame.get('config',frame.get('preregistered',{})))
            cfg=document.get('config',document)
            index=int(cfg['test_trials'])
            path=folder/'trajectories'/f'physical_{index:05d}.npz'
            if not path.is_file():raise FileNotFoundError(f'Preassigned s=0.2 trajectory zero: {path}')
            with np.load(path,allow_pickle=False) as record:
                frame['trajectory']={key:record[key] for key in record.files}
        data[component] = frame
    return data


def story_at(story, t=8):
    """Select an explicitly preassigned saved report, without recomputation."""
    prefix=f't{int(t)}__'
    chosen={key[len(prefix):]:value for key,value in story.items() if key.startswith(prefix)}
    if not chosen:
        raise ValueError(f'No saved preassigned story at N={t}')
    return chosen


def setup(story, path=None, *, n=8, s=.2, q=1., desired_deg=-20., interferer_deg=20., t=8):
    """ULA geometry, observed complex snapshots, and the rank-one covariance."""
    story=story if 'U' in story else story_at(story,t)
    snapshots = np.asarray(story['U'])
    if snapshots.ndim != 2 or snapshots.shape[1] != n:
        raise ValueError('Setup needs snapshots as time-by-sensor complex rows')
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), dpi=DPI,
                             layout='constrained', gridspec_kw={'width_ratios':[1.65, 1., .85]})
    ax = axes[0]
    sensor_x = np.linspace(-2.65, 2.65, n)
    ax.plot(sensor_x, np.zeros(n), color=GREY, lw=1.2)
    ax.scatter(sensor_x, np.zeros(n), s=48, facecolors='white', edgecolors=BLACK,
               linewidths=1.4, zorder=4)
    for j, x in enumerate(sensor_x):
        ax.text(x, -.28, str(j), ha='center', va='top', fontsize=9)
    ax.plot([0, 0], [0, 2.45], ls=':', lw=1, color=GREY)
    for angle, color, label in [(interferer_deg, ORANGE, r'Interferer: hidden $h$'),
                                 (desired_deg, GREEN, r'Desired $a$: decision only')]:
        theta = np.deg2rad(angle)
        start = 2.3*np.array([np.sin(theta), np.cos(theta)])
        ax.add_patch(FancyArrowPatch(start, .3*start, arrowstyle='-|>',
            mutation_scale=14, lw=1.8, color=color))
        ax.text(start[0]+(.18 if angle>=0 else -.18), start[1]+.1,
            label, color=color, ha='left' if angle>=0 else 'right', va='bottom', fontsize=10)
        lo, hi = sorted([90-angle, 90])
        ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=lo, theta2=hi, color=color, lw=1.1))
    ax.annotate(r'$\lambda/2$', xy=(sensor_x[1], -.8), xytext=(sensor_x[0], -.8),
        arrowprops=dict(arrowstyle='<->', color=GREY), ha='center', color=GREY)
    boxes = [(-2.8, r'$u_{1:t}$', 'Observed'), (-.8, r'$q(P_t)$', 'Inferred'),
             (1.2, r'$w_t$', 'Decision')]
    for x, value, label in boxes:
        ax.add_patch(FancyBboxPatch((x, -2.2), 1.35, .66, boxstyle='round,pad=.08',
            linewidth=1, edgecolor=GREY, facecolor=PALE))
        ax.text(x+.675, -1.87, value, ha='center', va='center', fontsize=14)
        ax.text(x+.675, -2.52, label, ha='center', fontsize=10, color=GREY)
    for x in (-1.35, .65):
        ax.annotate('', xy=(x+.5, -1.87), xytext=(x, -1.87),
                    arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.4))
    ax.annotate(r'$a$', xy=(1.87, -1.45), xytext=(2.7, -.8), color=GREEN,
                arrowprops=dict(arrowstyle='->', color=GREEN), ha='center')
    ax.set(xlim=(-3.9, 3.9), ylim=(-2.8, 3.05), title='Measurements, state and decision')
    ax.axis('off')
    shown = min(16, len(snapshots))
    im = axes[1].imshow(abs(snapshots[:shown].T), aspect='auto', origin='lower', cmap='Greys',
                         extent=(.5, shown+.5, -.5, n-.5), interpolation='nearest')
    axes[1].set(xlabel='Snapshot', ylabel='Sensor', title=r'Complex snapshots: $|u_{tj}|$ shown')
    axes[1].set_xticks([1, max(2, shown//2), shown]);axes[1].set_yticks([0,n-1])
    fig.colorbar(im, ax=axes[1], shrink=.73, label='Magnitude')
    eigenvalues = np.r_[s+q, np.full(n-1, s)]
    axes[2].bar(np.arange(1,n+1), eigenvalues, color=[ORANGE]+[GREY]*(n-1), width=.7)
    axes[2].axhline(s, color=GREY, ls=':', lw=1)
    axes[2].text(1.45, s+q, r'$s+q$', va='center', fontsize=11)
    axes[2].text(n-.3, s*1.12, r'$s$', ha='right', fontsize=11)
    axes[2].set(xlabel='Eigenvalue index', ylabel='Covariance eigenvalue',
                title='Model covariance spectrum', xticks=[1,n], ylim=(0,1.22*(s+q)))
    _clean(axes[2])
    return _finish(fig,path)


def information(data, path=None, *, n=8, s=.2, state='ULA 30 deg'):
    """Ambient isotropy and physical-angle information are separate panels."""
    f = _frame(data, 'information_matrix', ('n','s','state','i','j','analytic','empirical'))
    g = _slice(f,n=n,s=s,state=state)
    dim = int(max(g.i.max(),g.j.max()))+1
    expected = g.pivot(index='i',columns='j',values='analytic').reindex(index=range(dim),columns=range(dim)).to_numpy()
    measured = g.pivot(index='i',columns='j',values='empirical').reindex(index=range(dim),columns=range(dim)).to_numpy()
    scale = float(np.trace(expected)/dim)
    residual = (measured-expected)/scale
    fig, axes = plt.subplots(1,3,figsize=(14,4.7),dpi=DPI,layout='constrained')
    limit=max(.01,float(np.nanmax(abs(residual))))
    im=axes[0].imshow(residual,cmap='RdBu',vmin=-limit,vmax=limit,interpolation='nearest',origin='lower')
    axes[0].set(xlabel='Tangent coordinate',ylabel='Tangent coordinate',
                title='Score-covariance residual',xticks=[0,n-2,n-1,dim-1],yticks=[0,n-2,n-1,dim-1])
    fig.colorbar(im,ax=axes[0],shrink=.76,label=r'$(I_{\rm MC}-I)/I_0$')
    x=np.arange(dim)
    axes[1].axhline(1,color=BLACK,ls='--',lw=1.2,label='Analytic information')
    axes[1].scatter(x,np.diag(measured)/scale,color=BLUE,s=32,label='Score covariance')
    if 'curvature' in g:
        curvature=g.pivot(index='i',columns='j',values='curvature').reindex(index=range(dim),columns=range(dim)).to_numpy()
        axes[1].scatter(x,np.diag(curvature)/scale,facecolors='none',edgecolors=ORANGE,marker='s',s=34,label='Likelihood curvature')
    axes[1].axvline(n-1.5,color=GREY,lw=.6,alpha=.6)
    axes[1].set(xlabel=r'Real coordinates: $\mathrm{Re}(Z)$, then $\mathrm{Im}(Z)$',
                ylabel=r'Information / $I_0$',title='Equal ambient directions',ylim=(0,1.2),
                xticks=[0,n-2,n-1,dim-1],xticklabels=['1',str(n-1),str(n),str(dim)])
    axes[1].legend(fontsize=9);_clean(axes[1])
    a=_frame(data,'angle_information',('theta_deg','n','s','analytic','finite_derivative'))
    a=_slice(a,n=n,s=s).sort_values('theta_deg')
    axes[2].plot(a.theta_deg,a.analytic,color=BLUE,lw=2,label='Analytic')
    sample=a.iloc[np.linspace(0,len(a)-1,min(9,len(a)),dtype=int)]
    axes[2].scatter(sample.theta_deg,sample.finite_derivative,marker='o',facecolors='none',edgecolors=ORANGE,s=40,label='Independent derivative')
    axes[2].set(xlabel='Arrival angle (degrees)',ylabel=r'Angle information $(\mathrm{rad}^{-2})$',
                title='The steering curve changes with angle')
    axes[2].legend(fontsize=9);_clean(axes[2])
    return _finish(fig,path)


def tracking(bounds, trajectory, path=None, *, s=.2, beta=.95):
    """Physical covariance scan, sufficient-condition budget and valid bound."""
    f=_frame(bounds,'bounds',('trial','t','condition','eta','drift','q','valid','bound','actual_error'))
    U=np.asarray(trajectory.get('mode_projectors',trajectory.get('selected_memory_mode_projectors')))
    if U.ndim==4:
        choices=np.asarray(trajectory['betas']);U=U[int(np.argmin(abs(choices-beta)))]
    times=np.asarray(trajectory['times']);n=U.shape[-1]
    angles=np.linspace(-70,70,141)
    steering=np.exp(1j*np.pi*np.outer(np.arange(n),np.sin(np.deg2rad(angles))))/np.sqrt(n)
    q=float(np.asarray(trajectory.get('q',1.)).item())
    response=s+q*np.einsum('ia,tij,ja->ta',steering.conj(),U,steering,optimize=True).real
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),dpi=DPI,layout='constrained',gridspec_kw={'width_ratios':[1.2,1,1]})
    im=axes[0].pcolormesh(times,angles,response.T,cmap='Blues',shading='nearest',vmin=s,vmax=s+q)
    theta=np.rad2deg(trajectory['theta_rad'])
    axes[0].plot(times,theta,color='white',lw=3)
    axes[0].plot(times,theta,color=BLACK,lw=1.2,label='True arrival')
    axes[0].set(xlabel='Snapshot',ylabel='Scan angle (degrees)',title='Modal covariance scan')
    axes[0].legend(fontsize=9);fig.colorbar(im,ax=axes[0],shrink=.75,label='Scanning power')
    last=f[f.t==f.groupby('condition').t.transform('max')]
    if 's' in last:last=last[np.isclose(last.s,s)]
    groups=['condition']+(['beta'] if 'beta' in last else [])
    labels=[]
    for row,(key,g) in enumerate(last.groupby(groups,sort=False)):
        key=key if isinstance(key,tuple) else (key,)
        noise=float((g.eta/g.q).mean());drift=float((g.drift/g.q).mean())
        axes[1].barh(row,noise,color=BLUE,height=.58,label='Noise' if row==0 else None)
        axes[1].barh(row,drift,left=noise,color=ORANGE,height=.58,label='Drift' if row==0 else None)
        label=('Stationary control' if str(key[0])=='Long stationary control' else str(key[0]).replace('_',' '))
        if 'n' in g:label+=rf', $n={int(g.n.iloc[0])}$'
        if len(key)>1:label+=rf', $\beta={key[1]:g}$'
        labels.append(label)
        if 'neff' in g:
            axes[1].text(noise+drift+.06,row,rf'$N_{{\rm eff}}={g.neff.mean():.0f}$',va='center',fontsize=8)
    axes[1].axvline(.5,color=BLACK,ls='--',lw=1.2,label=r'Required: total $<1/2$')
    axes[1].set(yticks=range(len(labels)),yticklabels=labels,
                xlabel=r'Evaluation budget / gap $q$',title='Noise and motion consume the gap')
    axes[1].set_xlim(left=0);axes[1].invert_yaxis();axes[1].tick_params(axis='y',labelsize=8)
    axes[1].legend(fontsize=8);_clean(axes[1])
    valid=last[_ok(last.valid)]
    if len(valid):
        ceiling=max(float(valid.bound.max()),float(valid.actual_error.max()))*1.08
        floor=max(1e-6,min(float(valid.actual_error[valid.actual_error>0].min()),float(valid.bound.min()))*.75)
        axes[2].scatter(valid.bound,valid.actual_error,color=BLUE,s=18,alpha=.45)
        axes[2].plot([floor,ceiling],[floor,ceiling],color=GREY,ls='--',lw=1)
        axes[2].set(xscale='log',yscale='log',xlim=(floor,ceiling),ylim=(floor,ceiling),aspect='equal')
    else:
        _note(axes[2],'No valid sufficient conditions in this cohort',y=.45)
    axes[2].set(xlabel='Certified modal bound',ylabel='Actual modal error',title='Fixed-time valid certificates')
    _note(axes[2],f'Invalid: {int((~_ok(last.valid)).sum())}/{len(last)}')
    if len(valid) and 'n' in valid and valid.n.nunique()==1:
        axes[2].text(.03,.96,rf'Valid cohort: $n={int(valid.n.iloc[0])}$',transform=axes[2].transAxes,
            va='top',fontsize=9,color=GREY)
    _clean(axes[2])
    return _finish(fig,path)


def _gain_matrix(metrics, condition, references=None):
    f=_slice(metrics,condition=condition)
    times=sorted(f.t.unique());noises=sorted(f.s.unique(),reverse=True)
    references=references or ['mode','isotropic','scalar','loaded'];values=[];uncertain=[];names=[]
    for reference in references:
        for noise in noises:
            row=[];mask=[]
            for time in times:
                g=f[(f.t==time)&np.isclose(f.s,noise)]
                pivot=g.pivot(index='trial',columns='method',values='true_power')
                delta=pivot[reference]-pivot['mean']
                finite=np.isfinite(delta).all()
                denominator=float(pivot[reference].mean())
                if not finite:
                    row.append(np.nan);mask.append(True);continue
                gain=float(delta.mean());half=float(student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta)))
                row.append(100*gain/denominator);mask.append(gain-half<=0<=gain+half)
            values.append(row);uncertain.append(mask)
            q=float(f.q.iloc[0])
            names.append(f"{LABELS[reference]}, {10*np.log10(q/noise):.1f} dB")
    return np.asarray(values),np.asarray(uncertain),names,times


def decisions(data, path=None, *, condition='ula_static', s=.2, t=8):
    """Conditional improvement, complete gain map, and actual weight pattern."""
    metrics=_frame(data,'metrics',('condition','s','q','trial','t','method','true_power','posterior_power'))
    selected=_slice(metrics,condition=condition,s=s,t=t)
    actual=selected.pivot(index='trial',columns='method',values='true_power')
    expected=selected.pivot(index='trial',columns='method',values='posterior_power')
    x=(expected.isotropic-expected['mean']).to_numpy();y=(actual.isotropic-actual['mean']).to_numpy()
    valid=np.isfinite(x)&np.isfinite(y)
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.2),dpi=DPI,layout='constrained',gridspec_kw={'width_ratios':[1,1.35,1]})
    axes[0].axhline(0,color=GREY,ls='--',lw=1)
    axes[0].scatter(x[valid],y[valid],color=BLUE,s=24,alpha=.52,edgecolors='white',linewidths=.3)
    axes[0].set(xlabel='Posterior expected power saved',ylabel='Power saved at true covariance',
                title='Conditional and realized gain')
    _note(axes[0],rf'$N={t}$; {int(valid.sum())} independent training streams')
    if not valid.all():_note(axes[0],f'Unresolved pairs: {int((~valid).sum())}',ORANGE,y=.09)
    _clean(axes[0])
    matrix,uncertain,names,times=_gain_matrix(metrics,condition)
    all_values=[]
    for name in metrics.condition.unique():
        try:all_values.append(_gain_matrix(metrics,name)[0])
        except (KeyError,ValueError):continue
    vmax=max(1.,float(np.nanmax(abs(np.concatenate([a.ravel() for a in all_values])))))
    cmap=LinearSegmentedColormap.from_list('beam_gain',[ORANGE,'#FAFAFA',BLUE]);cmap.set_bad('#E5E7E9')
    im=axes[1].imshow(np.ma.masked_invalid(matrix),cmap=cmap,norm=TwoSlopeNorm(0,-vmax,vmax),aspect='auto')
    for i in range(len(names)):
        for j in range(len(times)):
            v=matrix[i,j]
            if not np.isfinite(v):
                axes[1].text(j,i,'--',ha='center',va='center',fontsize=9);continue
            axes[1].text(j,i,'0' if abs(v)<1e-10 else f'{v:+.2g}',ha='center',va='center',fontsize=8,
                         color='white' if abs(v)>.65*vmax else BLACK)
            if uncertain[i,j]:axes[1].scatter(j+.3,i-.3,s=16,facecolors='none',edgecolors=GREY,linewidths=.8)
    axes[1].set(xticks=range(len(times)),xticklabels=times,yticks=range(len(names)),yticklabels=names,
                xlabel='Snapshots',title='Mean power saved versus each control')
    axes[1].tick_params(axis='y',labelsize=8)
    fig.colorbar(im,ax=axes[1],shrink=.79,label=r'Power saved (\%)')
    if 1 in times:
        j=times.index(1)
        for i,name in enumerate(names):
            if name.startswith(LABELS['isotropic']):
                axes[1].add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor=PURPLE,lw=1.4))
    patterns=_angles(_frame(data,'weight_beampattern',('condition','s','t','method')))
    p=_slice(patterns,condition=condition,s=s,t=t)
    for method in ['mean','mode','oracle']:
        g=p[p.method==method].sort_values('angle_deg')
        if len(g):axes[2].plot(g.angle_deg,10*np.log10(np.maximum(g.gain,1e-5)),color=COLORS[method],ls='--' if method=='oracle' else '-',lw=1.6,label=LABELS[method])
    axes[2].axvline(-20,color=GREEN,ls=':',lw=1)
    axes[2].axvline(20,color=ORANGE,ls=':',lw=1)
    axes[2].set(xlabel='Scan angle (degrees)',ylabel=r'$|w^Ha(\theta)|^2$ (dB)',
                title='Actual weight beampattern',ylim=(-40,7))
    axes[2].legend(fontsize=9);_clean(axes[2])
    return _finish(fig,path)


def complete_decision_comparison(data,path=None):
    """Supplement: every method, noise, horizon and declared generation law."""
    metrics=_frame(data,'metrics')
    conditions=['haar','ula_static','ula_moving']
    references=['mode','pca','isotropic','scalar','loaded','grouse','oracle']
    maps=[_gain_matrix(metrics,c,references) for c in conditions]
    vmax=max(1.,max(float(np.nanmax(abs(a[0]))) for a in maps))
    cmap=LinearSegmentedColormap.from_list('complete_gain',[ORANGE,'#FAFAFA',BLUE]);cmap.set_bad('#E5E7E9')
    fig,axes=plt.subplots(1,3,figsize=(18,8.8),dpi=DPI,layout='constrained',sharey=True)
    for ax,condition,(matrix,uncertain,names,times) in zip(axes,conditions,maps):
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,norm=TwoSlopeNorm(0,-vmax,vmax),aspect='auto')
        for i in range(len(names)):
            for j in range(len(times)):
                value=matrix[i,j]
                label='--' if not np.isfinite(value) else ('0' if abs(value)<1e-10 else f'{value:+.2g}')
                ax.text(j,i,label,ha='center',va='center',fontsize=8,
                    color='white' if np.isfinite(value) and abs(value)>.65*vmax else BLACK)
                if uncertain[i,j] and np.isfinite(value):
                    ax.scatter(j+.31,i-.3,s=11,facecolors='none',edgecolors=GREY,lw=.7)
        ax.set(xticks=range(len(times)),xticklabels=times,yticks=range(len(names)),yticklabels=names,
            xlabel='Snapshots',title=CONDITIONS[condition])
        ax.tick_params(axis='y',labelsize=9)
    fig.colorbar(im,ax=axes,shrink=.75,label=r'Mean power saved by posterior mean (\%)')
    return _finish(fig,path)


def uncertainty(data, path=None, *, condition='ula_static', s=.2, t=8):
    """Pointwise functional intervals, a state-region image, and coverage/width."""
    scans=_angles(_frame(data,'scans',('condition','s','t','mean','truth','q25','q75','hpd_lower','hpd_upper')))
    g=_slice(scans,condition=condition,s=s,t=t).sort_values('angle_deg')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),dpi=DPI,layout='constrained')
    axes[0].fill_between(g.angle_deg,g.q025,g.q975,color=BLUE,alpha=.15,label=r'Central $95\%$')
    axes[0].fill_between(g.angle_deg,g.q25,g.q75,color=BLUE,alpha=.3,label=r'Central $50\%$')
    for ax in axes[:2]:
        ax.plot(g.angle_deg,g['mean'],color=BLUE,lw=1.8,label='Posterior mean')
        ax.plot(g.angle_deg,g.truth,color=BLACK,lw=1.3,label='True covariance')
        ax.set(xlabel='Scan angle (degrees)',ylabel='Covariance scanning power')
        _clean(ax)
    axes[0].set_title('Pointwise power quantiles')
    axes[0].legend(fontsize=8)
    axes[1].fill_between(g.angle_deg,g.hpd_lower,g.hpd_upper,color=PURPLE,alpha=.18,label=r'Image of state $95\%$ HPD')
    axes[1].set_title('One common state region')
    axes[1].legend(fontsize=8)
    for ax,flag in zip(axes[:2],['scalar_quantiles_ok','envelope_ok']):
        if flag in g and not _ok(g[flag]).all():
            _note(ax,f'Unresolved numerical reports: {int((~_ok(g[flag])).sum())}',ORANGE)
    lo=min(float(g.q025.min()),float(g.hpd_lower.min()),float(g.truth.min()))
    hi=max(float(g.q975.max()),float(g.hpd_upper.max()),float(g.truth.max()))
    for ax in axes[:2]:ax.set_ylim(lo-.04*(hi-lo),hi+.08*(hi-lo))
    components=_frame(data,'power_components',('condition','s','t','trial','covered','width'))
    components=components.copy();components['numerical_ok']=_ok(components.numerical_ok)
    components['covered']=_ok(components.covered).astype(float)
    c=_slice(components,s=s,t=t).groupby(['condition','s','t','trial'],as_index=False).agg(
        scalar_width=('width','mean'),scalar_coverage=('covered','mean'),
        numerical_ok=('numerical_ok','all'))
    for condition,h in c.groupby('condition',sort=False):
        width_column=_column(h,'response_width','clean_width','scalar_width')
        coverage_column=_column(h,'response_coverage','clean_coverage','scalar_coverage')
        width=h[width_column].to_numpy(float);covered=h[coverage_column].to_numpy(float)
        finite=np.isfinite(width)&np.isfinite(covered)
        if not finite.any():continue
        x=float(width[finite].mean());y=float(covered[finite].mean())
        marker={'haar':'o','ula_static':'s','ula_moving':'^'}.get(condition,'o')
        half=float(student_t.ppf(.975,finite.sum()-1)*np.std(covered[finite],ddof=1)/np.sqrt(finite.sum())) if finite.sum()>1 else 0.
        axes[2].errorbar(x,y,yerr=half,fmt=marker,color=BLUE,ms=7,capsize=3,label=CONDITIONS.get(condition,condition))
        if condition=='ula_static':
            aligned=_slice(components,condition=condition,s=s,t=t)
            aligned=aligned[np.isclose(aligned.angle,np.deg2rad(20.))]
            if len(aligned):
                axes[2].annotate(f'Aligned query: {aligned.covered.mean():.2g}',xy=(x,y),
                    xytext=(12,12),textcoords='offset points',fontsize=9,color=GREY)
        unresolved=int((~_ok(h.numerical_ok)).sum())
        if unresolved:_note(axes[2],f'{unresolved} unresolved trajectories retained',ORANGE)
    axes[2].axhline(.95,color=GREY,ls='--',lw=1)
    axes[2].set(xlabel='Mean scalar interval width',ylabel='True-power inclusion rate',
                title='Three fixed query angles',ylim=(0,1.02))
    axes[2].legend(fontsize=8);_clean(axes[2])
    return _finish(fig,path)


def state_coverage(data,path=None,*,n=8,s=.2):
    """Supplement: all eight fixed-state/horizon/region coverage cells."""
    f=_slice(_frame(data,'coverage',('state','t','trial','n','s','ball_covered','hpd_covered')),n=n,s=s)
    groups=list(f.groupby(['state','t'],sort=True))
    fig,ax=plt.subplots(figsize=(10.2,5.5),dpi=DPI,layout='constrained')
    z=1.959963984540054
    for row,((state,time),g) in enumerate(groups):
        if g.trial.duplicated().any():raise ValueError('Coverage cells must contain one row per independent stream')
        for method,color,offset,marker,label in [('hpd',PURPLE,-.13,'o','State HPD'),
                ('ball',BLUE,.13,'s','Posterior ball')]:
            count=int(_ok(g[method+'_covered']).sum());total=len(g);rate=count/total
            center=(rate+z*z/(2*total))/(1+z*z/total)
            half=z/(1+z*z/total)*np.sqrt(rate*(1-rate)/total+z*z/(4*total*total))
            lower=center-half;upper=center+half
            y=row+offset
            unresolved='state_ok' in g and not _ok(g.state_ok).all()
            ax.errorbar(rate,y,xerr=np.array([[rate-lower],[upper-rate]]),fmt=marker,
                color=color,mfc='white' if unresolved else color,ms=6,capsize=3,
                label=label if row==0 else None,clip_on=False)
            ax.text(.04,y,f'{count}/{total} = {rate:.3f}'+(' (unresolved)' if unresolved else ''),
                va='center',ha='left',fontsize=10,color=color)
    ax.axvline(.95,color=GREY,ls='--',lw=1.2,label=r'Nominal $95\%$')
    labels=[('Coordinate line' if state=='coordinate line' else r'ULA $20^\circ$')+rf', $N={int(time)}$'
        for (state,time),g in groups]
    ax.set(xlim=(0,1),ylim=(len(groups)-.55,-.55),yticks=range(len(groups)),yticklabels=labels,
        xticks=[0,.25,.5,.75,.95,1],xlabel='State coverage',title=r'Fixed-state regions: $n=8$')
    ax.legend(loc='center',bbox_to_anchor=(.59,.5),fontsize=9)
    _clean(ax)
    return _finish(fig,path)


def prediction_and_risk(data, path=None, *, s=.2, t=8, condition='ula_static'):
    """A labelled projection of P v, directional risk and trajectory-level bins."""
    cloud=_frame(data,'projection_cloud',('x','y','mean_x','mean_y','mode_x','mode_y','true_x','true_y'))
    cloud=_slice(cloud,condition=condition,s=s,t=t)
    g=cloud.iloc[0]
    fig,axes=plt.subplots(1,3,figsize=(14.6,4.8),dpi=DPI,layout='constrained')
    step=max(1,len(cloud)//1024);draws=cloud.iloc[::step]
    axes[0].scatter(draws.x,draws.y,s=9,color=BLUE,alpha=.18,edgecolors='none')
    for x,y,color,marker,label in [(g.mean_x,g.mean_y,BLUE,'D','Mean'),(g.mode_x,g.mode_y,ORANGE,'o','Mode'),(g.true_x,g.true_y,BLACK,'+','Truth')]:
        axes[0].scatter(x,y,s=70 if marker!='+' else 115,color=color,marker=marker,
                        edgecolors='white' if marker!='+' else None,zorder=5,label=label)
    axes[0].set(xlabel=r'$\mathrm{Re}(e_2^H P v)$',ylabel=r'$\mathrm{Im}(e_2^H P v)$',
                title='Two coordinates of a spatial response',aspect='equal')
    axes[0].legend(fontsize=9);_clean(axes[0])
    scans=_angles(_frame(data,'scans'))
    scan=_slice(scans,condition=condition,s=s,t=t).sort_values('angle_deg')
    if 'projection_risk' not in scan:
        report=story_at(data['story'],t)
        M=np.asarray(report['M']);n=len(M)
        queries=np.exp(1j*np.pi*np.outer(np.arange(n),np.sin(np.deg2rad(scan.angle_deg))))/np.sqrt(n)
        scan=scan.copy()
        scan['projection_risk']=np.einsum('ia,ij,ja->a',queries.conj(),M-M@M,queries,optimize=True).real
    axes[1].plot(scan.angle_deg,scan.projection_risk,color=BLUE,lw=2)
    axes[1].set(xlabel='Query direction (degrees)',ylabel=r'$v^H(M-M^2)v$',title='Risk depends on direction')
    _clean(axes[1])
    risk=_frame(data,'risk',('condition','s','t','trial','posterior_risk','projection_loss'))
    risk=_slice(risk,s=s,t=t)
    upper=0.
    for name,h in risk.groupby('condition',sort=False):
        h=h.copy();h['bin']=pd.qcut(h.posterior_risk,5,labels=False,duplicates='drop')
        points=[]
        for _,part in h.groupby('bin'):
            actual=part.projection_loss.to_numpy(float);pred=part.posterior_risk.to_numpy(float)
            if not np.isfinite(actual).all() or not np.isfinite(pred).all():continue
            x=float(pred.mean());y=float(actual.mean());half=float(student_t.ppf(.975,len(part)-1)*actual.std(ddof=1)/np.sqrt(len(part))) if len(part)>1 else 0.
            points.append((x,y,half));upper=max(upper,x,y+half)
        if points:
            a=np.asarray(points);marker={'haar':'o','ula_static':'s','ula_moving':'^'}.get(name,'o')
            axes[2].errorbar(a[:,0],a[:,1],yerr=a[:,2],fmt=marker,color={'haar':BLUE,'ula_static':GREEN,'ula_moving':ORANGE}.get(name,GREY),ms=6,capsize=3,label=CONDITIONS.get(name,name))
    axes[2].plot([0,upper*1.08],[0,upper*1.08],color=GREY,ls='--',lw=1)
    axes[2].set(xlabel='Binned posterior vector risk',ylabel='Observed vector loss',title='Independent random-query risk check')
    axes[2].legend(fontsize=8);_clean(axes[2])
    return _finish(fig,path)


def parameter_inference(data, path=None, *, s=.2, N=32):
    """Joint unknown powers, data amount, and prospective prediction/decision."""
    draws=_frame(data,'posterior_draws',('law','prior','N','trial','a','b'))
    prior=_category(draws,'prior','Default','default')
    d=_slice(draws,law='stationary',prior=prior,N=N,trial=0,s_true=s)
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.8),dpi=DPI,layout='constrained')
    axes[0].hist2d(d.b,d.a-d.b,bins=35,cmap='Blues')
    axes[0].scatter(s,1.,color=BLACK,marker='+',s=100,linewidths=1.8,label='True powers')
    sx=min(float(d.b.min()),s);ex=max(float(d.b.max()),s)
    sy=min(float((d.a-d.b).min()),1.);ey=max(float((d.a-d.b).max()),1.)
    axes[0].set_xlim(sx-.04*(ex-sx),ex+.04*(ex-sx))
    axes[0].set_ylim(sy-.04*(ey-sy),ey+.04*(ey-sy))
    axes[0].set(xlabel=r'Sensor variance $s=b$',ylabel=r'Interference power $q=a-b$',
                title='Joint power posterior')
    axes[0].legend(fontsize=9);_clean(axes[0])
    metrics=_frame(data,'metrics',('law','prior','s_true','N','trial','method','prospective_log_score'))
    power=_column(metrics,'actual_output_power','actualoutputpower')
    m=_slice(metrics,law='stationary',s_true=s)
    joint_name=_category(m,'method','Joint inference','Joint')
    m=m[(~m.method.eq(joint_name)) | m.prior.eq(prior)]
    joint=m[m.method.eq(joint_name)]
    for name,color,truth in [('b',BLUE,s),('q',GREEN,1.)]:
        center=_column(joint,name+'_median',name+'_q50',name+'_mean')
        low=_column(joint,name+'_q025',name+'_025',name+'_lower')
        high=_column(joint,name+'_q975',name+'_975',name+'_upper')
        chosen=joint[joint.trial.eq(0)].sort_values('N')
        axes[1].fill_between(chosen.N,chosen[low]/truth,chosen[high]/truth,color=color,alpha=.15)
        axes[1].plot(chosen.N,chosen[center]/truth,'o-',color=color,ms=4,label='Sensor variance' if name=='b' else 'Interference power')
        if 'parameter_mixing_ok' in chosen:
            unresolved=chosen[~_ok(chosen.parameter_mixing_ok)]
            if len(unresolved):axes[1].scatter(unresolved.N,unresolved[center]/truth,s=28,
                facecolors='white',edgecolors=color,zorder=5)
    axes[1].axhline(1,color=BLACK,ls='--',lw=1)
    axes[1].set(xlabel='Past snapshots used',ylabel='Posterior power / true power',title='Learning from one training stream')
    axes[1].legend(fontsize=9);_clean(axes[1])
    fixed=m[m.N.eq(N)]
    color_map={'Joint':BLUE,'Joint inference':BLUE,'Known parameters':GREY,'Known powers':GREY,
               'Validation fixed':GREEN,'Validation-fixed powers':GREEN,'Recorded-prior EM':ORANGE}
    for method,h in fixed.groupby('method',sort=False):
        x=-float(h.prospective_log_score.mean());y=float(h[power].mean())
        if {'parameter_mixing_ok','covariance_mean_mixing_ok'}.issubset(h):
            unresolved=method==joint_name and not (_ok(h.parameter_mixing_ok)&_ok(h.covariance_mean_mixing_ok)).all()
        else:unresolved=method==joint_name and 'mixing_ok' in h and not _ok(h.mixing_ok).all()
        if 'score_chain_mcse' in h:
            mcse=h.score_chain_mcse.to_numpy(float)
            if np.isfinite(mcse).all():
                axes[2].errorbar(x,y,xerr=1.96*np.linalg.norm(mcse)/len(h),fmt='none',
                    color=color_map.get(method,PURPLE),lw=1,capsize=2)
        axes[2].scatter(x,y,color=color_map.get(method,PURPLE) if not unresolved else 'white',
            edgecolor=color_map.get(method,PURPLE),s=60,label=method+(' (unresolved)' if unresolved else ''))
    if 'mixing_ok' in joint:
        shown=joint[(joint.N==N)&(joint.trial==0)]
        if len(shown):
            if 'parameter_mixing_ok' in shown and not _ok(shown.parameter_mixing_ok).all():
                _note(axes[0],'Parameter mixing unresolved',ORANGE)
            elif 'parameter_mixing_ok' not in shown and not _ok(shown.mixing_ok).all():
                _note(axes[0],'Mixing diagnostics unresolved',ORANGE)
            if 'full_state_tail_ok' in shown and not _ok(shown.full_state_tail_ok).all():
                _note(axes[0],'Full-state tails unresolved',GREY,y=.09)
    if 'full_state_tail_ok' in joint and not _ok(joint[joint.N.eq(N)].full_state_tail_ok).all():
        axes[2].text(.98,.04,'Joint full-state tails unresolved',transform=axes[2].transAxes,
            ha='right',va='bottom',fontsize=9,color=GREY)
    axes[2].set(xlabel='Prospective negative log score (MC)',ylabel='Next-covariance output power',
                title='Prediction and decisions use different losses')
    axes[2].legend(fontsize=8);_clean(axes[2])
    return _finish(fig,path)


def moving_parameter_inference(data,path=None,*,s=.2):
    """Keep matched-process concentration separate from an effective ULA fit."""
    draws=_frame(data,'posterior_draws',('law','prior','trial','kappa','s_true'))
    prior=_category(draws,'prior','Default','default')
    chosen=_slice(draws,prior=prior,trial=0,s_true=s)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),dpi=DPI,layout='constrained')
    for ax,law,title in zip(axes,['matched_motion','ula_motion'],
            ['Matched stochastic motion','Physical ULA path']):
        h=chosen[chosen.law.eq(law)]
        if h.empty:raise ValueError(f'Missing preassigned {law} concentration draws')
        for chain,g in h.groupby('chain',sort=True):
            ax.hist(g.kappa,bins=25,histtype='step',density=True,lw=1.3,label=f'Chain {int(chain)+1}')
        if law=='matched_motion':ax.axvline(20,color=BLACK,ls='--',lw=1.2,label='True concentration')
        else:ax.text(.97,.55,r'No true $\kappa$ for this path',transform=ax.transAxes,
            ha='right',va='top',fontsize=9,color=GREY)
        if 'metrics' in data:
            m=_slice(data['metrics'],law=law,s_true=s,trial=0)
            m=m[m.method.eq('Joint inference')&m.prior.eq(prior)]
            flag='parameter_mixing_ok' if 'parameter_mixing_ok' in m else 'mixing_ok'
            if flag in m and not _ok(m[flag]).all():
                ax.text(.97,.68,'Chain diagnostics unresolved',transform=ax.transAxes,
                    ha='right',va='top',fontsize=9,color=ORANGE)
        ax.set(xlabel=r'Process concentration $\kappa$',ylabel='Posterior density',title=title)
        ax.legend(fontsize=8);_clean(ax)
    return _finish(fig,path)


def process_and_cost(data, path=None):
    """Moment accuracy, density approximation and separately measured costs."""
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),dpi=DPI,layout='constrained')
    transition=_frame(data,'transition_checks')
    if {'analytic_real','empirical_real','analytic_imag','empirical_imag'}.issubset(transition):
        g=_slice(transition,n=8,kappa=20.,case='predictive mixture')
        low=np.inf;high=-np.inf
        for part,color,marker in [('real',BLUE,'o'),('imag',ORANGE,'s')]:
            x=g['analytic_'+part].to_numpy(float);y=g['empirical_'+part].to_numpy(float)
            error=g['mcse_'+part].to_numpy(float)
            axes[0].errorbar(x,y,yerr=1.96*error,fmt=marker,color=color,ms=3,alpha=.6,
                elinewidth=.6,label='Real entries' if part=='real' else 'Imaginary entries')
            low=min(low,x.min(),y.min());high=max(high,x.max(),y.max())
        axes[0].plot([low,high],[low,high],color=GREY,ls='--',lw=1)
        axes[0].set(xlabel='Exact predictive moment entry',ylabel='Independent transition sample',
                    title=r'Native complex moment: $n=8$')
        axes[0].legend(fontsize=8)
    elif {'analytic','empirical'}.issubset(transition):
        x=transition.analytic.to_numpy(float);y=transition.empirical.to_numpy(float)
        axes[0].scatter(x,y,color=BLUE,s=22,alpha=.65)
        low=min(x.min(),y.min());high=max(x.max(),y.max())
        axes[0].plot([low,high],[low,high],color=GREY,ls='--',lw=1)
        axes[0].set(xlabel='Exact predictive moment',ylabel='Independent transition sample',
                    title='Check the complex moment law')
    else:
        x=_column(transition,'samples','draws');y=_column(transition,'mean_error','moment_error')
        for key,h in transition.groupby('n'):
            axes[0].loglog(h[x],h[y],'o-',label=rf'$n={int(key)}$')
        axes[0].set(xlabel='Independent transition draws',ylabel='Moment error',title='Check the complex moment law')
        axes[0].legend(fontsize=9)
    _clean(axes[0])
    reference=_frame(data,'reference_checks')
    resid=_column(reference,'inverse_residual','mean_fit_residual','moment_residual')
    tv=_column(reference,'density_tv','process_density_tv','projection_tv')
    ok=_ok(reference.get('reference_numerical_ok',pd.Series(True,index=reference.index)))
    axes[1].scatter(np.maximum(reference.loc[ok,resid],1e-16),reference.loc[ok,tv],color=BLUE,s=27,alpha=.7,label='Resolved reference')
    if (~ok).any():
        axes[1].scatter(np.maximum(reference.loc[~ok,resid],1e-16),reference.loc[~ok,tv],color=ORANGE,s=45,marker='x',label='Unresolved reference')
        _note(axes[1],f'Unresolved: {int((~ok).sum())}/{len(ok)}',ORANGE)
    axes[1].set_xscale('log')
    axes[1].set(xlabel='Inverse-moment residual',ylabel='Predictive density total variation',
                title='Two-channel reference only')
    axes[1].legend(fontsize=8);_clean(axes[1])
    timing=_frame(data,'timings',('operation','n','seconds'))
    operations=list(timing.operation.unique())
    dimensions=[n for n in sorted(timing.n.unique()) if n in (8,16)]
    all_times=[]
    for j,n in enumerate(dimensions):
        values=[float(timing[(timing.n==n)&(timing.operation==op)].seconds.median())*1e3 for op in operations]
        all_times.append(values)
        axes[2].scatter(values,np.arange(len(operations)),s=58,color=BLUE if j==0 else PURPLE,
            label=rf'$n={int(n)}$',zorder=3)
    for row in range(len(operations)):
        axes[2].plot([values[row] for values in all_times],[row]*len(dimensions),color=GREY,lw=1,zorder=1)
    axes[2].set_xscale('log')
    time_values=np.asarray(all_times)
    left=float(time_values.min())/1.35;right=float(time_values.max())*1.35
    ticks=[m*10.**k for k in range(int(np.floor(np.log10(left)))-1,int(np.ceil(np.log10(right)))+1)
        for m in (1,2,5) if left<=m*10.**k<=right]
    axes[2].set_xlim(left,right);axes[2].set_xticks(ticks,labels=[f'{x:g}' for x in ticks])
    operation_labels={'Iid 8192 draws':'8192 independent draws','Full posterior report':'Full posterior report',
        'Inverse mean fit':'Inverse moment fit','Native mean':'Posterior mean'}
    axes[2].set(yticks=range(len(operations)),yticklabels=[operation_labels.get(op,op) for op in operations],xlabel='Median measured wall time (ms)',title='Moment and report costs',ylim=(len(operations)-.5,-.5))
    axes[2].legend(fontsize=9);_clean(axes[2])
    return _finish(fig,path)
