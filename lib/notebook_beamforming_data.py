"""Read-only numerical payloads for the editable beamforming notebook.

No Matplotlib import, plotting, fitting, sampling, or simulation occurs here.
Each payload is selected from the existing hash-verified paper CSV/NPZ files.
The notebook contains the actual plotting code and all visual choices.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .final_validation import require_gate, verify_component


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _SavedStudy:
    def __init__(self, root):
        self.root = Path(root).resolve()
        require_gate(self.root)
        self.components = {}
        self.input_hashes = {}

    def meta(self, component):
        if component not in self.components:
            self.components[component] = verify_component(self.root, 'beamforming', 'paper', component)
            name = f'results/beamforming/data/paper/{component}/manifest.json'
            self.input_hashes[name] = _sha(self.root/name)
        return self.components[component]

    def path(self, component, *names):
        artifacts = self.meta(component)['artifacts']
        for name in names:
            candidates = [p for p in artifacts if Path(p).name in (name+'.csv', name+'.csv.gz', name+'.npz')]
            if len(candidates) == 1:
                relative = candidates[0]
                expected = artifacts[relative]
                digest = expected['sha256'] if isinstance(expected, dict) else expected
                if _sha(self.root/relative) != digest:
                    raise RuntimeError('Saved plot data changed: '+relative)
                self.input_hashes[relative] = digest
                return self.root/relative
        raise KeyError(f'No unique saved {component} table/array for {names}')

    def table(self, component, *names):
        return pd.read_csv(self.path(component, *names))

    def array(self, component, *names):
        with np.load(self.path(component, *names), allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}


def _normalize(frame):
    frame = frame.copy()
    for destination, candidates in {'episode':('trial',), 'n':('N',), 'sigma2':('s',),
            'condition':('scenario','regime'), 't':('time',)}.items():
        if destination not in frame:
            for source in candidates:
                if source in frame:
                    frame = frame.rename(columns={source:destination})
                    break
    if 'method' in frame:
        frame['method'] = frame.method.replace({'isotropic':'iso','modal_loading':'mode_loaded',
            'covariance_loading':'scm_loaded','ledoit_wolf':'lw'})
    return frame


def _primary(frame):
    frame = _normalize(frame)
    if 'split' in frame and frame.split.eq('test').any():
        frame = frame[frame.split.eq('test')]
    if 'control' in frame:
        frame = frame[frame.control.eq('mean')]
    return frame


def _column(frame, *names):
    for name in names:
        if name in frame:
            return name
    raise KeyError(names)


def _finite_column(frame, *names):
    for name in names:
        if name in frame and np.isfinite(pd.to_numeric(frame[name], errors='coerce')).any():
            return name
    raise KeyError(names)


def _interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    half = float(student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return mean, mean-half, mean+half


def _saved(record, *names):
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(names)


def _mesh(A, polar=161, azimuth=321):
    """The existing adaptive display mesh; not an integration rule."""
    A=np.asarray(A)
    A=(A+A.conj().T)/2
    values, U = np.linalg.eigh(A.real)
    U = U[:, ::-1]
    gaps = values[-1]-values[-2::-1]
    concentration = max(1., .5*np.log1p(gaps[-1]))
    q = np.tanh(concentration*np.linspace(-1., 1., polar))/np.tanh(concentration)
    theta = np.arccos(np.clip(q[::-1], -1., 1.))
    base = np.linspace(0., 2*np.pi, azimuth)
    ratio = np.sqrt((1+gaps[0])/(1+gaps[1]))
    phi = np.unwrap(np.arctan2(ratio*np.sin(base), np.cos(base)))
    T, F = np.meshgrid(theta, phi, indexing='ij')
    local = np.stack((np.cos(T),np.sin(T)*np.cos(F),np.sin(T)*np.sin(F)), axis=-1)
    return local @ U.T


def _traces(frame, endpoint, time):
    result = []
    for method in ('known','exact_known','known_forgetting','misspecified','learned'):
        group = frame[frame.method.eq(method)]
        if group.empty:
            continue
        points = []
        for t, g in group.groupby(time, sort=True):
            sample = g.groupby('episode')[endpoint].mean().dropna()
            if len(sample):
                points.append((t, *_interval(sample)))
        if points:
            result.append(dict(method=method, points=np.asarray(points)))
    return result


def load_beamforming_plot_data(root, name, page=1):
    """Load one selected plot's local arrays, leaving all visual settings local.

    The returned ``input_hashes`` records precisely the numerical files read.
    User changes to notebook figure dimensions, dpi, labels or colors do not
    invalidate these numerical inputs or require an inference rerun.
    """
    root = Path(root).resolve()
    study = _SavedStudy(root)
    manifest = json.loads((root/'results/beamforming/figure_manifest.json').read_text())
    note = next(p for p in manifest['figures'][name+'.pdf']['pages'] if p['page']==page)
    data = dict(name=name, page=page, figsize=tuple(note['figure_size_in']))
    if name == 'setup':
        r = study.array('performance','stories','story')
        U = _saved(r,'setup_U','U')
        patterns = r.get('beampatterns', r.get('setup_beampatterns'))
        if patterns is None:
            patterns = abs(_saved(r,'setup_weights').conj() @ _saved(r,'scan_signatures').T)**2
        data.update(U=U, n=U.shape[-1], scan=np.rad2deg(_saved(r,'scan_angles','scan')),
            patterns=patterns, methods=_saved(r,'beampattern_methods','setup_methods'),
            desired_deg=float(np.rad2deg(np.asarray(r.get('desired_angle',0.)).ravel()[0])),
            interfering_deg=float(np.rad2deg(_saved(r,'interference_angle','setup_interference_angle','setup_interferer_angle'))))
    elif name == 'information':
        folder = root/'results/implementation_audit/paper/information'
        meta = json.loads((folder/'manifest.json').read_text())
        path = folder/'fisher_matrices.csv'
        if _sha(path) != meta['tables']['fisher_matrices.csv']:
            raise RuntimeError('Historical Fisher table changed')
        for p in (folder/'manifest.json',path):study.input_hashes[str(p.relative_to(root))]=_sha(p)
        frame = pd.read_csv(path).rename(columns={'s':'sigma2'})
        frame = frame[frame.complex_data.astype(bool)&frame.channel.eq('marginal')]
        eigenvalues = []
        for _, g in frame.groupby([k for k in ('case','channel','sigma2','L') if k in frame],sort=False):
            eigenvalues.append((np.linalg.eigvalsh(g.pivot(index='i',columns='j',values='expected').to_numpy()),
                np.linalg.eigvalsh(g.pivot(index='i',columns='j',values='score_covariance').to_numpy())))
        design = study.table('theory','information_loss')
        if 'complex_data' in design:design=design[design.complex_data.astype(bool)]
        ids=[k for k in ('sigma2','L','expected_conditional','expected_marginal') if k in design]
        if ids:design=design.drop_duplicates(ids)
        xname=_column(design,'snr','signal_to_noise','sigma2','L')
        curves=[]
        for key in ('conditional','marginal'):
            yname=_column(design,'expected_'+key,key,key+'_information',key+'_coefficient')
            curves.append(dict(method=key,x=design[xname].to_numpy(),y=design[yname].to_numpy()))
        both=np.concatenate([np.r_[a,b] for a,b in eigenvalues])
        low,high=both.min(),both.max()
        padding=max(.06*(high-low),.01*max(abs(low),abs(high)),1e-10)
        data.update(eigenvalues=eigenvalues,identity_limits=(max(low-padding,.8*low),high+padding),
            design=curves,design_xlabel=(r'$\tau^2/\sigma^2$' if xname=='snr' else r'Noise variance $\sigma^2$' if xname=='sigma2' else xname.replace('_',' ')))
    elif name == 'task_performance':
        frame=_primary(study.table('performance','stationary'))
        frame=frame[frame.condition.eq('static') & frame.n.isin([32,64,128])
                    & frame.t.eq(frame.n)]
        # The revised loss uses the saved exact covariance, not an independent
        # finite-snapshot estimate of output power. Deliberately no fallback.
        endpoint='true_ratio_db'
        if endpoint not in frame:
            raise KeyError('Stationary beamforming requires saved true_ratio_db')
        curves=[]
        for method in ('mean','mode','scm_loaded','great_window'):
            points=[]
            for n,g in frame[frame.method.eq(method)].groupby('n',sort=True):
                points.append((n,*_interval(g.groupby('episode')[endpoint].mean())))
            if points:curves.append(dict(method=method,points=np.asarray(points)))
        moving=_primary(study.table('performance','moving_episodes'))
        value=_column(moving,'true_ratio_db')
        groupcols=[k for k in ('condition','n','method','episode') if k in moving]
        moving=moving.groupby(groupcols,sort=False,dropna=False)[value].mean().reset_index()
        moving=moving[moving.condition.eq('smooth')&moving.n.eq(32)]
        base=moving[moving.method.eq('mean')].set_index('episode')[value]
        paired=[]
        for method in ('mode','iso','isotropic','great_window','great','great_ew'):
            other=moving[moving.method.eq(method)].set_index('episode')[value]
            common=base.index.intersection(other.index)
            if len(common):paired.append(dict(method=method,values=(other.loc[common]-base.loc[common]).to_numpy()))
        data.update(curves=curves,paired=paired,stationary_endpoint=endpoint,
            stationary_condition='static',stationary_dimensions=[32,64,128],
            stationary_selection='t=n; one exact-covariance loss per simulation')
    elif name == 'posterior_anisotropy':
        r=study.array('performance','stories','story')
        A=_saved(r,'section_A','setup_A');frame=_saved(r,'section_frame','setup_section_frame')
        B=(frame.conj().T @ A @ frame).real
        nodes=_mesh(B)
        logq=np.einsum('...i,ij,...j->...',nodes,B,nodes)-float(np.max(_saved(r,'section_eigenvalues','setup_eigenvalues')))
        peak=float(np.exp(np.max(logq)))
        stationary=_normalize(study.table('performance','stationary'))
        horizon=_column(stationary,'L','t','observations')
        stationary=stationary[stationary.n.eq(64)&stationary[horizon].eq(64)]
        value=_column(stationary,'true_ratio_db')
        mean=stationary[stationary.method.eq('mean')].set_index('episode')[value]
        iso=stationary[stationary.method.isin(['iso','isotropic'])].set_index('episode')[value]
        common=mean.index.intersection(iso.index);delta=np.sort((iso.loc[common]-mean.loc[common]).to_numpy())
        data.update(nodes=nodes,log_density=logq,density_limits=(peak/1e4,peak),
            mean_eigenvalues=_saved(r,'section_mean_eigenvalues','setup_mean_eigenvalues'),
            iso_eigenvalues=_saved(r,'section_iso_mean_eigenvalues','setup_iso_mean_eigenvalues'),
            differences=delta,cdf=np.arange(1,len(delta)+1)/len(delta))
    elif name == 'noise_adaptation':
        frame=_primary(study.table('prediction','adaptation'))
        frame=frame[frame.condition.eq('noise_change')]
        # t is the stored one-based measurement index. Physical-time columns
        # use different origins elsewhere and must not be converted back to t.
        time='t'
        if not np.isfinite(frame[time]).all() or not np.equal(frame[time],np.floor(frame[time])).all():
            raise ValueError('Variance reports require stored integer measurement indices')
        config=study.meta('prediction')['config']
        total=int(config['primary_beam_T'])
        if total!=256:
            raise ValueError('The declared noise-change figure requires T=256')
        measurement_index=np.arange(1,total+1,dtype=int)
        change_after=total//2
        schedules={
            'sigma2':np.where(measurement_index<=change_after,.2,.8),
            'tau2':np.ones(total),
        }
        variance={}
        for key in ('sigma2','tau2'):
            traces=[]
            for method in ('known','exact_known','known_forgetting','misspecified','learned'):
                g=frame[frame.method.eq(method)]
                if g.empty:continue
                grouped=g.groupby(time)[key+'_estimate']
                points=grouped.median()
                traces.append(dict(method=method,x=points.index.to_numpy(),y=points.to_numpy(),
                    low=grouped.quantile(.25).to_numpy(),high=grouped.quantile(.75).to_numpy()))
            recorded=frame.groupby(time)[key+'_true'].first()
            schedule=schedules[key]
            if not np.array_equal(recorded.to_numpy(),schedule[recorded.index.to_numpy(dtype=int)-1]):
                raise ValueError('Saved true variances disagree with the declared noise-change schedule')
            variance[key]=dict(traces=traces,x=measurement_index.copy(),truth=schedule,
                events=np.array([change_after+.5]) if key=='sigma2' else np.array([]),
                truth_step_where='mid',
                reporting_t=recorded.index.to_numpy(dtype=int),
                estimate_phase='Before receiving the measurement at the stored index')
        # Use the normalized mixture density before receiving u_t. Neither
        # the held-out power estimate nor the Gaussian score is a substitute.
        for endpoint in ('output_oracle_db','prequential_score'):
            if endpoint not in frame:
                raise KeyError('Noise adaptation requires saved '+endpoint)
        data.update(variance=variance,time_label=r'Measurement index $t$',
            reporting_t=np.sort(frame.t.unique()).astype(int),
            T=total,variance_block_length=int(config['primary_report_every']),
            task=_traces(frame,'output_oracle_db',time),
            score=_traces(frame,'prequential_score',time),
            task_endpoint='output_oracle_db',score_endpoint='prequential_score',
            beta_by_method={str(method):float(group.beta.iloc[0])
                            for method,group in frame.groupby('method',sort=False)})
    elif name == 'predictive_calibration':
        frame=_normalize(study.table('prediction','calibration_summary','prediction_calibration_summary'))
        if 'split' in frame and frame.split.eq('test').any():frame=frame[frame.split.eq('test')]
        if 'stage' in frame and frame.stage.eq('test').any():frame=frame[frame.stage.eq('test')]
        selection=note['selection'];regime=_column(frame,'regime','condition')
        target=_column(frame,'target','kind')
        frame=frame[frame[regime].eq(selection.get('condition',selection.get('regime')))&frame.t.eq(selection['t'])]
        panels=[]
        for kind in ('state_hpd','clean_response','noisy_observation'):
            subset=frame[frame[target].eq(kind)]
            if subset.empty and kind=='state_hpd':subset=frame[frame[target].eq('state_ball')]
            traces=[]
            for method in ('exact_known','known','known_forgetting','misspecified','learned'):
                g=subset[subset.method.eq(method)].sort_values('nominal')
                if len(g):traces.append(dict(method=method,x=g.nominal.to_numpy(),y=g.coverage.to_numpy(),
                    low=g.lower.to_numpy(),high=g.upper.to_numpy()))
            panels.append(dict(kind=kind,traces=traces))
        data.update(panels=panels,selection=selection)
    elif name == 'computation_time':
        summary=_normalize(study.table('computation','summary'))
        frame=summary[summary.case.eq('data_T64')]
        elapsed=_column(frame,'wall_median_ms','median_ms');curves=[]
        for method,interval,line in [('mean',1,'-'),('mode',1,'--'),('mean',16,'-.'),('mode',16,':')]:
            g=frame[frame.method.eq(method)&frame.report_every.eq(0)&frame.action_cadence.eq(interval)].sort_values('n')
            if len(g):curves.append(dict(method=method,interval=interval,line=line,n=g.n.to_numpy(),elapsed=g[elapsed].to_numpy()))
        n=32 if 32 in set(frame.n) else int(frame.n.max())
        group=frame[frame.n.eq(n)]
        if group.method.eq('adaptive_mean').any():
            group=group[group.method.isin(['mean','mean_report','adaptive_mean'])]
            group=group[group.method.ne('mean')|group.action_cadence.isin([8,32])]
        else:group=group[group.method.eq('mean')]
        group=group.sort_values(['method','action_cadence'])
        phases=[]
        for key in ('correction','action','report','fitting','overhead'):
            column=next((k for k in (key+'_median_ms',key+'_ms',key+'_seconds') if k in group),None)
            if column:phases.append(dict(key=key,values=group[column].to_numpy()*(1000 if column.endswith('_seconds') else 1.)))
        data.update(curves=curves,phase_rows=group[['method','action_cadence','em_iterations']].to_dict('records'),phases=phases)
    else:
        raise ValueError('Unknown editable beamforming figure: '+name)
    data['input_hashes']=dict(study.input_hashes)
    return data
