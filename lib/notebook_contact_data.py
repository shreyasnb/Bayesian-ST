"""Read-only arrays for editable contact notebook figures.

Only selection, deterministic display geometry and the original reporting
summaries are evaluated here. No observations, posterior, fit or benchmark is
produced. Imports complete before the notebook sets its visible rcParams; no
plotting routine is called by ``load_contact_plot_data``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import numpy as np

# Reuse unchanged numerical selection/statistic helpers. No renderer is called.
from .final_paper_visuals import (
    load_evidence, _normalize, _primary, _mean_interval, _episode_losses,
    _saved, _column, _finite_column)
from .distribution_geometry_visuals import bingham_sphere_mesh, real_bingham_log_density
from .final_information_visuals import contact_information_geometry, exact_contact_kl
from .final_validation import sha


@lru_cache(maxsize=12)
def _evidence(root, component):
    return load_evidence(Path(root), 'contact', 'paper', components=[component])


def contact_plot_input_hashes(root):
    """Exact saved numerical inputs, without binding user-editable plot code."""
    root = Path(root).resolve()
    hashes = {}
    for component in ('performance', 'prediction', 'computation'):
        evidence = _evidence(str(root), component)
        hashes.update(evidence.inputs)
        for name, digest in evidence.components[component]['artifacts'].items():
            hashes[name] = digest['sha256'] if isinstance(digest, dict) else digest
    return hashes


def _curve_rows(frame, endpoint, time, scale=1.):
    curves = []
    for method in ('known', 'exact_known', 'known_forgetting', 'misspecified', 'learned'):
        group = frame[frame.method.eq(method)]
        points = []
        for t, rows in group.groupby(time, sort=True):
            values = (rows.groupby('episode')[endpoint].mean()*scale).dropna()
            if len(values):
                mean, lower, upper = _mean_interval(values)
                points.append((t, mean, lower, upper))
        if points:
            curves.append(dict(method=method, points=np.asarray(points)))
    return curves


def load_contact_plot_data(root, name, page=1):
    """Return local saved arrays for one editable figure, without drawing it."""
    root = Path(root).resolve()
    if name in ('setup', 'information', 'task_performance', 'posterior_anisotropy'):
        evidence = _evidence(str(root), 'performance')
    elif name in ('noise_adaptation', 'predictive_calibration'):
        evidence = _evidence(str(root), 'prediction')
    elif name == 'computation_time':
        evidence = _evidence(str(root), 'computation')
    else:
        raise KeyError('Unknown contact notebook figure: '+name)
    if name == 'setup':
        saved = evidence.array('performance', 'story', 'stories')
        return dict(
            positions=_saved(saved, 'positions', 'setup_positions', 'sphere_positions').copy(),
            normals=_saved(saved, 'world_normal', 'normal', 'sphere_world_normal').copy(),
            V=_saved(saved, 'world_V', 'V', 'sphere_world_V').copy(),
            U=_saved(saved, 'world_U', 'U', 'sphere_world_U').copy(),
            clean=_saved(saved, 'clean', 'world_clean', 'sphere_world_clean', 'sphere_clean').copy(),
            radius=float(evidence.config('performance').get('radius', .05)))
    if name == 'information':
        saved = evidence.array('performance', 'anisotropy')
        geometry = contact_information_geometry(saved, evidence.config('performance')['anisotropy_sigma2'])
        coordinate = np.linspace(-.78, .78, 501)
        z1, z2 = np.meshgrid(coordinate, coordinate)
        grid = np.stack([z1, z2], axis=-1)
        angle = np.linspace(.25, 35., 201)
        delta = np.deg2rad(angle)
        first = np.column_stack([np.tan(delta), np.zeros_like(delta)])
        second = np.column_stack([np.zeros_like(delta), np.tan(delta)])
        marked = [np.array([np.tan(np.deg2rad(5.)), 0.]),
                  np.array([0., np.tan(np.deg2rad(30.))])]
        return dict(z1=z1, z2=z2, kl=exact_contact_kl(grid, geometry),
            angle_deg=angle, kl_first=exact_contact_kl(first, geometry),
            kl_second=exact_contact_kl(second, geometry),
            marked_points=np.asarray(marked), marked_angles=np.array([5., 30.]),
            marked_kl=np.array([float(exact_contact_kl(z, geometry)) for z in marked]),
            fisher_matrix=geometry['F'].copy())
    if name == 'task_performance':
        frame = _primary(evidence.frame('performance', 'metrics', 'episode_metrics'))
        frame = frame[frame.endpoint.eq('checkpoint') & frame.condition.eq('static_concentrated')]
        curves = []
        for method in ('mean', 'mode', 'regularized', 'great_window'):
            rows = []
            for t, group in frame[frame.method.eq(method)].groupby('t', sort=True):
                values = group.groupby('episode').response_mse.mean()*1e6
                mean, low, high = _mean_interval(values)
                rows.append((t, mean, low, high))
            if rows:curves.append(dict(method=method, points=np.asarray(rows)))
        moving = _episode_losses(evidence.frame('performance', 'episode_metrics', 'metrics'), 'response_mse')
        moving = moving[moving.condition.eq('sphere')]
        methods = [m for m in ('mean', 'mode', 'iso', 'isotropic', 'regularized') if moving.method.eq(m).any()]
        samples = {m:moving[moving.method.eq(m)].set_index('episode').response_mse for m in methods}
        common = samples[methods[0]].index
        for sample in samples.values():common = common.intersection(sample.index)
        absolute = []
        for method in methods:
            mean, low, high = _mean_interval(samples[method].loc[common].to_numpy()*1e6)
            absolute.append(dict(method=method, mean=mean, lower=low, upper=high))
        return dict(curves=curves, absolute=absolute, episodes=len(common))
    if name == 'posterior_anisotropy':
        saved = evidence.array('performance', 'anisotropy')
        A, Ai = _saved(saved, 'normal_A'), _saved(saved, 'iso_normal_A')
        nodes = [bingham_sphere_mesh(matrix, 161, 321).nodes for matrix in (A, Ai)]
        logq = [real_bingham_log_density(matrix, float(saved[key]), mesh)
                for matrix, key, mesh in zip((A, Ai), ('normal_logz', 'iso_normal_logz'), nodes)]
        query = _saved(saved, 'queries', 'unit_queries')
        selected = np.asarray(saved['query_family']) == 1
        norm2 = np.sum(abs(query[selected])**2, axis=-1)
        return dict(nodes=nodes, log_density=logq, mode=_saved(saved, 'mode_normal').copy(),
            theta=np.asarray(saved['query_angles'])[selected].copy(),
            risk_full=_saved(saved, 'risks_full', 'risk_full')[selected]/norm2,
            risk_iso=_saved(saved, 'risks_iso', 'risk_iso', 'risks_isotropic')[selected]/norm2)
    if name == 'noise_adaptation':
        frame = _normalize(evidence.frame('prediction', 'prediction_adaptation', 'adaptation'))
        if 'split' in frame:frame=frame[frame.split.eq('test')]
        if 'control' in frame:frame=frame[frame.control.eq('mean')]
        frame = frame[frame.condition.eq('noise_change')]
        time = _column(frame, 'time_s', 't')
        variance = []
        for method in ('known', 'exact_known', 'known_forgetting', 'misspecified', 'learned'):
            group = frame[frame.method.eq(method)]
            if group.empty:continue
            grouped = group.groupby(time).sigma2_estimate
            median = grouped.median()
            variance.append(dict(method=method, t=median.index.to_numpy(),
                median=median.to_numpy()*1e6, lower=grouped.quantile(.25).to_numpy()*1e6,
                upper=grouped.quantile(.75).to_numpy()*1e6))
        truth = frame.groupby(time).sigma2_true.first()
        changed = np.flatnonzero(abs(np.diff(truth.to_numpy())) > 1e-12)
        return dict(variance=variance, truth_t=truth.index.to_numpy(), truth=truth.to_numpy()*1e6,
            events=truth.index.to_numpy()[changed+1],
            future=_curve_rows(frame,_finite_column(frame,'future_task_loss','observed_response_mse'),time,1e6),
            scores=_curve_rows(frame,_finite_column(frame,'prequential_score','mixture_log_score'),time),
            time_label='Time (s)' if time=='time_s' else 'Number of observations')
    if name == 'predictive_calibration':
        if page not in (2, 6):raise ValueError('Contact notebook calibration uses pages 2 and 6')
        condition = 'prior_predictive' if page==2 else 'drifting'
        frame = _normalize(evidence.frame('prediction', 'prediction_calibration_summary',
            'calibration_summary', 'prediction_calibration', 'calibration'))
        if 'split' in frame and frame.split.eq('test').any():frame=frame[frame.split.eq('test')]
        if 'stage' in frame and frame.stage.eq('test').any():frame=frame[frame.stage.eq('test')]
        regime = _column(frame, 'regime', 'condition');target = _column(frame, 'target', 'kind')
        frame=frame[frame[regime].eq(condition) & frame.t.eq(64)]
        panels = []
        for kind in ('state_hpd', 'clean_response', 'noisy_observation'):
            subset=frame[frame[target].eq(kind)]
            if subset.empty and kind=='state_hpd':subset=frame[frame[target].eq('state_ball')]
            curves=[]
            for method in ('exact_known','known','known_forgetting','misspecified','learned'):
                group=subset[subset.method.eq(method)]
                values=[]
                for probability, rows in group.groupby('nominal',sort=True):
                    if 'coverage' in rows:
                        if len(rows)!=1:raise ValueError('Duplicated saved calibration summary')
                        row=rows.iloc[0];mean,low,high=row.coverage,row['lower'],row['upper']
                    else:
                        from scipy.stats import beta
                        binary=rows.groupby('episode').covered.mean()
                        if not np.isin(binary,[0.,1.]).all():raise ValueError('Expected binary inclusion per episode')
                        count=len(binary);successes=int(binary.sum());mean=successes/count
                        low=0. if successes==0 else beta.ppf(.025,successes,count-successes+1)
                        high=1. if successes==count else beta.ppf(.975,successes+1,count-successes)
                    values.append((probability,mean,low,high))
                if values:curves.append(dict(method=method, points=np.asarray(values)))
            panels.append(dict(target=kind,curves=curves))
        return dict(panels=panels, condition=condition, t=64)
    if name == 'computation_time':
        frame=_normalize(evidence.frame('computation','summary'))
        frame=frame[frame.case.eq('data_T64')]
        elapsed=_column(frame,'wall_median_ms','median_ms')
        scaling=[]
        for method,interval,style in [('mean',1,'-'),('mode',1,'--'),('mean',16,'-.'),('mode',16,':')]:
            group=frame[frame.method.eq(method)]
            if 'report_every' in group:group=group[group.report_every.eq(0)]
            group=group[group.action_cadence.eq(interval)].sort_values('n')
            if len(group):scaling.append(dict(method=method,interval=interval,style=style,
                n=group.n.to_numpy(),milliseconds=group[elapsed].to_numpy()))
        group=frame[frame.n.eq(3)]
        if group.method.eq('adaptive_mean').any():
            group=group[group.method.isin(['mean','mean_report','adaptive_mean'])]
            group=group[(group.method.ne('mean')) | group.action_cadence.isin([8,32])]
        else:group=group[group.method.eq('mean')]
        group=group.sort_values(['method','action_cadence'])
        phases=[]
        for key in ('correction','action','report','fitting','overhead'):
            column=next((c for c in (key+'_median_ms',key+'_ms',key+'_seconds') if c in group),None)
            if column is not None:
                phases.append(dict(key=key,values=group[column].to_numpy()*(1000 if column.endswith('_seconds') else 1.)))
        return dict(scaling=scaling, phases=phases, rows=group[['method','action_cadence','em_iterations']].to_dict('records'))
    raise AssertionError(name)
