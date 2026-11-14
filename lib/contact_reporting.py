from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


METHODS = ('mean', 'mode', 'isotropic', 'mean_identity')


def summarize(metrics):
    records = []
    for (endpoint, time, method), group in metrics.groupby(['endpoint', 't', 'method'], sort=True):
        values = group.response_mse.to_numpy(float) * 1e6
        if not np.isfinite(values).all() or len(values) < 2:
            raise ValueError('At least two finite independent contact trials are required')
        mean = float(values.mean())
        half = float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
        records.append(dict(condition='sphere', endpoint=endpoint, t=int(time),
            method='iso' if method == 'isotropic' else method, simulations=len(values),
            mean=mean, lower=mean-half, upper=mean+half))
    summary = pd.DataFrame(records)
    means = metrics[metrics.endpoint.eq('episode_average')].pivot(
        index='trial', columns='method', values='response_mse') * 1e6
    paired = []
    for control in ('mode', 'isotropic', 'mean_identity'):
        values = means[control] - means['mean']
        half = float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
        paired.append(dict(control=control, simulations=len(values), reduction=float(values.mean()),
            lower=float(values.mean()-half), upper=float(values.mean()+half)))
    return summary, pd.DataFrame(paired)


def save_tables(metrics, destination, write_metrics=True):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary, paired = summarize(metrics)
    if write_metrics:
        metrics.to_csv(destination / 'metrics.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    for endpoint in ('checkpoint', 'episode_average'):
        summary[summary.endpoint.eq(endpoint)].to_csv(destination / (endpoint+'_summary.csv'), index=False)
    paired.to_csv(destination / 'paired.csv', index=False)
    return summary, paired
