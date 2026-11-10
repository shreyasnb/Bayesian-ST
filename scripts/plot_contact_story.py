"""Render the five compact figures used by the revised contact walkthrough.

This reads saved experiment records; it does not regenerate mechanics, choose
a visually favourable trajectory, tune a policy, or reopen held-out selection.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lib.contact_mechanics import condition_spec
from lib.contact_story import (contact_scene, tracking_scene, posterior_scene,
    response_comparison, paired_prediction_errors)
from lib.plotting import plt, DPI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
        default=ROOT/'results/contact_tracking/uncertainty')
    parser.add_argument('--output', type=Path,
        default=ROOT/'results/contact_tracking/figures')
    parser.add_argument('--mean-results', type=Path,
        default=ROOT/'results/contact_tracking/mean')
    args = parser.parse_args()
    results = args.results.resolve()
    status = json.loads((results/'status.json').read_text())
    if status['status'] != 'completed':
        raise RuntimeError('Finish the frozen experiment before rendering its figures')
    document = json.loads((results/'config.json').read_text())
    config = document['config']
    # All visual choices were fixed before held-out results were generated.
    condition, trial, index, query_index, component = 'sphere_small', 0, 179, 1, 0
    with np.load(results/'trajectories'/f'{condition}_trial{trial}.npz') as record:
        arrays = {key: record[key] for key in record.files}
    data = dict(positions=arrays['positions'], normals=arrays['normals'],
        free_velocities=arrays['V'], observations=arrays['U'],
        velocities=arrays['velocities'], times=arrays['time'],
        forces=config['d']*arrays['V'], resistance=config['d']*np.eye(3),
        condition=condition_spec(condition))
    query = config['v0']*np.eye(3)[query_index]
    with np.load(args.mean_results/'trial0_paths.npz') as record:
        identity_prior = record[f'{condition}__0__identity__on__prior']
    identity_normals = np.linalg.eigh(identity_prior)[1][:, :, 0]
    identity_forecasts = np.eye(3)-np.einsum('ti,tj->tij', identity_normals, identity_normals)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = [
        contact_scene(data, index, path=args.output/'01_mechanics'),
        tracking_scene(data, arrays['raw__mode'], arrays['raw__forecast_mode'],
            index, path=args.output/'02_tracking', baseline_forecasts=identity_forecasts),
        posterior_scene(data, index, arrays['temperature__A'][index], query,
            arrays['temperature__mean'][index]@query, path=args.output/'03_plane_to_response')]
    story = pd.read_csv(results/'story.csv')
    story = story[(story.condition == condition) & (story.trial == trial)
        & (story['query'] == query_index) & (story.component == component)]
    series = {}
    for method, label in [('process_motion', 'Bingham process prediction'),
                          ('gaussian_motion', 'Gaussian process prediction')]:
        group = story[story.method == method].sort_values('t')
        if not group.interval_ok.astype(bool).all():
            raise ArithmeticError('An illustrative interval is unresolved; review its record')
        series[label] = dict(times=(group.t.to_numpy()-1)*config['dt'],
            **{key: group[key].to_numpy() for key in ['truth', 'mean', 'lower', 'upper']})
    figures.append(response_comparison(series, component='x', center_on_truth=True,
        path=args.output/'04_uncertainty'))
    metrics = pd.read_csv(results/'metrics.csv')
    figures.append(paired_prediction_errors(metrics, 'raw',
        ['process_motion', 'gaussian_motion'], condition=condition, time=600,
        reference_value='regularized_mse',
        labels={'process_motion': 'Bingham prediction', 'gaussian_motion': 'Gaussian prediction'},
        path=args.output/'05_monte_carlo'))
    for figure in figures:
        plt.close(figure)
    (args.output/'design.json').write_text(json.dumps(dict(
        condition=condition, trial=trial, index=index, query_index=query_index,
        response_component=component, posterior_cloud_seed=47013,
        band_methods=['process_motion', 'gaussian_motion'],
        band_centered_on_truth_for_scoring=True,
        posterior_cloud_count=256, cloud_for_illustration_only=True,
        scalar_intervals='Checked continuous-CDF equal-tailed intervals',
        trajectory_chosen_before_test=True, dpi=DPI, text_usetex=True), indent=2))


if __name__ == '__main__':
    main()
