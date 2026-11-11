"""Verified, read-only summaries of the original moving-interference cohort.

No simulation, parameter selection, inference or plotting is performed here.
The two motion conditions use the original recorded observations and settings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .final_validation import (
    HISTORICAL_SOURCE_PROOF, HISTORICAL_SOURCE_PROOF_SHA256,
    require_gate, verify_component,
)


CONDITIONS = ('smooth', 'abrupt')
METHODS = ('mean', 'mode', 'iso', 'great_window')
REFERENCES = ('mode', 'iso', 'great_window')
ORIGINAL = 'results/beamforming/data/paper/performance'
ORIGINAL_MANIFEST = ORIGINAL+'/manifest.json'
OUTPUT = 'results/beamforming_moving'
TIME_AVERAGE_DEFINITION = (
    'Arithmetic mean of true_ratio_db at the 32 reporting indices '
    't=8,16,...,256 within each simulation, followed by an across-simulation '
    'mean and Student 95% confidence interval. This is not the logarithm of '
    'an average power ratio and has no burn-in exclusion.'
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _digest(value):
    return value['sha256'] if isinstance(value, dict) else value


def _interval(values):
    values = np.asarray(values, float)
    _require(len(values)==256 and np.isfinite(values).all(),
             'Every reported interval requires all 256 finite simulations')
    mean = float(values.mean())
    half = float(student_t.ppf(.975, 255)*values.std(ddof=1)/16.)
    return dict(simulations=256, mean=mean, lower=mean-half, upper=mean+half)


def _artifact(meta, basename):
    matches = [name for name in meta['artifacts'] if Path(name).name==basename]
    _require(len(matches)==1, 'Expected one original saved artifact: '+basename)
    return matches[0]


def _signature(n, angle):
    return np.exp(1j*np.pi*np.arange(n)*np.sin(angle))/np.sqrt(n)


def _verified_original(root):
    """Require the original proof-bound producer and every original artifact."""
    require_gate(root)
    meta = verify_component(root, 'beamforming', 'paper', 'performance')
    proof_path = root/HISTORICAL_SOURCE_PROOF
    _require(sha(proof_path)==HISTORICAL_SOURCE_PROOF_SHA256,
             'The original producer proof changed')
    proof = json.loads(proof_path.read_text())
    _require(proof['component_manifests'].get(ORIGINAL_MANIFEST)
             ==sha(root/ORIGINAL_MANIFEST),
             'The moving cohort must use its exact original manifest')
    verified_sources = {}
    for name, digest in meta['sources'].items():
        digest = _digest(digest)
        path = root/name
        if not path.is_file() or sha(path)!=digest:
            archived = proof['source_files'].get(name)
            _require(archived is not None and archived['sha256']==digest,
                     'Missing proof for the original moving producer: '+name)
            path = root/archived['archive_path']
        _require(path.is_file() and sha(path)==digest,
                 'Original moving producer bytes changed: '+name)
        verified_sources[name] = dict(path=path.relative_to(root).as_posix(), sha256=digest)
    cfg = meta['config']
    required = dict(rank=1, moving_n=32, moving_T=256, sigma2=.2, tau2=1.,
                    overlap=.5, validation_episodes=64, test_episodes=256,
                    action_cadence=8, heldout_snapshots=256)
    for key, value in required.items():
        _require(cfg.get(key)==value, 'Unexpected original moving setting: '+key)
    _require(cfg['validation_seed']!=cfg['test_seed'],
             'Validation and test must have separate recorded seed channels')
    _require('Identity transport' in cfg['motion'],
             'The original moving experiment must use identity transport')
    return meta, proof, verified_sources


def _check_stories(stories, policies):
    """Check saved geometry and corrected states, without drawing new data."""
    checks = {}; phi0 = None; n = 32; total = 256; beta = .97
    eta = 1./(.2*(.2+1.))
    methods = ('mean','mode','isotropic','modal_loading','covariance_loading',
               'great_window','great_ew','grouse','ledoit_wolf','oracle')
    for condition in CONDITIONS:
        prefix = condition+'_n32_'
        angles = stories[prefix+'angles']
        _require(angles.shape==(total,), 'Saved moving angle horizon changed')
        start = float(angles[0])
        if phi0 is None:
            phi0 = start
        _require(start==phi0, 'Motion conditions use different initial directions')
        expected = (np.linspace(phi0, 3*phi0, total) if condition=='smooth'
                    else np.r_[np.full(128, phi0), np.full(128, 3*phi0)])
        np.testing.assert_array_equal(angles, expected)
        signatures = np.asarray([_signature(n, angle) for angle in expected])
        np.testing.assert_allclose(stories[prefix+'h'], signatures, rtol=1e-13, atol=1e-13)
        desired = _signature(n, 0.)
        np.testing.assert_allclose(stories[prefix+'a'], desired, rtol=0., atol=0.)
        np.testing.assert_allclose(abs(np.vdot(desired, signatures[0]))**2, .5,
                                   rtol=0., atol=2e-12)
        np.testing.assert_array_equal(stories[prefix+'sigma2'], np.full(total, .2))
        np.testing.assert_array_equal(stories[prefix+'tau2'], np.ones(total))
        np.testing.assert_allclose(stories[prefix+'time'], np.arange(1,total+1)*.01,
                                   rtol=0., atol=1e-15)
        U = stories[prefix+'U']
        _require(U.shape==(total,n), 'Saved moving measurement shape changed')
        maximum_parameter_error = 0.; maximum_weight_error = 0.
        for t in (32,128,256):
            key = prefix+f't{t}_'
            weights = beta**np.arange(t-1,-1,-1)
            A = eta*U[:t].T@(weights[:,None]*U[:t].conj())
            A -= np.trace(A)/n*np.eye(n)
            saved_A = stories[key+'A']
            np.testing.assert_allclose(A, saved_A, rtol=2e-12, atol=2e-11)
            maximum_parameter_error = max(maximum_parameter_error,
                float(np.max(np.abs(A-saved_A))))
            _, eigenvectors = np.linalg.eigh(saved_A)
            mode = np.outer(eigenvectors[:,-1], eigenvectors[:,-1].conj())
            np.testing.assert_allclose(mode, stories[key+'P'], rtol=1e-11, atol=1e-11)
            M = stories[key+'M']
            overlap = float(np.trace(M@mode).real)
            iso = overlap*mode+(1-overlap)/(n-1)*(np.eye(n)-mode)
            np.testing.assert_allclose(iso, stories[key+'M_iso'], rtol=1e-11, atol=1e-11)
            for method, matrix in [('mean',M),('mode',mode),('isotropic',iso)]:
                covariance = .2*np.eye(n)+matrix
                w = np.linalg.solve(covariance, desired)
                w /= np.vdot(desired,w)
                saved = stories[key+'weights'][methods.index(method)]
                np.testing.assert_allclose(w,saved,rtol=1e-11,atol=1e-11)
                maximum_weight_error = max(maximum_weight_error,
                    float(np.max(np.abs(w-saved))))
        checks[condition] = dict(saved_simulation=0, full_angle_schedule_exact=True,
            steering_vectors_match_angles=True, initial_desired_overlap=.5,
            uniform_initial_parameter=True, identity_transport_recursion_verified=True,
            shared_corrected_posterior_verified_at=[32,128,256],
            maximum_parameter_absolute_error=maximum_parameter_error,
            maximum_action_absolute_error=maximum_weight_error,
            selected_beta=float(policies[condition]['shared_beta']))
    return phi0, checks


def load_moving_evidence(root, config=None):
    """Return both saved motion conditions and independently computed summaries.

    ``config`` may declare the fixed settings to check; it cannot change them.
    Curves contain columns ``[t, mean, lower, upper]``. Paired differences are
    comparator minus posterior mean for the same simulation and measurement.
    """
    root = Path(root).resolve()
    meta, proof, sources = _verified_original(root)
    required_files = [_artifact(meta, name) for name in (
        'moving_checkpoints.csv.gz','selected.json','stories.npz','validation_curves.csv')]
    input_hashes = {ORIGINAL_MANIFEST:sha(root/ORIGINAL_MANIFEST),
                    HISTORICAL_SOURCE_PROOF:sha(root/HISTORICAL_SOURCE_PROOF)}
    input_hashes.update({name:_digest(meta['artifacts'][name]) for name in required_files})
    for record in sources.values():
        input_hashes[record['path']] = record['sha256']
    saved_policies = json.loads((root/required_files[1]).read_text())
    policies = {}
    validation = pd.read_csv(root/required_files[3])
    times = np.arange(8,257,8,dtype=int)
    selection_checks = {}
    for condition in CONDITIONS:
        matches = [p for p in saved_policies if p['condition']==condition
                   and p['n']==32 and p['stationary'] is False]
        _require(len(matches)==1, 'Expected one saved moving selection: '+condition)
        policy = matches[0]; policies[condition] = policy
        _require(policy['shared_beta']==.97, 'Saved Bingham beta differs from 0.97')
        np.testing.assert_array_equal(policy['times'], times)
        np.testing.assert_array_equal(policy['primary_times'], np.arange(32,257,8))
        _require(policy['selection_objective']==
            'Mean independent held-out measured output power; no oracle normalization',
            'The moving parameter-selection criterion changed')
        grid = validation[(validation.condition==condition)&(validation.n==32)
                          &(validation.stationary==False)]
        _require(grid.validation_episodes.eq(64).all(), 'Validation simulation budget changed')
        memories = grid[grid.family.eq('shared_mode_memory')]
        chosen = memories[memories.beta.eq(.97)]
        _require(len(chosen)==1 and np.isclose(chosen.loss.iloc[0],memories.loss.min(),rtol=1e-13),
                 'Saved beta does not match the recorded measured-power selection')
        great = policy['great_window']
        _require(great['window']==32 and great['inner_steps']==3
                 and great['stepsize']==.00625 and great['beta'] is None,
                 'Saved sliding-window GREAT settings changed')
        candidates = grid[grid.family.eq('great_window')]
        chosen = candidates[candidates.window.eq(great['window'])
                            &candidates.inner_steps.eq(great['inner_steps'])
                            &np.isclose(candidates.parameter,great['stepsize'],rtol=1e-13)]
        _require(len(chosen)==1 and np.isclose(chosen.loss.iloc[0],candidates.loss.min(),rtol=1e-13),
                 'Saved GREAT settings do not match the recorded validation selection')
        selection_checks[condition] = dict(validation_simulations=64,
            existing_beta_and_GREAT_selection_verified=True,
            validation_measurements=policy['primary_times'],
            selection_objective=policy['selection_objective'])
    with np.load(root/required_files[2],allow_pickle=False) as saved:
        stories = {name:saved[name] for name in saved.files}
    phi0, story_checks = _check_stories(stories, policies)
    declared = dict(n=32,T=256,beta=.97,sigma2=.2,tau2=1.,
        conditions=list(CONDITIONS),methods=list(METHODS),rank=1,
        validation_simulations=64,test_simulations=256,
        reporting_t=times.tolist(),dt=.01,phi0_rad=phi0,desired_overlap=.5,
        initial_parameter='A_0=0',transport='G_t=I_n',
        observation_model='Proper complex Gaussian marginal model',
        endpoint='true_ratio_db',time_average_definition=TIME_AVERAGE_DEFINITION)
    for name,value in (config or {}).items():
        _require(name in declared and value==declared[name],
                 'Requested configuration differs from the saved cohort: '+name)

    frame = pd.read_csv(root/required_files[0])
    frame = frame[frame.condition.isin(CONDITIONS)&frame.n.eq(32)
                  &frame.method.isin(('mean','mode','isotropic','great_window'))].copy()
    frame.rename(columns={'trial':'episode'},inplace=True)
    frame['method'] = frame.method.replace({'isotropic':'iso'})
    _require(not frame.duplicated(['condition','method','episode','t']).any(),
             'Duplicate moving simulation/checkpoint record')
    _require(len(frame)==2*4*32*256, 'Incomplete moving comparison')
    _require(np.isfinite(frame.t).all() and np.equal(frame.t,np.floor(frame.t)).all(),
             'Measurement indices must be stored integers')
    _require(frame.sigma2.eq(.2).all() and frame.tau2.eq(1.).all(),
             'The moving comparison requires fixed generating variances')
    bingham = frame[frame.method.isin(('mean','mode','iso'))]
    _require(bingham.beta.eq(.97).all() and bingham.same_parameter.astype(bool).all(),
             'Bingham comparisons must share their corrected posterior and beta')
    _require(frame.numerical_ok.astype(bool).all() and frame.mean_ok.astype(bool).all(),
             'Unresolved original moving records must not be dropped')
    _require(np.isfinite(frame.true_ratio_db).all(), 'Missing exact covariance output power')
    np.testing.assert_allclose(frame.true_ratio_db,
        10*np.log10(frame.true_power/frame.oracle_power),rtol=1e-12,atol=1e-12)
    great = frame[frame.method.eq('great_window')]
    _require(great.optimization_window.eq(32).all() and great.gradient_steps.eq(3).all()
             and great.gradient_stepsize.eq(.00625).all(),
             'Moving GREAT records do not use their saved selected settings')

    summary = []; paired_rows = []; paired_records = []; conditions = {}
    for condition in CONDITIONS:
        subset = frame[frame.condition.eq(condition)]
        curves = []
        for method in METHODS:
            points = []
            for t in times:
                values = subset[subset.method.eq(method)&subset.t.eq(t)].sort_values('episode')
                np.testing.assert_array_equal(values.episode,np.arange(256))
                stats = _interval(values.true_ratio_db)
                summary.append(dict(condition=condition,n=32,t=int(t),method=method,**stats))
                points.append((t,stats['mean'],stats['lower'],stats['upper']))
            curves.append(dict(method=method,points=np.asarray(points)))
        pivot = subset.pivot(index=['episode','t'],columns='method',values='true_ratio_db')
        for reference in REFERENCES:
            delta = pivot[reference]-pivot['mean']
            for (episode,t),value in delta.items():
                paired_records.append(dict(condition=condition,n=32,episode=int(episode),t=int(t),
                    reference=reference,difference_db=float(value)))
            for t in times:
                paired_rows.append(dict(condition=condition,n=32,t=int(t),reference=reference,
                    **_interval(delta.xs(t,level='t'))))
        conditions[condition] = dict(curves=curves,time=times.copy(),n=32,T=256,beta=.97,
            event=129 if condition=='abrupt' else None,
            angles=stories[condition+'_n32_angles'].copy(),
            selected_settings=policies[condition])

    simulation_means = frame.groupby(['condition','episode','method'],sort=True).true_ratio_db.agg(
        time_average_db='mean',reporting_points='size').reset_index()
    _require(simulation_means.reporting_points.eq(32).all(),
             'Time averages require exactly 32 reporting measurements per simulation')
    averages = []; average_pairs = []; average_differences = []
    for condition in CONDITIONS:
        selected = simulation_means[simulation_means.condition.eq(condition)]
        for method in METHODS:
            sample = selected[selected.method.eq(method)].sort_values('episode')
            np.testing.assert_array_equal(sample.episode,np.arange(256))
            averages.append(dict(condition=condition,method=method,**_interval(sample.time_average_db)))
        pivot = selected.pivot(index='episode',columns='method',values='time_average_db')
        for reference in REFERENCES:
            delta = pivot[reference]-pivot['mean']
            average_pairs.append(dict(condition=condition,reference=reference,**_interval(delta)))
            average_differences.extend(dict(condition=condition,episode=int(episode),
                reference=reference,difference_db=float(value)) for episode,value in delta.items())
    checks = dict(status='passed',original_sources_and_all_artifacts_verified=True,
        original_manifest_proof_bound=True,raw_rows_selected=len(frame),
        conditions=list(CONDITIONS),methods=list(METHODS),checkpoints_per_simulation=32,
        test_simulations_per_condition=256,validation_simulations_per_condition=64,
        exact_covariance_power_identity_checked=True,all_simulation_identifiers_present=True,
        all_native_means_resolved=True,selection=selection_checks,saved_geometry=story_checks,
        new_simulations=0,new_parameter_selections=0,new_posterior_fits=0)
    provenance = dict(original_manifest=ORIGINAL_MANIFEST,
        original_manifest_sha256=input_hashes[ORIGINAL_MANIFEST],
        source_proof=HISTORICAL_SOURCE_PROOF,
        source_proof_sha256=HISTORICAL_SOURCE_PROOF_SHA256,
        original_source_commit=proof['source_commit'],verified_source_files=sources,
        original_config=meta['config'],
        numerical_scope='Reuse of the original saved smooth and abrupt cohorts only.',
        interval_definition='Pointwise Student 95% interval across 256 independent test simulations.',
        paired_definition='Reference true_ratio_db minus posterior-mean true_ratio_db at the same simulation and measurement; positive favors the posterior mean.',
        time_average_definition=TIME_AVERAGE_DEFINITION,
        selection_definition='Original measured held-out power, selected using 64 disjoint validation simulations; no retuning or exact-covariance selection.')
    return dict(config=declared,conditions=conditions,checks=checks,provenance=provenance,
        input_hashes=input_hashes,source_hashes=meta['sources'],selected_settings=policies,
        checkpoint_losses=frame[['condition','n','episode','t','method','beta',
            'true_power','oracle_power','true_ratio_db','numerical_ok','mean_ok']].reset_index(drop=True),
        checkpoint_summary=pd.DataFrame(summary),paired_checkpoint_summary=pd.DataFrame(paired_rows),
        paired_checkpoint_differences=pd.DataFrame(paired_records),
        simulation_time_averages=simulation_means,time_average_summary=pd.DataFrame(averages),
        paired_time_average_summary=pd.DataFrame(average_pairs),
        paired_time_average_differences=pd.DataFrame(average_differences))
