"""Data-only normal-density diagnostics at a saved moving-contact snapshot.

The corrected parameter and physical trajectory are read from immutable paper
records. Existing moment and axial-moment routines evaluate that parameter and
fit its matched isotropic comparison law; no measurements are generated and no
tracker settings or saved estimator state are changed. Display nodes are never
used as integration weights. This module imports no plotting package.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .contact_posterior import normal_quadrature
from .contact_process_prediction import fast_moments, kernel_parameters
from .final_contact import mean_projector, isotropic_mean, mode_projector
from .final_validation import (
    HISTORICAL_SOURCE_PROOF, HISTORICAL_SOURCE_PROOF_SHA256,
    gate_path, require_gate, verify_component,
)
from .geometry import center
from .models import increment


MANIFEST = 'results/contact/data/paper/performance/manifest.json'
# Exact manifest bytes in dd399bacae3d7b37e31abb949b24740c1f07a4f3.
ORIGINAL_MANIFEST_SHA256 = '73d1be8e70beca6aaf76a50300c8f8653c0ce973c6fc97f9b8932ddd4dbd2044'
COLOR_MAP_STOPS = [(0., '#263a8b'), (.28, '#55209a'), (.53, '#b33695'),
                   (.72, '#ed78a4'), (.88, '#ffc85a'), (1., '#f4f52b')]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _native_mesh(B, polar=161, azimuth=321):
    """Existing adaptive unit-sphere mesh formula, in native coordinates."""
    values, basis = np.linalg.eigh(B)
    basis = basis[:, ::-1]
    gaps = values[-1]-values[-2::-1]
    concentration = max(1., .5*np.log1p(gaps[-1]))
    q = np.tanh(concentration*np.linspace(-1., 1., polar))/np.tanh(concentration)
    theta = np.arccos(np.clip(q[::-1], -1., 1.))
    base = np.linspace(0., 2*np.pi, azimuth)
    ratio = np.sqrt((1+gaps[0])/(1+gaps[1]))
    phi = np.unwrap(np.arctan2(ratio*np.sin(base), np.cos(base)))
    T, F = np.meshgrid(theta, phi, indexing='ij')
    local = np.stack((np.cos(T), np.sin(T)*np.cos(F), np.sin(T)*np.sin(F)), axis=-1)
    return local@basis.T


def _source_maps(root, meta):
    proof_path = root/HISTORICAL_SOURCE_PROOF
    _require(_sha(proof_path)==HISTORICAL_SOURCE_PROOF_SHA256,
             'The original source proof changed')
    proof = json.loads(proof_path.read_text())
    _require(proof['component_manifests'].get(MANIFEST,ORIGINAL_MANIFEST_SHA256)==_sha(root/MANIFEST),
             'The moving snapshot requires the exact original contact manifest')
    original = {}
    for name, record in meta['sources'].items():
        digest = record['sha256'] if isinstance(record, dict) else record
        if (root/name).is_file() and _sha(root/name)==digest:
            original[name] = digest
        else:
            saved = proof['source_files'].get(name)
            _require(saved is not None and saved['sha256']==digest,
                     'Missing original contact source proof: '+name)
            _require(_sha(root/saved['archive_path'])==digest,
                     'Changed original contact source archive: '+name)
            original[saved['archive_path']] = digest
    current = {name:_sha(root/name) for name in [
        'lib/contact_moving_snapshot.py', 'lib/final_contact.py',
        'lib/contact_process_prediction.py', 'lib/contact_posterior.py',
        'lib/geometry.py', 'lib/models.py', 'lib/final_validation.py']}
    return original, current


def load_moving_sphere_snapshot(root, t=96):
    """Return normalized full/matched-isotropic laws at corrected index ``t``.

    The display mesh and both densities use the saved sensor coordinates. The
    normal-line parameter is B=-A_t^+ by complement duality. Each response
    variance, when reported, uses the mean of its own probability law.
    """
    root = Path(root).resolve()
    if isinstance(t, bool) or int(t)!=t:
        raise ValueError('Use an integer corrected measurement index')
    t = int(t)
    require_gate(root)
    meta = verify_component(root, 'contact', 'paper', 'performance')
    cfg = meta['config']
    _require(cfg['n']==3 and cfg['k']==2 and cfg['T']==192,
             'Expected the original real Gr(2,3) moving-contact experiment')
    _require(1<=t<=cfg['T'], 'Snapshot lies outside the saved trajectory')
    tolerance = float(cfg['moment_tolerance'])
    _require(tolerance==5e-8 and cfg['algebra_tolerance']==1e-10,
             'The original numerical tolerances changed')
    artifacts = meta['artifacts']
    selected = {}
    for basename in ['story.npz', 'selected.json']:
        matches = [name for name in artifacts if Path(name).name==basename]
        _require(len(matches)==1, 'Expected one original '+basename)
        selected[basename] = matches[0]
    with np.load(root/selected['story.npz'], allow_pickle=False) as saved:
        story = {name:saved[name] for name in saved.files if name.startswith('sphere_')}
    policies = json.loads((root/selected['selected.json']).read_text())
    policies = [policy for policy in policies if policy['condition']=='sphere']
    _require(len(policies)==1, 'Expected one original sphere selection')
    beta = float(policies[0]['beta'])
    _require(beta==.9, 'The selected moving-sphere forgetting factor changed')

    index = t-1
    A = np.asarray(story['sphere_A'][index], float).copy()
    _require(A.shape==(3,3) and np.isfinite(A).all() and np.allclose(A,A.T,atol=1e-12),
             'Saved corrected parameter is not finite and symmetric')
    normal_saved = np.asarray(story['sphere_normal'][index], float)
    world_normal = np.asarray(story['sphere_world_normal'][index], float)
    R = np.asarray(story['sphere_R'][index], float)
    G = np.asarray(story['sphere_G'][index], float)
    previous_R = np.eye(3) if index==0 else story['sphere_R'][index-1]
    np.testing.assert_allclose(R.T@R, np.eye(3), rtol=0., atol=1e-12)
    np.testing.assert_allclose(R.T@world_normal, normal_saved, rtol=0., atol=1e-12)
    np.testing.assert_allclose(G, R.T@previous_R, rtol=0., atol=1e-12)
    np.testing.assert_allclose(story['sphere_P'][index],
        np.eye(3)-np.outer(normal_saved,normal_saved), rtol=0., atol=1e-12)
    np.testing.assert_allclose(story['sphere_positions'][index]/cfg['radius'],
                               world_normal, rtol=0., atol=1e-12)
    for key in ['U','V','clean']:
        np.testing.assert_allclose(R.T@story['sphere_world_'+key][index],
                                   story['sphere_'+key][index], rtol=0., atol=1e-12)
    # Verification only: reconstruct the recorded parameter, without fitting
    # or modifying the saved state used below for every diagnostic.
    reconstructed = np.zeros((3,3))
    for j in range(t):
        Gj = story['sphere_G'][j]
        reconstructed = center(beta*Gj@reconstructed@Gj.T
            +increment(story['sphere_U'][j],cfg['sigma2'],v=story['sphere_V'][j]))
    parameter_error = float(np.linalg.norm(reconstructed-A,2))
    _require(parameter_error<=cfg['algebra_tolerance'],
             'Saved A_t does not include exactly the first t corrections')

    report_times = np.asarray(story['sphere_report_times'],int)
    report_indices = np.flatnonzero(report_times==t)
    _require(len(report_indices)==1, 'Choose an actual saved posterior reporting index')
    report_index = int(report_indices[0])
    plane_mean, diagnostics = mean_projector(A)
    _require(diagnostics['numerical_ok'], 'Full normal moments were not resolved')
    _require(diagnostics['mean_resolution_error']<tolerance
             and diagnostics['log_normalizer_resolution_error']<tolerance,
             'Full moments exceed the original tolerance')
    plane_mode, mode, _ = mode_projector(A)
    saved_mean = story['sphere_M'][report_index]
    saved_mode = story['sphere_Pmode'][report_index]
    saved_iso = story['sphere_isotropic'][report_index]
    np.testing.assert_allclose(plane_mean,saved_mean,rtol=0.,atol=tolerance)
    np.testing.assert_allclose(plane_mode,saved_mode,rtol=0.,atol=cfg['algebra_tolerance'])
    expected_iso, _, _ = isotropic_mean(plane_mean,plane_mode)
    np.testing.assert_allclose(expected_iso,saved_iso,rtol=0.,atol=tolerance)

    # Fix the eigenvector sign independently of its arbitrary eigensolver sign,
    # then choose the same local antipodal representative for the true line.
    mode = mode.copy()
    if mode[np.argmax(np.abs(mode))]<0:mode = -mode
    true_normal = normal_saved.copy()
    if float(mode@true_normal)<0:true_normal = -true_normal
    normal_mean = np.eye(3)-plane_mean
    mode_projector_normal = np.outer(mode,mode)
    target = float(mode@normal_mean@mode)
    _require(1/3<target<1., 'A unique concentrated normal mode is required')
    upper = max(2.,2/max(1-target,1e-12))
    while kernel_parameters(upper).a<target:upper *= 2
    kappa = float(brentq(lambda value:kernel_parameters(value).a-target,
                         0.,upper,xtol=1e-11))
    B = -A
    B_iso = center(kappa*mode_projector_normal)
    iso = fast_moments(-B_iso, checked=True)
    _require(iso.numerical_ok and iso.resolution_error<tolerance
             and iso.logz_resolution_error<tolerance,
             'Matched isotropic moments were not resolved')
    plane_mean_iso = iso.mean
    normal_mean_iso = np.eye(3)-plane_mean_iso
    logz_full = float(diagnostics['log_normalizer']-np.trace(A))
    logz_iso = float(iso.log_normalizer+np.trace(B_iso))
    overlap_error = abs(float(mode@normal_mean_iso@mode)-target)
    mean_match_error = float(np.linalg.norm(plane_mean_iso-expected_iso,2))
    _require(overlap_error<1e-8 and mean_match_error<tolerance,
             'The isotropic comparison failed its mode-dispersion matching')

    # Independent existing sphere quadrature checks normalization and mean;
    # the display mesh is not used for any integration or fitting.
    quadrature_errors = {}
    for label,parameter,mean,logz in [
        ('full',B,normal_mean,logz_full),('isotropic',B_iso,normal_mean_iso,logz_iso)]:
        reference = normal_quadrature(parameter,64,192)
        error_mean = float(np.linalg.norm(reference['mean']-mean,2))
        error_logz = float(abs(reference['log_normalizer']-logz))
        _require(error_mean<tolerance and error_logz<tolerance,
                 'Independent normal-quadrature check failed for '+label)
        quadrature_errors[label] = dict(mean_error=error_mean,log_normalizer_error=error_logz)

    eigenvalues, basis = np.linalg.eigh(B)
    eigenvalues = eigenvalues[::-1];basis = basis[:,::-1]
    curvature = 2*(eigenvalues[0]-eigenvalues[1:])
    curvature_iso = 2*(np.linalg.eigvalsh(B_iso)[-1]-np.linalg.eigvalsh(B_iso)[-2::-1])
    _require(np.min(curvature)>0., 'The saved normal mode is not unique')
    angle = float(np.rad2deg(np.arccos(np.clip(abs(mode@normal_saved),0.,1.))))
    nodes = _native_mesh(B)
    np.testing.assert_allclose(np.sum(nodes*nodes,axis=-1),1.,rtol=0.,atol=2e-15)
    log_density_full = np.einsum('...i,ij,...j->...',nodes,B,nodes)-logz_full
    log_density_iso = np.einsum('...i,ij,...j->...',nodes,B_iso,nodes)-logz_iso
    log_peak = max(float(np.max(log_density_full)),float(np.max(log_density_iso)))
    density_limits = (float(np.exp(log_peak-4*np.log(10))),float(np.exp(log_peak)))
    log_true_full = float(normal_saved@B@normal_saved-logz_full)
    log_true_iso = float(normal_saved@B_iso@normal_saved-logz_iso)
    v = story['sphere_V'][index].copy()
    risk_full = float(v@(plane_mean-plane_mean@plane_mean)@v)
    risk_iso = float(v@(plane_mean_iso-plane_mean_iso@plane_mean_iso)@v)

    original_sources,current_sources = _source_maps(root,meta)
    input_hashes = {MANIFEST:_sha(root/MANIFEST),
        selected['story.npz']:_sha(root/selected['story.npz']),
        selected['selected.json']:_sha(root/selected['selected.json']),
        HISTORICAL_SOURCE_PROOF:HISTORICAL_SOURCE_PROOF_SHA256,
        str(gate_path(root).relative_to(root)):_sha(gate_path(root)),**original_sources}
    checks = dict(status='passed',condition='sphere',simulation=int(cfg['story_episode']),t=t,
        corrected_after_measurement=t,report_index=report_index,beta=beta,
        mean_tolerance=tolerance,algebra_tolerance=cfg['algebra_tolerance'],
        parameter_reconstruction_error=parameter_error,
        saved_mean_error=float(np.linalg.norm(plane_mean-saved_mean,2)),
        saved_mode_error=float(np.linalg.norm(plane_mode-saved_mode,2)),
        saved_isotropic_mean_error=float(np.linalg.norm(plane_mean_iso-saved_iso,2)),
        isotropic_kappa=kappa,isotropic_mode_overlap_error=overlap_error,
        isotropic_mean_match_error=mean_match_error,
        expected_squared_chordal_distance_full=1-target,
        expected_squared_chordal_distance_isotropic=1-float(mode@normal_mean_iso@mode),
        curvature_eigenvalues=curvature.tolist(),isotropic_curvature_eigenvalues=curvature_iso.tolist(),
        curvature_ratio=float(curvature.max()/curvature.min()),
        antipodal_mode_true_angle_deg=angle,
        log_density_at_true_normal_full=log_true_full,
        log_density_at_true_normal_isotropic=log_true_iso,
        density_at_true_normal_full=float(np.exp(log_true_full)),
        density_at_true_normal_isotropic=float(np.exp(log_true_iso)),
        independent_normal_quadrature=quadrature_errors,
        full_mean_resolution_error=diagnostics['mean_resolution_error'],
        full_log_normalizer_resolution_error=diagnostics['log_normalizer_resolution_error'],
        isotropic_mean_resolution_error=float(iso.resolution_error),
        isotropic_log_normalizer_resolution_error=float(iso.logz_resolution_error),
        sensor_frame_normal_verified=True,measured_transport_verified=True,
        fixed_radius_sphere=True,display_rotation=np.eye(3).tolist(),
        matched_normal_mode_projector_error=float(np.linalg.norm(
            mode_projector_normal-np.outer(np.linalg.eigh(B_iso)[1][:,-1],np.linalg.eigh(B_iso)[1][:,-1]),2)),
        display_mesh_used_for_integration=False,new_measurements=0,new_tracker_runs=0,
        isotropic_comparison_law_fitted=True)
    provenance = dict(original_manifest=MANIFEST,original_manifest_sha256=input_hashes[MANIFEST],
        original_story=selected['story.npz'],original_story_sha256=input_hashes[selected['story.npz']],
        original_sources=original_sources,source_hashes=current_sources,
        snapshot='Original moving-sphere simulation0, fixed corrected measurement index '+str(t),
        coordinate_frame='Saved sensor coordinates; R_t maps sensor to world, G_t=R_t^T R_{t-1}',
        density='q_B([n])=exp(n^T B n-log Z_1(B)), relative to normalized invariant measure',
        complement_identity='B=-A_t^+ and log Z_1(-A)=log Z_2(A)-tr(A)',
        curvature='Negative Hessian of log q_B at its mode in orthonormal normal-line tangent coordinates: 2(lambda_max(B)-lambda_j(B)); no chart-volume Jacobian',
        sign_rule='Mode largest-magnitude coordinate positive; true normal sign chosen to have nonnegative inner product with mode',
        matching='Same normal mode and expected squared chordal distance; each response variance uses its own law mean',
        interpretation='Physical tangent drift is not supplied by the measured coordinate transport G_t; this snapshot makes no posterior coverage claim')
    return dict(t=t,condition='sphere',simulation=int(cfg['story_episode']),beta=beta,
        A=A,B=B,B_iso=B_iso,nodes=nodes,log_density_full=log_density_full,
        log_density_isotropic=log_density_iso,density_limits=density_limits,
        log_normalizer_full=logz_full,log_normalizer_isotropic=logz_iso,
        normal_mean=normal_mean,normal_mean_isotropic=normal_mean_iso,
        plane_mean=plane_mean,plane_mean_isotropic=plane_mean_iso,mode=mode,
        true_normal=true_normal,true_normal_native=normal_saved.copy(),
        world_normal=world_normal.copy(),R=R.copy(),G=G.copy(),
        curvature_eigenvalues=curvature,isotropic_curvature_eigenvalues=curvature_iso,
        input_vector=v,response_variance_full=risk_full,response_variance_isotropic=risk_iso,
        camera=dict(elev=25.,azim=-50.),color_map_stops=list(COLOR_MAP_STOPS),
        checks=checks,provenance=provenance,input_hashes=input_hashes,
        source_hashes=current_sources,original_source_hashes=original_sources)


def export_moving_sphere_snapshot(root,t=96):
    """Save this diagnostic only under the new contact-diagnostics directory."""
    root=Path(root).resolve();data=load_moving_sphere_snapshot(root,t)
    folder=root/'results/contact_prediction_diagnostics/data';folder.mkdir(parents=True,exist_ok=True)
    stem='moving_sphere_t'+str(int(t))
    array_path=folder/(stem+'.npz');metadata_path=folder/(stem+'.json')
    arrays={name:value for name,value in data.items() if isinstance(value,np.ndarray)}
    arrays.update(log_normalizer_full=np.array(data['log_normalizer_full']),
        log_normalizer_isotropic=np.array(data['log_normalizer_isotropic']),
        density_limits=np.asarray(data['density_limits']))
    np.savez_compressed(array_path,**arrays)
    record={key:data[key] for key in ('t','condition','simulation','beta','checks','provenance',
        'input_hashes','source_hashes','original_source_hashes','camera','color_map_stops')}
    record.update(status='completed',array_path=str(array_path.relative_to(root)),
        array_sha256=_sha(array_path),density_limits=list(data['density_limits']))
    metadata_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return dict(array_path=str(array_path.relative_to(root)),metadata_path=str(metadata_path.relative_to(root)),
                output_hashes={str(path.relative_to(root)):_sha(path) for path in (array_path,metadata_path)})
