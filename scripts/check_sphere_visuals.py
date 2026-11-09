"""Audit sphere displays without rerunning or changing scientific experiments.

The audit checks exact changes of orthonormal display coordinates, rather than
approximating a posterior or treating a display mesh as an integration rule.
The default reference is the committed study preceding the display revision.
An optional snapshot can instead be taken before editing and compared later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE_COMMIT = 'ded5499e6974a0078022434235c4d2a5c9b25424'
GEOMETRY = 'results/simulation_redesign/paper/geometry/'
GEOMETRY_FILES = ('manifest.json', 'matrices.npz', 'directional.csv', 'distance_cdf.csv')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def _git_bytes(commit, name):
    return subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT,
                          check=True, capture_output=True).stdout


def snapshot(commit=None):
    """Record scientific sources, all 24 cohorts, and constructed geometry."""
    from lib.final_validation import (
        EXAMPLES, STAGES, COMPONENTS, component_path, require_gate,
        verify_component, science_sources, gate_path)
    from lib.validation import CORE

    def read(name):
        return _git_bytes(commit, name) if commit else (ROOT / name).read_bytes()

    def digest(name):
        return sha_bytes(read(name))

    gate_name = str(gate_path(ROOT).relative_to(ROOT))
    gate = json.loads(read(gate_name)) if commit else require_gate(ROOT)
    manifests = {gate_name: digest(gate_name)}
    sources = dict(gate['sources'])
    artifacts = {}
    for example in EXAMPLES:
        for stage in STAGES:
            for component in COMPONENTS:
                name = str(component_path(ROOT, example, stage, component).relative_to(ROOT))
                meta = (json.loads(read(name)) if commit else
                        verify_component(ROOT, example, stage, component))
                manifests[name] = digest(name)
                sources.update(meta['sources'])
                artifacts.update(meta['artifacts'])
    # Include the full original core closure; the final scientific hash does
    # not itself enumerate every test and benchmark file in its sources map.
    if commit:
        names = subprocess.run(['git', 'ls-tree', '-r', '--name-only', commit,
                                'benchmarks', 'tests'], cwd=ROOT, check=True,
                               capture_output=True, text=True).stdout.splitlines()
        closure = list(CORE) + [name for name in names if name.endswith('.py')]
    else:
        closure = list(CORE) + list(science_sources(ROOT)) + [
            str(path.relative_to(ROOT)) for directory in ('benchmarks', 'tests')
            for path in (ROOT / directory).rglob('*.py')]
    for name in closure:
        sources[name] = digest(name)
    for name in GEOMETRY_FILES:
        artifacts[GEOMETRY + name] = digest(GEOMETRY + name)
    return dict(schema=1, baseline_commit=commit or BASELINE_COMMIT,
                core_hash=gate['core_hash'], final_science_hash=gate['final_science_hash'],
                artifacts=dict(sorted(artifacts.items())),
                manifests=dict(sorted(manifests.items())),
                sources=dict(sorted(sources.items())))


def check_preservation(baseline):
    from lib.final_validation import require_gate
    gate = require_gate(ROOT)
    assert gate['core_hash'] == baseline['core_hash'], 'Core scientific source hash changed'
    assert gate['final_science_hash'] == baseline['final_science_hash'], 'Scientific source hash changed'
    counts = {}
    for category in ('artifacts', 'manifests', 'sources'):
        mapping = baseline[category]
        for name, expected in mapping.items():
            if isinstance(expected, dict):
                expected = expected['sha256']
            assert sha(ROOT / name) == expected, f'Frozen {category} changed: {name}'
        counts[category] = len(mapping)
    # Re-read all current component manifests, not merely their old hashes.
    current = snapshot()
    for category in ('artifacts', 'manifests', 'sources'):
        assert current[category] == baseline[category], f'Frozen {category} closure changed'
    return dict(status='passed', baseline_commit=baseline['baseline_commit'],
                core_hash=gate['core_hash'], final_science_hash=gate['final_science_hash'],
                unchanged_file_counts=counts,
                unchanged_files={key: baseline[key] for key in counts},
                scope='All 24 verify/pilot/paper component artifacts and manifests, '
                      'the original general gate, complete scientific/core source closure, '
                      'and the saved constructed geometry; exact SHA-256 equality.')


def sphere_laws(example):
    """Explicit coverage of all saved laws used by sphere plots, plus history.

    Historical time steps are extra stress checks. They are not new trials,
    and are never used to select or retune the displayed experimental result.
    """
    folder = ROOT / 'results' / example / 'data' / 'paper'
    laws = []
    if example == 'beamforming':
        path = folder / 'performance/stories.npz'
        with np.load(path, allow_pickle=False) as saved:
            prefixes = [key[:-len('_section_frame')] for key in saved.files
                        if key.endswith('_section_frame')]
            for prefix in prefixes:
                A, frame = saved[prefix+'_A'], saved[prefix+'_section_frame']
                restricted = frame.conj().T @ A @ frame
                assert np.allclose(restricted, restricted.conj().T, atol=1e-10)
                assert np.allclose(frame.conj().T @ frame, np.eye(3), atol=1e-10)
                laws.append(dict(name=prefix, A=restricted.real, frame=frame,
                                 ambient_A=A, saved_file=str(path.relative_to(ROOT)),
                                 group='beam_'+prefix.split('_t')[0],
                                 interpretation='Real-coefficient great-sphere section of '
                                     'the original complex posterior; not a marginal.',
                                 logz=float(saved[prefix+'_log_normalizer'])))
    else:
        path = folder / 'performance/anisotropy.npz'
        with np.load(path, allow_pickle=False) as saved:
            for prefix in ('', 'iso_'):
                laws.append(dict(name='anisotropy_'+(prefix or 'full'),
                                 A=saved[prefix+'normal_A'],
                                 logz=float(saved[prefix+'normal_logz']),
                                 group='contact_anisotropy',
                                 saved_file=str(path.relative_to(ROOT)),
                                 interpretation='Normalized normal-line posterior on RP^2.'))
        path = folder / 'performance/story.npz'
        with np.load(path, allow_pickle=False) as saved:
            for condition in ('static_balanced', 'static_concentrated', 'plane', 'sphere'):
                for index, plane_A in enumerate(saved[condition+'_A']):
                    laws.append(dict(name=f'{condition}_t{index+1}', A=-plane_A,
                                     group='history_'+condition,
                                     saved_file=str(path.relative_to(ROOT)),
                                     interpretation='Saved normal parameter -A_plane; '
                                         'additional history stress check, not a new trial.'))
        path = folder / 'prediction/stories.npz'
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                if key.endswith('_A'):
                    prefix = key[:-1]
                    plane_A = saved[key]
                    laws.append(dict(name=prefix[:-1], A=-plane_A,
                                     logz=float(saved[prefix+'log_normalizer'])-float(np.trace(plane_A)),
                                     group=prefix.split('_t')[0],
                                     saved_file=str(path.relative_to(ROOT)),
                                     interpretation='Saved causal pre-correction normal belief; '
                                         'normalizer psi_plane-trace(A_plane).'))
        path = ROOT / GEOMETRY / 'matrices.npz'
        with np.load(path, allow_pickle=False) as saved:
            for index, (A, logz) in enumerate(zip(saved['A'][:2], saved['log_normalizer'][:2])):
                laws.append(dict(name=f'constructed_A{index+1}', A=A, logz=float(logz),
                                 group='constructed_geometry', saved_file=str(path.relative_to(ROOT)),
                                 interpretation='Saved constructed identity; not data-generated performance.'))
    return laws


def _quadratic(nodes, A):
    return np.einsum('...i,ij,...j->...', nodes, A, nodes)


def check_law(law, rotation=None):
    from lib.final_paper_visuals import display_sphere_geometry
    A = np.asarray(law['A'], float)
    assert A.shape == (3, 3) and np.allclose(A, A.T, atol=1e-10)
    display = display_sphere_geometry(A, rotation=rotation)
    R, B = display['rotation'], display['parameter']
    native, shown = display['native_nodes'], display['nodes']
    eigenvalues = np.linalg.eigvalsh(A)
    scale = max(1., float(np.linalg.norm(A)))
    tolerance = 2e-11 + 128*np.finfo(float).eps*scale
    assert np.allclose(R.T @ R, np.eye(3), atol=2e-12, rtol=0)
    assert np.isclose(np.linalg.det(R), 1., atol=2e-12, rtol=0)
    assert np.allclose(B, R @ A @ R.T, atol=tolerance, rtol=0)
    assert np.allclose(shown, native @ R.T, atol=2e-12, rtol=0)
    supplied = display_sphere_geometry(A, nodes=native, rotation=R)
    assert np.array_equal(supplied['native_nodes'], native)
    assert np.allclose(supplied['nodes'], shown, atol=2e-12, rtol=0)
    unit_error = float(np.max(abs(np.sum(shown*shown, axis=-1)-1.)))
    assert unit_error < 2e-12
    assert native.ndim == 3 and native.shape[-1] == 3
    assert native.shape[0] >= 17 and native.shape[1] >= 33
    assert np.allclose(native[:, 0], native[:, -1], atol=2e-12, rtol=0)
    mode = np.asarray(display['native_mode'])
    assert np.allclose(display['mode'], R @ mode, atol=2e-12, rtol=0)
    assert np.isclose(mode @ A @ mode, eigenvalues[-1], atol=tolerance, rtol=0)
    assert np.allclose(native[0], native[0, 0], atol=2e-12, rtol=0)
    assert np.allclose(native[-1], native[-1, 0], atol=2e-12, rtol=0)
    # Two poles, a closed seam and one full equatorial winding establish full
    # angular mesh coverage; no cap is cropped, enlarged or integrated here.
    assert np.isclose(abs(native[0, 0] @ native[-1, 0]), 1., atol=2e-12)
    assert np.isclose(native[0, 0] @ native[-1, 0], -1., atol=2e-12)
    pole = native[0, 0]
    pivot = np.eye(3)[np.argmin(abs(pole))]
    tangent = pivot-pole*(pole @ pivot)
    tangent /= np.linalg.norm(tangent)
    second = np.cross(pole, tangent)
    equator = native[native.shape[0]//2]
    angles = np.unwrap(np.arctan2(equator @ second, equator @ tangent))
    assert np.isclose(abs(angles[-1]-angles[0]), 2*np.pi, atol=2e-10)
    q_native, q_display = _quadratic(native, A), _quadratic(shown, B)
    density_error = float(np.max(abs(q_native-q_display)))
    assert density_error < tolerance
    antipodal_error = float(np.max(abs(_quadratic(-shown, B)-q_display)))
    assert antipodal_error < tolerance
    gaps = np.sort(eigenvalues[-1]-eigenvalues[:-1])
    assert np.allclose(display['gaps'], gaps, atol=tolerance, rtol=0)
    assert np.allclose(np.sort(display['eigenvalues']), eigenvalues, atol=tolerance, rtol=0)
    contour_errors = {}
    for level, points in display['contours'].items():
        level = float(level)
        points = np.asarray(points)
        assert level in (1., 2.) and points.shape == (2, 241, 3)
        assert np.allclose(points[0], -points[1], atol=2e-12, rtol=0)
        assert np.allclose(points[:, 0], points[:, -1], atol=2e-12, rtol=0)
        assert np.max(abs(np.sum(points*points, axis=-1)-1.)) < 2e-12
        error = float(np.max(abs(eigenvalues[-1]-_quadratic(points, B)-level)))
        assert error < tolerance
        contour_errors[str(level)] = error
    for level in (1., 2.):
        assert (level in display['contours']) == (level < gaps[0]), 'Contour validity mismatch'
    ambient_error = None
    if 'frame' in law:
        frame = law['frame']
        ambient_nodes = native @ frame.T
        ambient_value = np.einsum('...i,ij,...j->...', ambient_nodes.conj(),
                                 law['ambient_A'], ambient_nodes).real
        ambient_error = float(np.max(abs(ambient_value-q_display)))
        assert ambient_error < tolerance
    positive = gaps[0] > 1e-10
    return dict(name=law['name'], saved_file=law['saved_file'],
                interpretation=law['interpretation'],
                eigenvalues=eigenvalues.tolist(), gaps=gaps.tolist(),
                curvature_ratio=float(gaps[-1]/gaps[0]) if positive else None,
                rotation=R.tolist(), determinant=float(np.linalg.det(R)),
                mesh_shape=list(shown.shape), unit_norm_error=unit_error,
                pointwise_log_density_error=density_error,
                ambient_section_log_density_error=ambient_error,
                antipodal_error=antipodal_error, contour_level_errors=contour_errors,
                full_mesh_angular_coverage=True,
                saved_normalizer_unchanged=law.get('logz'),
                density_scope='Quadratic log-density equality with the same saved '
                              'normalizer; no integration, sampling or estimator calls.'), R


def audit(example, preservation, baseline_sha):
    from lib.final_paper_visuals import CAMERA, PAPER_STYLE
    assert CAMERA == dict(elev=25., azim=-50.)
    assert PAPER_STYLE['text.usetex'] is True
    assert PAPER_STYLE['figure.dpi'] == PAPER_STYLE['savefig.dpi'] == 200
    records, shared = [], {}
    for law in sphere_laws(example):
        record, rotation = check_law(law)
        records.append(record)
        # Exercise the same-reference rotation interface independently of each
        # law's default orientation, as used by matched pairs and time panels.
        if law['group'] not in shared:
            shared[law['group']] = rotation
        check_law(law, rotation=shared[law['group']])
    return dict(schema=1, status='passed', example=example,
                audit_source_sha256=sha(Path(__file__)),
                display_source_sha256=sha(ROOT/'lib/final_paper_visuals.py'),
                baseline_snapshot_sha256=baseline_sha, preservation=preservation,
                laws_checked=len(records), shared_rotation_checks=len(records),
                plot_protocol=dict(camera=CAMERA, projection='orthographic',
                                   fixed_sphere_radius=1., text_usetex=True, dpi=200),
                laws=records,
                scope='Rendering-only numerical invariance, not a fresh claim of '
                      'statistical calibration or algorithm performance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path,
                        help='Write a pre-edit source/data snapshot, then exit.')
    parser.add_argument('--baseline', type=Path,
                        help='Compare against an existing pre-edit snapshot.')
    parser.add_argument('--baseline-commit', default=BASELINE_COMMIT,
                        help='Committed scientific baseline when no snapshot is supplied.')
    parser.add_argument('--example', choices=('beamforming', 'contact', 'both'), default='both')
    args = parser.parse_args()
    if args.snapshot:
        result = snapshot()
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(snapshot=str(args.snapshot),
                              files={name: len(result[name]) for name in
                                     ('artifacts', 'manifests', 'sources')})))
        return
    baseline = json.loads(args.baseline.read_text()) if args.baseline else snapshot(args.baseline_commit)
    preservation = check_preservation(baseline)
    baseline_sha = sha_bytes(json.dumps(baseline, sort_keys=True).encode())
    for example in ('beamforming', 'contact') if args.example == 'both' else (args.example,):
        result = audit(example, preservation, baseline_sha)
        folder = ROOT/'results'/example/'data/validation'
        folder.mkdir(parents=True, exist_ok=True)
        output = folder/'sphere_visual_checks.json'
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(example=example, status=result['status'],
                              laws_checked=result['laws_checked'], output=str(output),
                              preserved=preservation['unchanged_file_counts'])))


if __name__ == '__main__':
    main()
