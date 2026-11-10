"""Check contact likelihood geometry against the frozen Gaussian experiment.

This is a presentation audit. It generates no observations and changes no
scientific parameters, trials, fitted distributions or benchmark results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.final_information_visuals import (
    contact_information_geometry, chart_normals, exact_contact_kl, direct_contact_kl)
from lib.final_validation import require_gate, verify_component, sha


def check():
    gate = require_gate(ROOT)
    meta = verify_component(ROOT, 'contact', 'paper', 'performance')
    source = ROOT/'results/contact/data/paper/performance/anisotropy.npz'
    source_hash = sha(source)
    with np.load(source, allow_pickle=False) as saved:
        record = {name: saved[name] for name in saved.files}
    geometry = contact_information_geometry(record, meta['config']['anisotropy_sigma2'])
    coordinate = np.linspace(-1.2, 1.2, 41)
    first, second = np.meshgrid(coordinate, coordinate)
    z = np.stack([first, second], axis=-1)
    direct = direct_contact_kl(z, geometry)
    formula = exact_contact_kl(z, geometry)
    error = float(np.max(np.abs(direct-formula)))
    assert error < 2e-12
    assert direct.min() >= 0.
    n = chart_normals(z, geometry)
    unit_error = float(np.max(np.abs(np.sum(n*n, axis=-1)-1.)))
    assert unit_error < 2e-14

    # The Gaussian-mean expression, rather than the closed-form expression,
    # provides this independent central-difference Hessian at Z=0.
    step = 1e-5
    basis = step*np.eye(2)
    H = np.zeros((2, 2))
    for i in range(2):
        H[i, i] = (direct_contact_kl(basis[i], geometry)
                   +direct_contact_kl(-basis[i], geometry))/step**2
    H[0, 1] = H[1, 0] = (
        direct_contact_kl(basis[0]+basis[1], geometry)
        -direct_contact_kl(basis[0]-basis[1], geometry)
        -direct_contact_kl(-basis[0]+basis[1], geometry)
        +direct_contact_kl(-basis[0]-basis[1], geometry))/(4*step**2)
    hessian_error = float(np.linalg.norm(H-geometry['F'])/np.linalg.norm(geometry['F']))
    assert hessian_error < 2e-8

    delta = np.deg2rad(np.linspace(.25, 35., 101))
    ray_errors = []
    for i in range(2):
        ray = np.zeros((len(delta), 2))
        ray[:, i] = np.tan(delta)
        expected = .5*geometry['F'][i, i]*np.sin(delta)**2
        actual = direct_contact_kl(ray, geometry)
        ray_errors.append(float(np.max(np.abs(actual-expected))))
        assert ray_errors[-1] < 2e-12
        normal = chart_normals(ray, geometry)
        angles = np.arccos(np.clip(normal @ geometry['normal'], -1., 1.))
        assert np.allclose(angles, delta, rtol=0, atol=2e-12)

    levels = [.01, .05, .15, .3]
    phi = np.linspace(0., 2*np.pi, 241)
    contour_errors = []
    for level in levels:
        semiaxes = np.sqrt(2*level/(np.diag(geometry['F'])-2*level))
        points = np.column_stack([semiaxes[0]*np.cos(phi), semiaxes[1]*np.sin(phi)])
        contour_errors.append(float(np.max(np.abs(direct_contact_kl(points, geometry)-level))))
        assert contour_errors[-1] < 2e-12
    assert sha(source) == source_hash, 'A frozen numerical artifact changed'
    expected_hash = meta['artifacts'][str(source.relative_to(ROOT))]
    if isinstance(expected_hash, dict):
        expected_hash = expected_hash['sha256']
    assert source_hash == expected_hash

    marked = [np.array([np.tan(np.deg2rad(5.)), 0.]),
              np.array([0., np.tan(np.deg2rad(30.))])]
    return dict(status='passed', example='contact',
        producer='scripts/check_information_visuals.py', producer_sha256=sha(Path(__file__)),
        visual_source='lib/final_information_visuals.py',
        visual_source_sha256=sha(ROOT/'lib/final_information_visuals.py'),
        final_science_hash=gate['final_science_hash'],
        input_hashes={str(source.relative_to(ROOT)):source_hash,
            'results/contact/data/paper/performance/manifest.json':sha(
                ROOT/'results/contact/data/paper/performance/manifest.json')},
        checkpoint=geometry['t'], sigma2=geometry['sigma2'],
        fisher_matrix=geometry['F'].tolist(), fisher_eigenvalues=geometry['fisher_eigenvalues'].tolist(),
        direct_gaussian_KL_max_error=error, native_normal_unit_error=unit_error,
        finite_difference_Hessian_relative_error=hessian_error,
        physical_angle_ray_max_errors=ray_errors,
        exact_contour_max_errors=contour_errors, contour_levels=levels,
        marked_angles_deg=[5., 30.], marked_exact_KL=[float(direct_contact_kl(point, geometry)) for point in marked],
        checks=dict(saved_fisher_eigenvalues=True, direct_mean_KL_equals_closed_form=True,
            local_KL_Hessian_equals_Fisher=True, physical_normal_angles=True,
            equal_KL_contours=True, source_numerical_file_unchanged=True),
        scope='Analytic observation-law geometry at one already-saved design. '
              'KL contours are not posterior or confidence regions. '
              'No new observations, inference, fitting, benchmark selection or Monte Carlo repetitions.')


def main():
    result = check()
    output = ROOT/'results/contact/data/validation/information_visual_checks.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in (
        'status', 'checkpoint', 'fisher_eigenvalues', 'direct_gaussian_KL_max_error',
        'finite_difference_Hessian_relative_error', 'marked_exact_KL')}, indent=2))


if __name__ == '__main__':
    main()
