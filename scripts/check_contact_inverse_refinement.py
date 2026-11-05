"""Reproduce the 102-fit independent audit of contact moment inversion.

Targets and verification use ordinary, unwarped azimuth integration, separate
from the concentration-dependent angular rule used by the backend. This is
numerical debugging evidence, not additional trajectories or theorem tests.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import IntegrationWarning, quad
from scipy.special import dawsn, hyp1f1
from lib.contact_process_prediction import fast_moments, invert_contact_mean
from lib.geometry import center


BASE_PAIRS = [(0., 20.), (0., 100.), (0., 205.85), (0., 80000.),
              (.01, 20.), (.1, 100.), (.3518, 205.85), (.8, 1000.),
              (1.01, 2000.), (10., 80000.), (100., 1e6), (1e6, 1e6)]
CROSS_PAIRS = [(.01, 20.), (.1, 100.), (.3518, 205.85), (.8, 1000.),
               (1.01, 2000.), (0., 95.), (0., 99.9), (10., 80000.), (100., 1e6)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {str(key):portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def reference(a, b, epsabs, epsrel):
    """Plain physical azimuth; exact radial primitives, adaptive QUADPACK."""
    def integrand(phi, statistic):
        co, si = np.cos(phi), np.sin(phi)
        g = a*co*co+b*si*si
        if statistic == 0:
            return dawsn(np.sqrt(g))/np.sqrt(g) if g else 1.
        radial = (2/3)*hyp1f1(2., 2.5, -g)
        return (co*co if statistic == 1 else si*si)*radial
    values, errors = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        for statistic in range(3):
            value, error = quad(lambda phi:integrand(phi, statistic), 0., np.pi/2,
                epsabs=epsabs, epsrel=epsrel, limit=500)
            values.append(value)
            errors.append(error)
    values, errors = np.asarray(values), np.asarray(errors)
    bottom = values[1:]/values[0]
    normal = np.r_[1-bottom.sum(), bottom]
    return np.eye(3)-np.diag(normal), values, errors


def direct_radial_reference(a=.1, b=100.):
    """Additional check integrating exp[-g(1-z²)] directly in both angles."""
    def radial(phi, statistic):
        g = a*np.cos(phi)**2+b*np.sin(phi)**2
        def integrand(z):
            weight = np.exp(-g*(1-z*z))
            if statistic == 0:
                return weight
            return weight*(1-z*z)*(np.cos(phi)**2 if statistic == 1 else np.sin(phi)**2)
        return quad(integrand, 0., 1., epsabs=1e-12, epsrel=1e-12)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        integrals = np.array([quad(lambda phi:radial(phi, statistic), 0., np.pi/2,
            epsabs=1e-11, epsrel=1e-11, points=[np.pi/100, np.pi/20], limit=200)[0]
            for statistic in range(3)])
    bottom = integrals[1:]/integrals[0]
    return np.eye(3)-np.diag(np.r_[1-bottom.sum(), bottom]), integrals


def audit(out):
    rows, reference_rows, references, parameters = [], [], {}, {}
    def target(a, b, epsabs):
        key = (float(a), float(b), float(epsabs))
        if key not in references:
            M, integrals, errors = reference(a, b, epsabs, 1e-12)
            fine, fine_integrals, fine_errors = reference(a, b, epsabs/100, 1e-13)
            index = len(reference_rows)
            references[key] = (M, fine, index)
            reference_rows.append(dict(reference_id=index, gap1=a, gap2=b,
                epsabs=epsabs, epsrel=1e-12, fine_epsabs=epsabs/100, fine_epsrel=1e-13,
                quad_limit=500, angular_domain="[0,pi/2], unwarped physical azimuth",
                normalizer=integrals[0], normalizer_quad_error=errors[0],
                statistic1_integral=integrals[1], statistic1_quad_error=errors[1],
                statistic2_integral=integrals[2], statistic2_quad_error=errors[2],
                fine_normalizer=fine_integrals[0], fine_normalizer_quad_error=fine_errors[0],
                maximum_mean_refinement_difference=float(np.linalg.norm(M-fine, 2)),
                normal_mean0=float(1-M[0,0]), normal_mean1=float(1-M[1,1]),
                normal_mean2=float(1-M[2,2])))
        return references[key]

    def fit_one(group, a, b, M, fine_reference, reference_id, initial, initial_label,
                rotation=None, target_origin="Independent unwarped quadrature"):
        index = len(rows)
        Q = np.eye(3) if rotation is None else rotation
        requested, fine = Q@M@Q.T, Q@fine_reference@Q.T
        initial_A = None if initial is None else np.asarray(initial)
        parameters[f"fit_{index:03d}_target"] = requested
        parameters[f"fit_{index:03d}_reference"] = fine
        parameters[f"fit_{index:03d}_rotation"] = Q
        parameters[f"fit_{index:03d}_initial"] = (np.zeros((3,3)) if initial_A is None else initial_A)
        begin = perf_counter()
        row = dict(fit=index, group=group, gap1=a, gap2=b, reference_id=reference_id,
            initialization=initial_label, initial_supplied=initial_A is not None,
            target_origin=target_origin, status="unresolved", failure="")
        try:
            result = invert_contact_mean(requested, initial=initial_A)
            evaluated = fast_moments(result.A, checked=True, angular_order=512)
            target_error = float(np.linalg.norm(evaluated.mean-requested, 2))
            independent_error = float(np.linalg.norm(evaluated.mean-fine, 2))
            objective_ok = bool(result.objective <= result.initial_objective+5e-8)
            passed = bool(result.converged and result.residual+result.moment_uncertainty <= 1e-8
                and target_error < 1e-8 and independent_error < 1e-8 and objective_ok
                and evaluated.numerical_ok)
            row.update(status="passed" if passed else "unresolved", target_mean_error=target_error,
                independent_mean_error=independent_error, reported_residual=result.residual,
                reported_backend_uncertainty=result.moment_uncertainty,
                initial_objective=result.initial_objective, final_objective=result.objective,
                objective_guard_passed=objective_ok, iterations=result.iterations,
                evaluations=result.evaluations, inverse_seconds=result.seconds,
                verification_angular_order=evaluated.angular_order,
                verification_resolution_error=evaluated.resolution_error,
                message=result.message)
            parameters[f"fit_{index:03d}_fitted_A"] = result.A
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            row["failure"] = str(error)
        row["audit_wall_seconds"] = perf_counter()-begin
        rows.append(row)

    # The original 24-fit grid: two starts, no target from the tested backend.
    for a, b in BASE_PAIRS:
        M, fine, reference_id = target(a, b, 1e-12)
        for label, initial in [("uniform", None), ("known spectrum", center(np.diag([0., a, b])))]:
            fit_one("independent targets", a, b, M, fine, reference_id, initial, label)

    # The original hard-initial/easy-final cross-grid, including a
    # noncommuting initial matrix. No target-dependent tuning takes place.
    rng = np.random.default_rng(929315)
    rotation = np.linalg.qr(rng.normal(size=(3,3)))[0]
    initials = [("uniform", None),
        ("narrow ridge .1,100", center(np.diag([0., .1, 100.]))),
        ("narrow ridge .3518,205.85", center(np.diag([0., .3518, 205.85]))),
        ("axisymmetric ridge 99.9", center(np.diag([0., 0., 99.9]))),
        ("sharp pole 10,80000", center(np.diag([0., 10., 80000.]))),
        ("noncommuting rotated ridge", rotation@center(np.diag([0., .1, 100.]))@rotation.T)]
    for a, b in CROSS_PAIRS:
        M, fine, reference_id = target(a, b, 1e-17)
        for label, initial in initials+[("known spectrum", center(np.diag([0., a, b])))]:
            fit_one("cross-grid initial spectra", a, b, M, fine, reference_id, initial, label)

    # Preserve the original rotated-ridge target construction, and additionally
    # compare every fitted result with independent unwarped quadrature.
    for b in [95., 99., 99.9, 100., 101.]:
        A = center(np.diag([0., 0., b]))
        M = fast_moments(A, checked=True, angular_order=512).mean
        _, fine, reference_id = target(0., b, 1e-17)
        for replicate in range(3):
            Q = np.linalg.qr(rng.normal(size=(3,3)))[0]
            fit_one("rotated axisymmetric ridges", 0., b, M, fine, reference_id,
                Q@A@Q.T, f"rotated known spectrum {replicate}", rotation=Q,
                target_origin="Angular order512 doubled to1024, independently checked against unwarped quadrature")

    pd.DataFrame(rows).to_csv(out/"results.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(out/"reference_targets.csv", index=False)
    np.savez_compressed(out/"parameters.npz", **parameters)
    direct, integrals = direct_radial_reference()
    ordinary, _, _ = target(.1, 100., 1e-17)
    direct_difference = float(np.linalg.norm(direct-ordinary, 2))
    return rows, reference_rows, direct_difference, integrals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT/"results/contact_uncertainty/numerics/refinement")
    args = parser.parse_args()
    if (args.output/"checks.json").exists():
        raise RuntimeError("Preserve completed audit evidence; choose a fresh --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = ["scripts/check_contact_inverse_refinement.py", "lib/contact_process_prediction.py",
        "lib/contact_posterior.py", "lib/geometry.py", "lib/process_prediction.py",
        "lib/spectral_series.py", "lib/moments.py"]
    source_hashes = {name:sha(ROOT/name) for name in sources}
    begin = perf_counter()
    rows, references, direct_difference, direct_integrals = audit(args.output)
    if any(sha(ROOT/name) != expected for name, expected in source_hashes.items()):
        raise RuntimeError("Numerical sources changed during audit")
    failures = [row["fit"] for row in rows if row["status"] != "passed"]
    passed = len(rows) == 102 and not failures and direct_difference < 1e-10
    meta = dict(status="passed" if passed else "unresolved", fits=len(rows),
        groups={group:sum(row["group"] == group for row in rows)
            for group in ["independent targets", "cross-grid initial spectra", "rotated axisymmetric ridges"]},
        unresolved_fit_indices=failures,
        maximum_target_mean_error=max(row.get("target_mean_error", np.nan) for row in rows),
        maximum_independent_mean_error=max(row.get("independent_mean_error", np.nan) for row in rows),
        maximum_reported_residual_plus_backend=max(row.get("reported_residual", np.nan)+row.get("reported_backend_uncertainty", np.nan) for row in rows),
        maximum_reference_refinement_difference=max(row["maximum_mean_refinement_difference"] for row in references),
        independent_direct_radial_mean_difference=direct_difference,
        independent_direct_radial_integrals=direct_integrals,
        direct_radial_quad=dict(radial_epsabs=1e-12, radial_epsrel=1e-12,
            azimuth_epsabs=1e-11, azimuth_epsrel=1e-11, azimuth_points=[np.pi/100, np.pi/20],
            azimuth_limit=200),
        seconds=perf_counter()-begin,
        starting_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        sources=source_hashes,
        backend_sha256=source_hashes["lib/contact_process_prediction.py"],
        inverse_tolerance=1e-8, objective_tolerance=5e-8,
        verification="Returned fit is independently checked using unwarped azimuth references; extra backend order512 doubled to1024 verifies the same returned parameter",
        scope="Numerical resolution comparisons and QUADPACK estimates, not rigorous integration enclosures, calibration theorems, or additional independent physical trajectories",
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        tables={name:sha(args.output/name) for name in ["results.csv", "reference_targets.csv", "parameters.npz"]})
    (args.output/"checks.json").write_text(json.dumps(portable(meta), indent=2, allow_nan=False)+"\n")
    (args.output/"README.md").write_text(
        "# Contact inverse refinement audit\n\n"
        "Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_contact_inverse_refinement.py` "
        "from the repository root, using a fresh `--output` directory to preserve completed evidence.\n\n"
        "The102 fits reproduce the independent debugging audit:24 plain-azimuth quadrature targets with two starts, "
        "63 cross-grid starts (including difficult and noncommuting initial spectra), and15 rotated axisymmetric ridges. "
        "Every returned mean is compared with an unwarped quadrature reference. The rotated-ridge requested targets "
        "also preserve their original high-resolution backend construction; this is disclosed per row.\n\n"
        "The special-function radial primitives are checked separately by direct two-dimensional integration of "
        "the exponential integrand at the narrow-ridge regression target. Integration estimates and refinement "
        "differences are numerical diagnostics, not rigorous error bounds. No physical-data result or theorem-test count is added.\n\n"
        "`results.csv` retains every success/failure, objective, residual and refinement message. "
        "`reference_targets.csv` records all quadrature tolerances. `parameters.npz` stores every target, "
        "initial matrix, rotation and returned parameter. `checks.json` pins the exact backend and script bytes.\n")
    print(json.dumps(portable(meta), indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
