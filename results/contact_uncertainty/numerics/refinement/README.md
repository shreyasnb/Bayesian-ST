# Contact inverse refinement audit

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_contact_inverse_refinement.py` from the repository root, using a fresh `--output` directory to preserve completed evidence.

The102 fits reproduce the independent debugging audit:24 plain-azimuth quadrature targets with two starts, 63 cross-grid starts (including difficult and noncommuting initial spectra), and15 rotated axisymmetric ridges. Every returned mean is compared with an unwarped quadrature reference. The rotated-ridge requested targets also preserve their original high-resolution backend construction; this is disclosed per row.

The special-function radial primitives are checked separately by direct two-dimensional integration of the exponential integrand at the narrow-ridge regression target. Integration estimates and refinement differences are numerical diagnostics, not rigorous error bounds. No physical-data result or theorem-test count is added.

`results.csv` retains every success/failure, objective, residual and refinement message. `reference_targets.csv` records all quadrature tolerances. `parameters.npz` stores every target, initial matrix, rotation and returned parameter. `checks.json` pins the exact backend and script bytes.
