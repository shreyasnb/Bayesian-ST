# Development checkpoint findings

Branch: dev. Reviewed baseline: `57f520cbe860b73e13d71bae5c31391e2a009482`.
Original artifacts are preserved; this directory contains the extension only.
The baseline audit records the guide hash, software versions and original budgets.

| Checkpoint | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 0 | implemented and checked | Baseline tests, likelihood identities, equivariance, hashes and preserved outputs | [baseline/audit.json](baseline/audit.json); [validation.json](validation.json) |
| 1 | implemented and checked | Explicit identity, partial, exact and causal motion; sensor-fixed and co-rotating inputs | [motion/preregistered.json](motion/preregistered.json); [applications/predeclared_config.json](applications/predeclared_config.json) |
| 2 | implemented and checked | Exponential memory and rolling likelihood queue; independent mixture validation | [motion/tuning.csv](motion/tuning.csv); [motion/selection.json](motion/selection.json); [motion/paired.csv](motion/paired.csv) |
| 3 | implemented and checked | Geometry and causality checked; primary causal-versus-tuned-memory holdout outcome | [motion/paired.csv](motion/paired.csv); [motion/paths.csv](motion/paths.csv); [motion/cost.csv](motion/cost.csv) |
| 4 | implemented and checked | Rank/codimension-one transition moments and inverse projection; independent real-circle full-density reference | [process/reference_checks.json](process/reference_checks.json); [process/coverage_trials.csv](process/coverage_trials.csv) |
| 4 | unresolved | General-rank stochastic moment inversion and calibrated uncertainty; explicitly rejected | [process/status.json](process/status.json) |
| 5 | implemented and checked | Matched-process 200-trial calibration per kappa; four distinct coverages, widths, innovations and separate noise ablations | [process/coverage_summary.csv](process/coverage_summary.csv); [motion/coverage_paired_noise.csv](motion/coverage_paired_noise.csv); [applications/noise.csv](applications/noise.csv) |
| 6 | unresolved | array primary application versus scalar shrinkage | [applications/paired.csv](applications/paired.csv); [applications/tuning.csv](applications/tuning.csv) |
| 6 | empirically inconclusive | field primary application versus scalar shrinkage | [applications/paired.csv](applications/paired.csv); [applications/tuning.csv](applications/tuning.csv) |
| 7 | implemented and checked | Checkpoint handoff, sources, table checksums, commands, manuscript notes and measured cost | [manifest.json](manifest.json); [gate_binding.json](gate_binding.json); [README.md](README.md) |

## Memory and velocity

The common validation policy is beta=0.85, W=12, lag=32, damping=0.5.
It is shared across test laws and input frames; no hidden test speed selects a policy.
The lag grid was expanded using validation only. Its remaining boundary flag is retained.
The preselected fast-drift, sensor-fixed primary gain (tuned forgetting minus causal velocity) is 0.03102, with paired 95% interval [0.02667, 0.03536].
See [paired losses](motion/paired.csv) and [all motion findings](motion/findings.md), including transients, acceleration, turns, weak excitation, low SNR and corruption.
Learned rotations and rolling windows receive no deterministic tracking certificate.
Posterior/prior pilot sampling failures: 0/0.
Only eight trajectories per conditional posterior regime are evaluated; Wilson intervals are wide.
Final response coverage uses method-shared held-out noise in motion/coverage_paired_noise.csv.
The raw inherited method-specific query evaluation is preserved with an explicit postprocessing audit.

## Applications

One task-loss validation policy per application is frozen before independent tests.
Loading and memory form a Cartesian grid; scalar shrinkage, same-posterior mode,
EW PCA and the appropriate application references are included. Field GROUSE and
rolling-memory grids extend the earlier boundaries. Extra regimes are exploratory.

| Primary application vs shrinkage | Status | Selected validation failures | Test failures | Paired gain interval |
| --- | --- | ---: | ---: | --- |
| array | unresolved | 8 | 16 | [-0.01194, 0.01232] |
| field | empirically inconclusive | 0 | 0 | [-0.03383, 0.03826] |

Intervals with capped numerical failures do not support positive performance claims.
A negative or inconclusive comparison stays visible. Full tuning and trial-level
diagnostics are in applications/; point estimates do not repair failed numerical checks.

## Process uncertainty

The explicit isotropic transition is tested against an independent full-density real Gr(1,2) reference.
Kappa is a known simulation parameter. Fixed beta=0.85 controls are diagnostics, not tuned application competitors.
The predicted moment is exact conditional on the maintained preceding Bingham law; the density projection is approximate.
A reference fixture has inverse residual near 4e-17 and predicted-density TV near 0.0283. Those measure different errors.
General-rank inversion remains unresolved; it is not replaced by scalar rescaling.
See [reference checks](process/reference_checks.json) and [coverage with widths](process/coverage_summary.csv).

| Matched kappa | Method | Ball coverage | HPD coverage | Clean coverage | Noisy coverage | Trials |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Full-density reference | 0.950 | 0.955 | 0.960 | 0.955 | 200 |
| 4 | Projected transition | 0.950 | 0.950 | 0.965 | 0.955 | 200 |
| 20 | Full-density reference | 0.965 | 0.960 | 0.965 | 0.970 | 200 |
| 20 | Projected transition | 0.960 | 0.960 | 0.970 | 0.965 | 200 |

Coverage intervals are pointwise Wilson intervals; favorable finite results are not a general calibration theorem.
Deterministic drift and observation-mismatch failures remain in [process findings](process/findings.md).
Observation contamination produces undercoverage even for the full-density filter.
The two noise-learning ablations check frozen-prior objectives, not full state-space likelihood ascent.

## Cost and figures

motion/cost.csv and applications/metrics.csv separate prediction, update, moment and decision costs.
Process inversion is nested within prediction time; embedded moment costs are missing rather than falsely zero.
Reporting frequencies, worker counts and software versions are in the corresponding configurations.
manifest.json records hardware and notes that cloud wall-clock timings can overlap other studies.
Figures use TeX and 200 dpi. Main figures show motion error and uncertainty, paired application losses,
and coverage versus width. Extra tuning curves, conditions and runtime detail are supplementary.
[Motion and uncertainty](figures/motion.pdf), [paired application losses](figures/application_gains.pdf),
and [matched-process coverage and width](figures/process_coverage_width.pdf) are the three main figures.

## Commands

Run from the repository root with the documented Python/TeX dependencies.
Verify the delivered evidence and regenerate its figures:

```bash
python scripts/summarize_prediction_ablation.py
python scripts/plot_prediction_ablation.py
python scripts/write_dev_manifest.py
```

Recompute the numerical studies in fresh directories using the frozen scientific settings.
The same seeds reproduce the experiment; they do not create extra independent evidence:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_dev.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_prediction_ablation.py --workers 2 --output-dir results/dev_prediction/reproduced_motion
python scripts/check_motion_paired_noise.py --output-dir results/dev_prediction/reproduced_motion
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_process_prediction.py --output results/dev_prediction/reproduced_process
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_application_ablation.py --workers 2 --output-dir results/dev_prediction/reproduced_applications
```

The captured run gates remain in original run metadata. gate_binding.json verifies every
numerical source/table before associating those runs with the final gate after the certificate-path
fix and regression test. This is dependency-verified reuse, not additional independent simulation.
The motion reproduction command uses a fresh directory: the original producer deliberately
rejects its captured old gate rather than silently rewriting completed-run metadata.
See [development documentation](../../docs/development.md) and [manuscript notes](../../docs/dev_manuscript_notes.md).
