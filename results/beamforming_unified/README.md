# Unified passive beamforming study

This directory is separate from the historical array studies. It contains the native proper-complex rank-one experiment, its general-validation certificate, executed profiles, numerical flags, figures and historical source archive. The base is `origin/dev` at `3b47f8d7765d125fff6a1bbe8c8e9d43a8627517`; all task changes belong to branch `test`.

Read `findings.md` for observed results and execution status, `../../docs/beamforming_claims.md` for the claim map, and `../../docs/beamforming_theory.md` for derivations. The primary narrative is `../../notebooks/01_array.ipynb`.

## Reproduce

Run from the repository root with the installed repository requirements and working LaTeX/dvipng. The full-density reference requires SciPy 1.15 or newer (`sph_harm_y`); the executed environment uses 1.17.0. `lib.plotting` requires `text.usetex=True`, and all exports use vector PDF plus 200-dpi PNG. Simulation and rendering are separate: changing a layout does not regenerate data.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/validate_beamforming.py
python scripts/audit_beamforming_independent.py
python scripts/check_beamforming_information.py --profile quick
python scripts/check_beamforming_numerics.py --profile quick
python scripts/run_beamforming_decisions.py --profile quick --workers 2
python scripts/run_beamforming_tracking.py --profile quick --component all --workers 2
python scripts/run_beamforming_inference.py --profile quick --workers 2
python scripts/summarize_beamforming_tracking.py --profile quick
python scripts/plot_beamforming_unified.py --profile quick
python scripts/check_beamforming_information.py --profile paper
python scripts/check_beamforming_numerics.py --profile paper
python scripts/run_beamforming_decisions.py --profile paper --workers 2
python scripts/run_beamforming_tracking.py --profile paper --component all --workers 2
python scripts/run_beamforming_inference.py --profile paper --workers 2
python scripts/summarize_beamforming_tracking.py --profile paper
python scripts/plot_beamforming_unified.py --profile paper
python scripts/summarize_beamforming.py --profile paper
python scripts/make_beamforming_notebook.py
python scripts/execute_beamforming_notebook.py
python scripts/manifest_beamforming_artifacts.py
python scripts/check_beamforming_handoff.py
```

Each scientific runner records configuration/source hashes before generation and supports verified checkpoint resumption. Do not edit science sources in the middle of a run or suppress a stale-cache error. A source change requires fresh validation and regenerated affected results in a new or explicitly archived output directory. The historical archive is verified by `lib.beamforming_history.verify_historical`; its old validation certificate is matched to its archived source tree rather than the evolving current test suite.

A completed verified cache returns immediately. To regenerate omitted worker records from scratch while preserving the published evidence, use a fresh directory:

```bash
python scripts/run_beamforming_decisions.py --profile paper --workers 2 --output /tmp/beamforming_regenerated/decisions
python scripts/run_beamforming_tracking.py --profile paper --workers 2 --output-root /tmp/beamforming_regenerated
python scripts/run_beamforming_inference.py --profile paper --workers 2 --output /tmp/beamforming_regenerated/inference
```

Scientific random draws are seeded; runtime, timestamps and their resulting byte hashes naturally vary in a fresh run. Full Gibbs states are regenerated in memory by `stationary_gibbs`/`full_path_gibbs` with the recorded dataset, prior and chain seed; their `JointResult.states` field is available to callers but intentionally not serialized by the compact runner. Incomplete tracking resumes assume preassigned NPZ display arrays have not been manually damaged; complete component manifests verify those arrays. Preserve a damaged incomplete directory and use a fresh output directory rather than trusting its scalar checkpoints alone.

Quick profiles are engineering checks with smaller independent cohorts. They expose finite-sampling diagnostics, and are not a replacement for paper evidence. The numerical paper budget was increased to 16,384 independent training draws and 16,384 independent checking draws before scientific paper comparisons, based on development accuracy checks. Scientific quick and paper decision/tracking/inference evaluation streams are disjoint. Mathematical score/normalizer checks can share predeclared states without making a learned performance comparison.

## Artifacts and omitted raw data

Component `manifest.json` or `config.json` files pin scientific sources and compact table hashes. The figures manifest pins presentation sources and exports. The root artifact manifest records retained files, excluded raw files and regeneration commands. Worker checkpoints, full trajectory/raw state arrays and intermediate development copies can be large; where excluded they are reproducible from the declared component configuration and seeds. Exclusion does not remove the compact trial-level metric, pairing, mixing or numerical-status records. The native complex-line numerical module is new; earlier shared mathematical implementation files remain unchanged.

`archive/` retains the earlier executed notebook and exact numerical/producer source snapshots. Older `results/array_*`, `case1_*` and `decision_study_*` artifacts remain at their existing paths, byte for byte. `development/` records bounded engineering pilots and the independent audit, including unfavorable stress outcomes. These do not count as independent paper test trajectories.
