# Final simulation protocol

This protocol implements the author's
[final guide](final_simulation_guide.md), using `v5.tex` as the main manuscript
and `supplementary(14).tex` as its supplement. See
[mathematical checks and interpretation](final_manuscript_checks.md).

## Evidence and stages

Final vector PDFs are in `results/beamforming/` and `results/contact/`.
Each application has 15 individual study PDFs, `paper_summary.pdf`, three
supplementary bundles, external caption notes and a figure manifest.
Numerical records live in `data/{verify,pilot,paper}/{component}/` and
`tables/{verify,pilot,paper}/{component}/`. Components are `theory`,
`performance`, `prediction` and `computation`.

Primary performance and causal-prediction cohorts use stage-disjoint seed
sets. Mathematical checks and engineering fixtures reuse declared fixed seeds;
their pilot prefixes are not additional independent results. New primary
task and causal-adaptation studies use 64 validation and 256 test episodes per
declared condition. Counts for mathematical integration, local accuracy,
calibration, block-delay and historical cohorts differ; their configurations
and tables give the actual repetition units and budgets. A query direction,
held-out snapshot or time point is not an independent episode. Intervals are
pointwise; exploratory grids have no multiplicity correction.

Every component binds its configuration, scientific producers and numerical
artifacts by SHA-256. The general gate runs the complete original theory
notebook and every current regression in an isolated output directory.
Source changes invalidate that gate. Plotting verifies saved evidence and
does not simulate, fit, resample or choose a new trajectory.

Provisional producers and outputs that required a methodological correction
remain in development folders with preservation maps. They are not additional
independent results and do not enter the final figures. Checkpoint caches are
regenerable and excluded from Git.

## Reproduce

Use Python 3.12, working LaTeX, and the dependencies in `pyproject.toml`.
Every figure uses `text.usetex=True`. Paper figures have a nominal width of
7 inches and 9-point text. Preview images and rasterized density layers use
200 dpi; PDF axes, lines and text remain vector objects.

From the repository root:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/validate_final_simulations.py
```

Run these commands first with `--stage verify`, then `--stage pilot`, then
`--stage paper`. Frozen existing outputs are hash-checked and reused; changed
scientific configurations require preserving the previous results and using
a fresh output directory, rather than relabeling them.

```bash
python scripts/run_final_theory.py --stage paper --example both
python scripts/run_final_beamforming.py --stage paper --workers 2
python scripts/run_final_contact.py --stage paper --workers 2
python scripts/run_final_prediction.py --stage paper --example both --workers 2
```

After those processes have finished, measure computation **in isolation**:

```bash
python scripts/run_final_computation.py --stage paper --isolated
```

Concurrent simulation execution times are engineering metadata, not benchmark
timings. The computation runner warms operations, randomizes their order and
records repeated single-thread cycle times, phase breakdowns and memory with
an explicit measurement scope. Task-state timings use saved policy settings.

Render and check the saved results:

```bash
python scripts/plot_final_simulations.py --stage paper --example both
python scripts/make_final_notebooks.py --stage paper --example both
python scripts/execute_final_notebooks.py --example both
python scripts/build_final_inventory.py
python scripts/check_final_simulations.py
```

The notebook maker requires completed figure evidence. The execution command
uses clean kernels and saves both notebooks before the last check. They verify
and display saved figures; executing them does not repeat the Monte Carlo
study. See `completion_report.md` for the tested commands and revision.

## Historical evidence

`archive/a91faab/preservation.json` binds 229 original source/notebook files
and 2,358 historical result files from the starting revision. The original
result files remain in place. Old certificates are not refreshed to pretend
that a new source tree produced old data. In particular,
`04_implementation_audit.ipynb` remains a historical walkthrough whose matching
source is available at commit `a91faab` and in that archive.

Original 8-/16-sensor and moving 6-sensor beamforming cohorts retain their
recorded counts. The new 32-/64-/128-sensor task cohorts are distinct from
synthetic spectral integration and dimension-cost studies. General-rank
Fisher checks do not establish general-rank moment or evidence scalability.
