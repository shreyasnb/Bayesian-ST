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
7 inches with 11-point axis text and 10-point ticks/legends. Preview images and rasterized density layers use
300 dpi; PDF axes, lines and text remain vector objects.

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

Render and package the saved results. PDF page extraction and the independent
actual-export audit use the separate presentation dependency:

```bash
python -m pip install -r requirements-presentation.txt
python scripts/plot_final_simulations.py --stage paper --example both
python scripts/export_draft_figures.py
python scripts/check_sphere_visuals.py
python scripts/check_information_visuals.py
python scripts/check_final_pdf_layout.py --include-exports --write
python scripts/make_final_notebooks.py --stage paper --example both
python scripts/execute_final_notebooks.py --example both
python scripts/build_final_inventory.py --content-commit eee4c1e30882c5530eda8d25fe6afcc349f0ddf7
python scripts/check_final_simulations.py
```

The current display revision restores the original native sphere coordinates,
oblique camera and continuous blue-purple-pink-yellow palette on the original shared logarithmic scale. Thin gray curves
are front-visible coordinate great circles, rather than probability contours.
The mode-facing rotation and white density contours remain removed. Color
positions change only the palette; they do not represent probability quantiles. Density values and physical angular
widths were never changed. The original n=64, t=64 array checkpoint is restored.
The array sphere is a real-coefficient section of a complex posterior, not a
physical radiation field or a normalized sphere marginal; the setup figure
shows the physical beam response. Contact normals retain their measured wrist
coordinates and antipodal interpretation.

For manuscript insertion, the current vector PDFs are also organized in
`results/main/{beamforming,contact}/` and
`results/supplementary/{beamforming,contact}/`. The main selection contains
four single-page figures per application. Contact task performance displays
absolute MSE intervals for every method, including the posterior mean; its
paired MSE differences are retained on a supplementary page. Processing-time
figures state the measured 64-observation workload, with memory on a separate
supplementary page. Contact information uses the same saved inputs to visualize
exact Gaussian KL under physical plane rotations, with the Fisher matrix as
its local Hessian; the old eigenvalue validation remains in the implementation
notebook and numerical tables. Source captions, page selectors and hashes are
in `results/draft_figure_manifest.json` and the folder READMEs. Both original application catalogues remain complete.

Published exporters use the shared `DPI=300` from `lib/plotting.py`; editable
notebook cells expose their own `DPI=300` defaults. Seven-inch paper pages yield 2100-pixel-wide previews.
PDF text and curves remain vector objects: dpi applies to their raster layers
and PNG previews. Notebook images are scaled to fit the screen; full-resolution
PNG and PDF links are supplied beside every image. Preserved historical output
files retain their original metadata and are not relabeled as 300-dpi exports.

PyMuPDF and Pillow are used only for presentation export/auditing, separate
from the scientific environment. Run the complete sequence above after a
rendering change, since both the draft selections and export audits bind the
current source PDFs and figure manifests.

The sphere audit compares against commit `ded5499`: all 574 frozen numerical
artifacts, 25 scientific manifests and 132 scientific/core source files must
remain byte-identical. It checks native-coordinate density evaluation and
unit-sphere coverage for 20 beamforming and 464 contact laws. A separately
identified fixed-rotation stress test checks mathematical invariance; those
rotations and auxiliary contour tests are not used in the final figures.
Reports live under each application's `data/validation/`. No tuning, simulation
or benchmark timing is repeated by a plotting or export command.

The notebook maker requires completed figure evidence. The execution command
uses clean kernels and saves both notebooks before the last check. They verify numerical inputs and draw each figure using the full Matplotlib
code written in its notebook cell; executing the application notebooks does not
repeat the Monte Carlo study. See `completion_report.md` for the tested commands and revision.

## Historical evidence

`archive/a91faab/preservation.json` binds 229 original source/notebook files
and 2,358 historical result files from the starting revision. The original
result files remain in place. Old certificates are not refreshed to pretend
that a new source tree produced old data. In particular,
the original field and implementation-audit notebooks, including their old
plotting assertions and outputs, remain available at commit `a91faab` and in
the archive. Reproducing those original simulations requires that historical
source/environment. The current root `02_field.ipynb` and
`04_implementation_audit.ipynb` instead read verified historical arrays and
render editable 300-dpi figures in the current environment, without rerunning
the old producers. The current application studies are `01_array.ipynb` and
`03_contact.ipynb`.

Original 8-/16-sensor and moving 6-sensor beamforming cohorts retain their
recorded counts. The new 32-/64-/128-sensor task cohorts are distinct from
synthetic spectral integration and dimension-cost studies. General-rank
Fisher checks do not establish general-rank moment or evidence scalability.

The [plotting protocol](plotting_protocol.md) compares the implementation
notebook's typography with the paper layout. The shared current profile uses
11-point main/axis text, 10-point ticks/legends and 1.6-point lines at 300 dpi;
additional panel space preserves legibility at the seven-inch insertion width.
The new contact likelihood illustration is checked independently with
`python scripts/check_information_visuals.py`, comparing the direct Gaussian
means, exact KL formula and numerical Hessian. It introduces no new trials.

## Editable notebook plotting

All five root notebooks expose their figure code directly. Per-cell DPI,
FIGSIZE, artists, labels, colors and PDF saves can be changed independently.
Data-only helpers read the saved arrays, while editable exports are kept under
`results/notebook_exports/`. The published figure catalogue and raw numerical
records remain frozen. See [the editing workflow](editable_notebooks.md).
The theory notebook's original numerical checks write their fresh diagnostic
outputs to `results/notebook_runs/00_theory/`; field and audit notebooks read
historical data without rerunning the old simulation producers.
