# Implementation-guide checks

This study extends commit 6e78cc9 on branch test. It preserves the preceding notebooks, complete redesign outputs and their producer sources in [archive/previous](archive/previous). The [guide](../../docs/implementation_guide.md), [coverage audit](../../docs/implementation_guide_audit.md), [results](findings.md) and [claim manifest](claim_manifest.json) distinguish implementation checks from negative or inconclusive empirical outcomes.

The computational walkthrough is [04_implementation_audit.ipynb](../../notebooks/04_implementation_audit.ipynb). The application narratives retain six main figures each and link to this supplement. All density spheres use a continuous blue–violet–pink colormap with the original normalized density, logarithmic color scale and fixed unit radius. New figure names contain the guide's B/C/S IDs. Exact aliases of older main panels are marked as aliases.

The available specification consists of the guide's explicit equations and the supplied PDF theory/appendices. main.tex and supplementary.tex were not supplied. Exact TeX agreement, including the separate complex conditional extension's manuscript constants, remains unverified.

## Verify the committed handoff

Use the repository dependencies and working LaTeX/dvipng. These commands read the committed evidence without recomputing it:

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_implementation_audit.py
~~~

The check verifies both smoke and paper components, scientific and presentation producers, all figure exports, current validation, every embedded notebook image and the prior archives.

## Reproduce

Scientific configurations, seeds, variance conventions, parameter grids, tolerance budgets and statuses are in each component manifest. The five components are information, information_counts, tracking, em and reliability. Smoke results check execution and references; they do not establish a performance advantage.

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/archive_implementation_audit.py
python scripts/validate_implementation_audit.py
python scripts/run_guide_information.py --profile smoke
python scripts/run_guide_information_counts.py --profile smoke
python scripts/run_guide_tracking.py --profile smoke
python scripts/run_block_em_study.py --profile smoke
python scripts/run_guide_reliability.py --profile smoke --workers 2
python scripts/run_guide_information.py --profile paper
python scripts/run_guide_information_counts.py --profile paper
python scripts/run_guide_tracking.py --profile paper
python scripts/run_block_em_study.py --profile paper
python scripts/run_guide_reliability.py --profile paper --workers 2
python scripts/plot_geometry_studies.py --profile quick
python scripts/plot_geometry_studies.py --profile paper
python scripts/plot_implementation_audit.py --profile smoke
python scripts/plot_implementation_audit.py --profile paper
python scripts/summarize_simulation_redesign.py --profile paper
python scripts/summarize_implementation_audit.py
python scripts/make_geometry_notebooks.py
python scripts/make_implementation_audit_notebook.py
python scripts/execute_implementation_audit.py
python scripts/manifest_simulation_redesign.py
python scripts/manifest_implementation_audit.py
python scripts/check_implementation_audit.py
~~~

Completed scientific caches verify configuration, producer and table hashes. Preserve obsolete runs and pass a fresh output directory when changing a configuration; do not rewrite failed diagnostic subsets. After a scientific change, repeat general validation, affected studies and coverage review. The finalized coverage plan is bound to its reviewed scientific component manifests; changed scientific evidence needs a new review before publication. Figure regeneration and notebook execution have separate source/content checks.

The earlier scaling cohort remains isolated under results/simulation_redesign. Its serial warmed measurements are the runtime evidence. Concurrent guide-study elapsed times are not a latency comparison. To regenerate the original decision cohorts in a new directory, use their recorded configurations and earlier protocol; keep their original certificates intact.

## Interpretation and preservation

The new block EM uses one stationary state and one boundary prior excluding the entire block. Every E-step rebuilds that posterior and checks the same normalized evidence. It is distinct from the historical recorded-prior objective. Known-noise theorem checks and adaptive fits remain separate.

Prior-predictive, fixed-state and drifting calibration are different experiments. Validation-calibrated intervals do not automatically achieve matched test coverage. All inclusion intervals, interval-width failures, finite-iteration limits, mixture/surrogate score comparisons and mismatch outcomes remain in the tables.

Review identified a discrete-quadrature region error and a friction stress that could reverse motion. Both superseded runs retain their exact producers and failed diagnostics under development. The final real-state regions use continuous radial integration, and the final friction force is clamped to permit sticking and guarantee nonpositive friction power. The full general gate was repeated after those corrections and the observation-count additions. Unaffected calibration, scoring and scale-fitting outputs were byte-compared with the provisional run; only the intended friction rows changed.

Worker checkpoints and raw pytest XML are regenerable and excluded from Git. Compact raw results, all numerical flags, reviewed figures and producer snapshots are retained in the artifact manifest. All notebook execution is recorded separately in execution.json, avoiding circular report/execution hashes. main and dev are unchanged.
