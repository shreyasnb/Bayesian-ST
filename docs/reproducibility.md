# Reproducing the simulations

The validated numerical environment uses CPython 3.12 on Linux. The direct
Python dependencies are pinned in `requirements-lock.txt`. Transitive Jupyter
and plotting dependencies are resolved by pip; this file is not a complete
platform-specific environment lock. The notebook outputs record the numerical
versions, seeds, sampler budgets and simulation settings used for each run.

Use the existing checkout in an isolated cloud task. A Git worktree is not
needed for this workflow. Run commands below from the repository root.

## Python and TeX

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script creates `.venv` by default, installs the pinned direct
dependencies without disabling TLS or package verification, installs this
repository in editable mode, and registers the virtual environment's `python3`
kernel. Set `BAYESIAN_ST_PYTHON` to a CPython 3.12 interpreter and
`BAYESIAN_ST_VENV` to a writable alternative environment path if needed.
Installation requires access to PyPI and its distribution hosts.

All plots use `matplotlib.rcParams['text.usetex'] = True`, configured centrally
by `lib.plotting`. TeX rendering is a requirement, rather than a fallback to
Matplotlib's mathtext. On a Debian or Ubuntu machine, an administrator can install
the rendering prerequisites using:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    texlive-latex-extra texlive-fonts-recommended cm-super dvipng
```

`scripts/setup.sh` does not run administrator commands. The prepared cloud
machine has `dvipng` and `cm-super` extracted under `/workspace/.local/tex`, in
addition to its system TeX installation. `lib.plotting` activates that path when
present. A fresh machine should install the packages above or provide equivalent
TeX tools. Missing TeX dependencies must be fixed before running the notebooks;
turning off `text.usetex` changes the requested plotting configuration.

## Validation order

Run the general notebook before creating or executing the two case studies:

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 python scripts/execute_notebook.py notebooks/00_theory.ipynb
```

The notebook executes the complete required regression suite, numerical checks,
nonvacuous tracking-bound fixture, independent-chain diagnostics, and measured
scalability checks. Only its final successful cell writes
`results/validation.json`. The certificate records a SHA-256 hash of the
numerical core and benchmark implementation. The case-study generator and
example notebooks reject a missing, failed or stale certificate through
`lib.validation.require_gate`; rerun the general notebook after changing the
core. Do not create the example notebooks to bypass a failure. Diagnose and
resolve it, or record the concrete theoretical or numerical blocker.

After a passing certificate, use the example generator provided in `scripts`
and execute each generated case notebook with the same runner:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/execute_notebook.py notebooks/01_array.ipynb
OPENBLAS_NUM_THREADS=1 python scripts/execute_notebook.py notebooks/02_field.ipynb
```

The example filenames and generator command are listed in the root README.
`execute_notebook.py` runs from the repository root, writes outputs back into
the notebook even on failure, and propagates an execution exception as a nonzero
exit. It does not treat partially executed outputs as a successful run. Run
cells in order with a fresh kernel when working interactively. Set
`OMP_NUM_THREADS=1` as well when comparing runtimes across BLAS implementations.
The GitHub Actions workflow tests the library and executes the general notebook
with a fresh TeX installation. That workflow is configuration; a CI pass is
established only after the remote job runs.

## Interpretation and numerical budgets

The gate validates the stated finite examples and workloads. It cannot prove
all parameter regimes, universal MCMC convergence, unrestricted scalability or
frequentist coverage from a single trajectory. Chain disagreement, effective
sample sizes and Monte Carlo errors must accompany sampled posterior summaries.
Credible threshold uncertainty and numerical optimization gaps are separate
quantities. A tight dual certificate does not certify a sampled region's
probability mass.

Conditional tracking-bound checks use fixed true noise variance, deterministic
inputs, a specified terminal time and the full retained history. Invalid bound
conditions stay visible; they are not clipped into passing bounds. Noise-learning
runs are evaluated separately. EM uses fixed stored priors in each fit and does
not replay historical corrections. Exact deterministic E-steps establish the
EM identities in the tests; sampled E-steps require numerical ascent checks and
must not inherit the exact theorem merely by using its formulas.

Tune baseline settings on separate validation trajectories with matched rank,
candidate counts and observation budgets. Oracle covariances are evaluation
references and never participate in tuning. Posterior-mean and modal comparisons
share a Bingham parameter sequence and fitted variance estimates. Report
held-out outcomes and model-mismatch results even when a reference outperforms
the proposed decision rule; no performance ranking is a validation assertion.

The notebook runner permits up to 3,600 seconds per cell by default. Set
`BAYESIAN_ST_NOTEBOOK_TIMEOUT` to a longer limit for larger repetition counts.
The array study performs substantially more high-dimensional posterior sampling
than the field study; modal-only timing is not representative of its full cost.

The recorded array study uses four spawned worker processes on a four-core
budget. Independent `(scenario, repetition)` jobs preserve the serial data and
random seeds exactly and share the same validation settings. Atomic JSON
checkpoints under `results/checkpoints/array` validate source/parameter hashes,
software versions, result checksums, row counts and scenario identities before
reuse. Completed jobs survive an interrupted notebook; aggregate tables retain
deterministic order. Set `workers=1` for serial execution. Checkpoints are local
scratch; the complete aggregate tables are committed.

The first serial array attempt exceeded its 1,800-second notebook-cell limit.
The final run uses parallel checkpoints and a 7,200-second per-cell limit, preserving
all scientific budgets. The field results were reused after verifying that all
eight field numerical dependency functions were unchanged verbatim, the old
provenance hash matched, and every saved table checksum was valid. That reuse is
recorded in `case2_config.json`; it does not represent a new field simulation.

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_parallel.py`
checks exact scientific equality between serial and two-worker execution on a
small smoke protocol, plus exact JSON restoration. This low-draw protocol tests
execution and caching only; it is not a posterior-accuracy study.

## Main figures and supplementary comparisons

The main application figures now show native complex n=16, rank-three array
responses, posterior output-power distributions, and a real n=8, rank-three
conditional field study. They include evolving response intervals, projected
posterior draws, repeated-data error distributions and the noise/drift terms
in the theorem condition. Dense method/parameter maps and the stationary
Gr(1,3) decision diagnostic remain supplementary. Coordinate projections are
visual summaries; inference uses every ambient dimension.

After the core gate passes, the example notebooks invoke these commands and
verify their cached tables and numerical dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_drift_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_array_story.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_array_envelopes.py
python scripts/make_example_notebooks.py
python scripts/execute_notebook.py notebooks/01_array.ipynb
python scripts/execute_notebook.py notebooks/02_field.ipynb
python scripts/summarize_results.py
python scripts/write_manifest.py
```

The drift study uses 64 independent trajectories and posterior computations
on the first 16 at six preassigned times. Its 672 reported summaries include
exactly transported static/known-motion controls, rather than treating these
coupled controls as independent Monte Carlo replications. Sampling uses two
chains, spectral and scalar-response diagnostics, and adaptive budgets capped
at 12,000 retained draws and 4,000 burn-in sweeps per chain. The representative
trajectory is always trial zero. Fast-drift failures are retained. This is a
conditional-observation experiment, distinct from the original passive field
reconstruction study.

The array story uses repeat zero of the original static/drifting cases with
identical data and tuned settings. All six targets receive new independent
chain evaluations. Its light response envelopes project an estimated 95%
state HPD region using the regional optimization; central response quantiles
are distinct and shown separately. The region threshold comes from the
original validated target and retains its Monte Carlo error. Optimization
gaps do not certify that threshold's probability mass or frequentist coverage.

All plots retain TeX and use figure/export DPI 200. The field response fans
join quantiles at reporting times for display, not as continuous-time or
simultaneous confidence bands. Clean-response uncertainty differs from noisy
observation prediction; both are retained in the raw tables.

## Deterministic spectral moments

`lib.spectral_series.rank_one_bingham` evaluates rank-one means and normalized
log integrals using the real/complex Dirichlet series, with adaptive truncation
and reported truncation bounds. `lib.moments.mean_bingham` selects this path
for rank/codimension one when no sampler-specific options are provided.
`backend='mcmc'` or explicit draws/burn-in retains sampling.
`lib.moments.log_normalizer` also uses exact complement duality for
codimension one and rejects unsupported general ranks.

```bash
OPENBLAS_NUM_THREADS=1 python scripts/check_spectral_series.py
OPENBLAS_NUM_THREADS=1 python scripts/check_paper_realification.py
```

The theory notebook includes this backend's independent analytic/integration
checks and measured timings through n=64. Error bounds cover truncation in
exact arithmetic, not floating-point rounding or a Fisher/Hessian approximation.
See [the appendix audit](latent_paper_audit.md) for why low-degree general-rank
Jack approximations and unrestricted realification were not adopted.

## Earlier comparisons and cache provenance

The two application notebooks now use larger 200 dpi canvases and exports,
with TeX retained. Figures show paired effect maps, individual paired trials,
spatial residuals, coverage forests, risk calibration, and computation tradeoffs.
Open circles in effect maps identify intervals containing zero. Intervals are
pointwise paired Student intervals, not simultaneous confidence bands.

Run `OPENBLAS_NUM_THREADS=1 python scripts/run_decision_study.py` to reproduce
the added stationary real Gr(1,3) diagnostic. The default study has 400 test
trajectories, 160 independent validation trajectories, two noise levels, and
five snapshot counts. GROUSE, scalar shrinkage, and diagonal loading use nine
validation candidates per applicable task and horizon. Selected boundaries are
recorded. All 4,000 posterior means undergo order-doubling checks; production
moment timing excludes those reference audits and region integrations.

Original array/field simulation tables are reused unchanged. The existing
extended-chain accuracy follow-up is also preserved:
notebooks check its numerical source dependencies and raw-table checksum while
excluding display-only plotting settings. Reformatting a figure does not require
repeating those Monte Carlo simulations. The new controlled-study cache has its
own source, configuration, version and table checksums.

The spectral-backend revision changes the core digest but leaves every numeric
dependency used by the historical application and decision tables unchanged.
`scripts/revalidate_cached_results.py --from-commit 864f8d1` verified original
table checksums, unchanged source files, and unchanged sampler/quadrature
definitions before associating those caches with the newly passed gate.
`results/revision_provenance.json` and each updated metadata file retain the
original digests and the proof. This is dependency-verified reuse, not a claim
that the historical Monte Carlo simulations were repeated. New uncertainty
studies were executed separately. Reproducing from scratch does not require
this migration step; absent caches are generated by the corresponding scripts.
