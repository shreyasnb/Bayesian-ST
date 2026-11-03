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
