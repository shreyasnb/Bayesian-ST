# Bayesian subspace tracking

Implementation and simulations for **Bayesian Subspace Tracking with Grassmann Bingham Distributions** (supplied L4DC 2027 draft). The code separates exact conjugate parameter updates from numerical posterior moments, credible regions, and frozen-prior noise fitting.

## Run

Use Python 3.12 and a working TeX installation (`latex`, `dvipng`, `type1cm`, and `type1ec`). Every Matplotlib figure uses `text.usetex = True`; there is no fallback to non-TeX rendering.

```bash
bash scripts/setup.sh
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
python scripts/execute_notebook.py notebooks/00_theory.ipynb
python scripts/make_example_notebooks.py
python scripts/execute_notebook.py notebooks/01_array.ipynb
python scripts/execute_notebook.py notebooks/02_field.ipynb
```

The general notebook must pass before the example generator or example simulations can run. Its certificate hashes the numerical core, benchmark code, tests, controlled theory studies, and theory notebook generator. A source change invalidates the certificate. Example tables have separate source/configuration hashes and checksum-verified caching. Set `refresh=True` in an example to resimulate. Regenerating notebooks replaces their saved outputs.

See [reproducibility](docs/reproducibility.md) for installation, numerical budgets, and CI. The executed example notebooks use eight repetitions per scenario, 72 training observations, and adaptive independent chains with a predeclared cap. The array runner uses four workers and resumable checkpoints; separate longer-chain follow-ups retain and investigate original capped failures. These are reproducible pilot studies; increase repetitions for precise lower-tail and coverage claims.

## Notebooks

| File | Purpose |
| --- | --- |
| [00_theory](notebooks/00_theory.ipynb) | Debugging and validation: likelihoods, excitation, counterexamples, drift bounds, posterior risks, credible geometry, EM, sampling, and scaling |
| [01_array](notebooks/01_array.ipynb) | Complex array beamforming: orientation, snapshots, drift, held-out power, SINR, mismatch, regional leakage, coverage, and cost |
| [02_field](notebooks/02_field.ipynb) | Passive field reconstruction, known calibration patterns, separate noise fitting, predictive scores, mismatch, coverage, and cost |

Plots use larger canvases, paired-effect heatmaps, spatial residual maps, coverage forests, and cost–accuracy comparisons. Both display and export resolution are 200 dpi, with TeX fonts and PDF exports. Executed notebooks embed the corresponding plots. Raw tables, configurations, numerical certificates, and PDF figures are under `results/`.

## Library

`lib/` contains reusable implementation files. Benchmark algorithms remain in `benchmarks/case1/` and `benchmarks/case2/`.

```python
import numpy as np
from lib.filter import BinghamFilter
from lib.moments import sample_bingham

rng = np.random.default_rng(7)
tracker = BinghamFilter(n=8, k=3, s=0.2, tau2=1.5, beta=0.98)
for u in rng.normal(size=(20, 8)):
    basis = tracker.update(u)  # optionally supply known orthogonal motion G
posterior = sample_bingham(tracker.A, 3, rng, draws=12000, burnin=4000)
# Inspect posterior.diagnostics and independent chains before using its mean.
```

For known projected inputs, omit `tau2` and call `update(u, v=v)`. For complex data use `complex_data=True`; the likelihood correctly omits the real Gaussian factor of one half. `lib.em.fit_noise` pairs each observation with its own stored predicted prior and supports deterministic likelihood safeguards. `lib.theory.regional_bound` returns a feasible projector and dual bound, including tied-eigenvalue cases. Small real rank-one quadrature and adaptive continuous credible thresholds provide independent references.

## Scientific interpretation

- With uniform initialization, fixed noise, and matching forgetting, the marginal Bingham mode equals EW PCA. The experiments explicitly check this equivalence. The distribution contributes posterior decisions, directional risk, region geometry, and noise updates.
- Forgetting is approximate prediction. The fixed-time tracking bound uses true, fixed noise and full historical excitation; invalid conditions are shown rather than converted into bounds.
- Exact EM ascent holds for fixed stored priors with exact E-steps. Sampled high-dimensional fits are labelled as having **unverified ascent**.
- MCMC diagnostics and credible-threshold errors are reported separately from regional optimization gaps. A capped failed diagnostic stays visible in the results.
- Credible posterior mass does not guarantee frequentist coverage under deterministic paths, forgetting, or model mismatch. Coverage plots include finite-repetition uncertainty.
- GROUSE, EW PCA, rolling PCA, and loaded-covariance MVDR are implemented with citations and matched validation budgets. GREAT, GeRoST, and a paper-specific Bayesian array tracker are not silently approximated or relabelled.

[Claims and checks](docs/claims.md) map equations to validation. [Draft notes](docs/draft_notes.md) describe numerical clarifications to Algorithm 1 and limitations. [Simulation protocol](docs/simulations.md) specifies data generation, tuning, and interpretation. See [recorded evidence](results/README.md) and [simulation findings](docs/results.md) for results and limitations.

See [interpretation](docs/interpretation.md) for what the comparisons support, the stronger shrinkage/loading controls, and when the posterior computation is worthwhile. The added controlled study uses exact low-dimensional quadrature, 400 test trajectories and 160 independent validation trajectories; it is distinct from the drifting application pilots.
