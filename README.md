# Bayesian subspace tracking

Implementation and simulations for **Bayesian Subspace Tracking with Grassmann Bingham Distributions** (supplied L4DC 2027 draft). The code separates exact conjugate parameter updates from numerical posterior moments, credible regions, and frozen-prior noise fitting.

The `test` branch adds an executed [passive beamforming walkthrough](notebooks/01_array.ipynb): eight-sensor complex rank-one estimation, Fisher information, a passive covariance tracking certificate, posterior decisions, functional uncertainty, joint power inference and stochastic prediction. Follow its [reproduction commands](results/beamforming_unified/README.md), [findings](results/beamforming_unified/findings.md) and [claim map](docs/beamforming_claims.md). The earlier complex rank-three and real low-data array studies remain in a source-verified historical appendix. The older generic example generator below reconstructs the historical array notebook; use `make_beamforming_notebook.py` for the new walkthrough.

## Run

Use Python 3.12 and a working TeX installation (`latex`, `dvipng`, `type1cm`, and `type1ec`). Every Matplotlib figure uses `text.usetex = True`; there is no fallback to non-TeX rendering.

```bash
bash scripts/setup.sh
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
python scripts/validate_beamforming.py
python scripts/plot_beamforming_unified.py --profile paper
python scripts/summarize_beamforming.py --profile paper
python scripts/make_beamforming_notebook.py
python scripts/execute_beamforming_notebook.py
```

These commands validate and render the committed unified paper evidence. See the [full protocol](results/beamforming_unified/README.md) to regenerate every quick/paper simulation and its findings before rendering. The general notebook must pass before formal array simulations and notebook generation; its isolated certificate hashes the numerical core, benchmarks, tests and new scientific sources. A source change invalidates the certificate. Component tables have separate source/configuration hashes and verified caching. Regenerating a notebook replaces its saved outputs. The original `make_example_notebooks.py` workflow reconstructs historical examples and should not replace the new array walkthrough.

See [reproducibility](docs/reproducibility.md) for installation and historical numerical budgets. The historical array pilots use eight repetitions per scenario, 72 observations and capped adaptive chains, with separate longer-chain investigations. The unified decision study instead uses 128 validation and 256 test trajectories per condition/noise; its parameter-learning and full-density-reference cohorts have separately stated smaller budgets. Earlier executed field/contact notebooks and their original certificates are preserved rather than silently rerun by the new protocol.

## Notebooks

| File | Purpose |
| --- | --- |
| [00_theory](notebooks/00_theory.ipynb) | Debugging and validation: likelihoods, excitation, counterexamples, drift bounds, posterior risks, credible geometry, EM, sampling, and scaling |
| [01_array](notebooks/01_array.ipynb) | Passive complex beamforming: information, tracking, posterior decisions, uncertainty, unknown powers, process approximation and cost; historical studies in the appendix |
| [02_field](notebooks/02_field.ipynb) | Known-input prediction and drift bounds; supplementary passive reconstruction, noise fitting, predictive scores, coverage, and cost |
| [03_contact](notebooks/03_contact.ipynb) | Contact walkthrough: force balance, null-space identification, causal motion prediction, response uncertainty, and independent trajectory checks |

Main plots show response uncertainty, projected posterior draws, decision distributions, and the conditions behind the tracking bound. Detailed heatmaps, spatial residuals, coverage forests, and cost comparisons remain supplementary. Both display and export resolution are 200 dpi, with TeX fonts and PDF exports. Executed notebooks embed the corresponding plots. Raw tables, configurations, numerical certificates, and PDF figures are under `results/`.

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

The latest [computational audit](docs/latent_paper_audit.md) checks the supplied latent-information paper's appendices. A new adaptive spectral backend evaluates real/complex rank-one means and normalizers (and codimension-one quantities by duality). It rejects unresolved truncation budgets and does not substitute unchecked general-rank Jack truncations.

The historical array figures emphasize **16-channel rank-three prediction bands**; the field/contact studies also use higher-dimensional states. The new array narrative above uses a native complex eight-channel rank-one model. [Modeling and drift](docs/modeling_and_drift.md) explains why identity-frame modal certificates and credible-region calibration are distinct, with explicit informative and failed regimes.

The `dev` branch adds the supplied implementation checkpoints: rolling likelihood memory, causal Grassmann velocity prediction, and a separate stochastic assumed-density predictor. See [development](docs/development.md) and the [checkpoint handoff](results/dev_prediction/README.md). Its certificate, tuning, independent holdouts, numerical failures and figures are isolated under `results/dev_prediction/`; historical notebooks and results are preserved.

The [frozen contact pilot](results/contact_pilot/findings.md) tests physically generated constrained motion on an idealized maintained bilateral guide with known resistance. Its notebook is [archived](results/contact_tracking/archive/03_contact_f09d3fe.ipynb); its results and the field study are preserved. The historical `make_contact_notebook.py` generator reconstructs that original pilot and replaces the contact notebook, so use the tracking generator below for the revised walkthrough.

The revised [contact notebook](notebooks/03_contact.ipynb) connects the known null-space projection model to one mechanical trajectory before comparing independent holdouts. It separates stale-state bias, causal motion prediction, spread calibration, an explicit Bingham process filter and an equally informed recursive Gaussian. Five main figures replace the earlier dense collection of scans. Temperature and transition parameters are defined where they first appear; all plots use TeX and 200 dpi. No code from the supplied style-reference notebook is used.

See the [tracking protocol](docs/contact_tracking.md), [findings](results/contact_tracking/findings.md), and [process/shape assessment](results/contact_uncertainty/findings.md). Reproduction is isolated from the historical results:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/validate_contact_uncertainty.py
OPENBLAS_NUM_THREADS=1 python scripts/run_contact_tracking.py --workers 4
OPENBLAS_NUM_THREADS=1 python scripts/run_contact_tracking_uncertainty.py --workers 4
OPENBLAS_NUM_THREADS=1 python scripts/run_contact_matched.py --skip-development --workers 2
python scripts/summarize_contact_tracking.py
python scripts/make_contact_tracking_notebook.py
OPENBLAS_NUM_THREADS=1 python scripts/execute_notebook.py notebooks/03_contact.ipynb
```

The numerical gate runs the complete general notebook and all regression tests into a new directory before the contact notebook can be generated. Held-out selections, source hashes, report failures, raw paired results and timing scopes remain recorded. Mean matching does not imply exact density closure or calibrated physical-contact coverage. The seven-dimensional numerical feasibility check is a separate bounded follow-up; no new robot-arm or controller validation is claimed.
