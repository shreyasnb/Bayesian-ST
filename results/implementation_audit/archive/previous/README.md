# Bayesian subspace tracking

Implementation and simulations for **Bayesian Subspace Tracking with Grassmann Bingham Distributions**. The current `test` branch redesigns the beamforming and contact-normal notebooks around the supplied fifth draft's models, theory and appendices. Its empirical simulation section does not define these experiments.

Start with the executed [beamforming notebook](notebooks/01_array.ipynb) or [contact-normal notebook](notebooks/03_contact.ipynb). Each develops the observation model, posterior geometry, decision, independent comparisons and computational cost in order. Genuine spherical Bingham densities, explicitly labeled complex density sections, physical contact trajectories and uncertainty displays replace the earlier dense collections of parameter scans. Figures use LaTeX, 200 dpi and PDF/PNG exports.

The [findings](results/simulation_redesign/findings.md), [scientific protocol](docs/simulation_redesign.md), [notation map](docs/simulation_redesign_notation.md) and [independent checks](docs/simulation_redesign_checks.md) state what each comparison establishes. All benchmark outcomes and numerical/local-approximation limitations remain available, including inconclusive comparisons and losses.

## Run

Use Python 3.12 and working TeX tools (`latex`, `dvipng`, `type1cm`, `type1ec`). Every Matplotlib figure requires `text.usetex=True`. Set up with `bash scripts/setup.sh`; activate `.venv` if that script created it.

To verify and display the committed paper evidence:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/validate_simulation_redesign.py
python scripts/plot_geometry_studies.py --profile paper
python scripts/summarize_simulation_redesign.py --profile paper
python scripts/make_geometry_notebooks.py
python scripts/execute_geometry_notebooks.py
```

See the [complete reproduction commands](results/simulation_redesign/README.md) for all quick/paper simulations, scaling and handoff checks. The general validation runs the complete theoretical notebook and regression suite into a new directory before paper simulations and example generation. Changes to scientific sources invalidate its certificate; component tables, figures, findings and saved notebooks have separately verified provenance. Regeneration replaces saved outputs, so preserve the original evidence and use a fresh output directory when changing a scientific configuration.

## Notebooks

| File | Purpose |
| --- | --- |
| [00_theory](notebooks/00_theory.ipynb) | General validation: likelihoods, excitation, drift bounds, posterior risks, regions, EM, sampling and scaling; original preserved |
| [01_array](notebooks/01_array.ipynb) | Proper-complex passive beamforming: directional uncertainty, posterior decisions, independent 8-/16-sensor comparisons and scaling to 128 |
| [02_field](notebooks/02_field.ipynb) | Historical field reconstruction and drift study; preserved |
| [03_contact](notebooks/03_contact.ipynb) | Known-input contact projection: force balance, spherical normal uncertainty, soft decisions, physical tracking and generalized-coordinate examples |

Each application has six main figures and links to detailed supplementary evidence. The contact examples use an ideal maintained bilateral constraint with known isotropic resistance. Higher-dimensional examples test one linear constraint in generalized coordinates; they do not assert validation of a complete robot controller.

## Library and benchmarks

Reusable code is in `lib/`, with original benchmark files in `benchmarks/case1/` and `benchmarks/case2/`. The new studies use `array_geometry_study.py`, `contact_geometry_study.py`, `anisotropy_study.py` and `simulation_scaling.py`. Presentation is separate from simulation and inference.

```python
import numpy as np
from lib.filter import BinghamFilter
from lib.moments import sample_bingham

rng = np.random.default_rng(7)
tracker = BinghamFilter(n=8, k=3, s=0.2, tau2=1.5, beta=0.98)
for u in rng.normal(size=(20, 8)):
    basis = tracker.update(u)
posterior = sample_bingham(tracker.A, 3, rng, draws=12000, burnin=4000)
# Inspect diagnostics and independent chains before using sampled moments.
```

For known projected inputs, omit `tau2` and call `update(u, v=v)`. For proper-complex data use `complex_data=True`; its likelihood omits the real Gaussian factor of one half. The adaptive spectral backend supplies real/complex rank-one means and normalizers, with codimension-one duality. Unsupported general-rank calculations are not replaced by unchecked truncations. See the [computational audit](docs/latent_paper_audit.md).

The beamforming comparisons include independently validated continuous covariance loading, data-adaptive Ledoit--Wolf shrinkage, regularized modal covariance and GROUSE. Contact comparisons include conditional projector least squares, residual PCA, rank-informed ridge regression, the same-parameter local Gaussian and a recursive extended Kalman filter. All use their stated observations and independently selected parameters; covariance-only methods do not acquire an invented projector prediction.

## Interpretation

- With uniform initialization, fixed noise and matching memory, the marginal Bingham mode equals weighted covariance PCA; the conditional mode equals spectral projector least squares. The posterior contributes directional means, uncertainty and task decisions.
- The draft's same-mode/same-distance-distribution pair shows why a radius alone cannot determine directional risk or a task-optimal action. Selected-data identities and random-training performance are reported separately.
- Credible posterior mass, repeated-data coverage, mean accuracy and tail precision are different checks. A local Gaussian can fail its chart-validity check even with finite numerical output.
- Discounting is approximate prediction. A supplied coordinate rotation is different from unknown state drift. The printed modal certificate has explicit excitation and drift conditions; invalid cases remain visible and do not certify credible coverage.
- Scaling separates parameter corrections, dense eigenanalysis, moments, solves and optional uncertainty reports. A point estimate can be substantially cheaper; the results identify when the added distribution changes a decision.
- The current comparisons hold noise fixed. The draft's fixed-block EM objective differs from the older frozen-per-observation-prior implementation in `lib.em`; the new notebooks do not claim a noise-learning advantage.

## Historical studies

Earlier executed notebooks, numerical-certificate source trees and 1,020 historical result artifacts are preserved by the [archive manifest](results/simulation_redesign/archive/preservation.json). The original `main` and `dev` branches are unchanged. Historical generators reconstruct their own notebook versions and should not overwrite the current walkthroughs.

The previous [unified beamforming study](results/beamforming_unified/README.md), [contact tracking study](docs/contact_tracking.md), [contact pilot](results/contact_pilot/findings.md), [development checkpoints](docs/development.md) and [field results](docs/results.md) remain available under their original assumptions and producer versions. Stochastic process approximation and joint-parameter experiments there are additional historical studies, rather than claims about the fifth draft's Algorithm 1. See also [modeling and drift](docs/modeling_and_drift.md), [draft notes](docs/draft_notes.md), and the earlier [reproducibility guide](docs/reproducibility.md).
