# Prediction and process uncertainty on dev

This extension follows the supplied *Bayesian subspace tracking: Implementation
checkpoints for the L4DC experiments* (4 November 2026). The reviewed baseline
is `57f520cbe860b73e13d71bae5c31391e2a009482`, exactly the initial checkout.
The guide's scientific procedures are the user-authorized specification.
Its example development outcomes are reference data, not acceptance targets.
The attachment hash and original simulation seeds, versions, budgets, table
digests and notebook execution checks are recorded in
`results/dev_prediction/baseline/audit.json`.

All new results live under `results/dev_prediction/`. Historical tables,
figures, notebooks and certificates are preserved. The new general certificate
is separate because the numerical dependency set has changed. To execute the
historical examples with their original cache certificate, use the reviewed
baseline checkout; on dev, a stale historical certificate is deliberately not
silently accepted.

## Models and information

`lib/prediction.py` adds a queue of fixed-noise likelihood increments and a
causal short-geodesic forecast. A rolling likelihood window assumes local
stationarity and supplies no motion. Exponential forgetting changes retained
age; its long-run mean age is beta/(1-beta). The velocity predictor estimates
an orthogonal/unitary rotation from corrected subspaces through t-1, then
prepares the next prior before seeing the current measurement. Zero damping
reproduces identity prediction. Cut-locus safeguards and initialization use
identity. A learned rotation is random through past observations and does not
inherit the deterministic-transformation tracking theorem.

Exact motion and half-motion controls are labeled separately and use truth
only as declared synthetic controls. All applicable methods within a comparison
receive the same supplied frame. Sensor-fixed inputs are evaluated alongside
co-rotating inputs. Unknown-motion policies are frozen on independent validation
streams; no test speed or hidden state chooses memory or damping.

`lib/process_prediction.py` implements the isotropic transition proportional to
exp(kappa trace(P C)), C=G P_previous G*. Its normalizer is independent of C.
The contraction rho is derived from kappa and the real/complex invariant
measure, not fitted independently. The predicted first moment is exact given
the maintained preceding Bingham law. Its projection back into one Bingham
density is an assumed-density approximation. A small inverse-moment residual
does not establish accurate HPD mass or full-density prediction.

Only rank-one and codimension-one deterministic inversion are supported.
General-rank inversion is explicitly unresolved. Uniform and zero-process-noise
limits, field-specific measures, equivariance, numerical objective decrease,
moment residuals and independent transition samples are checked. A separate
real Gr(1,2) convolution reference tests the complete prediction and posterior
densities and credible summaries before calibration conclusions are drawn.
Moment matching and position/velocity tracking have precedents listed in the
guide; this extension does not claim these procedures as new contributions.

## Reproduction and validation

Run from the repository root using the documented Python and TeX environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_dev.py
```

`validate_dev.py` executes the complete existing general notebook with only its
output directory redirected. It saves the executed copy under the extension
directory and writes a passing certificate only after every cell succeeds.
The original notebook is unchanged. New prediction/process/application code,
all tests and benchmark implementations enter the numerical content hash.

Numerical correctness and empirical superiority have separate status. Full
tuning curves, selected boundaries, unsuccessful methods, invalid theorem
conditions, capped sampler failures and finite-repetition uncertainty must stay
visible. Coverage replicates are independent trajectories, not posterior draws.
Intervals across additional conditions are exploratory unless specified as
primary. Natural-parameter HPD thresholds, geometric ball radii and response
interval widths have different units and are not interchangeable.

Measurement-noise fitting is a separate stored-prior ablation. Neither a
stochastic prior approximation nor sampled E-steps establishes complete
state-space likelihood ascent. Innovation energy is compared with
n*s + v*(M-M^2)*v before assimilation; it is a trace identity, not a chi-squared
calibration test or a full predictive covariance.

The final checkpoint status, commands, measured outcomes, supported claims and
remaining limitations are in `results/dev_prediction/README.md`. All main and
supplementary figures retain `text.usetex=True` and 200 dpi.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_prediction_ablation.py --workers 2 --output-dir results/dev_prediction/reproduced_motion
python scripts/check_motion_paired_noise.py --output-dir results/dev_prediction/reproduced_motion
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_process_prediction.py --output results/dev_prediction/reproduced_process
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_application_ablation.py --workers 2 --output-dir results/dev_prediction/reproduced_applications
python scripts/summarize_prediction_ablation.py
python scripts/plot_prediction_ablation.py
python scripts/write_dev_manifest.py
```

The main motion comparison uses one common validation-selected policy across
the declared rate mixture. Its final streams are 3296411/3296412 for validation
and testing, distinct from the guide's development streams. The lag grid was
expanded from 16 to 32 using validation before final testing; a remaining edge
selection is reported. Application streams are 582031/582032, with a separate
noise-fit stream. Array and passive-field horizons are 32 and 128 respectively.
Application budgets are bounded pilots, so a capped accuracy failure can leave
a comparison unresolved even when its observed gain interval looks favorable.

The motion reproduction command uses a fresh directory because the original
producer preserves its captured gate and refuses to silently replace that
completed-run association. Verification of the delivered motion results uses
the source-checked `gate_binding.json`, not a rewrite of the captured metadata.

The final certificate adds a regression for relative repository paths. Original
run certificates remain in their metadata. `gate_binding.json` verifies each
numerical dependency and table before associating the unchanged simulations
with that final certificate; it does not describe additional Monte Carlo trials.

For the fixed-noise interfaces:

```python
from lib.prediction import PredictiveBinghamFilter
from lib.process_prediction import stochastic_predict

tracker = PredictiveBinghamFilter(8, 3, s=0.03, prediction='rolling',
                                  likelihood_window=12)
prior = tracker.predict()       # uses no current observation
tracker.update(u, v=v)          # assimilates the observation exactly once

# A is a real symmetric 3x3 rank-one concentration parameter.
prediction = stochastic_predict(A, k=1, kappa=20.0)
assert prediction.fit.converged
# prediction.prior is an approximate density, with an accurately fitted moment.
```
