# Application checkpoint findings

The fixed-noise application pilot completed with eight independent validation
trajectories and sixteen independent test trajectories per condition. A single
policy per method/application was frozen on matched-drift validation data before
any test evaluation. Numerical checks and empirical performance have separate
statuses.

| Check | Status | Evidence |
| --- | --- | --- |
| Cartesian loaded-covariance memory/loading tuning | Implemented and checked | `tuning.csv`, `validation_trials.csv` |
| Expanded GROUSE/window families and validation-only boundary rule | Implemented and checked; selected open parameters are interior | `predeclared_config.json`, `selected.json` |
| One correction and retained observation-time priors | Implemented and checked | `tests/test_application_ablation.py` |
| Matching marginal mode versus equally predicted EW PCA | Implemented and checked | `metrics.csv`, maximum chordal discrepancy 8.17e-8; projector-matrix discrepancy checked below 1e-8 |
| Primary passive field comparison | Empirically inconclusive | `paired.csv`: every paired 95% interval contains zero; all selected-validation and primary-test posterior diagnostics pass |
| Primary complex array comparison | Numerically unresolved | `tuning.csv`: 8/8 selected validation failures; `metrics.csv`: 16/16 primary test failures at the fixed sampling cap |
| Separate rank-one frozen-prior noise fitting | Implemented and checked; not state-space likelihood ascent | `noise.csv`: all 96 fitted windows pass the deterministic frozen-objective safeguard |

The primary array mean held-out powers are 0.5367 for posterior means, 0.5369
for tuned scalar shrinkage, 0.5548 for loaded covariance, and 0.5563 for the
same-posterior mode. These figures are retained exploratory outputs, not a
supported ranking: posterior accuracy is unresolved under the deliberately
bounded 4,000-draw cap. A favorable loaded-covariance paired interval must not
be used as a positive main-paper claim. Longer validated chains or a validated
general-rank deterministic moment backend are needed before such a claim.

The primary field mean reconstruction losses are 0.7410 for the posterior mean,
0.7432 for its mode/tuned EW PCA/scalar shrinkage, 0.7253 for rolling PCA, and
0.7711 for GROUSE. All corresponding paired intervals include zero. This pilot
supports neither a positive gain claim nor a reliable loss ranking. Field matched,
partial-motion and sparse-corruption targets all pass the recorded posterior
diagnostics; unequal-power field data have 7/16 capped failures.

Across all regimes there are 160 posterior test targets and 102 capped failures:
95/96 array targets and 7/64 field targets. There are 39/80 validation posterior
failures across all candidates. Sampling-budget failures are distinct from model
misspecification and credible undercoverage. The budgets were not increased
based on test outcomes; every failure stays visible.

Cost is measured on the recorded shared machine with two application workers
and one BLAS thread per worker. Terminal posterior moments average about 1.92 s
in the primary array regime and 1.25 s in the primary field regime. Cumulative
point-method times are roughly 0.003–0.020 s at these horizons. These are
implementation/session measurements, not asymptotic speed claims or a favorable
compute tradeoff established by this pilot. `tails_cost.csv` reports output-power
loss tails and array-only lower-tail SINR separately.

The separate real n=3, rank-one noise ablation uses the selected field beta,
eight trajectories and four fits per trajectory. Mean terminal learned
measurement variances are approximately 0.171 under stationarity, 0.196 under
slow drift and 0.330 under fast drift, against true variance 0.2. This illustrates
that a learned measurement variance can absorb missing dynamics. It is not a
process-dispersion estimate or full state-space likelihood ascent result.

The recorded reporting revalidation corrects an application key in the SINR
summary merge and adds selected-validation accuracy qualifications. All raw
losses, numerical-source files, selected policies, seeds and budgets are
unchanged; the original generation source hash and captured run gate are retained
in `config.json`. The immutable predeclared configuration remains the one saved
before validation. A later gate association is recorded separately.
