# Recorded results

`validation.json` records the general notebook's successful finite-budget
validation, numerical versions and content hash. `*_config.json` and `*.csv`
contain independently reproducible case configurations and raw results;
configuration files also checksum the tables. PDF figures are retained for
export, and the executed notebooks embed rendered plots.

The required general notebook passed before the example notebooks were
created. Its scope includes 47 checks (45 tests and two subtests), rank-three modal scaling through n=128,
independent real/complex high-concentration chains, deterministic EM ascent,
certified regional extrema, and controlled fixed-time tracking checks.
The zero-violation check uses 128 independent noise trials at a predeclared
terminal time; its one-sided 95% binomial upper bound is about 0.0231.

Case-study coverage, performance, and numerical diagnostics are experimental
outcomes rather than tests that require a preferred ranking. See the executed
notebooks and raw tables. Sampled high-dimensional EM ascent is unverified;
low-ESS scores and capped sampler failures, when present, remain visible.

The field “Matched” scenario matches the observation model but still has
unmodelled motion. Its nominal 95% ball and HPD regions have 0/8 terminal
coverage. This observed overconfidence is a substantive limitation of the
approximate prediction/tuning protocol, not a calibrated-coverage result.

The original array budget flags 22/168 posterior evaluations. Separate
longer-chain checks passed for all 22 targets; the largest absolute relative
change in population beamformer output power was 0.366%. Both original flags
and follow-up diagnostics are retained in `case1_accuracy.csv`/`.json`.

The main figures additionally use `array_story_*` (native complex n=16, rank 3)
and `drift_study_*` (real n=8, rank 3). The latter has 64 modal trajectories,
posterior evaluations on the first 16, and 672 reported summaries passing the
predeclared diagnostics. Static/known-motion controls are orthogonally coupled.
Slow identity prediction has valid modal certificates but 0/16 terminal ball
coverage; this illustrates a distinction between a modal-error theorem and
credible calibration. Fast nominal-frame protocols fail the theorem condition.
The full assumptions and qualifications are in `docs/modeling_and_drift.md`.

`array_envelope_*` projects estimated 95% state HPD regions into response bounds
at evaluated steering angles. This differs from central pointwise response
quantiles. Threshold sampling error remains separate from the recorded
optimization gaps. Neither is a frequentist confidence guarantee.

`spectral_series_checks.json` records independent checks for the new real/complex
rank-one deterministic backend. `paper_audit_checks.json` records the exact
realification counterexample and its Monte Carlo corroboration.
`revision_provenance.json` records dependency-verified reuse of unchanged
historical tables after the core digest changed; it does not describe a rerun.
