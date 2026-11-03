# Recorded results

`validation.json` records the general notebook's successful finite-budget
validation, numerical versions and content hash. `*_config.json` and `*.csv`
contain independently reproducible case configurations and raw results;
configuration files also checksum the tables. PDF figures are retained for
export, and the executed notebooks embed rendered plots.

The required general notebook passed before the example notebooks were
created. Its scope includes 39 checks (37 tests and two subtests), rank-three modal scaling through n=128,
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
