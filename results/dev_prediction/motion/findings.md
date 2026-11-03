# Conditional motion and memory findings

All test filters receive identical observations within each trajectory. Unknown-motion
settings are selected once using independent validation over a declared mixture.
Selected beta=0.85, W=12, lag=32, damping=0.5.
Boundary selections: beta=False, W=False, lag=True, damping=False.

The primary condition is fast drift with sensor-fixed inputs and terminal chordal
error; other paired intervals are exploratory pointwise intervals. No method is
selected using a hidden test rate. Partial motion is a declared synthetic control.

The primary paired gain (tuned forgetting minus causal velocity) is 0.03102
with 95% interval [0.02667, 0.03536]; it supports lower error.

See summary.csv and paths.csv for all regimes and initial/forecast errors;
cost.csv separates prediction from correction, and tuning.csv preserves every
validation candidate. bounds.csv retains failed sufficient conditions and excludes
learned rotations, finite windows, and corrupted observations from theorem claims.
Valid certificate violations: 0/1145.

posterior.csv and coverage.csv retain ball, HPD, clean-response and noisy-observation
coverage and widths separately, including every capped sampler failure. The terminal
innovation check uses independently sampled priors before correction. It is a trace
identity; it is not a chi-squared rule. Temporal innovation correlation and noise
learning are not evaluated by this bounded terminal pilot.

Method-shared held-out query evaluation: use posterior_paired_noise.csv and
coverage_paired_noise.csv for the final noisy-response coverage comparison.
The inherited raw summary used method-specific held-out noise; its original
rows and producer metadata remain preserved. This paired-evaluation repair
changes no inference, tuning, training data or posterior draws. See
paired_noise_evaluation.json for the independent query seed and checksums.
