# Manuscript changes supported by the development checkpoints

## Separate memory, prediction, and process uncertainty

The earlier fast-drift failure at beta=0.97 is not sufficient evidence that
identity-frame tracking is intrinsically poor. That setting retains mean age
32.3 samples. A validation-selected beta=0.85 retains mean age 5.7, while a
12-sample likelihood window has mean age 5.5. The new comparisons select one
policy over a declared mixture and include sensor-fixed inputs. Do not select
a distinct memory using each test trajectory's hidden speed.

Add the rolling fixed-noise likelihood product as a local-stationarity control.
Its prior removes the outgoing increment before storing the current Record;
the observation is then corrected once. Transported windows and historical
noise refits are different variants and are not implied by this implementation.

The causal velocity variant continues the shortest fitted Grassmann geodesic
between past corrected estimates. It uses initialization and cut-locus guards
and must be described as a learned local forecast, including its lag, damping,
and optional speed cap. Since its rotation depends on past noise, do not apply
the current deterministic-transformation tracking theorem to it without a new
proof. Marginal EW PCA with the same prediction still shares its mode.

The preselected fast-motion, sensor-fixed comparison is distinct from slow,
low-SNR, acceleration, turn and corruption stresses. An average improvement
does not justify saying velocity helps everywhere. Preserve transients and
regimes where common memory or velocity worsens error. The remaining selected
lag boundary limits conclusions about the best velocity history.

## State the assumed-density model precisely

For the transition proportional to exp(kappa trace(P G P_previous G*)), the
predicted first moment is exact conditional on the maintained previous law.
Rho is determined by kappa, dimension, rank and real/complex invariant measure.
Moment matching back to one Bingham is a forward-KL assumed-density projection,
not exact density closure or scalar power forgetting. The likelihood correction
is conjugate for that projected prior. Numerical inverse-moment error and
distributional approximation error require different checks.

The rank-one/codimension-one implementation passes independent tests. General
rank inversion remains unresolved and explicitly unsupported. The real-circle
reference fixture has a moment residual near 4e-17 while predicted-density TV
is about 0.0283. A small moment residual alone therefore cannot certify HPD
probability, tails, or cumulative filtering accuracy.

Matched-process coverage at known simulation kappa is finite low-dimensional
evidence for the approximation. It does not establish general-rank calibration,
learned-kappa calibration, or robustness to arbitrary motion and observation
mismatch. The fixed forgetting control is a diagnostic, not a tuned application
benchmark. Observation contamination causes credible undercoverage even for
the independent full-density filter, so that failure cannot be assigned solely
to Bingham projection.

## Keep numerical and empirical acceptance separate

Application posterior means are tuned on their actual task loss, alongside a
same-posterior modal decision, independently tuned scalar shrinkage, and the
appropriate application references. Loading and memory are Cartesian choices.
Report candidate counts, validation cost, boundary checks, and the fact that
the state/motion robustness cases reuse a matched-drift policy rather than
retuning with hidden test information.

Selected-validation or test sampler failures leave positive performance claims
unresolved, even when a paired gain interval excludes zero. Do not remove those
targets or treat additional chain draws as additional trajectories. The current
application intervals are exploratory pointwise intervals across independent
trajectories; they do not establish superiority across all conditions.

Report update, prediction, moment, decision and optimization costs with their
actual frequencies. Process inversion is nested in prediction cost, not added
again. These cloud wall-clock measurements are not universal complexity or
dedicated-machine runtime guarantees. Point tracking alone can still favor a
less expensive method.

Keep state balls, HPD regions, clean-response intervals and noisy-observation
intervals separate, including widths and finite-trial uncertainty. A large
measurement variance can mask state undercoverage in the noisy prediction.
Natural-parameter HPD loss thresholds have different units from geometric
radii and response widths. The innovation energy identity is not a chi-squared
test or full predictive covariance formula.

Noise learning uses complete stored-prior windows in separate ablations.
Safeguarded ascent concerns a fixed-prior surrogate; it does not establish
complete state-space likelihood ascent after approximate prediction. Noise
estimates must not be interpreted as estimates of unmodelled process motion.

Numerical commands, checkpoint outcomes, trial counts and evidence links are in
`results/dev_prediction/README.md`. Retain all unsuccessful and unresolved
outcomes when selecting final paper figures.
