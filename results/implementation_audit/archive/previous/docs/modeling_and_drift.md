# What identity prediction does and does not assert

Setting `G_t = I` is exact when the state is stationary under the deterministic
motion model. With physical drift it is a nominal frame choice: no known motion
is supplied. It does not integrate over unknown motion. Multiplying the previous
parameter by `beta` is an approximate prediction rule, not in general the
predictive law of a specified Grassmann process-noise model.

## Three different claims

1. **Conjugate correction and known-motion prediction.** These identities hold
   under their stated Gaussian observation and deterministic-motion models.
   Setting a nominal frame to identity does not alter the correction identity.
2. **Theorem 5's modal error bound.** This real conditional-model theorem allows
   residual deterministic physical motion, charging it through `d_t`. It gives
   a fixed-time certificate only if `epsilon < gamma/2`. It neither promises
   that this condition always holds nor certifies posterior credible coverage.
3. **Credible-region and predictive-risk calibration.** These are conditional on
   the maintained distribution. Under unmodelled motion that distribution can
   be overconfident, even when its mode tracks reasonably and Theorem 5 is
   informative. Numerical evaluation accuracy cannot repair a wrong state model.

The original passive field coverage result is not a test of Theorem 5: its
observations marginalize the generating vector, while that theorem assumes
known projected inputs and the true fixed noise variance. If a passive-model
tracking certificate is required, its covariance/noise concentration argument
must be established separately rather than borrowing the conditional bound.

## A useful regime and a failure regime

The new conditional experiment uses all eight coefficient dimensions, rank
three, 128 measurements, norm-two preassigned input cycles, true variance 0.03,
and a Haar initial state. Fixed orthogonal motion rotates the same canonical
input and noise streams, providing paired exact and nominal-frame controls.
Slow and fast residual rates are 0.003 and 0.025, and forgetting is fixed at
0.97 before evaluation. The stronger excitation/SNR/history was designed from
the sufficient condition, not selected to achieve favorable test coverage.
The audit also records that the old 72-step, variance-0.2 conditional setting
has an invalid condition even with exact frame motion.

At the terminal time, across 64 independent trajectories:

| Prediction | Average modal error | Average valid bound | Valid conditions |
| --- | ---: | ---: | ---: |
| Exact frame | 0.057 | 0.369 | 64/64 |
| Identity, slow drift | 0.198 | 0.836 | 64/64 |
| Identity, slow drift, forgetting | 0.124 | 1.005 | 64/64 |
| Identity, fast drift | 1.164 | unavailable | 0/64 |
| Identity, fast drift, forgetting | 0.599 | unavailable | 0/64 |

The distance ceiling is 1.732. No valid terminal certificate was violated.
The slow identity-frame bound is therefore nonvacuous, although conservative.
Forgetting lowers actual error and the drift term here, but raises the
noise-to-gap ratio by reducing retained excitation. A better point estimate
can thus have a looser sufficient bound. The figures show the two terms and
this tradeoff explicitly. Results at several times are not a simultaneous
confidence guarantee; `delta=0.05` is specified separately at each fixed time.

Posterior calculations use the first 16 preassigned trajectories, two chains,
separate scalar-functional diagnostics, and adaptive budgets. All 672 reported
protocol/time summaries met those diagnostics. Static/known-frame posterior
draws are transported exactly and are not extra independent replications.
At the terminal time the nominal 95% subspace ball covered 15/16 states under
exact prediction, 0/16 with slow drift and identity/no forgetting, and 15/16
with slow drift and forgetting. Both fast nominal-frame cases had 0/16.
Sixteen trials give wide Wilson intervals. The favorable slow-forgetting result
is empirical evidence for that protocol, not a calibration theorem.

The coexistence of a valid modal bound and 0/16 credible coverage is the central
lesson: these are different mathematical statements. It does not make all the
theory vacuous. It does rule out interpreting the approximate maintained law as
a generally calibrated posterior under arbitrary unknown drift.

## Suggested replacement wording in the draft

For Section 3.2, after introducing identity motion and forgetting:

> When frame motion is unavailable, identity is used as a nominal frame. If
> the physical subspace varies beyond that frame, the forgetting update is an
> approximate prediction rule rather than the exact prediction of a specified
> stochastic motion model. Theorem 5 accounts for residual variation in the
> modal estimate when its sufficient condition holds. Credible summaries refer
> to the maintained Bingham distribution; their calibration for the physical
> state must be assessed under the stated state process.

For the experiments, distinguish **matched observation covariance** from
**correct state dynamics**. Report the theorem condition alongside its bound,
keep invalid regimes, and do not label missing certificates as successful
bounds. Retain credible undercoverage even when reconstruction error is good.
Do not claim that the nominal credible ball is the theorem's tracking radius.

## Best next modeling step

If calibrated Bayesian uncertainty under unknown motion is a central claim,
specify an actual transition kernel. For example, draw a skew-symmetric random
increment with an independently calibrated covariance and apply its exponential
to the supplied frame. Predict by integrating that kernel against the previous
posterior. Exact prediction generally gives a mixture, so Bingham mixtures,
particles, or a separately validated moment projection are natural options.
The conjugate correction remains useful within that extension.

A single Bingham projection can preserve tractable storage, but its approximation
error and region calibration must then be tested. Learning process uncertainty
is different from fitting measurement noise or choosing beta for reconstruction
MSE. Inflating a radius after seeing test truths, clipping concentration, or
renaming the failure does not establish the desired guarantee.

For the current draft, the defensible immediate route is to retain the exact
results and conditional modal theorem, label unknown-motion prediction as
approximate, and present uncertainty as model-conditional. Add a transition
model before making a stronger unknown-drift calibration claim. If that claim
was intended as the main novelty, the present experiments reveal a substantive
modeling gap that requires new work, rather than a cosmetic caveat.
