# Mathematical specification and interpretation

The current study uses **v5.tex** as the main manuscript and
**supplementary(14).tex** as its supplement, as confirmed by the author.
Their SHA-256 values and the guide hash are recorded in
[specification.json](../results/beamforming/data/specification.json).
The supplied numerical prose is not a source of new simulation results.
Neither manuscript has been edited.

## Implemented conventions

| Manuscript calculation | Implementation and independent check |
| --- | --- |
| Real and proper-complex Gaussian corrections | `lib/final_theory.py`, `tests/test_final_theory.py`; complex precision has no real factor of one half |
| Discounted, transported recursion | Exact singleton/block reconstruction, including changing observation variances and noncommuting frame changes; no division by block length |
| Information and KL | Independent scores, Hessians, exact KL identities, degenerate excitation, and common-frame reference gaps |
| Sharp modal certificate | Midpoint initialization radius; aggregate conditional drift norm; eligible only when epsilon is below half the reference gap |
| Posterior curvature | Invariant-density Hessian distinguished from graph-density volume correction and from repeated-data estimator spread |
| Fixed-boundary variance estimation | `lib/block_em.py`; likelihood constants, a fixed preceding prior, pooled zero-signal boundary and joint variance constraints |
| Causal scale adaptation | `lib/causal_variance.py`; completed disjoint blocks fit once; estimates affect future corrections only |
| Task prediction | Mean projector and response uncertainty; separate state, clean-response, noisy-response and normalized predictive-score diagnostics |

The proper-complex **marginal** tracking expression is implemented from the
supplement. A separately derived complex conditional helper in the historical
code is not represented as a theorem in the supplied manuscripts.

## Meaning of identity transport

Setting G_t to the identity is valid as an inference choice when motion is
unknown. It does not assert that a moving physical subspace is stationary.
The manuscript explicitly describes forgetting under unknown motion as an
approximate prediction. The modal tracking certificate includes a deterministic
drift contribution and can be ineligible when motion, short data or weak
excitation dominates the reference gap. Ineligibility is not a theorem
violation and does not certify uncertainty coverage.

Exact correction for a chosen Bingham prior and calibrated repeated-data
coverage of an approximate dynamic filter are different statements. Coverage
failures under unmodeled motion therefore limit a dynamic uncertainty claim;
they do not refute conjugacy, Fisher identities or the conditional modal
bound. Known wrist-frame motion is transported for every eligible method;
unknown physical curvature remains drift. A random-motion Bayesian prediction
would generally require a Bingham mixture or another approximation.

The contact application assumes maintained contact, known isotropic resistance
and independent direct velocity errors. Under a known anisotropic resistance,
a coordinate change must transform both the input/output relation and the
sensor covariance. Raw isotropic velocity noise generally becomes anisotropic
after that transformation. Correct transformed noise and uncorrected noise are
separate mismatch cohorts here; variance fitting does not restore an incorrect
likelihood or a lost contact constraint.
The plant follows a preassigned force program. Estimated response maps and soft
commands are evaluated on that common recorded stream; estimated commands do
not regenerate separate closed-loop trajectories. This tests prediction and
decision costs, not closed-loop stability or robot safety.

## Algorithm and manuscript clarifications

The supplied variance section already specifies a preceding fixed prior and
future use of fitted variances. The implementation makes the schedule explicit:
save the prior before a disjoint block, align its observations and inputs in a
common known frame, fit the stationary-block likelihood, retain the existing
streaming parameter, and apply the new scales to the next observation. Do not
re-add the fitted block posterior. If Algorithm 1 is expanded to include
learning, include these operations and the learning cadence explicitly.

The fixed-variance modal bound does not cover this empirical-Bayes recursion.
Within-block physical drift is an approximation and can inflate fitted noise.
Variance floors prevent singular likelihoods; they are not a cure for an
incorrect state model. A finite EM budget and approximate moments do not by
themselves guarantee an exact EM ascent statement.

All new policy selection uses independent held-out **measured** observations.
Simulated truth, drift and oracle information are reserved for evaluation and
theoretical validation. Validation and test episodes use disjoint fixed seed
sets, and test outcomes do not change the tuning grid or repetition count.

## Scope of possible advantages

The marginal mode equals covariance PCA under matching fixed scales, memory
and a uniform prior. Finite-gradient GREAT can be cheaper. The relevant extra
outputs of the distribution are directional posterior means, uncertainty,
task-dependent decisions, predictive scores and scale fitting. Mean-versus-mode
improvement alone does not establish an anisotropy advantage: the matched
isotropic control and independently validated loading are required.

Contact GREAT sees measured responses only; the conditional method also knows
the applied input. That comparison cannot attribute every gain to uncertainty.
Native dimension scaling here applies to rank one or codimension one, not to
general-rank normalized inference. Numerical accuracy of a mean does not
automatically certify second moments or quantiles.

The calibration cohorts use n=3. Their clean scalar targets are a unit query's
projected energy, q^H P q, for beamforming and the first coordinate of P q for
contact. The noisy target is the first real measurement coordinate (the real
part for complex data). These checks do not establish joint vector coverage
or high-dimensional region calibration.

GREAT uses the independently checked finite Grassmann exponential update in
[great_reference.md](great_reference.md). Direct primary-PDF and author-code
parity remain unverified because their hosts were denied by the environment's
network policy. This limitation is retained in the completion report.
