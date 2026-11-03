# Simulation findings

These are the recorded eight-trajectory pilot runs. Comparisons below use
all scenarios and retain unfavorable outcomes and numerical failures.
See the notebooks for individual scenarios and uncertainty intervals.

## Array

168 passive posterior evaluations; 22 failed the predeclared
sampling diagnostics at the maximum budget.
Capped failures are retained and require separate accuracy follow-up.

| Scenario | Loaded covariance | Mode | Oracle | Posterior mean |
| --- | --- | --- | --- | --- |
| Contamination | 2.5655 | -0.2496 | 4.1642 | 0.1391 |
| Correlation | 3.7296 | 3.9342 | 4.2446 | 3.9692 |
| Drift | 2.0857 | 2.9349 | 4.1642 | 3.0418 |
| Far from interference | 4.5845 | 4.3831 | 5.0357 | 4.4055 |
| Near interference | -1.9228 | -1.5061 | -0.6070 | -1.4985 |
| Static | 3.6776 | 3.9866 | 4.1642 | 3.9791 |
| Unequal powers | 3.7493 | 3.5123 | 4.4909 | 3.5803 |

Largest regional primal–dual gap: 1e-06.

Separate longer-chain checks passed for 22/22 original capped failures.
Largest absolute relative change in population output power: 0.3657%.
Original flags remain unchanged; extended checks use new independent random streams.

## Field

72 passive posterior evaluations; 0 failed the predeclared
sampling diagnostics at the maximum budget.
All selected passive evaluations met the stated diagnostics; this is not a convergence proof.

| Scenario | EW PCA | GROUSE | Mode | Oracle | Posterior mean | Rolling PCA |
| --- | --- | --- | --- | --- | --- | --- |
| Matched | 0.7981 | 0.8141 | 0.7981 | 0.6570 | 0.7979 | 0.7203 |
| Sparse corruption | 2.2456 | 3.5443 | 2.2456 | 1.5912 | 2.2209 | 2.4542 |
| Unequal powers | 0.6445 | 1.6050 | 0.6445 | 0.5500 | 0.6305 | 0.6060 |

Known-calibration posterior evaluations with failed diagnostics: 0 of 72.

Sampled high-dimensional EM ascent is unverified. Predictive scores
must be read with their importance ESS, sampler ESS and chain disagreement.

## Controlled decision study

This separate stationary real Gr(1,3) study uses identical observations,
known variances, 400 independent test trajectories, 160 independent
validation trajectories, and nine-candidate reference families.
It tests posterior decisions under the stated joint prior model.

| Task | Reference | Loss reduction | Paired gain interval |
| --- | --- | ---: | --- |
| Projection | Mode | 23.19% | [0.02191, 0.03717] |
| Projection | Shrunk PCA | 10.36% | [0.00762, 0.01501] |
| Reconstruction | Mode | 10.54% | [0.01812, 0.04594] |
| Reconstruction | Shrunk PCA | 5.36% | [0.00615, 0.02464] |
| Beamforming | Loaded covariance | 2.28% | [0.00726, 0.01228] |
| Beamforming | Mode | 5.61% | [0.01724, 0.03256] |
| Beamforming | Shrunk PCA | 2.15% | [0.00546, 0.01297] |

Rows above use noise variance 0.2 and four observations; all other
predeclared regimes, including statistical ties, remain in the notebooks.
Intervals concern paired absolute-loss gains, not the displayed percentage.
Shrunk PCA and strongly loaded covariance capture much of the benefit.
See [interpretation](interpretation.md) for the cost and scope qualifications.

## Interpretation

Marginal Bingham and matching EW PCA modes coincide. Mean/modal changes
therefore concern posterior task decisions. Individual realized errors
need not obey the posterior expected-risk ordering. No universal
performance or frequentist-coverage claim follows from this pilot.

No inconsistency in the stated mathematical assumptions was found.
The required numerical qualifications and Algorithm 1 clarifications
are recorded in `draft_notes.md`.
