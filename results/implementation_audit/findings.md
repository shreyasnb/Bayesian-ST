# Implementation-guide results

The complete general notebook and all **241 regression tests** passed. The audit maps 57 requested checks: 1 failed, 4 inconclusive, 1 not applicable, 51 passed.

The guide and supplied PDF theory/appendices are available; main.tex and supplementary.tex are absent from this checkout. Exact source agreement remains inconclusive. The proper-complex conditional certificate is an explicitly derived conservative extension, not a claim of matching unavailable supplementary constants. The optional 2017 tracker is not relabeled from another algorithm.

## Algebra and information

Information study: 0 algebra failures and 0 unresolved score cases. Final-step likelihood-Hessian and Gaussian-KL relative errors are 1.003e-06 and 1e-06. Maximum independent native/reference mean error is 6.024e-13.

Real and complex measures, tangent dimensions and Gaussian normalization are checked separately. Recorded inputs are prescribed independently. Discounted score covariance uses squared weights, while expected Hessians use the weights themselves. The additional observation-count cohort directly simulates independent observations per dataset; its count axis is not a posterior-draw or Monte Carlo-repetition count. Its complete precision diagnostics are retained in paper/information_counts/manifest.json and summary.csv.

## Tracking and task comparisons

The tracking study retains 7680 reports in 60 fixed-time cells: 2048 valid sufficient conditions, 2048 envelopes below the geometric ceiling, and 0 eligible violations. Invalid cells have no modal-violation interval. The exact binomial intervals, including nonzero upper limits after zero violations, are in paper/tracking/summary.csv. These are pointwise reports, not a trajectory-wide guarantee.

The maximum perturbation-decomposition residual divided by maximum(1, gap, allowance) is 4.72e-14. The absolute residual and declared relative tolerance are retained separately.

The revised marginal constants, initialization radius, aggregate drift and memory-limit preflight are implemented separately from the legacy certificate. A permanently invalid fixed-memory regime is not repaired by extending its horizon.

The added moving-array study uses snapshots alone and identity transport. Positive paired gain means comparator minus posterior-mean output power. Intervals are pointwise Student intervals across independent episodes. Held-out snapshots are averaged within each episode at each checkpoint. Validation averages reporting times within each episode; checkpoint comparisons keep independent episodes as their repetition unit. This is not a multiplicity-adjusted ranking. The known-covariance row is an oracle reference.

| Reference | Cells | Gain | Loss | Crosses zero | Unresolved |
| --- | --- | --- | --- | --- | --- |
| $P_{\widehat{\mathcal X}_t}$ | 20 | 13 | 2 | 5 | 0 |
| $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 20 | 3 | 0 | 17 | 0 |
| $P_{\widehat{\mathcal X}_t}$, $\beta=1$ | 20 | 11 | 3 | 6 | 0 |
| Loaded covariance | 20 | 16 | 3 | 1 | 0 |
| EW covariance | 20 | 20 | 0 | 0 | 0 |
| Ledoit--Wolf | 20 | 19 | 0 | 1 | 0 |
| GROUSE | 20 | 19 | 1 | 0 | 0 |
| Known covariance | 20 | 0 | 20 | 0 | 0 |

## Fixed-boundary block EM

The stationary-block study has 96 independent blocks and 360 starts/fits, 0 numerical failures, 0 rejected updates, and 360 fits meeting the declared finite-iteration tolerance. Maximum independent normalized-evidence error is 3.233e-13; minimum recorded evidence increment is -7.39e-13, compared with the declared ascent tolerance.

Every E-step rebuilds from one boundary prior excluding the entire stationary block. The joint constrained variance update, posterior-uncertainty residual and zero-signal absorbing boundary are checked. This objective differs from the historical recorded-prior routine. Block ascent does not establish a global optimum, monotone online scores or convergence under drift.

## Calibration, scores and changed physics

Prior-predictive calibration draws states from the stated prior and data from its likelihood. Fixed-state and drifting-state inclusion answer different questions. State regions, clean-response intervals and noisy predictive intervals are separate targets. The following nominal-0.95 reports use independent test episodes; exact binomial intervals and unresolved reports remain visible.

| Model | Regime | Target | Episodes | Inclusion | 95% interval | Unresolved |
| --- | --- | --- | --- | --- | --- | --- |
| Beamforming | Prior predictive | State HPD region | 128 | 0.9531 | [0.9008, 0.9826] | 0 |
| Beamforming | Prior predictive | Clean response | 128 | 0.9531 | [0.9008, 0.9826] | 0 |
| Beamforming | Prior predictive | Noisy observation | 128 | 0.9062 | [0.842, 0.9506] | 0 |
| Beamforming | Fixed state | State HPD region | 128 | 0.9453 | [0.8906, 0.9777] | 0 |
| Beamforming | Fixed state | Clean response | 128 | 0.9531 | [0.9008, 0.9826] | 0 |
| Beamforming | Fixed state | Noisy observation | 128 | 0.8906 | [0.8233, 0.9389] | 0 |
| Beamforming | Drifting state | State HPD region | 128 | 0.1016 | [0.0552, 0.1674] | 0 |
| Beamforming | Drifting state | Clean response | 128 | 0.6797 | [0.5915, 0.7594] | 0 |
| Beamforming | Drifting state | Noisy observation | 128 | 0.9375 | [0.8806, 0.9726] | 0 |
| Contact | Prior predictive | State HPD region | 128 | 0.9609 | [0.9112, 0.9872] | 0 |
| Contact | Prior predictive | Clean response | 128 | 0.9531 | [0.9008, 0.9826] | 0 |
| Contact | Prior predictive | Noisy observation | 128 | 0.9297 | [0.8707, 0.9673] | 0 |
| Contact | Fixed state | State HPD region | 128 | 0.9531 | [0.9008, 0.9826] | 0 |
| Contact | Fixed state | Clean response | 128 | 0.9609 | [0.9112, 0.9872] | 0 |
| Contact | Fixed state | Noisy observation | 128 | 0.9609 | [0.9112, 0.9872] | 0 |
| Contact | Drifting state | State HPD region | 128 | 0.3281 | [0.2478, 0.4167] | 0 |
| Contact | Drifting state | Clean response | 128 | 0.6719 | [0.5833, 0.7522] | 0 |
| Contact | Drifting state | Noisy observation | 128 | 0.9453 | [0.8906, 0.9777] | 0 |

Matched-width comparisons select their multiplier on independent validation episodes. A validation target that is not attained is explicitly flagged; it does not become a matched-coverage claim. See matched_widths.csv for achieved test coverage and widths.

Predictive scores use a new observation excluded from fitting and state assimilation. The normalized Bingham mixture and Gaussian moment surrogate are different predictive densities. Their paired score difference and clean-response risk discrepancy are reported with episode-level uncertainty.

| Model | Regime | Mixture minus Gaussian score | 95% interval | Realized minus predicted response risk | Unresolved |
| --- | --- | --- | --- | --- | --- |
| Beamforming | Prior predictive | -0.009803 | [-0.02585, 0.00624] | -0.0009334 | 0 |
| Beamforming | Fixed state | -0.004836 | [-0.01678, 0.007103] | -0.0008011 | 0 |
| Beamforming | Drifting state | 0.03457 | [0.002574, 0.06658] | 0.1023 | 0 |
| Contact | Prior predictive | 0.0003649 | [-0.0008078, 0.001538] | 0.002956 | 0 |
| Contact | Fixed state | 0.00111 | [-0.001244, 0.003463] | -0.0009152 | 0 |
| Contact | Drifting state | -0.000103 | [-0.002774, 0.002568] | 0.08442 | 0 |

Separate interventions change sensor noise at fixed motion or change motion at fixed sensor noise. The next table reports the final block. An increased fitted noise under motion mismatch is not recovery of the sensor variance. Plug-in fits omit scale uncertainty and do not inherit the fixed-scale theorem. Nonconverged finite-iteration fits are not silently dropped.

| Model | Intervention | Episodes | Mean fitted/true sensor variance | Numerically unresolved | Finite-iteration limits |
| --- | --- | --- | --- | --- | --- |
| Beamforming | Noise change | 24 | 0.9838 | 0 | 4 |
| Beamforming | Motion change | 24 | 1.119 | 0 | 0 |
| Contact | Noise change | 24 | 1.029 | 0 | 0 |
| Contact | Motion change | 24 | 1.127 | 0 | 0 |

Mismatch studies retain desired-signal contamination, unequal additional interference powers, anisotropic damping, dissipative friction with sticking, and contact loss. They do not validate the matched likelihood under those changes. Independent KKT identities and the transformed sensor covariance are checked for known anisotropic damping. Differenced position noise has negative lag-one velocity-error covariance and is not independent velocity sensor noise.

## Numerical qualifications and preservation

All failed jobs, unresolved integration/quantile flags, finite-iteration limits and raw paired comparisons remain in the component tables. Low-dimensional reference accuracy does not certify general-rank integration. Runtime evidence is the separately isolated earlier S1 study; concurrent audit-study elapsed times are execution metadata, not comparative latency.

Review found and corrected a friction force that could reverse motion; the provisional results and exact producers are preserved under development. The corrected force clamps the Coulomb magnitude at the available tangent force and allows sticking. The complete general validation was repeated after this correction. The earlier 6e78cc9 notebooks, figures, data and producer sources remain byte-exact in archive/previous.

The sphere density palette is continuous blue through violet to pink. It does not change density normalization, credible thresholds or the fixed sphere radius. Main figures retain their original experimental scope; filename aliases do not supply additional independent evidence.

See claim_manifest.json for every claim, routine, metric, figure ID, assumption and status; README.md for reproduction; and notebooks/04_implementation_audit.ipynb for the chronological computational walkthrough.
