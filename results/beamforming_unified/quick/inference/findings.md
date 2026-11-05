# Ordered-power and process inference

Observations are native proper complex passive snapshots. Sensor variance is b=s and total parallel variance is a=s+q. The prior is one globally normalized product IG(a)IG(b) restricted to a>b; it is not an independent prior on s and q.

The default inverse-gamma shapes are 3,3 and scales 3,1; sensitivity scales are 1.5,.5. The default concentration prior is Gamma(shape2,rate.04), mean50; sensitivity rate is .02, mean100. These choices were fixed before the paper data.

The quick profile uses 2 independent stationary data sets per noise, at N=8,32,64, and 1 moving data sets per law/noise at T=24. There are 2 independently initialized chains with 128 retained draws and 64 burn-in sweeps for stationary fits; moving budgets are 64 and 32. These are bounded parameter-inference pilots, separate from the 256-trajectory decision experiment.

| Generation law | Joint fits | Calculation failures | Scalar checks pass | Covariance-mean checks pass | All checks pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| stationary | 24 | 0 | 10 | 14 | 0 |
| matched_motion | 4 | 0 | 0 | 0 | 0 |
| ula_motion | 4 | 0 | 0 | 0 | 0 |

Scalar diagnostics check a,b,q and, when applicable, kappa. Covariance-mean diagnostics check entries of the paired joint forecast covariance. The strict overall flag additionally requires all monitored state-component tails to agree across chains. Separate flags retain that distinction; failed tail or scalar checks are never turned into successful inference by dropping trajectories.

A moving fit with unresolved kappa cannot support a reliable concentration-learning or parameter-coverage claim. A finite returned posterior sample is not proof of mixing. The short data-only development pilot already found slow process-concentration mixing; budgets were fixed rather than extended until a favorable result appeared.

## Conditioning and comparisons

Every fit sees only the prefix through N. Prospective density and decision power use observation/state N+1, before that observation is assimilated. Predictive density is a Monte Carlo average of full normalized complex Gaussian likelihoods with paired parameter/state draws. Moving predictions simulate an exact complex transition per draw; chain-based score MCSE and the separately labelled IID diagnostic remain recorded.

Decision weights use E[bI+(a-b)P_next] with all dependence retained. The product-of-means discrepancy is a diagnostic only. Unit gain and conditional posterior expected-power excess identities are checked numerically. These are expected-power decisions, not maximizers of expected SINR or expected log-SINR.

Stationary known/fixed-power controls use the exact native rank-one mean for one common projector. Moving known/fixed-power controls use a single-Bingham assumed-density approximation. First two matched-motion data sets per noise additionally receive a known-power/known-kappa full-path Gibbs control, shared across both prior sensitivities. Comparisons involving that approximation do not isolate the effect of learning alone.

Recorded-prior EM stores every pre-correction prior once, then holds the priors fixed. Its ascent check concerns that normalized fixed objective only. The reported final state uses the last stored prior plus one fitted correction; no whole-history refit is passed off as the same ascent statement.

The physical ULA law has deterministic changing angle. It has no true process concentration: its fitted kappa describes an assumed random-motion model. Integrated unknown-parameter state marginals are mixtures, not one exact Bingham recursion.

## Independent numerical validation

Tests compare the global ordered-prior normalizer with independent integration; conditional shape/energy ratios with the full passive likelihood; both moving neighbors under arbitrary unitary transports; the T-1 concentration normalizers; a log-kappa slice update including its Jacobian; joint covariance averaging; and an independent native complex CP1 scalar-grid reference. The CP1 reference validates the hierarchy, not native eight-channel state tails.

Numerical calculation failures: 0; maximum unit-gain error 2.69e-16; maximum posterior optimality identity residual 7.49e-16.

Full state arrays are excluded from published artifacts. Seeded regeneration uses scripts/run_beamforming_inference.py with the recorded profile and source hashes. Compact preassigned scalar draws, all per-fit summaries and mixing diagnostics are retained.
