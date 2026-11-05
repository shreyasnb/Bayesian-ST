# Unified passive beamforming results

This report is generated from source- and artifact-verified saved tables. No simulations, fitting, posterior sampling or tuning occur in the summary command.

## Execution status

| Item | Status | Qualification |
| --- | --- | --- |
| General theory/regression gate | Verified passed | Current original-core and passive-science fingerprints match the saved gate |
| numerics | Completed and source/artifact verified | Numerical and approximation qualifications are reported separately |
| information | Completed and source/artifact verified | Numerical and approximation qualifications are reported separately |
| decisions | Completed and source/artifact verified | Numerical and approximation qualifications are reported separately |
| tracking | Completed and source/artifact verified | Numerical and approximation qualifications are reported separately |
| process | Completed and source/artifact verified | Numerical and approximation qualifications are reported separately |
| inference | Completed and source/artifact verified | Numerical and approximation qualifications are reported separately |
| Figures | Source/artifact verified | 22 PDF/PNG exports; presentation manifest checked |
| Historical appendices | Archive verified | Original rank-three and real low-data artifacts retain their source/certificate snapshot |
| Array notebook | Separate execution manifest | See results/beamforming_unified/execution.json for the executed notebook, figure and findings hashes, complete cell execution and saved error checks |

## Decision comparisons

The complete map contains 42 predeclared condition/noise/horizon cells per comparator, with 256 independent test trajectories and 128 independent validation trajectories per condition/noise. A saved paired 95% Student interval for baseline minus posterior-mean loss lies above zero for a gain and below zero for a loss. Crossing zero means insufficient evidence of a difference, not proven equality. Exact ties are the analytic one-snapshot control. The intervals are pointwise exploratory comparisons; these cell counts are not a multiple-testing-adjusted discovery claim. Queries, snapshots and posterior draws within one trajectory are not extra replications.

For true-covariance power, the posterior mean records 42/42 gains and 0 losses against the mode; 42/42 gains and 0 losses against GROUSE; 10/42 gains and 28 losses against validated regularized PCA; and 5/42 gains and 7 losses against matched isotropization. These distinctions matter: improvement over a point mode does not by itself demonstrate an advantage over regularization or prove a benefit from anisotropy.

The regularized-PCA comparison separates by generation law: Haar stationary 10 gains/0 losses in 14 cells; ULA stationary 0 gains/14 losses in 14 cells; ULA moving 0 gains/14 losses in 14 cells. The largest observed relative power gain over matched isotropization is 0.1515%; anisotropy contributes little and its signs are mixed. The isotropized posterior retains data-dependent spread, whereas validated regularized PCA uses one scalar per validation condition/horizon shared across test trajectories. The static ULA likelihood remains correct: its fixed steering geometry and decision-risk weighting differ from Haar-prior averaging. This decision difference does not contradict the stationary state-invariance argument. Moving ULA motion introduces a separate process mismatch.

| Endpoint | Comparator | Gain | Crosses zero | Loss | Exact tie | Unresolved | Not defined | Cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True-covariance power | Posterior mode | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| True-covariance power | EW PCA | 41 | 1 | 0 | 0 | 0 | 0 | 42 |
| True-covariance power | Matched isotropization | 5 | 24 | 7 | 6 | 0 | 0 | 42 |
| True-covariance power | Regularized PCA | 10 | 4 | 28 | 0 | 0 | 0 | 42 |
| True-covariance power | Loaded covariance | 13 | 3 | 26 | 0 | 0 | 0 | 42 |
| True-covariance power | GROUSE | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| True-covariance power | True-covariance oracle | 0 | 0 | 42 | 0 | 0 | 0 | 42 |
| Held-out power | Posterior mode | 40 | 2 | 0 | 0 | 0 | 0 | 42 |
| Held-out power | EW PCA | 39 | 3 | 0 | 0 | 0 | 0 | 42 |
| Held-out power | Matched isotropization | 3 | 30 | 3 | 6 | 0 | 0 | 42 |
| Held-out power | Regularized PCA | 8 | 9 | 25 | 0 | 0 | 0 | 42 |
| Held-out power | Loaded covariance | 12 | 7 | 23 | 0 | 0 | 0 | 42 |
| Held-out power | GROUSE | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Held-out power | True-covariance oracle | 0 | 0 | 42 | 0 | 0 | 0 | 42 |
| Spatial-response loss | Posterior mode | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Spatial-response loss | EW PCA | 41 | 0 | 1 | 0 | 0 | 0 | 42 |
| Spatial-response loss | Matched isotropization | 25 | 10 | 1 | 6 | 0 | 0 | 42 |
| Spatial-response loss | Regularized PCA | 41 | 0 | 1 | 0 | 0 | 0 | 42 |
| Spatial-response loss | Loaded covariance | 0 | 0 | 0 | 0 | 0 | 42 | 42 |
| Spatial-response loss | GROUSE | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Spatial-response loss | True-covariance oracle | 0 | 0 | 42 | 0 | 0 | 0 | 42 |
| Posterior expected power | Posterior mode | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Posterior expected power | EW PCA | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Posterior expected power | Matched isotropization | 36 | 0 | 0 | 6 | 0 | 0 | 42 |
| Posterior expected power | Regularized PCA | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Posterior expected power | Loaded covariance | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Posterior expected power | GROUSE | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| Posterior expected power | True-covariance oracle | 42 | 0 | 0 | 0 | 0 | 0 | 42 |

Loaded covariance supplies no point-subspace action in this implementation, so its spatial-response loss is not defined. Those saved NaN pairs are not numerical failures. Posterior expected-power rows evaluate the model objective under the same full posterior; true-covariance and held-out power are separate empirical endpoints.

One snapshot: 1536/1536 trajectory-level equalities; maximum `||M-M_iso||_F` 5.68e-15. Maximum mode/PCA projector difference 1.97e-14; conditional expected-power identity residual 1.28e-15. A one-snapshot gain over the mode therefore cannot demonstrate anisotropy. The true-covariance oracle uses scoring truth and is a lower reference, not an implementable method.

True-covariance power by generation law, combining both declared noise levels and all seven horizons:

| Law | Comparator | Gain | Crosses zero | Loss | Exact tie | Unresolved |
| --- | --- | --- | --- | --- | --- | --- |
| Haar stationary | Posterior mode | 14 | 0 | 0 | 0 | 0 |
| Haar stationary | EW PCA | 14 | 0 | 0 | 0 | 0 |
| Haar stationary | Matched isotropization | 2 | 10 | 0 | 2 | 0 |
| Haar stationary | Regularized PCA | 10 | 4 | 0 | 0 | 0 |
| Haar stationary | Loaded covariance | 12 | 2 | 0 | 0 | 0 |
| Haar stationary | GROUSE | 14 | 0 | 0 | 0 | 0 |
| ULA stationary | Posterior mode | 14 | 0 | 0 | 0 | 0 |
| ULA stationary | EW PCA | 14 | 0 | 0 | 0 | 0 |
| ULA stationary | Matched isotropization | 2 | 7 | 3 | 2 | 0 |
| ULA stationary | Regularized PCA | 0 | 0 | 14 | 0 | 0 |
| ULA stationary | Loaded covariance | 1 | 0 | 13 | 0 | 0 |
| ULA stationary | GROUSE | 14 | 0 | 0 | 0 | 0 |
| ULA moving | Posterior mode | 14 | 0 | 0 | 0 | 0 |
| ULA moving | EW PCA | 13 | 1 | 0 | 0 | 0 |
| ULA moving | Matched isotropization | 1 | 7 | 4 | 2 | 0 |
| ULA moving | Regularized PCA | 0 | 0 | 14 | 0 | 0 |
| ULA moving | Loaded covariance | 0 | 1 | 13 | 0 | 0 |
| ULA moving | GROUSE | 14 | 0 | 0 | 0 | 0 |

Decision uncertainty reports: 3072, with 0 unresolved means, 0 unresolved state-mass checks and 0 unresolved scalar checks. The 9216 queried scalar intervals belong to 3072 trajectory/time reports and are not independent trajectory repeats. All reports remain in the tables.

Recorded decision costs at `N=64`, median milliseconds across declared conditions and trajectories. Update/eigensystem clocks are cumulative through that horizon; the mean evaluation and decision solve are calls at the reporting horizon. They do not represent continuously reporting every posterior tail, and are not a portable speed guarantee.

| Method | Update | Eigensystem | Current mean call | Decision |
| --- | --- | --- | --- | --- |
| Posterior mean | 0.602 | 0 | 9.06 | 0.173 |
| EW PCA | 0.592 | 2.74 | 0 | 0.0749 |
| GROUSE | 9.5 | 0 | 0 | 0.0728 |
| Loaded covariance | 0.604 | 0 | 0 | 0.0739 |

## Information and deterministic checks

Passive Fisher verification uses 4 fixed-state/noise cases with 131,072 independent proper-complex snapshots per case. Maximum score-covariance relative matrix error is 0.0185; maximum normalized off-diagonal 0.0133; maximum expected-likelihood curvature relative error 1e-06. Score-covariance discrepancies are Monte Carlo error; curvature error is a separate finite-difference check. The analytic information is isotropic in the stated real tangent coordinates. Physical-angle information varies with the steering-curve tangent.

Native numerical checks: 0/12 unresolved ordinary mean targets; 2 deliberately out-of-range series targets retained; 0/24 unresolved transition-moment checks; 0/9 unresolved vector-risk checks. Inverse moment fits: 0/18 unresolved, maximum residual 3.01e-09. Posterior scalar precision: 0/9 unresolved fixed-budget reports. Accurate means and exact iid sampling laws do not remove empirical-quantile uncertainty.

Whole-call native numerical costs, measured on a shared host. Contention can separate CPU time from wall time; these are not dedicated latency measurements or a complexity law. The full posterior report contains a mean evaluation, two iid batches and uncertainty calculations; inverse fits contain forward evaluations. These nested categories must not be added as disjoint costs.

| Complex dimension | Operation | Repeated timings | Median wall ms | Median CPU ms |
| --- | --- | --- | --- | --- |
| 8 | Native mean | 3 | 32.6 | 8.6 |
| 8 | Iid 8192 draws | 3 | 20.2 | 11.7 |
| 8 | Full posterior report | 3 | 196 | 100 |
| 8 | Inverse mean fit | 6 | 125 | 62.4 |
| 16 | Native mean | 3 | 37.3 | 14 |
| 16 | Iid 8192 draws | 3 | 82.3 | 30.6 |
| 16 | Full posterior report | 3 | 496 | 199 |
| 16 | Inverse mean fit | 6 | 384 | 150 |

## Fixed-time certificates and state coverage

The certificate uses known fixed powers, deterministic weights/transports and a truth-dependent drift for evaluation. Invalid sufficient conditions remain invalid; no posterior radius is substituted. The observations at each predeclared reporting time give pointwise guarantees.

| Cohort | Fixed-time reports | Valid | Valid bound violated | Smallest valid bound | Largest valid bound |
| --- | --- | --- | --- | --- | --- |
| Long stationary control | 768 | 256 | 0 | 0.346 | 0.346 |
| Moving ULA | 4096 | 0 | 0 | — | — |

Fixed-state inclusion, with pointwise 95% Wilson intervals across independent trajectories. Numerical failures remain included and are separately counted.

| State | N | HPD inclusion [CI] | Ball inclusion [CI] | State unresolved | Scalar unresolved |
| --- | --- | --- | --- | --- | --- |
| coordinate line | 8 | 250/256 = 0.977 [0.950, 0.989] | 249/256 = 0.973 [0.945, 0.987] | 0 | 0 |
| coordinate line | 32 | 238/256 = 0.930 [0.892, 0.955] | 239/256 = 0.934 [0.896, 0.958] | 0 | 0 |
| ULA 20 degrees | 8 | 234/256 = 0.914 [0.873, 0.943] | 238/256 = 0.930 [0.892, 0.955] | 0 | 0 |
| ULA 20 degrees | 32 | 245/256 = 0.957 [0.925, 0.976] | 244/256 = 0.953 [0.920, 0.973] | 0 | 0 |

Across the 8 predeclared state/time/region entries, observed inclusion ranges from 0.914 to 0.977. The lowest is HPD at ULA 20 degrees, N=8: 234/256, with pointwise Wilson interval [0.873, 0.943]. This deviation remains visible. These finite, dependent comparisons do not establish exact 95% coverage at every state and are not a blanket empirical confirmation. The theorem concerns ideal equivariant regions of exact posterior mass; finite training/check batches and trajectory counts are separate qualifications.

Physical moving-ULA state inclusion is a separate calibration check. These are fixed-time, independent-trajectory intervals within each cell; policies and times share trajectories and cannot be pooled as independent repetitions.

| s | Memory policy | Time | HPD inclusion [CI] | Ball inclusion [CI] | Numerically resolved |
| --- | --- | --- | --- | --- | --- |
| 1.0 | All history | 64 | 3/32 = 0.094 [0.032, 0.242] | 3/32 = 0.094 [0.032, 0.242] | 32/32 |
| 1.0 | All history | 128 | 2/32 = 0.062 [0.017, 0.201] | 6/32 = 0.188 [0.089, 0.353] | 32/32 |
| 1.0 | Selected memory | 64 | 5/32 = 0.156 [0.069, 0.318] | 7/32 = 0.219 [0.110, 0.388] | 32/32 |
| 1.0 | Selected memory | 128 | 7/32 = 0.219 [0.110, 0.388] | 12/32 = 0.375 [0.229, 0.547] | 32/32 |
| 0.2 | All history | 64 | 0/32 = 0.000 [0.000, 0.107] | 0/32 = 0.000 [0.000, 0.107] | 32/32 |
| 0.2 | All history | 128 | 0/32 = 0.000 [0.000, 0.107] | 0/32 = 0.000 [0.000, 0.107] | 32/32 |
| 0.2 | Selected memory | 64 | 0/32 = 0.000 [0.000, 0.107] | 0/32 = 0.000 [0.000, 0.107] | 32/32 |
| 0.2 | Selected memory | 128 | 0/32 = 0.000 [0.000, 0.107] | 0/32 = 0.000 [0.000, 0.107] | 32/32 |

At s=0.2, every one of 4 predeclared memory/time cells has HPD inclusion 0/32. Numerical reports resolve 128/128 of these evaluations. Accurate posterior masses do not ensure frequentist state coverage under unmodeled drift. The selected memory minimizes independent-validation decision power, not a coverage loss. This observed calibration failure is substantive and must remain visible when interpreting the uncertainty plots.

Moving-ULA uncertainty: 0/256 unresolved state checks and 0/256 unresolved scalar checks. The ideal stationary state-inclusion argument applies to known powers, uniform initialization, beta=1 and exact mass. Forgetting, unmodeled ULA drift, learned powers and assumed-density prediction do not inherit it. Central scalar intervals have a further support-boundary limitation: an exactly aligned query has truth one, above every finite nondegenerate 97.5% posterior quantile. A correct state-coverage result therefore does not imply uniform central scalar inclusion. A common state-region image and separate marginal intervals answer different questions.

## Process projection and parameter inference

The complex dimension-two full-density reference has 4/192 unresolved full-density reference reports. Maximum reference refinement total variation is 1.48e-05, maximum scalar-CDF discrepancy 1.24e-07. The single-Bingham filter differs from that reference by total variation 0.00166–0.145 and mean operator error 0.000251–0.0684 at declared checkpoints. These are density-approximation discrepancies, separate from a small inverse-fit residual and separate from reference discretization error. The joint reference status includes density-validity and region-mass checks; small refinement discrepancies alone do not resolve a failed flag. The CP1 reference is not an eight-channel exact filtering result.

Native process reports: 0/9216 unresolved means. Tails were evaluated for 1152/9216 reports on a preassigned trajectory subset; 0 state checks and 0 scalar checks are unresolved. Unevaluated tails are not counted as resolved. Exact transition first moments do not establish exact dynamic regions.

Actual matched-process trajectory costs can vary substantially from the representative inverse-fit fixture. These are cumulative prediction-call wall times through the final reporting horizon, not complete pipeline times: score evaluation, posterior reporting and simulation are separate. Shared-host contention also affects the wall distribution; no dedicated-latency or end-to-end complexity claim follows.

| Complex dimension | kappa | Final T | Trajectories | Resolved means | Median prediction s | Mean prediction s | Maximum prediction s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4.0 | 24 | 256 | 256 | 0.092 | 0.11 | 0.422 |
| 2 | 20.0 | 24 | 256 | 256 | 0.147 | 0.148 | 0.423 |
| 8 | 4.0 | 24 | 256 | 256 | 0.418 | 1.06 | 12.8 |
| 8 | 20.0 | 24 | 256 | 256 | 1.44 | 2.02 | 21.2 |

Joint inference uses proper ordered variance priors and paired power/state/concentration draws. The following are counts passing finite-chain diagnostics, not proofs of convergence. All fits, including unresolved ones, remain in performance tables.

| Law | Prior | Fits | Calculation failure | Parameter flag | State-mean flag | Covariance-mean flag | State-tail agreement | All flags | Median fit s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stationary | Default | 384 | 0 | 301 | 384 | 384 | 98 | 89 | 1.85 |
| Stationary | Sensitivity | 384 | 0 | 317 | 384 | 384 | 87 | 78 | 1.86 |
| Matched process | Default | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 26.5 |
| Matched process | Sensitivity | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 26.2 |
| Moving ULA | Default | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 28.5 |
| Moving ULA | Sensitivity | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 30.1 |

Concentration diagnostics: 0/64 kappa traces meet the declared scalar ESS/R-hat/tail-agreement thresholds. There is no ground-truth kappa for deterministic ULA motion. Failure of a state-tail or scalar check must not be described as a resolved full posterior, even when the decision covariance mean has better diagnostics.

Prospective metadata identify 2084 prefix/next-observation scores; 0 recorded-prior EM traces decrease beyond tolerance. EM ascent concerns its frozen-prior objective only. Predictive scoring uses the held-out next observation without assimilation. Joint covariance averaging preserves E[qP] and its concentration/state dependence; replacing it by a product of separate means defines a different control.

| Law | Method | Recorded fits | Calculation failure | Median fit s |
| --- | --- | --- | --- | --- |
| Stationary | Known powers | 384 | 0 | 0.00622 |
| Stationary | Validation-fixed powers | 384 | 0 | 0.00536 |
| Stationary | Recorded-prior EM | 384 | 0 | 2.3 |
| Stationary | Joint inference | 768 | 0 | 1.86 |
| Matched process | Known powers | 16 | 0 | 0.659 |
| Matched process | Validation-fixed powers | 16 | 0 | 0.715 |
| Matched process | Recorded-prior EM | 16 | 0 | 1.46 |
| Matched process | Joint inference | 32 | 0 | 26.3 |
| Matched process | Known full-path Gibbs | 4 | 0 | 17.4 |
| Moving ULA | Known powers | 16 | 0 | 0.872 |
| Moving ULA | Validation-fixed powers | 16 | 0 | 0.832 |
| Moving ULA | Recorded-prior EM | 16 | 0 | 1.95 |
| Moving ULA | Joint inference | 32 | 0 | 29.3 |

These inference cohorts and Monte Carlo budgets differ from the main fixed-power 256-trajectory comparison. Conditional sampler correctness does not establish full-path Gibbs mixing, calibrated kappa intervals, or a universal predictive gain over validated fixed parameters.

## Interpretation and paper placement

The framework offers posterior decisions and uncertainty diagnostics at an extra computational cost. Its mode is weighted covariance PCA under matching assumptions. Any empirical advantage of the full mean must be judged against matched isotropization, validated scalar regularization, broad covariance loading and geometric tracking; the complete comparison map above retains gains, intervals crossing zero, losses, analytic equalities and unresolved calculations. Conditional posterior expected-power optimality does not guarantee lower true-covariance power on every trajectory or in every regime.

GROUSE is a reasonable choice when a cheap point subspace estimate is the main requirement. The posterior framework is useful when downstream decisions, directional risk, state regions or joint parameter uncertainty justify the additional work. Exact likelihood correction and exact first-moment prediction remain valid mathematical ingredients, while density projection, quantile Monte Carlo and Gibbs mixing have separate limitations. Identity transport permits drift to be modeled or bounded; it does not make unmodeled motion disappear.

Main-paper candidates are the setup, complete decision/anisotropy comparison and a carefully explained uncertainty/response example (`01_setup`, `04_decisions`, `05_uncertainty`, `06_prediction_and_risk`). A compact tracking validity panel can accompany a tracking claim. Full information matrices, detailed bounds, all-regime comparison panels, process/reference discrepancies, inference diagnostics and cost tables belong in the supplement. Unresolved concentration or state-tail results support a documented limitation, not a claim of calibrated full dynamic uncertainty.

The earlier real conditional theorem remains in its original theory/contact or field fixtures. Historical rank-three and real low-data figures are separately archived supplementary experiments, not new rank-one passive results. See `docs/beamforming_claims.md` for the claim-by-claim scope map.

## Provenance

Profile: `paper`. Summary generator SHA-256: `c333c1bd69325553a8603236087fee364cda8025e1a43920009c5b0fbc82370e`. Component manifests below were rechecked after reading the tables. Their artifact hashes cover the input CSVs and declared scientific sources.

| Verified manifest/config | SHA-256 |
| --- | --- |
| results/beamforming_unified/paper/numerics/config.json | 8dad17c684a1f57c83d1052bdf0960dcb560f5bbde859bf5532671b1cb675d22 |
| results/beamforming_unified/paper/information/config.json | e6f9d4974ee3e7c0605a5a66a36743dec89c71e37464c6c7f35b6508de55d50c |
| results/beamforming_unified/paper/decisions/manifest.json | e534f5eb9e429e2a2ea018a0304e9b2dc44b66445c9029a3163209410f9b6d4d |
| results/beamforming_unified/paper/tracking/manifest.json | f00f23f0f15d79ecdb44cbd39fc28812d04f1199f0dbe55600f9e71e9b026158 |
| results/beamforming_unified/paper/process/manifest.json | 97190284b555fb97805fcd1b81a83e1b487cd9dbcdb96219e1bfc6417469ab08 |
| results/beamforming_unified/paper/inference/manifest.json | d56ebdbc105ee5bf467559c98a86cad04892c0eb61384e8ad6e019a365d19a0d |
| results/beamforming_unified/paper/figures/manifest.json | cda10691a3cd060480599769b4f1b66b6c71c80c04fd701c209d251006df4b4a |
