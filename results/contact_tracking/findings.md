# Contact tracking findings

The study preserves the earlier contact pilot and uses fresh roots 6245201/6245202: 16 validation and 64 test trajectories per condition, with nine queries averaged within each trajectory. The primary task is clean vector-response MSE, with the terminal small-sphere comparison predeclared for inference. Time-average MSE and clean interval score are retained secondary comparisons. One curved policy is shared across both radii and the one-axis force program. The plane is selected separately; anisotropic resistance never tunes.

The comparison is paired across **64 independent small-sphere trajectories**. Positive paired gains below mean lower loss for the Bingham process prediction; intervals are pointwise 95% Student intervals.

| Question | Held-out result |
| --- | --- |
| Does the causal forecast reduce lag? | Terminal mode MSE changes from 1.085 to 0.126 (mm/s)², a 88.4% reduction. Time-average MSE improves 43.0%. These gains are available to point methods. |
| What does spread calibration change? | With T=2, clean coverage changes from 98.0% to 100.0%; width from 0.938 to 1.326 mm/s; interval score from 0.979 to 1.326 mm/s (lower is better). The mode is unchanged. |
| Does process Bingham beat the cheap Gaussian? | It has no resolved terminal MSE advantage. The Gaussian-minus-Bingham paired MSE difference is -0.001381 [-0.006124, 0.003362] (mm/s)² (-0.62% of Gaussian loss). |
| What does computation cost? | Measured filter cost, including the common forecast tracker, is 2.38 versus 2.05 ms/observation. One 27-component uncertainty report averages 145 versus 73 ms. |

For uncertainty, Bingham/Gaussian clean-component coverage is **99.5%/99.5%**, with mean interval width **1.395/1.392 mm/s** and interval score **1.405/1.402 mm/s** (lower is better). The paired Gaussian-minus-Bingham score difference is -0.0028 [-0.0081, 0.0026] mm/s. Lower MSE alone would not establish better uncertainty.

The whole-trajectory tradeoff differs from the terminal result: time-average MSE is 1.162 for the raw forecast, 0.578 for process Bingham and 0.600 (mm/s)² for the Gaussian. Point memory was selected for terminal MSE, so this is a secondary comparison of the retained policies, not a search for the best possible time-average point estimator. Temperature was fitted on the curved-condition validation mixture; more coverage on this one endpoint can mean unnecessarily wide intervals.

An exploratory case shows where process uncertainty is useful: when the second and third force components are divided by five, the temperature control covers 94.8% at width 2.12 mm/s. Process Bingham gives 95.4% at 1.71 mm/s; the Gaussian gives 95.5% at 1.69 mm/s. Useful uncertainty propagation is supported there, while the full Bingham family still has no demonstrated premium over the Gaussian.

Time-average loss, same-parameter Gaussian checks, all misses and the other contact conditions are in the tables below. Cost scope: median of three separate serial runs of preassigned small-sphere trajectory 0; shared forecast included. Average report cost does not guarantee a 50 ms deadline. These numbers separate improved dynamics from any additional benefit of a distribution. The finite validation scan selected a lag of 32 at the extended boundary; it does not establish a globally optimal forecast.


## Small-sphere endpoint

| Prediction | MSE (mm/s)² | Time-average MSE (mm/s)² | Clean coverage | Width (mm/s) | Interval score (mm/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bingham, selected memory/forecast | 0.1262 | 1.1625 | 0.980 | 0.938 | 0.979 |
| Bingham, calibrated spread | 0.1264 | 1.1442 | 1.000 | 1.326 | 1.326 |
| Bingham process, identity motion | 0.4827 | 0.6542 | 0.978 | 1.854 | 1.961 |
| Bingham process, causal motion | 0.2254 | 0.5775 | 0.995 | 1.395 | 1.405 |
| Gaussian process, identity motion | 0.4965 | 0.6770 | 0.976 | 1.869 | 1.996 |
| Gaussian process, causal motion | 0.2240 | 0.6005 | 0.995 | 1.392 | 1.402 |

The regularized point reference has terminal MSE 0.1263 (mm/s)². Its scale rho is optimized continuously on validation for the improved practical forecast. Its paired loss minus Bingham process loss is -0.09906 [-0.1379, -0.0602] (mm/s)². A positive comparison against this point alone is not evidence of superiority over the Gaussian uncertainty control.

## Mean tracking and causality

Original mode minus causal mode terminal MSE is 0.9583 [0.8232, 1.093] (mm/s)²; the time-average difference is 0.9153 [0.4262, 1.404] (mm/s)². Predictions use observations through t-1; the current observation is read only for correction. The mode remains equal to transported weighted projector least squares. Noiseless runs keep the same mechanics, forces, variance and memory; truth alignment is a diagnostic, never a practical tuning candidate.

The selected finite-family causal forecast uses beta=0.9, lag=32 and damping=1 on the curved conditions. Lag 32 won the single permitted extension from 16 and remains a boundary. All finite-grid conclusions are conditional on that search. Shorter exponential memory did not replace beta=0.8 as its family winner.

## Distribution shape

| Same-parameter comparison | Local minus full MSE, 95% paired interval (mm/s)² | Full/local clean coverage | Full/local width (mm/s) | Unresolved full/local |
| --- | ---: | ---: | ---: | ---: |
| Bingham, calibrated spread | -1.475e-06 [-1.654e-06, -1.296e-06] (mm/s)² | 1.000/1.000 | 1.326/1.325 | 0/0 |
| Bingham process, identity motion | -5.629e-06 [-7.542e-06, -3.715e-06] (mm/s)² | 0.978/0.978 | 1.854/1.850 | 0/0 |
| Bingham process, causal motion | -1.251e-06 [-1.559e-06, -9.429e-07] (mm/s)² | 0.995/0.995 | 1.395/1.393 | 0/0 |

## State regions

| Method | Chordal-ball inclusion | HPD or local-ellipse inclusion | Region type |
| --- | ---: | ---: | --- |
| Bingham, selected memory/forecast | 62/64 | 63/64 | Invariant-density Bingham HPD |
| Bingham, calibrated spread | 64/64 | 64/64 | Invariant-density Bingham HPD |
| Bingham process, identity motion | 64/64 | 63/64 | Invariant-density Bingham HPD |
| Bingham process, causal motion | 64/64 | 64/64 | Invariant-density Bingham HPD |
| Gaussian process, identity motion | 64/64 | 63/64 | Local Gaussian ellipse |
| Gaussian process, causal motion | 64/64 | 64/64 | Local Gaussian ellipse |

These are terminal state inclusions, not componentwise response coverage. Counts use the same 64 independent trajectories; they are not simultaneous-time guarantees. A Gaussian local ellipsoid is not an invariant-density Grassmann HPD region.

## Mean, mode and regularized point predictions

| Filter | Mode MSE (mm/s)² | Mean MSE (mm/s)² | Regularized mode MSE (mm/s)² |
| --- | ---: | ---: | ---: |
| Bingham, selected memory/forecast | 0.12620 | 0.12618 | 0.12633 |
| Bingham, calibrated spread | 0.12620 | 0.12641 | 0.12633 |
| Bingham process, identity motion | 0.48304 | 0.48268 | 0.48291 |
| Bingham process, causal motion | 0.22537 | 0.22538 | 0.22539 |
| Gaussian process, identity motion | 0.49690 | 0.49653 | 0.49674 |
| Gaussian process, causal motion | 0.22399 | 0.22400 | 0.22401 |

Every row uses that filter's own forecast and process policy. Regularization predicts rho*P_mode*w, with rho fitted continuously on pooled terminal validation MSE. It is not a coarse scalar grid. The posterior mean is optimal for squared loss under its own maintained distribution; it need not dominate its mode on every fixed physical path.

## Posterior risk and directional error

| Method | Posterior vector risk (mm/s)² | Actual vector MSE (mm/s)² | Actual / predicted |
| --- | ---: | ---: | ---: |
| Bingham, calibrated spread | 0.4150 | 0.1264 | 0.305 |
| Bingham process, causal motion | 0.4544 | 0.2254 | 0.496 |
| Gaussian process, causal motion | 0.4526 | 0.2240 | 0.495 |

The risk is the within-posterior full-vector quantity w^T(M-M^2)w, averaged over the nine queries. Actual error uses the same full-vector loss, not a single-coordinate variance. A difference measures descriptive risk calibration in this physical experiment; equality is not guaranteed under the misspecified dynamics.

| Method | Physical direction | Signed bias (deg) | RMS error (deg) | RMS posterior SD (deg) |
| --- | --- | ---: | ---: | ---: |
| Bingham, calibrated spread | Motion direction | 0.135 | 0.810 | 1.267 |
| Bingham, calibrated spread | Cross-motion direction | -0.020 | 0.948 | 1.875 |
| Bingham process, causal motion | Motion direction | -0.018 | 1.109 | 1.478 |
| Bingham process, causal motion | Cross-motion direction | -0.016 | 1.243 | 1.850 |
| Gaussian process, causal motion | Motion direction | 0.013 | 1.106 | 1.474 |
| Gaussian process, causal motion | Cross-motion direction | -0.019 | 1.239 | 1.848 |

Directions are the actual mechanical normal-motion direction projected at the estimated normal and its perpendicular. Geometry enters scoring only. Antipodal signs are aligned in scoring so arbitrary eigensolver signs do not cancel bias. Bingham widths use shortest projective logarithms; Gaussian covariance is a local-coordinate approximation. These standard deviations are not central credible intervals or an HPD claim.

The same-parameter comparison isolates reporting shape. The recursive Gaussian additionally uses a local transition and a linearized observation update. Its normalized predictive log score is the linearized Gaussian score after a six-observation, data-only Bingham warm-up. Both methods pay for that initialization and for the common causal-forecast tracker. The Gaussian local ellipsoid is not an invariant-density Grassmann HPD region.

The secondary time-average raw-minus-process gain is 0.5849 [0.2737, 0.8962] (mm/s)², while Gaussian-minus-Bingham time-average gain is 0.02296 [-0.02353, 0.06945] (mm/s)². Memory and regularization were fitted to terminal point loss and process spread to predictive NLL. This comparison does not optimize all methods for the time-average endpoint. Tiny same-parameter reporting differences are descriptive and are not a certified tail-accuracy advantage.

## Other conditions and numerical flags

| Condition | Method | MSE (mm/s)² | Clean coverage | Width (mm/s) | Unresolved reports |
| --- | --- | ---: | ---: | ---: | ---: |
| Stationary plane | Bingham, calibrated spread | 0.0028 | 0.966 | 0.123 | 0 |
| Stationary plane | Bingham process, causal motion | 0.0028 | 0.966 | 0.123 | 0 |
| Stationary plane | Gaussian process, causal motion | 0.0028 | 0.965 | 0.122 | 0 |
| 20 cm sphere | Bingham, calibrated spread | 0.3970 | 0.996 | 2.067 | 0 |
| 20 cm sphere | Bingham process, causal motion | 0.4513 | 0.987 | 1.744 | 0 |
| 20 cm sphere | Gaussian process, causal motion | 0.4446 | 0.987 | 1.740 | 0 |
| 5 cm sphere | Bingham, calibrated spread | 0.1264 | 1.000 | 1.326 | 0 |
| 5 cm sphere | Bingham process, causal motion | 0.2254 | 0.995 | 1.395 | 0 |
| 5 cm sphere | Gaussian process, causal motion | 0.2240 | 0.995 | 1.392 | 0 |
| Forces mostly along one axis | Bingham, calibrated spread | 0.8946 | 0.948 | 2.120 | 0 |
| Forces mostly along one axis | Bingham process, causal motion | 0.9102 | 0.954 | 1.713 | 0 |
| Forces mostly along one axis | Gaussian process, causal motion | 0.9124 | 0.955 | 1.694 | 0 |
| Anisotropic resistance | Bingham, calibrated spread | 50.8445 | 0.083 | 1.660 | 0 |
| Anisotropic resistance | Bingham process, causal motion | 50.2218 | 0.140 | 1.484 | 0 |
| Anisotropic resistance | Gaussian process, causal motion | 50.2941 | 0.131 | 1.460 | 0 |

For forces mostly along one axis, the second and third force components are divided by five. Local Fisher information and actual retained-history turning are recorded independently. Anisotropic resistance breaks the assumed mobility map: its truth is the KKT response, not an isotropic projector surrogate. In the Stage 1 mode-only ablation, removing sensor noise does not remove its large response-prediction error; Stage 2 does not evaluate noiseless coverage.

Numerical flags concern mean integration, region integration, response quantiles and local chart validity separately. A fitted predictive moment does not certify density tails. The software gate can pass while empirical coverage fails; the gate does not test whether the physical simulation obeys a fitted stochastic transition.

## Cost and scope

The Bingham/Gaussian process filters cost 2.376/2.052 ms per observation including their shared forecast tracker, with uncertainty reports averaging 145.2/73.4 ms each for nine queries and three components. Cost scope: Median of three separate serial runs of preassigned small-sphere trajectory 0; shared forecast included. Component clocks are nested: inverse-fit time is inside prediction, and warm-up inside the Gaussian path. The end-to-end wall clock avoids adding nested clocks twice. Contended in-study timings remain in the original tables. These are implementation measurements, not hardware-independent rates or worst-case latency guarantees.

Cheap point and local Gaussian methods should remain the default accuracy/cost references. A full Bingham distribution needs a demonstrated advantage in decision loss, calibrated width/score, or broad global geometry to justify additional computation; a gain from using known force, enforcing a projector, or predicting motion does not establish that advantage. Posterior means are expected-response predictions, not contact controllers.

For these held-out contacts, the primary MSE and interval-score intervals do not establish a Bingham-process advantage over the recursive Gaussian. The raw motion predictor is better at the terminal endpoint; process propagation helps the retained policies over the full trajectory. This supports a motion/identification example, not a claim that full-distribution computation is necessary for this robotics task. The optional uncertainty-based abstention decision was not triggered by a material demonstrated premium over the cheap control.

The identity nominal frame does not make the deterministic modal theory vacuous. That bound pays for actual residual drift and can lose its sufficient condition. Credible-region calibration on arbitrary drifting paths is a different statement and is not implied by that bound. Learned G, temperature and assumed-density projection require separate assumptions and empirical or theoretical justification.

No manuscript is changed. A physical seven-joint extension is deferred to the bounded numerical feasibility check: increasing ambient dimension cannot repair unmodelled dynamics, and known kinematic constraints must be supplied to the structured competitor.
