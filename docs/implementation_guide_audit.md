# Implementation-guide audit

This audit starts from commit `6e78cc9911e7e78452633f13dd947afa669c3470` on `test`. It distinguishes the uploaded `implementation_prompt.md` from manuscript text and from historical experiment specifications. The user requested an audit and additions against this guide; its prospective test descriptions are not scientific evidence. The baseline inventory is in [`coverage_plan.json`](../results/implementation_audit/coverage_plan.json), with 57 individual checks, the guide's SHA-256, the responsible routine, the metric, the figure IDs and the assumptions.

The guide names `main.tex` and `supplementary.tex`, but neither file exists in this checkout. Agreement with those exact source files is therefore **inconclusive**. Explicit formulas in the uploaded guide can be implemented and tested. The supplied `L4DC2027_SB_BM_CM-5.pdf` is separately identified by SHA-256 in `provenance.json`; only its theory and appendices are used here. Agreement with that PDF does not establish agreement with unavailable TeX revisions. No manuscript has been edited.

## What was already present

The previous redesign already supplied strong static proper-complex beamforming comparisons, an independently solved ideal bilateral contact plant, matched-isotropy controls, posterior task identities, normal-line sphere densities, higher-dimensional rank-one/complement calculations, and a serial scaling study. Its main notebooks contain independent validation/test episodes and retain losses to adaptive covariance references and the extended Kalman filter. Its archived results remain evidence under their recorded producer versions.

That implementation did **not** establish the guide's revised fixed-boundary block EM procedure. `lib/em.py` and `beamforming_inference.recorded_prior_em` use separately recorded per-observation priors, a different objective. Likewise, the earlier passive covariance certificate has different constants from the guide's specified revised marginal bound. Neither historical routine can be renamed to satisfy the new requirement. Existing fixed-state uncertainty plots also do not substitute for the requested joint prior-predictive calibration experiment.

## Checks and figure map

The following table records the baseline and the scope of the additions. The finalized checklist now binds nine completed scientific component manifests and four reviewed figure manifests. Implementation requirements, empirical outcomes and unavailable-source comparisons have separate statuses; the table does not turn a planned test or an unavailable reference into evidence.

| IDs | Claim and required evidence | Existing support at the baseline | Required current check |
| --- | --- | --- | --- |
| B1 | Snapshot-only interference training; physical array and available measurements | `array_geometry_study.dataset` and passive-scatter/PCA tests | Keep latent coefficients, true geometry and desired steering out of correction; identity-transport primary motion study |
| B2a; C2 | Marginal, conditional and mixed Fisher information | Real/complex passive information helpers and conditional theory tests | Score covariance, negative expected Hessian and Gaussian KL curvature in aligned charts; several finite-difference steps; independent recorded designs; `w` versus `w^2` |
| B2b | Expected information differs from realized posterior curvature | Modal covariance/PCA equality and posterior eigengap plots | Repeated-data gap distributions and exact modal Hessian/eigengap identity |
| B3; C4 | Fixed-time modal bounds | Older passive certificate; real conditional draft wrapper | Exact guide constants, initialization, drift and noise terms; decomposition checks; preflight limits; geometric ceiling; valid/informative flags and per-time binomial intervals |
| B4 | Directional information beyond mode and one radius | Exact same-mode/same-distance-distribution pair and axial controls | Retain specified axial law for probability reports, verify the equivalent modal-loading identity, and separate selected-data identities from Monte Carlo performance |
| B5; C3 | Posterior responses and expected decisions | Strong static covariance controls, identical-likelihood point estimates, ridge, residual PCA, local Gaussian and EKF | Fresh held-out effects with paired episode intervals; fixed posterior for action ablations; fixed action rule for filter comparisons; current-state predictions distinguished from forecasts |
| B6; C5 | Reliability and proper predictive scoring | Numerical tail checks and explicit fixed-state support-boundary limitations | Joint prior-predictive calibration, separate fixed/drifting-state coverage, clean/noisy targets, widths, risk calibration and normalized pre-observation mixture scores |
| B7a; C6a | Stationary fixed-boundary block EM | Legacy recorded-prior EM only | Rebuild every E-step from one boundary prior excluding the block; ordered-variance and expected-residual updates; normalized evidence/reference agreement; interior multistarts; absorbing zero-signal boundary; safeguarded ascent |
| B7b; C6b | Scale adaptation and mismatch | Historical additional models under different objectives | Known/misspecified fixed scales, block EM and residual-only estimates; separate noise changes from motion changes; plugin uncertainty clearly scoped |
| C1 | Contact mechanics and known physical inputs | Independent KKT plant for a plane and curved guide | Keep true normals/surface radius out of estimation; transformed damping and sensor-noise law checked separately |
| C2; C4 | Controlled excitation and motion stresses | Equal-energy static command programs | Fixed deterministic normal paths while varying designs; weak excitation and abrupt motion labeled as stresses; physical radius/force changes not treated as clean information interventions |
| C5 | Sensor/model mismatch | Matched isotropic bilateral contact only | Known anisotropic damping with transformed covariance; unmodeled damping/friction/contact loss as mismatch; correlated velocity errors from differenced positions |
| B6 | Array mismatch | Exact rank-one/equal-power snapshot model | Desired-signal contamination and unequal higher-rank powers separated from the exact likelihood |
| S1 | Computational feasibility and numerical limits | Serial one-thread scaling through `N=128`; independent low-dimensional references | Keep means, tails, eigensystems and full estimators separate; reference resolution and numerical flags explicit; no arbitrary-rank claim |

A verified implementation of the optional 2017 Bingham tracking reference is absent. Its use is **not applicable**, rather than a missing result being replaced by a similarly named routine. The verified modal/PCA, diagonal-loading, Ledoit--Wolf, GROUSE, conditional least-squares, ridge and Gaussian comparisons remain individually identified.

## Interpretation rules

Posterior expected-power or expected-command optimality is conditional on the stated belief and loss. It does not require a win at every fixed physical state or against validation-selected regularization. The beamforming objective is output power at unit desired gain, not posterior expected SINR. A contact expected-penalty command is not a guaranteed feasible controller. A covariance-only method does not acquire an invented projector-response estimate.

The isotropized mean preserves trace, mode and expected squared chordal loss. It defines a decision control; probability comparisons require a specified distribution. Proper-complex rank-one dimension two has no nontrivial complementary directional-dispersion control. Contact normals are unoriented: `n` and `-n` describe one plane. Complex higher-dimensional sphere pictures are labeled density sections, not normalized marginals of the entire state.

For known anisotropic damping, scaled dynamics and sensor covariance must be transformed together. If physical velocity noise has covariance `sigma^2 I`, scaling by `D^(1/2)` gives covariance `sigma^2 D`, generally outside the isotropic conjugate likelihood. Physical covariance `sigma^2 D^(-1)` instead becomes isotropic. An exact mechanical projection identity alone does not validate the wrong transformed likelihood.

A finite local-Gaussian result can fail its chart-validity check. Such cases remain unresolved, rather than validated wins for the global posterior. Numerical integration accuracy, posterior mass, prior-averaged calibration, fixed-state repeated-data inclusion and deterministic modal bounds are distinct checks. Zero observed bound violations must retain the independent episode denominator and a binomial interval; prescribed-time results do not imply trajectory-wide coverage.

## Acceptance record

The final claim manifest must map **claim → routine → metric → figure → assumptions → status**. Its only status values are `passed`, `failed`, `inconclusive` and `not applicable`. Smoke data establish execution and algebra checks, not paper performance. Every final result must bind the current source and configuration, preserve raw failed diagnostics, use independent validation/test episodes and avoid tuning after test outcomes. Revised scientific sources invalidate the numerical gate before example regeneration.

The scientific claim report is produced only after completed current-source studies and reviewed figure exports. Notebook generation and clean-kernel execution then have their own execution record and final handoff checks. A scientific report does not claim that a later notebook execution has already occurred. Historical notebooks and results are preserved under their original hashes; new results live under `results/implementation_audit/`.

The claim map also names the reused PDF results explicitly: local versus uniform identifiability and worst-orientation information; posterior mean-response risk and soft-action identities; and RMS/HPD/ball geometry with regional task bounds. They are backed by the current general tests and unchanged source-verified geometry results. Finite checks illustrate the stated results and do not replace their mathematical proofs.

## Completed scientific review

The current general gate passes **241 tests**. All five new paper components and all four reused paper components pass their source/table verification; 567 files from the previous handoff remain byte-exact in the separate archive. Those checks concern executed scientific evidence. Saved notebook execution is verified separately after report generation.

| Component | Completed checks | Scope |
| --- | --- | --- |
| Information | No failed algebra/score cases; final-step Hessian and KL relative errors about `1.0e-6`; native/reference mean error below `6.1e-13` | Includes real rank-two cases, prescribed mixed observations, distinct real/complex measures and independent reference refinement |
| Information versus count | 24 cells, 8,192 independent datasets per cell; no precision flags; maximum standardized product discrepancy `2.727` | Actual observations per dataset are `1,4,16,64`, at `s=0.2,1,4` and `tau2=1.3`; correlated prefixes are not extra replicates |
| Tracking | 7,680 fixed-time reports in 60 cells; 2,048 valid informative reports in 16 cells; zero eligible violations | Each eligible cell has 128 independent episodes and a two-sided 95% upper violation limit `0.028408`; invalid cells have no violation interval |
| Block EM | 96 independent blocks, 360 fits; all stationary fits meet the declared tolerance; no numerical failures or rejected updates | Minimum evidence increment `-7.39e-13` is inside the `1e-8` ascent tolerance; this is fixed-objective numerical ascent, not a global/online convergence result |
| Reliability | All 1,120 jobs complete; zero native or finite-integration precision flags | Numerical resolution does not imply calibration, model correctness or a performance advantage |

The implementation requirements and empirical outcomes have separate rows in the claim map. All 24 joint prior-predictive posterior summaries contain the nominal mass in their pointwise 95% binomial intervals; this finite experiment supplies no simultaneous or multiplicity-adjusted guarantee. By contrast, at nominal 0.95 the stationary inference model under drifting paths includes the true state in only 13/128 beamforming HPD reports and 42/128 contact HPD reports. Fixed-state beamforming noisy-response inclusion is 114/128. These are retained negative calibration diagnostics, not contradictory claims about an assumed-stationary Bayesian identity or the modal perturbation theorem.

All requested scale-comparison procedures run, but four of 192 selected adaptation fits reach their finite iteration budget. The finite values remain numerically resolved; universal adaptation convergence is inconclusive. Width multipliers are selected independently on validation episodes, and one validation target is unattained at the prescribed cap. Some test inclusions also miss the intended matched coverage. Such rows do not establish a width advantage at achieved matched test coverage. No credible calibration for fitted-scale uncertainty is claimed.

The optional 2017 reference remains unavailable, and exact agreement with absent TeX sources remains inconclusive. The derived proper-complex conditional certificate is explicitly separated from manuscript-certified real conditional and guide-specified marginal formulas. Negative comparisons and unsupported extensions are not converted into passed scientific claims by the fact that a program executed.

Final presentation review approved all 12 new paper layouts and both recolored main profiles. Independent checks confirm all 17 requested figure IDs, current source/input/export hashes, actual LaTeX, large 200-dpi PNGs, and PDF/SVG exports for the new guide figures. The coverage audit is finalized against these completed artifacts. The strict report generator records acceptance of the actual scientific claim manifest when it publishes that report; clean notebook execution and the requested push have their own later handoff records.
