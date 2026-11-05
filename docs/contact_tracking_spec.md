# Contact tracking: motion, uncertainty, and a higher-dimensional extension

Work in shreyasnb/Bayesian-ST, starting from the current dev branch. Record the actual starting commit. The snapshot reviewed for this prompt is f09d3fecd3af11fff5af708c8d1d31a35db75ca4.

Read AGENTS.md instructions wherever applicable. Create an isolated experiment and results directory. Preserve the frozen contact pilot, historical outputs, and manuscript files. This task is an implementation study, not a request to revise the paper or to produce a positive result.

## Objective

Find out whether the curved-contact failures arise mainly from stale state estimates, insufficient predictive uncertainty, or both. Determine whether retaining a Bingham distribution offers useful accuracy or calibration beyond a cheap, equally informed Gaussian approximation.

Finish the small contact study before implementing a new general-rank example.

## Current evidence to reproduce and inspect

Read these sources first:

- lib/contact_mechanics.py
- lib/contact_study.py, especially _snapshots and policy selection
- lib/contact_posterior.py
- lib/prediction.py, especially PredictiveBinghamFilter and geodesic_forecast
- lib/process_prediction.py, especially stochastic_predict and its supported ranks
- results/contact_pilot/{findings.md,selected.json,tuning.csv,coverage.csv,paired.csv}
- results/dev_prediction/motion/findings.md
- results/dev_prediction/process/findings.md

The original contact experiment uses n=3, k=2, d=10 N s/m, v0=0.02 m/s, dt=0.05 s, T=600, and known velocity-noise standard deviation sigma=0.001 m/s. It uses identity state prediction. The selected curved beta is 0.8, the lowest tested value in {1,.98,.95,.9,.8}.

The small-sphere mean query MSE is 1.08252e-6 (m/s)^2. Its improvement over its own mode is about 0.17%, and over pooled scalar shrinkage about 3.82%. Terminal nominal 95% clean-component coverage is about 66.4%; HPD state coverage is 23/64. Large-sphere clean coverage is about 98.5%, so curvature does not cause uniform failure across regimes.

Plant/projection and integration checks already pass. Accurate integration of the maintained Bingham density does not validate that density as a filtering posterior under unknown motion.

## Stage 1: isolate tracking bias and improve the mean prediction

Keep the observation model and true sensor variance fixed.

1. Repeat matched noisy and noiseless *mode-only* runs on the same mechanical trajectories. Remove measurement noise without changing the mechanics, input program, fixed update scaling, or memory. Record forecast error and corrected error separately. This isolates errors caused by treating old tangent planes as the current plane.

2. Log actual normal rotation per observation, rotation over the retained history, weighted input excitation, and local Fisher eigenvalues. On a sphere, dot(n)=dot(p)/R. Relate errors to realized rotation and information, rather than sphere radius alone.

3. Compare these causal policies:
   - Exponential identity memory. Start with beta in {.2,.4,.6,.7,.8,.85,.9,.95,.98,1}.
   - Rolling likelihood windows W in {1,2,3,5,8,12,20}, using the existing rolling implementation.
   - The existing causal Grassmann velocity forecast. Start with beta in {.6,.8,.9,.95}, lag in {2,8,16}, and damping in {.25,.5,1}. Start with a bounded mode-only validation scan and refine only promising candidates.
   - An oracle alignment using the true tangent changes, strictly as a diagnostic. Never label it a feasible estimator, use it to select practical policies, or infer a coverage theorem from truth-dependent rotations.

Use only information available through t-1 to predict the state at t. Never estimate the forecast using u_t or the current true normal. Record initialization, cut-locus fallbacks, and capped rotations. If a finite tuning boundary wins, extend it once on validation before freezing.

Shorter memory need not improve noisy performance: it reduces lag but also discards useful information. A preliminary check on eight existing validation trajectories per curved condition found residual noiseless error at beta=.8, while smaller beta increased noisy pooled mode loss. Treat this as a diagnostic, not a new held-out result.

Compare the same forecast and memory policies for point and posterior methods. The equality between the mode and matched weighted projector least squares remains a correctness check.

## Stage 2: separate the mean forecast from uncertainty spread

Only carry the few promising Stage 1 policies into posterior calculations.

A. Add a simple calibration control. Keep the raw recursion and modal estimate fixed, and evaluate q_{A_t/T} with a constant temperature T selected on validation. Use a small grid such as {.5,1,2,4,8}; expand only if a finite boundary wins. This changes posterior spread without changing its mode. It is generalized posterior calibration, not an estimate of physical measurement noise. Retain the true sensor variance when forming noisy-observation predictions.

B. Reuse stochastic_predict at n=3,k=2. Compare identity and causal nominal G with an explicit transition concentration selected on validation. A useful coarse kappa range may include {20,100,500,2000,10000,infinity}; inspect the implied angular step and refine only as needed. Reuse warm starts and cached transition coefficients. Preserve the distinction between the exact predicted first moment and its approximate projection back to one Bingham density. Log every inverse-moment failure and its time cost.

C. Include a cheap normal-line Gaussian filter using two local rotation coordinates, the same available inputs, and the same causal nominal forecast. Tune process spread independently of memory. Also consider a local Laplace approximation to the same Bingham parameter. Recenter coordinates when needed and respect n and -n equivalence. This is essential to distinguish a benefit of Bingham inference from a benefit of using any reasonable motion model or uncertainty calculation.

At least one covariance/temperature policy must be selected by a proper validation score, not coverage alone: intervals can trivially cover by becoming very wide. Use clean-response interval score where validation truth is available, and/or pre-correction predictive log score. State which information each criterion requires.

Do not fit a different noise variance merely to hide motion-model error. Keep actual sensor noise fixed in the primary study. A later variance-learning ablation must distinguish sensor-quality changes from tangent motion.

## Evaluation and scope

Use a small validation-only pilot first. Freeze a manageable set of policies, then evaluate on fresh held-out seeds, paired across methods. Use the existing 16-validation/64-test scale for the final retained comparison, or justify a smaller bounded pilot explicitly. Preserve the stationary plane and both radii, weak excitation, and a clearly separated anisotropic-resistance mismatch stress.

Primary task: clean query-velocity MSE averaged over the existing nine queries within each trajectory. Report time-averaged error as well as the predeclared terminal endpoint. Give paired intervals across independent trajectories. Retune scalar shrinkage fairly under the improved memory/forecast, alongside the point baseline.

Report:
- Forecast and corrected chordal error.
- Mean versus mode versus tuned shrinkage query losses.
- State-ball, HPD, and clean scalar-response coverage and widths.
- Proper interval score and pre-correction predictive score.
- Risk versus actual query loss, including directional diagnostics.
- Correction, forecast, moment, inverse-fit, and interval costs at their actual frequencies.
- Separate numerical convergence checks and all unresolved reports.

Use deterministic rank/codimension-one calculations where available. Do not rerun expensive unchanged plant checks or old notebooks without a reason. Keep current noise, process approximation, and numerical error distinct. The existing known-motion/balanced-input coverage result does not automatically apply to learned forecasts, fitted temperatures, or stochastic assumed-density prediction.

Write a compact findings table answering:
1. Does a causal forecast remove most of the lag?
2. Does temperature calibration already explain the uncertainty improvement?
3. Does the explicit transition improve the accuracy/width/calibration tradeoff beyond the cheap Gaussian control?
4. Is that improvement worth the measured cost?

A result that favors the Gaussian or point method is a valid outcome.

## Stage 3: higher dimensions, only after the preceding decision

Gr(3,7) has 12 local rotation coordinates, versus 2 for Gr(2,3). It is a harder general-rank integration problem, not a remedy for missing dynamics. The repository already contains a controlled Gr(3,8) motion experiment. Identify what new physical or computational question another example answers.

First assess Gr(6,7): a seven-joint arm maintaining one regular surface-contact constraint. This retains the rank/codimension-one numerical path, although runtime still needs measurement.

A physical Gr(3,7) candidate is a seven-joint arm maintaining fixed end-effector orientation (three independent constraints) while sliding on an unknown smooth surface (one constraint). For regular configurations, form J_c(q) in R^{4x7}. Under the stipulated isotropic joint admittance,
    d qdot = tau + J_c(q)^T lambda,
    J_c(q) qdot = 0,
so
    qdot = P_{ker J_c(q)} (tau/d).

Generate motion from this constrained mechanical system, rather than prescribing rotations of a hidden subspace. Check rank, constraint drift, force balance, joint limits, and units. State the assumptions that make joint damping and normalized noise isotropic.

The fixed-orientation constraints and robot kinematics are known structure. Give the same available noisy encoder positions and kinematics to every applicable estimator, and include a structured constraint/null-space comparator. Otherwise the experiment can unfairly reward an unstructured 12-parameter estimator for ignoring useful known geometry. This example does not automatically explore every possible Gr(3,7) direction.

Before a large experiment, benchmark general-rank posterior means and tails on representative n=7,k=3 parameters. mean_bingham uses the general-rank sampler here, and process_prediction currently rejects general-rank inverse-moment fitting. Do not silently route this example through rank-one formulas. Validate independent chains, effective sample sizes, Monte Carlo error, sensitivity to initialization, and wall time. Keep numerical error below any claimed between-method gain. A general-rank process projection requires a separately validated implementation.

Normalize comparisons by actual input information and noise, reporting both total input energy and Fisher eigenvalues. Increasing n while keeping arbitrary component scales fixed is not an informative scaling experiment.

Stop at a documented feasibility result if the general-rank numerical cost is prohibitive. Do not add unsupported calibration or scalability claims.

Deliver a reproducible isolated experiment, frozen configurations, concise figures and tables, and a short recommendation for the paper. Do not change the manuscript in this task.

