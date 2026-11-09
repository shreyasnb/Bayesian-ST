# Implement and evaluate the revised Bayesian subspace tracker

Use main.tex and its included supplementary.tex as the specification. The primary experiment learns from measured vectors alone; contact additionally uses recorded inputs. Report negative or inconclusive results. Do not tune until a desired advantage appears, invent results, or present planned tests as evidence.

## Repository and reproducibility

- Work in https://github.com/shreyasnb/Bayesian-ST on branch **test only**. Inspect remotes, branches and uncommitted work first. If test is absent, create it from the intended dev revision. If it exists, continue without resetting or overwriting work. Never commit or push changes to main or dev, and never force-push. Push test when implementation and checks are complete.
- Preserve existing notebooks and archived results. Add clearly named study modules and results folders. Record the source commit, model, variance convention, dimensions, seeds, parameter grids, selected settings, tolerances and source hashes in a manifest.
- Inspect the current checkout rather than assuming names are unchanged. Useful entry points include lib/models.py, lib/filter.py, lib/em.py, lib/theory.py, lib/plotting.py, lib/moments.py, lib/reference.py, lib/array_story.py, lib/experiments.py and lib/contact_mechanics.py. Contact-specific prediction, evidence and quadrature modules may already exist. Reuse verified components.
- Use separate training, validation and test episodes, common random numbers across paired methods, and immutable test seeds. Average within an episode before estimating uncertainty across episodes. Publish raw tables and failed diagnostics.

## 1. Algebra and information

Start with low-dimensional deterministic references, then smoke studies before full Monte Carlo runs.

1. Compare normalized likelihoods with direct Gaussian implementations. Check trace-gauge invariance, restored scalar terms in evidence, additive corrections, fixed-state block versus sequential assimilation, and one assimilation per observation. Verify marginal Bingham-mode/EW-PCA equality at matching scales, initialization and memory.
2. Implement real marginal Fisher information for one fixed subspace, known positive scales and L independent observations:
   \[
   F_{\rm marg}=L\,\frac{(\tau^2)^2}{s(s+\tau^2)}I_{k(N-k)}.
   \]
   Retain the conditional formula using recorded V. Check information additivity for an independently prescribed mixture of observation types. Never reconstruct hidden inputs from observations to evaluate conditional information.
3. Check Fisher information through empirical score covariance, negative expected likelihood Hessian, and Gaussian KL curvature in the paper's tangent chart and vectorization. Repeat finite differences at several step sizes. Discounted Hessians use weights w; discounted score covariance uses w squared. These are different matrices.
4. Check the realized Bingham Hessian against selected/complementary eigenvalue gaps. Keep posterior curvature distinct from expected likelihood information. The isotropic marginal model has isotropic expected information; a finite-data posterior need not be isotropic.
5. Audit complex conventions. Proper-complex likelihood increments omit the real factor 1/2. In real/imaginary tangent coordinates, the marginal Fisher coefficient is twice the real-model coefficient at matching scales, and the tangent dimension doubles. Verify directly. Realification uses covariance one-half of the block matrix with rows (Re R, -Im R) and (Im R, Re R), preserving complex structure. Passing complex arrays to the old real bound routine is invalid.

Use independent quadrature or a trusted small-state reference for moments and normalizers. Real rank-one/codimension-one sphere routines do not automatically validate complex Grassmann integration. Record accuracy, chain disagreement, effective sample size, reference resolution and failures.

**Figures:** analytical/empirical Fisher spectra; error heatmaps over signal-to-noise ratio and count; exact KL contours against Fisher ellipses; distributions of posterior directional curvature across repeated datasets. Align tangent bases before matrix comparisons.

## 2. Tracking bounds

Implement the revised theorem from the supplied files, including exact constants, initialization, drift terms and sufficient conditions. Older equation numbers and lib/theory.py may differ.

For marginal data use the exact paper quantities
\[
w_{t,j}=\beta^{t-j},\quad m_t=\sum_jw_{t,j},\quad
b_t=\sum_jw_{t,j}^2,\quad a_0=\min_{c\in\mathbb R}\|A_0-cI\|_2,
\]
\[
\eta=\frac{\tau^2}{2s(s+\tau^2)},\quad
h_t=N\log9+\log(2/\delta),\quad \gamma_t=\eta\tau^2m_t,
\]
\[
d_t=\eta\tau^2\Big\|\sum_jw_{t,j}(P_{\mathcal X_j}-P_{\mathcal X_t})\Big\|_2,
\quad \varepsilon_t=\beta^ta_0+d_t+
4\eta(s+\tau^2)(\sqrt{b_th_t}+h_t).
\]
If epsilon_t < gamma_t/2, the fixed-time modal error is at most 2 sqrt(k) epsilon_t/gamma_t with probability at least 1-delta. For native proper-complex data use eta_C=tau^2/[s(s+tau^2)] and h_C=2N log9+log(2/delta), conjugate transposes, and the same conservative fluctuation constant 4. Use the conditional theorem and its separate complex extension from supplementary.tex; do not substitute marginal constants there.

- Marginal bounds use population covariance, weighted covariance fluctuations and the marginal reference eigengap. Conditional bounds use recorded inputs. Check both matrix decompositions before checking final errors.
- Theorem runs require fixed true scales, fixed memory, deterministic subspace paths and deterministic transforms/designs where stated. Hold the path fixed across repeated signal/noise realizations. Fitted noise, data-derived rotations or adaptive inputs invalidate assumptions unless separately covered.
- Begin with a static state, beta=1, correct model and a sufficiently long horizon. Seek a nonvacuous regime on validation data; a short smoke run can have no valid times. Then vary drift and memory independently. Expensive moment integration is needed only at selected checkpoints.
- Preflight before Monte Carlo: static A0=0 gives epsilon/gamma=4(1+s/tau^2)(sqrt(b_t h_t)+h_t)/m_t. For beta<1 its limit is 4(1+s/tau^2)[sqrt(h_t(1-beta)/(1+beta))+h_t(1-beta)]. Use h_C for complex data. A fixed memory setting can leave the bound permanently vacuous; a longer horizon alone will not fix that. Distinguish gap validity from an envelope below the geometric ceiling sqrt(min(k,N-k)).
- Save actual operator perturbation, theoretical allowance, gap, drift, noise, initialization, distance ceiling, sufficient-condition flag, modal error and violations. Never replace a failed sufficient condition with a zero bound or discard it.
- Estimate fixed-time violation frequencies across independent episodes at preassigned times, with binomial intervals. Trajectory-wide claims need an allocated failure budget. Ground-truth quantities are evaluation-only, not online confidence radii.

**Figures:** error, theorem envelope and distance ceiling; shaded insufficient-condition intervals; validity phase diagram over drift/memory; perturbation allowance against actual perturbation; static sample-size scaling. Report the informative fraction.

## 3. Primary example: beamforming from snapshots alone

Use interference-only training:
\[
u_t=h_tc_t+e_t,\quad \|h_t\|=1,\quad
c_t\sim{\cal CN}(0,\tau^2),\quad e_t\sim{\cal CN}(0,sI).
\]
Neither c_t nor hidden geometry enters the estimator. Start at rank one and modest sensor count. A ULA steering path can generate h_t; specify the desired steering vector independently. Explain interference-only training. Desired-signal contamination is a separate mismatch test.

- Compare static and deterministically drifting signatures. The primary tracker receives no oracle motion. Separate identity transport/tuned fixed memory from stochastic prediction and causal motion estimation.
- Change sensor noise, interference power and direction separately. Higher-rank unequal powers are model mismatch, not exact equal-power likelihood data.
- Compare posterior-mean covariance beamforming, its own modal covariance at identical scales, EW covariance/PCA, tuned diagonal loading, and the 2017 Bingham tracker if its implementation is verified. Do not relabel another routine as that baseline.
- Isolate anisotropy with M_iso=alpha P_hat+xi(I-P_hat), alpha=tr(P_hat M)/k and xi=(k-tr(P_hat M))/(N-k). This preserves trace, mode and expected squared chordal loss. Verify that its covariance equals (s+tau^2 xi)I+tau^2(alpha-xi)P_hat, so its beamformer equals a modal beamformer with loading ell=(s+tau^2 xi)/(alpha-xi)-s when alpha>xi; handle equality as isotropic covariance. Include validation-tuned loading. Hold the posterior fixed for decision ablations and the decision fixed for filter comparisons. Use N>=3 for rank-one anisotropy tests; N=2 has no extra dispersion for this comparison. M_iso alone specifies a decision control, not a predictive law: use a specified isotropic Bingham distribution for probability comparisons.
- Use E[sI+tau^2 P | data]. Factor this into sI+tau^2 M only for fixed or explicitly conditioned plug-in scales. EM-based intervals condition on fitted scales and omit their uncertainty. The beamformer minimizes posterior expected output power at unit desired gain; it does not maximize posterior expected SINR.

**Main panels:** B1 physical array/angle setup; B4 directional posterior dispersion and power fans at preassigned checkpoints; B5 matched, paired held-out decision costs. Supplementary panels cover Fisher information, curvature, bounds, scores, calibration, and adaptation as indexed below.

## 4. Secondary example: contact robotics

Use an independently solved ideal bilateral contact plant:
\[
d\dot q=f+J_c(q)^\top\lambda_c,\qquad J_c(q)\dot q=0,
\qquad v=f/d,\quad u=\dot q+e.
\]
The estimator sees applied force and noisy instantaneous velocity, not true normals, surface radius or geometry-derived transport. Keep maintained contact, known damping, negligible inertia and friction assumptions explicit.

- Include a plane and curved guides. Solve KKT mechanics independently and check the projection residual.
- The theorem controls use directly observed velocity plus independent Gaussian sensor noise. Differencing noisy positions produces correlated velocity errors and belongs in a separate sensor-model test.
- Separate physical trajectories from theorem controls. Radius and force changes can alter both excitation and subspace speed. Freeze a deterministic subspace path while varying only the recorded design for clean Fisher/bound tests; label these controlled observation tests.
- Compare posterior-mean response, its own modal response, tuned scalar shrinkage and verified paired-input least squares with local Gaussian/bootstrap uncertainty. All receive identical pairs. Include weak excitation and abrupt motion as declared stresses.
- Known anisotropic damping requires the scaled-coordinate construction and the correct transformed noise law. Unmodelled damping anisotropy, friction and contact loss are mismatch cases.
- Evaluate fresh held-out directions, posterior prediction risk, and realized admissibility-decision costs. A supplied force is a physical input; arbitrary hidden simulator vectors must not be substituted for available measurements.

**Main panels:** C1 physical path, tangent planes and recorded inputs; C2 excitation/information versus direction; C3 held-out response clouds and realized response/command costs. Supplementary panels contain bound, coverage, noise-learning and mismatch checks.

## 5. Noise learning and reliability

The revised block EM assumes one constant subspace and a fixed boundary prior excluding the entire block. Every E-step rebuilds from that boundary prior. The old product of likelihoods under separately recorded single-sample priors is a different fitting objective.

- Implement disjoint stationary-block references first. Check marginal ordered-variance updates and conditional expected residuals, including subspace uncertainty and noise floors.
- Use positive interior signal-variance starts and compare several starts by normalized evidence. With a uniform boundary prior, tau^2_old=0 gives M=(k/N)I, a=b and another zero signal-variance update. Include this absorbing boundary as a declared initialization test.
- Evaluate full normalized block evidence independently, restoring trace-centering constants. Exact reference EM must not decrease this fixed objective beyond tolerance. Approximate integration needs a safeguard and diagnostics.
- Compare calibrated fixed scales, misspecified fixed scales, evidence/EM fitting and residual-only fitting. Score a new observation before fitting or assimilation. Test sensor-noise changes at fixed motion, then motion changes at fixed sensor variance. Record whether motion mismatch inflates learned noise.
- Keep fixed-scale theorem checks separate from adaptive tracking. Fixed-block ascent does not establish monotone online scores, convergence under drift or an inferred forgetting factor.
- Report nominal/empirical coverage, widths at matched coverage, predictive log score and risk calibration. Distinguish state regions, clean-response intervals and noisy predictive intervals. Overlapping time samples are not independent coverage trials.
- For exact static beta=1 inference with fixed known scales, first draw independent states from the stated prior and data from the likelihood. Check credible mass over this joint prior-predictive experiment. Then report fixed-state and drifting-path repeated-data coverage separately; the first calibration identity does not imply the latter.
- Compute predictive log scores before assimilation from c_t(theta)+psi(A_t^-+B_t(theta))-psi(A_t^-), using the full normalized singleton likelihood and the correct real/complex normalizer. The Gaussian with covariance sI+tau^2 M matches only a predictive moment. Label scores from that surrogate separately from mixture scores.

## Stages, figures and acceptance

First run algebra tests and references: N<=6, rank one and one rank-two real case, three smoke seeds, selected moment checkpoints. Smoke runs check execution and numerical validity, not performance. Add a long-horizon static bound case if short runs are vacuous. Resolve failed algebra before expensive studies.

Then freeze dimensions, grids, checkpoints, tuning rules and test seeds. Choose independent episode counts for a declared Monte Carlo precision and report achieved intervals. Separate runtime studies from reference-quality integration studies.

Acceptance gates: deterministic likelihood/increment identities agree to scaled numerical precision; Fisher checks agree within declared finite-difference and Monte Carlo errors; reference moments and evidence stabilize under tighter quadrature/sampling settings; exact block EM respects evidence ascent tolerance. Record each tolerance in configuration. A statistical bound check must report violation intervals and informative times, rather than declare success from zero observed violations. Decision claims require paired held-out effects with uncertainty and matched controls. Failures remain in the report.

Retain lib/plotting.py conventions: actual LaTeX, serif fonts, simple mathematical labels, restrained colors and **200 dpi** PNGs. Export matching PDF figures and SVG where supported. Use geometry, contours, heatmaps, uncertainty fans, calibration curves and paired distributions instead of only lines with error bars. Comparable panels share scales. Label what every interval represents.

Deliver executable notebooks, smoke/full configurations, raw results, diagnostics and a manifest mapping **claim -> routine -> metric -> figure -> assumptions -> status**. Status is passed, failed, inconclusive or not applicable. Cover every main theorem and task identity in the supplied draft. Validate algebra and references before full studies, and report Monte Carlo uncertainty. No improvement, poor calibration or vacuous bounds are legitimate outcomes.

Use these figure IDs in filenames, notebook headings and the manifest. B1/B4/B5 and C1/C2/C3 replace main.tex placeholders; others support supplementary.tex.

| ID | Claim or check | Figure/metric |
|---|---|---|
| B1; C1 | Observation law and available measurements | Array/surface geometry; independent plant residuals |
| B2a; C2 | Fisher information and excitation | Score/Hessian/KL agreement; spectra and directional ellipses |
| B2b | Expected information versus realized curvature | Posterior eigenvalue gaps across repeated datasets |
| B3; C4 | Fixed-time modal bounds | Error/envelope; perturbation decomposition; informative fraction and violations |
| B4 | Posterior anisotropy | Directional dispersion and matched-isotropy power maps |
| B5; C3 | Posterior prediction and decisions | Paired held-out costs; response risk calibration |
| B6; C5 | Uncertainty calibration | Coverage/width, reliability maps, pre-observation proper scores |
| B7a; C6a | Block EM | Normalized evidence ascent and reference agreement |
| B7b; C6b | Reliable scale adaptation | Noise recovery; separate noise-change and motion-change panels |
| S1 | Computational feasibility | Runtime, moment accuracy, effective samples and failure counts |

Provide a results note identifying evidence, numerical failures, unsupported claims and manuscript plots. Execute notebooks from a clean kernel, inspect figure labels/clipping, and push only reviewed changes to test.

