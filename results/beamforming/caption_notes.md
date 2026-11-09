# Proposed figure captions

These notes are external to the title-free PDFs. Page and panel conditions are in figure_manifest.json.

## setup.pdf

Available measurements and task geometry. Beamforming uses interference-only proper-complex snapshots; the desired unit steering vector enters the distortionless decision. The displayed beampattern uses a predeclared saved episode, not another independent repetition. Contact uses an independently solved maintained bilateral constraint with known resistance, applied input and directly measured noisy velocity. Positions and normals are generation/scoring information.


## task_performance.pdf

Task losses on common independent test episodes. Held-out disturbance or query losses are averaged within each episode before computing pointwise Student 95\% intervals. Only the posterior-mean curve carries a confidence band in the main absolute-loss panels. The contact paired panel isolates the same-data modal, matched-isotropic and regularized spectral controls; response-only GREAT remains in its absolute-loss panel and the full supplementary grid. Contact paired points show mean loss differences and pointwise Student 95\% confidence intervals across independent episodes, including every episode outlier; they are not posterior intervals. Beamforming paired boxes show the distribution of episode differences, not confidence intervals. The same-$A_t$ modal and matched-isotropic controls isolate posterior moments; GREAT and application-specific controls have separately validated settings and their stated information. Beamforming power ratios are $10\log_{10}(P/P_{\mathrm{oracle}})$, where the denominator is true-covariance oracle output power. Oracle quantities are evaluation-only. This cohort has 256 independent test episodes per condition; snapshots and queries are averaged within each episode.


## posterior_anisotropy.pdf

a real-coefficient section $h=U_3x$, $x\in S^2$, of the ambient complex-line density, shown relative to its mode; this is not a normalized marginal or a full posterior. A matched isotropic mean is a task control, not an invented probability law. Visible anisotropy and empirical task benefit are separate comparisons. Complementary mean eigenvalues compare the full and isotropic means. The lower panel is the empirical distribution of paired isotropic-minus-full held-out power-ratio losses at n=L=64, in dB; it is not a confidence interval.


## noise_adaptation.pdf

Actual causal variance learning from completed nonoverlapping past blocks. Each observation is corrected once; a fit updates only subsequent variances while retaining the streamed subspace belief. Known coordinate changes are aligned before block fitting. Curves show saved trajectory/episode summaries; variance bands are interquartile repeated-episode spread, not a posterior over variance. Future-task and pre-observation score intervals are pointwise 95\% intervals across independent episodes. Each page retains one predeclared noise-, motion-, or signal-power intervention. The contact supplementary density pages show one predeclared episode: panels (a,b) of the first density page use $t=32,96$, and panel (a) of the following density page uses $t=192$; pre-correction $A_t^-$ is converted to the wrist-frame normal law $q_{-A_t^-}$ with normalizer $\psi_1(-A_t^-)=\psi_2(A_t^-)-\operatorname{tr}(A_t^-)$. The three orthographic spheres share a density scale; panel (b) of the following density page shows the physical-query risk at all three checkpoints, $v_t^T(M_t^--(M_t^-)^2)v_t$ at the recorded physical query. This is individual belief uncertainty, distinct from the repeated-episode variance bands and from calibration. Empirical-Bayes plug-in scales omit their uncertainty and are outside the fixed-true-variance theorem.

Page 1: noise_change.
Page 2: smooth.
Page 3: power_change.

## information.pdf

Independent Gaussian Fisher, score and curvature checks. Expected information is distinguished from finite-data posterior curvature. All matrix comparisons use aligned tangent coordinates; proper-complex information uses real/imaginary coordinates and its correct factor. Recorded-input excitation and marginalized-input information use the stated common data law. Monte Carlo dataset count differs from observations per dataset. Reused mathematical reference cohorts keep their actual source hashes and repetition counts.


## posterior_curvature.pdf

Invariant-density Hessian and local accuracy diagnostics. Empirical bounded log-coordinate spread is compared with $H_A^{-1}$ only under recorded locality conditions; broad/nonlocal cases remain visible. The stationary inverse-Fisher study uses fixed known variances and $\beta=1$, reports estimation bias and local fractions, and treats inverse Fisher as a local reference rather than a global intrinsic Cramer--Rao equality. The real graph-density volume term is a separate quantity, retained in the tables. Forgetting sampling references use squared weights.


## tracking_bound.pdf

Recomputed sharp fixed-time envelope $\sqrt{\min(k,n-k)}\,\epsilon_t/(\gamma_t-\epsilon_t)$ under $\epsilon_t<\gamma_t/2$, with the exact aggregate conditional drift and the appropriate real/complex constants. True path, gap and drift are analysis quantities, not online confidence radii. Invalid conditions remain marked and receive no violation test. Error bands describe repeated-episode spread. The geometric ceiling is not used to clip the theorem. Phase tables and exact binomial violation intervals retain every declared condition.


## block_updates.pdf

Exact discounted aggregation, delayed actions and constant-subspace approximation are separate checks. Aggregation retains the boundary factor $\beta^L$, within-block weights, per-observation variances and recorded noncommuting transport. Numerical zeros are displayed at a $10^{-16}$ plotting floor; full tolerances remain in the tables. Delayed actions are evaluated only after a block arrives, with common endpoints and physical output age reported separately. The block-likelihood reference resets to the uniform prior, $A_0=0$, and uses $q_{\sum_{j=t-L+1}^t B_j}$ with $\beta=1$ within the raw $L$-observation window. It does not retain the singleton history. Under motion its one-state likelihood is an approximation, distinct from the algebraic aggregation identity. Processing/action cadence cost is reported in the computation output.


## additional_comparisons.pdf

Complete declared dimension, memory and method comparisons. Heatmaps show comparator-minus-full posterior-mean loss; positive values favor the full mean. Hatching marks cells whose paired pointwise Student 95\% interval includes zero. No simultaneous-coverage claim is made. Test episodes never tune a setting. Paired differences include losses and inconclusive intervals; broad grids are exploratory without simultaneous coverage. Contact GREAT is response-only and cannot isolate the benefit of known inputs. The same-parameter local Gaussian and recursive EKF are distinct supplementary references with their validity diagnostics retained. Identical modal/PCA or conditional spectral least-squares solutions are not counted as independent algorithms.

Page 4: static.
Page 5: smooth.
Page 6: abrupt.
Page 7: noise_change.
Page 8: power_change.

## variance_estimation.pdf

Fixed-boundary stationary-block EM reference. Every E-step reconstructs the full block from one saved boundary prior. Evidence curves are shifted by their own initial value, so final heights do not rank starts; absolute normalized evidence selects starts. Both positive starts and the uniform-prior zero-signal boundary are retained. Noise floors and pooled isotropic M-steps are tested in the numerical tables. Contact residual updates include posterior response uncertainty, not only a squared residual using the mean projector. These reference blocks are not online adaptive-tracker episodes.


## predictive_calibration.pdf

Actual-recursion calibration with distinct state, clean-response and noisy-next-observation targets. Pages separate joint prior-predictive sampling, fixed-state repetitions and drift diagnostics. Vertical intervals are pointwise exact binomial uncertainty across independent episodes, not credible uncertainty. All intervals are evaluated at their declared time before the scored future observation is seen. Appended pages show 95% report-size distributions at t=64 for every probability regime. State-ball sizes are chordal radii, obtained by taking the square root of each recorded squared-distance upper threshold; HPD energy thresholds are not used as geometric widths. Scalar sizes are upper minus lower in the n=3 fixture units. Boxes show the median and interquartile range across independent episodes, with 1.5-IQR whiskers and every outlier; this spread is not a posterior distribution or a confidence interval over width. Proper-score checks are separate. Validation-fitted widths do not establish a matched-coverage advantage if achieved test coverage misses its target. Numerical mass and quantile precision are separate from mean accuracy. Both applications use n=3 reliability fixtures. On the inclusion pages, panel (a) checks latent-subspace inclusion in its HPD set; panel (b) reports the clean scalar $q^H Pq$ for a known unit complex query $q$; panel (c) reports the noisy scalar $\operatorname{Re}(u_{t,1})$. Contact observations are real. On the appended size pages, panels (a), (b), (c) show the state-ball chordal radius, clean scalar width and noisy scalar width, respectively. These are 128 independent test episodes per fixed checkpoint and regime.

Page 1: prior_predictive.
Page 2: prior_predictive.
Page 3: fixed_state.
Page 4: fixed_state.
Page 5: drifting.
Page 6: drifting.
Page 7: prior_predictive.
Page 8: fixed_state.
Page 9: drifting.

## predictive_scores.pdf

Excluded-observation proper scores and response risk under the actual causal recursion. The normalized Bingham-mixture predictive density and Gaussian moment surrogate are distinct densities. Snapshot/query errors are averaged within each episode before comparison. The risk identity predicts total squared clean-response error; a full directional covariance requires second projector moments. Neither pointwise identity lines nor a single trajectory establish a universal score improvement or calibration guarantee.


## model_mismatch.pdf

Declared mismatch experiments under the actual recursion. Beamforming changes training contamination, source powers/geometry, sensor covariance or motion speed. Contact changes constrained mechanics or sensor construction, including dissipative friction/sticking and contact loss. Known anisotropic resistance requires transformed measurements and the correct sensor covariance; isotropic physical velocity noise becomes $\sigma^2D$. Position differencing induces correlated velocity errors. Increased posterior spread is not a repair of an incorrect likelihood.


## computation_time.pdf

Warmed single-thread pipeline cost at actual action, reporting and learning schedules. Action cadence is labelled in observations and is distinct from the fixed $L=32$ observations per EM fit. EM bars use two or four iterations, and optional uncertainty reports occur every 32 observations. Correction, modal/mean action, uncertainty report and variance fitting are nonoverlapping phases; a native moment routine already includes its eigensystem. Medians and upper quantiles use the declared randomized repetitions. Phase stacks sum per-phase medians and need not equal the median full-cycle wall time. Stored state and tracked allocation have separate scopes; native workspaces and RSS are not silently included. Timings are host measurements, not worst-case deadlines. Synthetic spectra and statistical-state task fixtures remain distinct. Task/cost trade-off points use the same predeclared episode 0, not the population task-loss mean; clock repetitions quantify execution variability only.


## numerical_accuracy.pdf

Independent accuracy at the requested targets. Normalizer, mean, second moments, derivative and quantile diagnostics retain separate flags. Small-state quadrature and high-precision native complex-line references have their own resolution limits. A first-derivative truncation certificate is not a floating-point, second-moment or tail certificate. Broad and highly concentrated unresolved cases are retained; rank/codimension-one feasibility does not establish arbitrary-rank integration.


## paper_summary.pdf

Selected main panels from the same saved arrays used by the individual figures. Contact emphasizes the physical input/constraint model and directional posterior uncertainty; beamforming emphasizes moderate-dimensional task performance, causal variance learning and matched-cadence computational cost. No simulation, posterior resampling, tuning or trajectory selection occurs during summary composition. Caption conditions and full numerical records remain in the associated figure entries.


## supp_information_tracking.pdf

Independent Gaussian Fisher, score and curvature checks. Expected information is distinguished from finite-data posterior curvature. All matrix comparisons use aligned tangent coordinates; proper-complex information uses real/imaginary coordinates and its correct factor. Recorded-input excitation and marginalized-input information use the stated common data law. Monte Carlo dataset count differs from observations per dataset. Reused mathematical reference cohorts keep their actual source hashes and repetition counts. Invariant-density Hessian and local accuracy diagnostics. Empirical bounded log-coordinate spread is compared with $H_A^{-1}$ only under recorded locality conditions; broad/nonlocal cases remain visible. The stationary inverse-Fisher study uses fixed known variances and $\beta=1$, reports estimation bias and local fractions, and treats inverse Fisher as a local reference rather than a global intrinsic Cramer--Rao equality. The real graph-density volume term is a separate quantity, retained in the tables. Forgetting sampling references use squared weights. Recomputed sharp fixed-time envelope $\sqrt{\min(k,n-k)}\,\epsilon_t/(\gamma_t-\epsilon_t)$ under $\epsilon_t<\gamma_t/2$, with the exact aggregate conditional drift and the appropriate real/complex constants. True path, gap and drift are analysis quantities, not online confidence radii. Invalid conditions remain marked and receive no violation test. Error bands describe repeated-episode spread. The geometric ceiling is not used to clip the theorem. Phase tables and exact binomial violation intervals retain every declared condition.


## supp_comparisons.pdf

Task losses on common independent test episodes. Held-out disturbance or query losses are averaged within each episode before computing pointwise Student 95\% intervals. Only the posterior-mean curve carries a confidence band in the main absolute-loss panels. The contact paired panel isolates the same-data modal, matched-isotropic and regularized spectral controls; response-only GREAT remains in its absolute-loss panel and the full supplementary grid. Contact paired points show mean loss differences and pointwise Student 95\% confidence intervals across independent episodes, including every episode outlier; they are not posterior intervals. Beamforming paired boxes show the distribution of episode differences, not confidence intervals. The same-$A_t$ modal and matched-isotropic controls isolate posterior moments; GREAT and application-specific controls have separately validated settings and their stated information. Beamforming power ratios are $10\log_{10}(P/P_{\mathrm{oracle}})$, where the denominator is true-covariance oracle output power. Oracle quantities are evaluation-only. This cohort has 256 independent test episodes per condition; snapshots and queries are averaged within each episode. Exact discounted aggregation, delayed actions and constant-subspace approximation are separate checks. Aggregation retains the boundary factor $\beta^L$, within-block weights, per-observation variances and recorded noncommuting transport. Numerical zeros are displayed at a $10^{-16}$ plotting floor; full tolerances remain in the tables. Delayed actions are evaluated only after a block arrives, with common endpoints and physical output age reported separately. The block-likelihood reference resets to the uniform prior, $A_0=0$, and uses $q_{\sum_{j=t-L+1}^t B_j}$ with $\beta=1$ within the raw $L$-observation window. It does not retain the singleton history. Under motion its one-state likelihood is an approximation, distinct from the algebraic aggregation identity. Processing/action cadence cost is reported in the computation output. Complete declared dimension, memory and method comparisons. Heatmaps show comparator-minus-full posterior-mean loss; positive values favor the full mean. Hatching marks cells whose paired pointwise Student 95\% interval includes zero. No simultaneous-coverage claim is made. Test episodes never tune a setting. Paired differences include losses and inconclusive intervals; broad grids are exploratory without simultaneous coverage. Contact GREAT is response-only and cannot isolate the benefit of known inputs. The same-parameter local Gaussian and recursive EKF are distinct supplementary references with their validity diagnostics retained. Identical modal/PCA or conditional spectral least-squares solutions are not counted as independent algorithms. a real-coefficient section $h=U_3x$, $x\in S^2$, of the ambient complex-line density, shown relative to its mode; this is not a normalized marginal or a full posterior. A matched isotropic mean is a task control, not an invented probability law. Visible anisotropy and empirical task benefit are separate comparisons. Complementary mean eigenvalues compare the full and isotropic means. The lower panel is the empirical distribution of paired isotropic-minus-full held-out power-ratio losses at n=L=64, in dB; it is not a confidence interval.

Page 6: static.
Page 7: smooth.
Page 8: abrupt.
Page 9: noise_change.
Page 10: power_change.

## supp_variance_prediction_cost.pdf

Fixed-boundary stationary-block EM reference. Every E-step reconstructs the full block from one saved boundary prior. Evidence curves are shifted by their own initial value, so final heights do not rank starts; absolute normalized evidence selects starts. Both positive starts and the uniform-prior zero-signal boundary are retained. Noise floors and pooled isotropic M-steps are tested in the numerical tables. Contact residual updates include posterior response uncertainty, not only a squared residual using the mean projector. These reference blocks are not online adaptive-tracker episodes. Actual causal variance learning from completed nonoverlapping past blocks. Each observation is corrected once; a fit updates only subsequent variances while retaining the streamed subspace belief. Known coordinate changes are aligned before block fitting. Curves show saved trajectory/episode summaries; variance bands are interquartile repeated-episode spread, not a posterior over variance. Future-task and pre-observation score intervals are pointwise 95\% intervals across independent episodes. Each page retains one predeclared noise-, motion-, or signal-power intervention. The contact supplementary density pages show one predeclared episode: panels (a,b) of the first density page use $t=32,96$, and panel (a) of the following density page uses $t=192$; pre-correction $A_t^-$ is converted to the wrist-frame normal law $q_{-A_t^-}$ with normalizer $\psi_1(-A_t^-)=\psi_2(A_t^-)-\operatorname{tr}(A_t^-)$. The three orthographic spheres share a density scale; panel (b) of the following density page shows the physical-query risk at all three checkpoints, $v_t^T(M_t^--(M_t^-)^2)v_t$ at the recorded physical query. This is individual belief uncertainty, distinct from the repeated-episode variance bands and from calibration. Empirical-Bayes plug-in scales omit their uncertainty and are outside the fixed-true-variance theorem. Actual-recursion calibration with distinct state, clean-response and noisy-next-observation targets. Pages separate joint prior-predictive sampling, fixed-state repetitions and drift diagnostics. Vertical intervals are pointwise exact binomial uncertainty across independent episodes, not credible uncertainty. All intervals are evaluated at their declared time before the scored future observation is seen. Appended pages show 95% report-size distributions at t=64 for every probability regime. State-ball sizes are chordal radii, obtained by taking the square root of each recorded squared-distance upper threshold; HPD energy thresholds are not used as geometric widths. Scalar sizes are upper minus lower in the n=3 fixture units. Boxes show the median and interquartile range across independent episodes, with 1.5-IQR whiskers and every outlier; this spread is not a posterior distribution or a confidence interval over width. Proper-score checks are separate. Validation-fitted widths do not establish a matched-coverage advantage if achieved test coverage misses its target. Numerical mass and quantile precision are separate from mean accuracy. Both applications use n=3 reliability fixtures. On the inclusion pages, panel (a) checks latent-subspace inclusion in its HPD set; panel (b) reports the clean scalar $q^H Pq$ for a known unit complex query $q$; panel (c) reports the noisy scalar $\operatorname{Re}(u_{t,1})$. Contact observations are real. On the appended size pages, panels (a), (b), (c) show the state-ball chordal radius, clean scalar width and noisy scalar width, respectively. These are 128 independent test episodes per fixed checkpoint and regime. Excluded-observation proper scores and response risk under the actual causal recursion. The normalized Bingham-mixture predictive density and Gaussian moment surrogate are distinct densities. Snapshot/query errors are averaged within each episode before comparison. The risk identity predicts total squared clean-response error; a full directional covariance requires second projector moments. Neither pointwise identity lines nor a single trajectory establish a universal score improvement or calibration guarantee. Declared mismatch experiments under the actual recursion. Beamforming changes training contamination, source powers/geometry, sensor covariance or motion speed. Contact changes constrained mechanics or sensor construction, including dissipative friction/sticking and contact loss. Known anisotropic resistance requires transformed measurements and the correct sensor covariance; isotropic physical velocity noise becomes $\sigma^2D$. Position differencing induces correlated velocity errors. Increased posterior spread is not a repair of an incorrect likelihood. Warmed single-thread pipeline cost at actual action, reporting and learning schedules. Action cadence is labelled in observations and is distinct from the fixed $L=32$ observations per EM fit. EM bars use two or four iterations, and optional uncertainty reports occur every 32 observations. Correction, modal/mean action, uncertainty report and variance fitting are nonoverlapping phases; a native moment routine already includes its eigensystem. Medians and upper quantiles use the declared randomized repetitions. Phase stacks sum per-phase medians and need not equal the median full-cycle wall time. Stored state and tracked allocation have separate scopes; native workspaces and RSS are not silently included. Timings are host measurements, not worst-case deadlines. Synthetic spectra and statistical-state task fixtures remain distinct. Task/cost trade-off points use the same predeclared episode 0, not the population task-loss mean; clock repetitions quantify execution variability only. Independent accuracy at the requested targets. Normalizer, mean, second moments, derivative and quantile diagnostics retain separate flags. Small-state quadrature and high-precision native complex-line references have their own resolution limits. A first-derivative truncation certificate is not a floating-point, second-moment or tail certificate. Broad and highly concentrated unresolved cases are retained; rank/codimension-one feasibility does not establish arbitrary-rank integration.

Page 2: noise_change.
Page 3: smooth.
Page 4: power_change.
Page 5: prior_predictive.
Page 6: prior_predictive.
Page 7: fixed_state.
Page 8: fixed_state.
Page 9: drifting.
Page 10: drifting.
Page 11: prior_predictive.
Page 12: fixed_state.
Page 13: drifting.
