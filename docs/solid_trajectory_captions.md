# Figure captions and statistical definitions

Copy-pastable LaTeX is in [solid_trajectory_captions.tex](solid_trajectory_captions.tex). Each caption states the actual quantity, method colors, number of saved coordinates and meaning of any band. The combined moving figure and its two individual exports reuse the same data.

## Statistics

A Student 95% interval for a Monte Carlo mean is $\bar y\pm t_{0.975,255}s/\sqrt{256}$. This describes uncertainty in the mean across the independent simulations. An empirical interquartile band is $[Q_{0.25}(y),Q_{0.75}(y)]$ across those 256 values at each saved index. It describes simulation-to-simulation variation. Neither is a posterior credible region. Bands are pointwise, rather than simultaneous.

The moving-power curves average individual dB values; they do not take the logarithm of an averaged linear power. The contact error already averages three clean response errors; conversion to squared mm/s multiplies by $10^6$, without dividing by three again. Sphere full-run summaries use all 48 saved action evaluations per simulation, rather than averaging the six displayed checkpoints.

## Captions

### results/main/beamforming/task_performance_output_power.pdf

Stationary beamforming: $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$ from~\eqref{eq:beam-power-comparison}, after $T=200$ measurements for each $n\in\{32,64,128\}$. Blue, orange, green and dark gray solid curves denote the posterior mean $\overline P_T^+$, mode $P_{\widehat{\mathcal X}_T}$, GREAT (sliding window) and MVDR with regularized sample covariance, respectively. Each curve joins three arithmetic means over 256 Monte Carlo simulations; blue shading gives pointwise Student 95\% confidence intervals for the posterior-mean averages.

### results/main/beamforming/noise_adaptation_page1.pdf

Causal variance estimation with $n=32$ and $T=256$. Black, gray and blue curves use supplied true variances, fixed estimates and EM estimates, respectively. The purple dashed line at $t=129$ marks the noise-variance change from $\sigma_t^2=0.2$ to $0.8$; $\tau_t^2=1$. The top-left and top-right panels show the variances available before $u_t$: solid estimate curves join medians at the eight saved indices $t=32,64,\ldots,256$, and the blue bands contain the pointwise empirical 25th--75th percentiles across 256 Monte Carlo simulations. Black references give the true schedules. The bottom-left and bottom-right panels show arithmetic means over these simulations of $\ell_t$ in~\eqref{eq:beam-power-comparison} and $\log\pi_t(u_t)$ in~\eqref{eq:beam-predictive-density}, evaluated from $q_{A_t^-}$ before correction at those same indices; all three methods have pointwise Student 95\% confidence bands. EM uses blocks of $L=32$ measurements and two iterations; a fit completed at $t$ is first used at $t+1$.

### results/main/beamforming/moving_interference_smooth_output_power.pdf

Moving interference with a smooth direction change: $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$ in~\eqref{eq:beam-power-comparison}, with $n=32$ and $\beta=0.97$. Blue, orange, pink and green solid curves denote the posterior mean $\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$, matched isotropic mean $\overline P_t^{\mathrm{iso}}$ and GREAT (sliding window), respectively. Curves are arithmetic means over 256 Monte Carlo simulations at the 32 saved indices $t=8,16,\ldots,256$; matching shaded bands contain the pointwise empirical 25th--75th percentiles across those simulations.

### results/main/beamforming/moving_interference_abrupt_output_power.pdf

Moving interference with an abrupt direction change: $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$ in~\eqref{eq:beam-power-comparison}, with $n=32$ and $\beta=0.97$. Blue, orange, pink and green solid curves denote the posterior mean $\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$, matched isotropic mean $\overline P_t^{\mathrm{iso}}$ and GREAT (sliding window), respectively. Curves are arithmetic means over 256 Monte Carlo simulations at the 32 saved indices $t=8,16,\ldots,256$; matching shaded bands contain the pointwise empirical 25th--75th percentiles across those simulations. The purple dashed line at $t=129$ marks the interference-direction change.

### results/main/beamforming/moving_interference_comparison.pdf

Moving interference: smooth direction change (left) and abrupt direction change (right). $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$ is defined in~\eqref{eq:beam-power-comparison}; $n=32$ and $\beta=0.97$. Blue, orange, pink and green solid curves denote the posterior mean $\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$, matched isotropic mean $\overline P_t^{\mathrm{iso}}$ and GREAT (sliding window), respectively. Curves are arithmetic means over 256 Monte Carlo simulations at the 32 saved indices $t=8,16,\ldots,256$; matching shaded bands contain the pointwise empirical 25th--75th percentiles across those simulations. The purple dashed line at $t=129$ marks the interference-direction change in the right panel.

### results/main/contact/information_rotation.pdf

Anisotropy of the conditional measurement information for the stationary plane with normal $e_3$. Orange and blue solid curves show the exact $D_{\mathrm{KL}}(p_0\Vert p_z)$ for $z=(\tan\delta,0)^\top$ and $z=(0,\tan\delta)^\top$, respectively, where $n(z)=(e_3-z_1e_1-z_2e_2)/\sqrt{1+\|z\|^2}$ and $p_z$ is the joint conditional likelihood of the 16 measurements from~\eqref{eq:contact-model}. Curves evaluate 201 angles from $0.25^\circ$ to $35^\circ$; the orange and blue points mark $5^\circ$ and $30^\circ$. The thin neutral segment joins the highlighted values $D_{\mathrm{KL}}=0.23957$ at $5^\circ$ and $0.23077$ at $30^\circ$: nearly equal divergences arise at very different rotation angles. Their actual heights are retained. These are deterministic likelihood comparisons, with no Monte Carlo averaging or confidence band.

### results/main/contact/posterior_anisotropy_spheres.pdf

Posterior normal densities for one saved stationary-plane simulation at corrected time $t=16$. Left: $q_{-A_t^+}([n])$; right: the isotropic Bingham law with the same modal line and expected squared chordal distance, whose plane mean is~\eqref{eq:isotropic-control}. The blue--purple--pink--yellow colors show normalized density relative to invariant measure on one common logarithmic scale, from lower to higher density. Orange points mark the common modal line; gray curves are coordinate great circles. Antipodal normals represent the same plane. Each surface evaluates one saved distribution; it is not a Monte Carlo average or a confidence region.

### results/main/contact/posterior_anisotropy_risk.pdf

Posterior response variance $r_t(v(\theta))/v_0^2$ from~\eqref{eq:contact-response-variance}, for $v(\theta)=v_0(\cos\theta\,e_1+\sin\theta\,e_2)$ and $v_0=0.02\,\mathrm{m/s}$ at $t=16$. Blue and pink solid curves use the full and matched isotropic laws, respectively, each with its own posterior mean. The curves evaluate the same saved posterior pair at 181 angles over $[0,2\pi]$; they are deterministic posterior variances, with no Monte Carlo averaging or confidence band.

### results/main/contact/task_performance_page1.pdf

Velocity prediction error $\mathcal E_t(D_t)$ in~\eqref{eq:contact-prediction-error}, in $(\mathrm{mm}/\mathrm{s})^2$. In the stationary-plane panel (left), blue, orange, brown and green solid curves denote the posterior mean $\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$, regularized matrix $\widetilde M_t$ and GREAT (sliding window). Each joins arithmetic means over 256 Monte Carlo simulations at $t=1,4,16,32$; blue shading gives pointwise Student 95\% confidence intervals for the posterior-mean averages. In the sphere panel (right), blue, orange, pink and brown points denote the posterior mean, mode, matched isotropic mean and regularized matrix $\widetilde M_t$. Each point first averages the error over all 48 indices $t=4,8,\ldots,192$ within each simulation, then averages the 256 resulting values; horizontal bars are Student 95\% confidence intervals for every displayed method. Here $\widetilde M_t=\rho P_{\widehat{\mathcal X}_t}+(1-\rho)(2/3)I_3$.

### results/main/contact/prediction_error_stationary.pdf

Stationary-plane velocity prediction error $\mathcal E_t(D_t)$ in~\eqref{eq:contact-prediction-error}, in $(\mathrm{mm}/\mathrm{s})^2$. Blue, orange and pink solid curves denote $\overline P_t^+$, $P_{\widehat{\mathcal X}_t}$ and $\overline P_t^{\mathrm{iso}}$, respectively. Each curve joins arithmetic means over 256 Monte Carlo simulations at the four saved indices $t=1,4,16,32$. Blue shading gives pointwise Student 95\% confidence intervals for the posterior-mean averages.

### results/main/contact/prediction_error_sphere.pdf

Moving-sphere velocity prediction error $\mathcal E_t(D_t)$ in~\eqref{eq:contact-prediction-error}, in $(\mathrm{mm}/\mathrm{s})^2$. Blue, orange and pink solid curves denote the posterior mean $\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$ and matched isotropic mean $\overline P_t^{\mathrm{iso}}$ with measured transport $G_t=R_t^\top R_{t-1}$. The dark gray solid curve is the posterior mean with $G_t=I_3$. Each curve joins arithmetic means over the same 256 Monte Carlo simulations at all 48 original action indices $t=4,8,\ldots,192$. Blue shading gives pointwise Student 95\% confidence intervals for the measured-transport posterior-mean averages. The ordering varies with time; there is no programmed event at $t=128$.

### results/supplementary/contact/prediction_error_references.pdf

Time-averaged sphere velocity prediction error from~\eqref{eq:contact-prediction-error}, in $(\mathrm{mm}/\mathrm{s})^2$. Blue, orange, pink, brown, green and light blue points denote the posterior mean, mode, matched isotropic mean, regularized matrix $\widetilde M_t$, GREAT (sliding window) and GREAT (exponential weighting), respectively. Each point averages $\mathcal E_t(D_t)$ over all 48 action indices $t=4,8,\ldots,192$ within each simulation and then over 256 Monte Carlo simulations. Horizontal bars are Student 95\% confidence intervals for these means. Here $\widetilde M_t=\rho P_{\widehat{\mathcal X}_t}+(1-\rho)(2/3)I_3$.

### results/supplementary/contact/moving_sphere_posterior_spheres.pdf

Normal uncertainty for the saved moving-sphere simulation at corrected time $t=96$. Left: the actual normal-line posterior $q_{-A_t^+}([n])$; right: the isotropic Bingham law with the same modal line and expected squared chordal distance. Blue--purple--pink--yellow colors show normalized density relative to invariant measure on a common logarithmic scale, from lower to higher density. Orange circles mark the modal line and black stars the true normal in the current sensor frame; the same antipodal representative is used in both panels. This is one saved simulation, with no Monte Carlo averaging or confidence band.

### Combined contact density and response variance

Directional posterior uncertainty at $t=16$. The two spheres show the full normal density $q_{-A_t^+}([n])$ (left) and a matched isotropic Bingham law (right), with equal modal line and expected squared chordal distance. Their common blue--purple--pink--yellow logarithmic scale gives normalized density relative to invariant measure; orange points mark the mode and gray curves are coordinate great circles. Below, blue and pink solid curves give $r_t(v(\theta))/v_0^2$ in~\eqref{eq:contact-response-variance} under the full and isotropic laws, respectively, for $v(\theta)=v_0(\cos\theta\,e_1+\sin\theta\,e_2)$, $v_0=0.02\,\mathrm{m/s}$. Each curve uses its corresponding mean and evaluates 181 angles. All panels concern one saved posterior pair, not Monte Carlo averages or confidence regions.

## Audit boundaries

The individually checked captions cover all 11 main PDFs and the two supplementary contact diagnostics. The machine-readable inventory also lists all 87 declared application-notebook export routes and all 38 preserved canonical documents. Its supplementary notes and canonical captions are explicitly inherited metadata where their statistics were not independently rechecked; no universal 256-simulation or confidence-band claim is applied to that catalogue.

The updated contact sphere history uses blue for measured transport and dark gray for its identity-transport control, as explicitly requested. The 48 original action indices were independently replay-verified; the retained supplementary tracking figure still uses its six stored checkpoints.

The supplied numerical-section TeX still contains the earlier $t=n$ comparison and $v(\theta)$ in $\operatorname{span}(e_2,e_3)$. Its stationary paragraph must match fixed $T=200$, and its response-variance paragraph must match the current saved family in $\operatorname{span}(e_1,e_2)$. All equation references used above already exist in that TeX source. No new equation labels or guide-specific definitions are assumed.

Numeric evidence and exact group counts are recorded in `results/solid_trajectory_update/caption_inventory.json`. This is a read-only statistical audit, not a replacement figure-production manifest.

## Supplementary connected-artist figures

The following 41 captions use inspected saved-data artists and their source tables. Timing repetitions, deterministic quadrature, binary coverage fractions, EM optimization paths and Monte Carlo error means retain their separate definitions. All saves and displays were disabled during the audit.

### results/supplementary/beamforming/setup_page1.pdf

Array geometry, the first 32 measured snapshots from one saved $n=64$ simulation, and the corresponding distortionless beampatterns. The right panel evaluates $10\log_{10}|w^{\mathrm H}b(\varphi)|^2$ at 161 arrival angles; blue, orange and pink curves use the posterior mean, mode and matched isotropic mean. Vertical references mark the desired and interference directions. This is one saved realization, not a Monte Carlo average; no confidence band is shown.

### results/supplementary/beamforming/information_page1.pdf

Fisher-information checks for the complex marginal model. Left: eigenvalues of the score-covariance estimate, based on 65,536 independent score realizations per case, against the analytical eigenvalues; the diagonal marks equality. Right: the analytical information coefficients for recorded inputs (orange) and marginalized inputs (blue), at the two saved noise variances. Connected coefficients are deterministic evaluations, not Monte Carlo means or confidence bands.

### results/supplementary/beamforming/stationary_dimensions_T200.pdf

Stationary beamforming: $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$ from~\eqref{eq:beam-power-comparison}, after $T=200$ measurements for each $n\in\{32,64,128\}$. Blue, orange, green and dark gray solid curves denote the posterior mean $\overline P_T^+$, mode $P_{\widehat{\mathcal X}_T}$, GREAT (sliding window) and MVDR with regularized sample covariance, respectively. Each curve joins three arithmetic means over 256 Monte Carlo simulations; blue shading gives pointwise Student 95\% confidence intervals for the posterior-mean averages.

### results/supplementary/beamforming/posterior_anisotropy_eigenvalues.pdf

Complementary eigenvalues of the posterior mean (blue) and matched isotropic mean (pink) for the saved $n=64$, $t=T=200$ posterior. The 63 displayed eigenvalues lie outside the modal line. These are deterministic spectra of one saved posterior pair; no Monte Carlo averaging or confidence interval is used.

### results/supplementary/beamforming/posterior_anisotropy_empirical_cdf.pdf

Empirical distribution of $\ell_T^{\mathrm{iso}}-\ell_T^{\mathrm{mean}}$ at $n=64$ and $T=200$, using~\eqref{eq:beam-power-comparison}. Each of the 256 values compares the matched isotropic and full posterior-mean beamformers in the same Monte Carlo simulation. Positive values indicate lower output power for the full mean. The blue curve is the empirical cumulative distribution, not a confidence band; the vertical reference marks zero.

### results/supplementary/beamforming/predictive_calibration_page2.pdf

Prior-predictive repetitions at $t=64$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.98$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/beamforming/predictive_calibration_page6.pdf

Subspace-drift repetitions at $t=64$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.98$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/beamforming/computation_time_scaling.pdf

Computational time for 64 measurements at the six saved ambient dimensions $3,8,16,32,64,128$. Blue and orange solid curves denote posterior-mean and modal decisions, respectively; each method is shown with decisions after every measurement and after every 16 measurements, as labeled. Values are medians of seven randomized repeated executions after a warm-up; uncertainty reports are disabled. These are repeated clocks on a fixed workload, not Monte Carlo accuracy simulations. No confidence band is shown.

### results/supplementary/beamforming/tracking_error_supp_page1.pdf

Smooth moving interference with $n=32$. Left: chordal subspace distance $d_c(\widehat{\mathcal X}_t,\mathcal X_t)=\|P_{\widehat{\mathcal X}_t}-P_{\mathcal X_t}\|_{\mathrm F}/\sqrt2$. Orange, green, light blue and brown curves denote the mode, GREAT (sliding window), GREAT (exponential weighting) and GROUSE, respectively. Right: $\|D_t-P_{\mathcal X_t}\|_{\mathrm F}/\sqrt2$ for the posterior mean (blue), mode (orange) and matched isotropic mean (pink). For the two means this is a matrix error, since $D_t$ need not be a projector. Solid curves join arithmetic means over 256 Monte Carlo simulations at $t=8,16,\ldots,256$ (32 saved indices); all matching bands are pointwise Student 95\% confidence intervals for the means.

### results/supplementary/beamforming/tracking_bound_supp_page1.pdf

Tracking-bound diagnostic for the stationary uniform-prior reference. Blue shows the median chordal error across 128 independent simulations at $t=64,256,1024,4096,16384$; its shading is the empirical pointwise 25th--75th percentile range. Pink gives the deterministic perturbation bound and black the distance ceiling; crosses retain conditions failing $\epsilon_t<\gamma_t/2$. The bound and ceiling are mathematical references, not confidence intervals. Bottom left: the 128 terminal realized operator perturbations against their deterministic allowance. Bottom right: eligibility of the 24 saved forgetting-factor/motion-rate pairs; these cells are deterministic.

### results/supplementary/beamforming/tracking_bound_supp_page3.pdf

Tracking-bound diagnostic for the zero-signal fixture. Blue shows the median chordal error across 128 independent simulations at $t=16384$; its shading is the empirical pointwise 25th--75th percentile range. Pink gives the deterministic perturbation bound and black the distance ceiling; crosses retain conditions failing $\epsilon_t<\gamma_t/2$. The bound and ceiling are mathematical references, not confidence intervals. Only one measurement index is saved in this fixture; its comparisons are isolated scientific points.

### results/supplementary/beamforming/tracking_bound_supp_page4.pdf

Tracking-bound diagnostic for the nonuniform-prior fixture. Blue shows the median chordal error across 128 independent simulations at $t=256$; its shading is the empirical pointwise 25th--75th percentile range. Pink gives the deterministic perturbation bound and black the distance ceiling; crosses retain conditions failing $\epsilon_t<\gamma_t/2$. The bound and ceiling are mathematical references, not confidence intervals. Only one measurement index is saved in this fixture; its comparisons are isolated scientific points.

### results/supplementary/beamforming/posterior_curvature_supp_page1.pdf

Posterior curvature and local estimation error. Top: bounded projective log-coordinate variances computed by deterministic quadrature for six saved spectra, against the corresponding diagonal entries of $H_A^{-1}$; blue marks eligible local cases and orange open triangles retain cases outside that regime. Bottom left: tangent mean squared error (blue) over 256 independent estimation simulations at $L=4,16,64,256,1024$, with pointwise Student 95\% confidence bands; gray gives the deterministic inverse-Fisher trace. Bottom right: the local fraction (blue) and normalized estimation bias (orange), computed from those same 256 simulations. These local references do not assert a global equality or global posterior-covariance bound.

### results/supplementary/beamforming/block_updates_supp_page1.pdf

Block aggregation and delayed decisions at $L=1,4,16,32$. Left: maximum saved aggregation discrepancies in the parameter, mean and mode (blue, orange and green), with a $10^{-16}$ display floor; these are algebra checks, not Monte Carlo averages. The center panel gives exact covariance-based $\ell_t$ for delayed exact aggregation (blue) and a constant-state block likelihood (orange), averaged over 64 independent simulations for each of the four block lengths. Right: the recorded action delay (orange), averaged within each block-length cohort. No confidence band is shown. The constant-state likelihood and delayed streamed decisions are distinct procedures.

### results/supplementary/beamforming/variance_estimation_supp_page1.pdf

EM variance estimation for a fixed complex marginal block. Left: log-evidence increases from four initializations in the first saved trial; blue, pink, orange and gray curves are identified by their displayed initial $(\widehat\sigma_0^2,\widehat\tau_0^2)$ and contain 9, 10, 11 and 3 stored iterates, respectively. Right: distributions of $\widehat\sigma^2/\sigma^2$ (blue) and $\widehat\tau^2/\tau^2$ (pink) across 24 independent blocks for each initialization. Boxes show medians and interquartile ranges, whiskers extend to the most extreme samples within 1.5 interquartile ranges, and outliers are retained. These distributions are not confidence intervals for means; the gray zero-signal initialization is a separate boundary case.

### results/supplementary/beamforming/noise_adaptation_supp_page2.pdf

Causal variance estimation under smooth interference motion. Black, gray and blue denote supplied true variances, fixed estimates and EM estimates, respectively. The top panels join medians of the saved premeasurement noise- and signal-variance estimates over 256 Monte Carlo simulations at the eight indices $t=32,64,\ldots,256$; blue shading contains their empirical 25th--75th percentiles. The bottom panels join arithmetic means of exact covariance-based $\ell_t$ and the pre-correction Bingham-mixture $\log\pi_t(u_t)$ at those same indices; matching bands for all methods are pointwise Student 95\% confidence intervals for means. Reference variance schedules are deterministic. An EM fit completed at $t$ affects measurements from $t+1$.

### results/supplementary/beamforming/noise_adaptation_supp_page3.pdf

Causal variance estimation under a signal-variance change. Black, gray and blue denote supplied true variances, fixed estimates and EM estimates, respectively. The top panels join medians of the saved premeasurement noise- and signal-variance estimates over 256 Monte Carlo simulations at the eight indices $t=32,64,\ldots,256$; blue shading contains their empirical 25th--75th percentiles. The bottom panels join arithmetic means of exact covariance-based $\ell_t$ and the pre-correction Bingham-mixture $\log\pi_t(u_t)$ at those same indices; matching bands for all methods are pointwise Student 95\% confidence intervals for means. Reference variance schedules are deterministic. An EM fit completed at $t$ affects measurements from $t+1$. The reference schedule changes at measurement 129; the displayed boundary is halfway between indices 128 and 129.

### results/supplementary/beamforming/predictive_calibration_supp_page1.pdf

Prior-predictive repetitions at $t=32$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.98$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/beamforming/predictive_calibration_supp_page3.pdf

Fixed-state repetitions at $t=32$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.98$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/beamforming/predictive_calibration_supp_page4.pdf

Fixed-state repetitions at $t=64$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.98$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/beamforming/predictive_calibration_supp_page5.pdf

Subspace-drift repetitions at $t=32$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.98$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/beamforming/computation_time_supp_page2.pdf

Memory for the posterior-mean implementation after 64 measurements, at ambient dimensions $3,8,16,32,64,128$. Blue denotes persistent array storage and orange the tracked Python/NumPy peak allocation, in KiB. Curves connect the six saved memory measurements for the declared decision interval with uncertainty reports disabled. These are allocation measurements, not Monte Carlo accuracy simulations; no confidence band is shown.

### results/supplementary/beamforming/paper_summary_supp_page1.pdf

Beamforming overview. Top left: fixed-$T=200$ output-power means over 256 simulations at $n=32,64,128$; blue, orange, dark gray and green denote the posterior mean, mode, regularized sample covariance and GREAT (sliding window), with a Student 95\% band for the posterior mean only. Top right: paired smooth-motion differences of $10\log_{10}[29^{-1}\sum_{j=4}^{32}10^{\ell_{8j}/10}]$ between each method and the posterior mean, over 256 simulations, summarized by medians, interquartile boxes, 1.5-IQR whiskers and retained outliers. Bottom left: premeasurement noise-variance medians at eight reports, with the EM empirical interquartile band; black, gray and blue denote true, fixed and EM variances. The reference variance change is displayed between indices 128 and 129. Bottom right: recorded median execution times against the matching fixed-realization output powers for the earlier $t=n$ timing cohort; these three-point curves are not a $T=200$ Monte Carlo cost--accuracy comparison. Timing medians use seven repetitions.

### results/supplementary/contact/noise_adaptation_page1.pdf

Causal noise-variance estimation for a fixed plane in a rotating sensor frame, with a change in measurement noise. Black, gray and blue denote supplied true variances, fixed estimates and EM estimates, respectively. Left: medians of the six saved premeasurement variance estimates over 256 independent simulations, with the EM empirical 25th--75th percentile band and the deterministic true-variance reference. Center and right: arithmetic means of squared prediction error against independent noisy query responses and pre-correction $\log\pi_t(u_t)$, with pointwise Student 95\% confidence bands for all methods. These six reports correspond to $t=32,64,\ldots,192$. The velocity error averages three held-out noisy query responses and differs from the three-input clean error in~\eqref{eq:contact-prediction-error}; its units are $(\mathrm{mm}/\mathrm{s})^2$. The horizontal axis is elapsed time $(t-1)0.05$ s, so the reports are at $1.55,3.15,4.75,6.35,7.95,9.55$ s; the sampled variance reference changes at the first report after the physical change.

### results/supplementary/contact/predictive_calibration_page2.pdf

Prior-predictive repetitions at $t=64$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.9$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/contact/predictive_calibration_page6.pdf

Subspace-drift repetitions at $t=64$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.9$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/contact/computation_time_scaling.pdf

Computational time for 64 measurements at the six saved ambient dimensions $3,8,16,32,64,128$. Blue and orange solid curves denote posterior-mean and modal decisions, respectively; each method is shown with decisions after every measurement and after every 16 measurements, as labeled. Values are medians of seven randomized repeated executions after a warm-up; uncertainty reports are disabled. These are repeated clocks on a fixed workload, not Monte Carlo accuracy simulations. No confidence band is shown.

### results/supplementary/contact/tracking_error_supp_page1.pdf

Moving-sphere contact. Left: chordal subspace distance $d_c(\widehat{\mathcal X}_t,\mathcal X_t)=\|P_{\widehat{\mathcal X}_t}-P_{\mathcal X_t}\|_{\mathrm F}/\sqrt2$. Orange, green and light blue curves denote the mode, GREAT (sliding window) and GREAT (exponential weighting), respectively. Right: $\|D_t-P_{\mathcal X_t}\|_{\mathrm F}/\sqrt2$ for the posterior mean (blue), mode (orange) and matched isotropic mean (pink). For the two means this is a matrix error, since $D_t$ need not be a projector. Solid curves join arithmetic means over 256 Monte Carlo simulations at $t=4,16,32,64,128,192$ (6 saved indices); all matching bands are pointwise Student 95\% confidence intervals for the means.

### results/supplementary/contact/tracking_bound_supp_page1.pdf

Tracking-bound diagnostic for the stationary uniform-prior reference. Blue shows the median chordal error across 128 independent simulations at $t=64,256,1024,4096,16384$; its shading is the empirical pointwise 25th--75th percentile range. Pink gives the deterministic perturbation bound and black the distance ceiling; crosses retain conditions failing $\epsilon_t<\gamma_t/2$. The bound and ceiling are mathematical references, not confidence intervals. Bottom left: the 128 terminal realized operator perturbations against their deterministic allowance. Bottom right: eligibility of the 24 saved forgetting-factor/motion-rate pairs; these cells are deterministic.

### results/supplementary/contact/tracking_bound_supp_page3.pdf

Tracking-bound diagnostic for the concentrated-input fixture. Blue shows the median chordal error across 128 independent simulations at $t=16384$; its shading is the empirical pointwise 25th--75th percentile range. Pink gives the deterministic perturbation bound and black the distance ceiling; crosses retain conditions failing $\epsilon_t<\gamma_t/2$. The bound and ceiling are mathematical references, not confidence intervals. Only one measurement index is saved in this fixture; its comparisons are isolated scientific points.

### results/supplementary/contact/tracking_bound_supp_page4.pdf

Tracking-bound diagnostic for the nonuniform-prior fixture. Blue shows the median chordal error across 128 independent simulations at $t=256$; its shading is the empirical pointwise 25th--75th percentile range. Pink gives the deterministic perturbation bound and black the distance ceiling; crosses retain conditions failing $\epsilon_t<\gamma_t/2$. The bound and ceiling are mathematical references, not confidence intervals. Only one measurement index is saved in this fixture; its comparisons are isolated scientific points.

### results/supplementary/contact/posterior_curvature_supp_page1.pdf

Posterior curvature and local estimation error. Top: bounded projective log-coordinate variances computed by deterministic quadrature for six saved spectra, against the corresponding diagonal entries of $H_A^{-1}$; blue marks eligible local cases and orange open triangles retain cases outside that regime. Bottom left: tangent mean squared error (blue) over 256 independent estimation simulations at $L=4,16,64,256,1024$, with pointwise Student 95\% confidence bands; gray gives the deterministic inverse-Fisher trace. Bottom right: the local fraction (blue) and normalized estimation bias (orange), computed from those same 256 simulations. These local references do not assert a global equality or global posterior-covariance bound.

### results/supplementary/contact/block_updates_supp_page1.pdf

Block aggregation and delayed decisions at $L=1,4,16,32$. Left: maximum saved aggregation discrepancies in the parameter, mean and mode (blue, orange and green), with a $10^{-16}$ display floor; these are algebra checks, not Monte Carlo averages. The center panel gives velocity prediction error for delayed decisions (blue), the common endpoint (orange) and a constant-state block likelihood (pink), averaged over 32 independent simulations for each of the four block lengths. Right: the recorded action delay (orange), averaged within each block-length cohort. No confidence band is shown. The constant-state likelihood and delayed streamed decisions are distinct procedures.

### results/supplementary/contact/variance_estimation_supp_page1.pdf

EM noise-variance estimation for a fixed conditional contact block. Left: log-evidence increases in one saved trial from initial variances $0.02$ (blue), $0.2$ (pink) and $0.6$ (orange), with 7, 8 and 8 stored iterates. Center: distributions of $\widehat\sigma^2/\sigma^2$ across 24 independent blocks per initialization; boxes show medians and interquartile ranges, 1.5-IQR whiskers and retained outliers. Right: mean-response and posterior-expected residual updates, each normalized by $\sigma^2$; the diagonal marks equality. The paths are single-trial optimizations and the boxes are distributions, not Monte Carlo mean confidence bands.

### results/supplementary/contact/noise_adaptation_supp_page2.pdf

Causal noise-variance estimation with faster sphere motion. Black, gray and blue denote supplied true variances, fixed estimates and EM estimates, respectively. Left: medians of the six saved premeasurement variance estimates over 256 independent simulations, with the EM empirical 25th--75th percentile band and the deterministic true-variance reference. Center and right: arithmetic means of squared prediction error against independent noisy query responses and pre-correction $\log\pi_t(u_t)$, with pointwise Student 95\% confidence bands for all methods. These six reports correspond to $t=32,64,\ldots,192$. The velocity error averages three held-out noisy query responses and differs from the three-input clean error in~\eqref{eq:contact-prediction-error}; its units are $(\mathrm{mm}/\mathrm{s})^2$.

### results/supplementary/contact/predictive_calibration_supp_page1.pdf

Prior-predictive repetitions at $t=32$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.9$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/contact/predictive_calibration_supp_page3.pdf

Fixed-state repetitions at $t=32$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.9$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/contact/predictive_calibration_supp_page4.pdf

Fixed-state repetitions at $t=64$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.9$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/contact/predictive_calibration_supp_page5.pdf

Subspace-drift repetitions at $t=32$. Left to right: empirical coverage of a latent-subspace highest-posterior-density region, a clean scalar-response interval and a noisy scalar-observation interval. Black curves use true variances with $\beta=1$; purple curves use true variances with $\beta=0.9$; gray and blue curves use fixed and EM variance estimates. Each curve joins coverage fractions from 128 independent test simulations at the three credible levels $0.5,0.8,0.95$. Error bars are pointwise exact binomial 95\% confidence intervals, not posterior intervals; the dashed diagonal indicates equal coverage and credible level.

### results/supplementary/contact/computation_time_supp_page2.pdf

Memory for the posterior-mean implementation after 64 measurements, at ambient dimensions $3,8,16,32,64,128$. Blue denotes persistent array storage and orange the tracked Python/NumPy peak allocation, in KiB. Curves connect the six saved memory measurements for the declared decision interval with uncertainty reports disabled. These are allocation measurements, not Monte Carlo accuracy simulations; no confidence band is shown.

### results/supplementary/contact/paper_summary_supp_page1.pdf

Contact overview. The top spheres show one full and matched isotropic stationary normal posterior at $t=16$, on a shared normalized logarithmic density scale; these are deterministic distributions, not Monte Carlo averages. Bottom left: each the corresponding posterior law’s normalized response variance $r_t(v(\theta))/v_0^2$ at 181 angles, using $v(\theta)=v_0(\cos\theta\,e_2+\sin\theta\,e_3)$; blue and pink denote the full and isotropic means. This is the supplementary $e_2,e_3$ scan, distinct from the main $e_1,e_2$ scan. Bottom right: paired method-minus-posterior-mean differences in the sphere error averaged over all 48 action indices within each of 256 simulations. Orange, pink, brown and green boxes denote the mode, isotropic mean, regularized matrix and GREAT (sliding window); boxes show medians, interquartile ranges, 1.5-IQR whiskers and retained outliers, not mean confidence intervals.


## Transport interpretation

Copy-pastable simulation-section text, using the existing `eq:recursion` label
in the uploaded `v5.tex`:

```tex
Using $G_t=R_t^\top R_{t-1}$ in~\eqref{eq:recursion} transports previous posterior information into the current sensor coordinates, while forgetting accommodates additional physical tangent-plane motion approximately. Across the same 256 sphere simulations, the paired change $\Delta\mathcal E=\mathcal E_{192}-\mathcal E_{128}$ is $-1.84$ with 95\% confidence interval $[-3.42,-0.25]$ for the measured-transport posterior mean and $14.93$ with interval $[12.80,17.07]$ for $G_t=I_3$, in $(\mathrm{mm}/\mathrm{s})^2$. The identity control is nevertheless better at $t=160$ ($1.81$ versus $4.45$), so these diagnostics support frame compensation in this experiment without a universal pointwise ordering or a causal claim about sensor-rotation speed.
```

The dense sphere history uses all 48 original action indices after independent replay verification and identifies the gray identity-transport control. The four supplementary transport captions distinguish Monte Carlo projector-increment means from deterministic frame/input schedules. There is no programmed change at t=128.

### results/supplementary/beamforming/computation_time_phases.pdf

Computational phases for 64 measurements at $n=32$. Each row uses its labeled decision interval and EM iteration count. Blue, orange, pink, green and gray segments denote parameter correction, task decisions, uncertainty reports, EM fitting and remaining execution time, respectively. Each segment is the median of that component over seven repeated executions after a warm-up; a stack therefore sums component medians, rather than representing a confidence interval or independent Monte Carlo accuracy simulation. The 6 workloads retain their saved settings.

### results/supplementary/contact/computation_time_phases.pdf

Computational phases for 64 measurements at $n=3$. Each row uses its labeled decision interval and EM iteration count. Blue, orange, pink, green and gray segments denote parameter correction, task decisions, uncertainty reports, EM fitting and remaining execution time, respectively. Each segment is the median of that component over seven repeated executions after a warm-up; a stack therefore sums component medians, rather than representing a confidence interval or independent Monte Carlo accuracy simulation. The 7 workloads retain their saved settings.

### results/supplementary/contact/transport_paired_rise.pdf

Empirical distributions of the paired endpoint change $\Delta\mathcal E=\mathcal E_{192}(D_{192})-\mathcal E_{128}(D_{128})$, in $(\mathrm{mm}/\mathrm{s})^2$. Blue and dark gray curves use the posterior mean with measured transport $G_t=R_t^\top R_{t-1}$ and with $G_t=I_3$, respectively. Each empirical cumulative distribution retains all 256 simulation-level changes, including both tails; positive values indicate a larger error at $t=192$ in the same simulation. These are empirical distributions, with no confidence band. The vertical dashed line marks zero change, not a motion event.

### results/supplementary/contact/transport_projector_change.pdf

Changes of the generating tangent-plane projector during sphere motion. Blue shows $\|P_t-P_{t-1}\|_{\mathrm F}$ in sensor coordinates; dark gray shows $\|P_t-G_tP_{t-1}G_t^\top\|_{\mathrm F}$ after compensating the measured frame rotation. Solid curves are arithmetic means over 256 Monte Carlo simulations at the 191 indices $t=2,3,\ldots,192$; the initial normal and resulting physical path vary across simulations. No confidence band is displayed. The compensated increment equals the world-frame projector increment within each simulation.

### results/supplementary/contact/transport_excitation.pdf

Directional excitation of the known inputs. Let $S_t=\beta G_tS_{t-1}G_t^\top+v_tv_t^\top$, $s_t=\beta s_{t-1}+1$, and $H_t=S_t/(\sigma^2s_t)$, with $S_0=0$, $s_0=0$ and the original $\beta=0.9$. Blue, orange and pink solid curves give the increasingly ordered eigenvalues $\lambda_1(H_t)$, $\lambda_2(H_t)$ and $\lambda_3(H_t)$, evaluated at the 48 action indices $t=4,8,\ldots,192$. The vertical scale is logarithmic. The input and measured-frame schedules are shared deterministic quantities, so these curves are not Monte Carlo error means and no uncertainty band is drawn.

### results/supplementary/contact/transport_curvature.pdf

Tangent curvature of the normal posterior during sphere motion. For increasingly ordered eigenvalues of the plane parameter $A_t^+$, define $c_{\min,t}=2[\lambda_2(A_t^+)-\lambda_1(A_t^+)]$ and $c_{\max,t}=2[\lambda_3(A_t^+)-\lambda_1(A_t^+)]$. Blue and orange curves give these curvatures with measured transport; dark gray and light gray give the minimum and maximum with $G_t=I_3$. Curves are arithmetic means over 256 Monte Carlo simulations at the 48 action indices $t=4,8,\ldots,192$; matching bands give pointwise Student 95\% confidence intervals for the means. These are angular tangent curvatures of the maintained normal posterior, rather than prediction-error or coverage estimates.
