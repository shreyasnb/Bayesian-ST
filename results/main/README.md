# Main draft figures

The revised main numerical section uses two figures:

- [contact/contact_comparison.pdf](contact/contact_comparison.pdf): posterior velocity variance above contact tracking error, composed at 6.5 inches wide.
- [beamforming/moving_interference_comparison.pdf](beamforming/moving_interference_comparison.pdf): smooth and abrupt interference motion side by side, with one common legend.

The individual PDFs below retain their filenames. Information rotation, posterior spheres and the 2×2 beamforming noise-adaptation figure have resized copies under `results/supplementary/`. See [figure sizing](../../docs/paper_figure_sizing.md) for dimensions and export instructions.

## Current notebook PDFs

Running a figure cell with `SAVE_PDF=True` updates its local PDF and the same-named manuscript copy. The visible `DRAFT_SECTION` and `DRAFT_PATH` set the destination; `SAVE_PDF=False` updates neither copy.

- [beamforming/noise_adaptation_page1.pdf](beamforming/noise_adaptation_page1.pdf).
- [beamforming/task_performance_output_power.pdf](beamforming/task_performance_output_power.pdf).
- [contact/information_rotation.pdf](contact/information_rotation.pdf).
- [contact/posterior_anisotropy_risk.pdf](contact/posterior_anisotropy_risk.pdf).
- [contact/posterior_anisotropy_spheres.pdf](contact/posterior_anisotropy_spheres.pdf).
- [contact/task_performance_page1.pdf](contact/task_performance_page1.pdf).

## Caption notes

### beamforming/task_performance_output_power.pdf

Stationary beamforming: $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$ from~\eqref{eq:beam-power-comparison}, after $T=200$ measurements for each $n\in\{32,64,128\}$. Blue, orange, green and dark gray solid curves denote the posterior mean $\overline P_T^+$, mode $P_{\widehat{\mathcal X}_T}$, GREAT (sliding window) and MVDR with regularized sample covariance, respectively. Each curve joins three arithmetic means over 256 Monte Carlo simulations; blue shading gives pointwise Student 95\% confidence intervals for the posterior-mean averages.

### beamforming/noise_adaptation_page1.pdf

Causal variance estimation with $n=32$ and $T=256$. Black, gray and blue curves use supplied true variances, fixed estimates and EM estimates, respectively. The purple dashed line at $t=129$ marks the noise-variance change from $\sigma_t^2=0.2$ to $0.8$; $\tau_t^2=1$. The top-left and top-right panels show the variances available before $u_t$: solid estimate curves join medians at the eight saved indices $t=32,64,\ldots,256$, and the blue bands contain the pointwise empirical 25th--75th percentiles across 256 Monte Carlo simulations. Black references give the true schedules. The bottom-left and bottom-right panels show arithmetic means over these simulations of $\ell_t$ in~\eqref{eq:beam-power-comparison} and $\log\pi_t(u_t)$ in~\eqref{eq:beam-predictive-density}, evaluated from $q_{A_t^-}$ before correction at those same indices; all three methods have pointwise Student 95\% confidence bands. EM uses blocks of $L=32$ measurements and two iterations; a fit completed at $t$ is first used at $t+1$.

### contact/information_rotation.pdf

Anisotropy of the conditional measurement information for the stationary plane with normal $e_3$. Orange and blue solid curves show the exact $D_{\mathrm{KL}}(p_0\Vert p_z)$ for $z=(\tan\delta,0)^\top$ and $z=(0,\tan\delta)^\top$, respectively, where $n(z)=(e_3-z_1e_1-z_2e_2)/\sqrt{1+\|z\|^2}$ and $p_z$ is the joint conditional likelihood of the 16 measurements from~\eqref{eq:contact-model}. Curves evaluate 201 angles from $0.25^\circ$ to $35^\circ$; the orange and blue points mark $5^\circ$ and $30^\circ$. The thin neutral segment joins the highlighted values $D_{\mathrm{KL}}=0.23957$ at $5^\circ$ and $0.23077$ at $30^\circ$: nearly equal divergences arise at very different rotation angles. Their actual heights are retained. These are deterministic likelihood comparisons, with no Monte Carlo averaging or confidence band.

### contact/posterior_anisotropy_spheres.pdf

Posterior normal densities for one saved stationary-plane simulation at corrected time $t=16$. Left: $q_{-A_t^+}([n])$; right: the isotropic Bingham law with the same modal line and expected squared chordal distance, whose plane mean is~\eqref{eq:isotropic-control}. The blue--purple--pink--yellow colors show normalized density relative to invariant measure on one common logarithmic scale, from lower to higher density. Orange points mark the common modal line; gray curves are coordinate great circles. Antipodal normals represent the same plane. Each surface evaluates one saved distribution; it is not a Monte Carlo average or a confidence region.

### contact/posterior_anisotropy_risk.pdf

Posterior response variance $r_t(v(\theta))/v_0^2$ from~\eqref{eq:contact-response-variance}, for $v(\theta)=v_0(\cos\theta\,e_1+\sin\theta\,e_2)$ and $v_0=0.02\,\mathrm{m/s}$ at $t=16$. Blue and pink solid curves use the full and matched isotropic laws, respectively, each with its own posterior mean. The curves evaluate the same saved posterior pair at 181 angles over $[0,2\pi]$; they are deterministic posterior variances, with no Monte Carlo averaging or confidence band.

### contact/task_performance_page1.pdf

Velocity prediction error $\mathcal E_t(D_t)$ in~\eqref{eq:contact-prediction-error}, in $(\mathrm{mm}/\mathrm{s})^2$. In the stationary-plane panel (left), blue, orange, brown and green solid curves denote the posterior mean $\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$, regularized matrix $\widetilde M_t$ and GREAT (sliding window). Each joins arithmetic means over 256 Monte Carlo simulations at $t=1,4,16,32$; blue shading gives pointwise Student 95\% confidence intervals for the posterior-mean averages. In the sphere panel (right), blue, orange, pink and brown points denote the posterior mean, mode, matched isotropic mean and regularized matrix $\widetilde M_t$. Each point first averages the error over all 48 indices $t=4,8,\ldots,192$ within each simulation, then averages the 256 resulting values; horizontal bars are Student 95\% confidence intervals for every displayed method. Here $\widetilde M_t=\rho P_{\widehat{\mathcal X}_t}+(1-\rho)(2/3)I_3$.

## Optional moving-interference candidates

These independent exports are candidates for review. They do not replace the stationary presentation or select a manuscript block. Their editable source and summaries are in [the moving notebook](../../notebooks/beamforming_moving_interference.ipynb).

- [moving_interference_abrupt_output_power.pdf](beamforming/moving_interference_abrupt_output_power.pdf).
- [moving_interference_comparison.pdf](beamforming/moving_interference_comparison.pdf).
- [moving_interference_smooth_output_power.pdf](beamforming/moving_interference_smooth_output_power.pdf).
