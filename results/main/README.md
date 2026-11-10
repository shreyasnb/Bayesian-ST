# Main draft figures

The revised main numerical section uses exactly six PDFs: two for the array and four for constrained motion. The contact sphere pair and response-variance plot are combined in TeX.

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

Array output power relative to its minimum under the unit-gain constraint: $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$, where $R_t=\sigma_t^2I_n+\tau_t^2P_{\mathcal X_t}$ and $p_t^\star=(a^{\mathrm H}R_t^{-1}a)^{-1}$. The interference direction is fixed and estimates use $t=n$ measurements at $n=32,64,128$, with $\beta=1$. Curves show averages over 256 Monte Carlo simulations, using saved exact covariance powers. Shading is a 95\% confidence interval for the average obtained with the posterior mean. Settings retain their earlier measured-data selection criterion.

### beamforming/noise_adaptation_page1.pdf

Beamforming with true variances, fixed variance estimates and EM variance estimates at $n=32$, $T=256$ and variance-estimation block length $L=32$. Panels (a,b) show the variances available before receiving $u_t$. The true noise variance is $0.2$ for $t\leq128$ and $0.8$ for $t\geq129$; the true signal variance is one. The change is drawn between measurements 128 and 129. Variance bands contain the middle 50\% of estimates across 256 Monte Carlo simulations. Panel (c) shows $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$, using the prior mean $\overline P_t^-$ and saved exact covariance powers. Panel (d) shows the normalized Bingham-mixture density $\log\pi_t(u_t)$, with $\pi_t(u)=\int p(u\mid\mathcal X,\widehat\theta_t)q_{A_t^-}(\mathcal X)\,d\mu(\mathcal X)$, evaluated before correction. Power and log-density bands are 95\% confidence intervals for Monte Carlo means. The unchanged forgetting factors are $0.98$, $0.90$ and $0.98$ for true, fixed and EM variances. An EM fit uses two iterations and affects measurements from $t+1$.

### contact/information_rotation.pdf

Change in the conditional measurement distribution under rotations of the saved plane with true normal $e_3$, using the original $t=16$ measurements. The angle $\delta$ is measured in degrees. The curves evaluate $D_{\mathrm{KL}}(p_0\|p_z)$ along $z=(\tan\delta,0)^\top$ and $z=(0,\tan\delta)^\top$, where $n(z)=(e_3-z_1e_1-z_2e_2)/\sqrt{1+\|z\|^2}$ and $p_z$ is the corresponding joint conditional likelihood. Markers retain the $5^\circ$ and $30^\circ$ rotations. This standalone figure has no inherited panel letter.

### contact/posterior_anisotropy_spheres.pdf

Directional posterior uncertainty at $t=16$. Panels (a,b) show the full and matched isotropic normal densities, normalized relative to invariant measure and drawn on a common logarithmic scale. Antipodal normals represent one plane. The point marks the common modal line; gray curves are coordinate great circles. Saved coordinates, camera, density parameters and logarithmic limits are unchanged. These panels are combined with the separate response-variance panel (c) in TeX.

### contact/posterior_anisotropy_risk.pdf

Panel (c) of the directional posterior-uncertainty figure. For $v(\theta)=v_0(\cos\theta\,e_2+\sin\theta\,e_3)$ with $v_0=0.02\,\mathrm{m/s}$, the ordinate is $r_t(v(\theta))/v_0^2$, where $r_t(v)=v^\top[\overline P_t^+-(\overline P_t^+)^2]v$. The two curves use $\overline P_t^+$ and $\overline P_t^{\mathrm{iso}}$, respectively, each with its own distribution. The saved unit-input curves equal this normalized quantity; no velocity scaling is applied to them.

### contact/task_performance_page1.pdf

Mean squared velocity prediction error $\mathcal E_t(D_t)=\tfrac13\sum_{j=1}^3\|D_tv_t^{(j)}-P_{\mathcal X_t}v_t^{(j)}\|^2$, with $v_t^{(j)}=v_0R_t^\top e_j$ and $v_0=0.02\,\mathrm{m/s}$. Panel (a) uses stationary inputs near $e_1$ at measurement indices $t=1,4,16,32$. Panel (b) uses a sphere of radius $0.05$ m; errors are averaged over $t=4,8,\ldots,192$ within each simulation before averaging the 256 values. Multiplication by $10^6$ converts the saved squared-velocity error to $(\mathrm{mm}/\mathrm{s})^2$; no additional factor of three is used. Confidence intervals are shown for the posterior mean in (a) and every method in (b).
