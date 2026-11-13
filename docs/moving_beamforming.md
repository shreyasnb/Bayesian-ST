# Moving-interference beamforming

The optional notebook [beamforming_moving_interference.ipynb](../notebooks/beamforming_moving_interference.ipynb) presents two moving-interference experiments from the original saved cohort. It uses the recorded measurements and selected settings. No simulation, posterior fit, or parameter selection is repeated.

## Model and comparisons

The array has $n=32$ sensors and receives $T=256$ measurements,

$$
u_t=h_tc_t+e_t,\qquad c_t\sim\mathcal{CN}(0,1),\qquad
 e_t\sim\mathcal{CN}(0,0.2I_n).
$$

Here $h_t=b(\varphi_t)$ is the unit half-wavelength ULA steering vector, and the desired direction is $a=b(0)$. The initial angle satisfies $|a^{\mathrm H}b(\varphi_0)|^2=1/2$. The two cases are

$$
\varphi_t=\varphi_0+2\varphi_0\frac{t-1}{255}
\quad\text{and}\quad
\varphi_t=\begin{cases}\varphi_0,&t\leq128,\\3\varphi_0,&t\geq129.\end{cases}
$$

The tracker starts with $A_0=0$, uses $G_t=I_n$, and applies forgetting with $\beta=0.97$. Thus the sensor coordinates remain fixed while the interference direction changes. Each plotted state includes the measurement at index $t$.

The four curves use the posterior mean $\overline P_t^+$, the mode $P_{\widehat{\mathcal X}_t}$, the matched isotropic mean $\overline P_t^{\mathrm{iso}}$, and GREAT with a sliding window. For rank one, writing $\alpha_t=\operatorname{tr}(\overline P_t^+P_{\widehat{\mathcal X}_t})$, the isotropic mean is

$$
\overline P_t^{\mathrm{iso}}
=\alpha_tP_{\widehat{\mathcal X}_t}
+\frac{1-\alpha_t}{n-1}(I_n-P_{\widehat{\mathcal X}_t}).
$$

It preserves the mode and posterior expected squared chordal distance. All methods receive the same measurements within a simulation. The variances are supplied. The existing settings were selected using measured held-out output power in 64 disjoint validation simulations for each motion case. The saved GREAT window has 32 measurements, three gradient steps per update, and step size $0.00625$; no settings are retuned for these figures.

For each candidate matrix $D_t$, the MVDR weight uses $\sigma^2I_n+\tau^2D_t$. Evaluation uses the generating covariance,

$$
R_t=0.2I_n+h_th_t^{\mathrm H},\qquad
p_t^\star=(a^{\mathrm H}R_t^{-1}a)^{-1},\qquad
\ell_t=10\log_{10}\!\left(\frac{w_t^{\mathrm H}R_tw_t}{p_t^\star}\right).
$$

The latent direction is used to score the simulation, not to update the estimators or select their settings. Lower $\ell_t$ is better, and the beamformer using the true covariance has $\ell_t=0$.

## Summaries and interpretation

Curves show means across 256 Monte Carlo simulations. Shaded regions show the pointwise 25th–75th percentiles across simulations. Each uses the stored indices $t=8,16,\ldots,256$ and individual `true_ratio_db` records. These are empirical performance bands, not posterior credible regions or confidence intervals for the mean. The numerical tables retain Student 95% confidence intervals. The notebook exposes the shading choices and full plotting code directly.

For the tables below, each simulation first contributes $\frac1{32}\sum_{j=1}^{32}\ell_{8j}$. The mean and its interval then use these 256 simulation averages, including the initial transient. All entries are in dB.

| Method | Smooth direction: mean [95% CI] | Abrupt direction: mean [95% CI] |
|---|---:|---:|
| Posterior mean | 0.3298 [0.3227, 0.3368] | 0.6802 [0.6686, 0.6918] |
| Mode | 0.3307 [0.3233, 0.3381] | 0.6646 [0.6525, 0.6768] |
| Isotropic mean | 0.3318 [0.3247, 0.3390] | 0.6948 [0.6829, 0.7066] |
| GREAT (sliding window) | 0.2962 [0.2900, 0.3025] | 0.6936 [0.6840, 0.7033] |

Paired differences use the same simulation and measurement indices. Positive values of reference minus posterior mean favor the posterior mean.

| Motion | Reference | Difference [paired 95% CI] (dB) |
|---|---|---:|
| Smooth | Mode | 0.00089 [-0.00077, 0.00255] |
| Smooth | Isotropic mean | 0.00205 [0.00175, 0.00234] |
| Smooth | GREAT (sliding window) | -0.03355 [-0.03745, -0.02964] |
| Abrupt | Mode | -0.01556 [-0.01771, -0.01340] |
| Abrupt | Isotropic mean | 0.01458 [0.01381, 0.01536] |
| Abrupt | GREAT (sliding window) | 0.01346 [0.00621, 0.02070] |

Retaining directional uncertainty improves on the matched isotropic mean in both cases, although the smooth-motion difference is small. The posterior mean and mode are similar during smooth motion; their paired interval includes zero. Sliding-window GREAT performs better on the smooth time average. In the abrupt case the mode performs best among these four comparisons, while the posterior mean improves on sliding-window GREAT and the isotropic mean. These experiments show different transient behavior and a measurable effect of anisotropy; they do not establish universal superiority of one method or address computational cost. The [separate comparison audit](beamforming_comparison_audit.md) includes exponentially weighted GREAT, additional optimization budgets and computational costs.

## Files and checks

The original input is `results/beamforming/data/paper/performance/moving_checkpoints.csv.gz`, verified against its original manifest and archived producer sources. Derived per-simulation values, checkpoint summaries, time averages, and paired differences are stored under `results/beamforming_moving/data/` and `results/beamforming_moving/tables/`.

The individual figures retain editable $(15,5)$-inch canvases, 300 dpi, and `text.usetex = True`. Their PDF and PNG exports are saved locally under `results/beamforming_moving/figures/` and copied to `results/main/beamforming/` using the separate names `moving_interference_smooth_output_power` and `moving_interference_abrupt_output_power`. A third editable cell exports `moving_interference_comparison.pdf` and its preview on a compact 7-by-2.8-inch canvas, with smooth motion on the left, abrupt motion on the right, and one shared legend below. All are optional main-paper candidates; stationary exports remain separate. See the [plotting update](application_plotting_update.md) for the replacement manuscript command and caption.

Run the read-only audit with:

```bash
.venv/bin/python scripts/check_moving_beamforming.py --report /tmp/moving-beamforming-check.json
```

It independently recomputes every displayed mean and interquartile band from the original rows, checks the table confidence intervals and paired differences, verifies the notebook's execution and visible plotting controls, and compares all three local/main PDF and PNG pairs. The moving-data loader also checks saved steering vectors, motion schedules, posterior-parameter recursion, and shared posterior actions without running inference.


## Figure interpretation and captions

The individual smooth/abrupt figures and the combined figure use the same
32 saved reporting indices and the same 256 independent simulations. Solid
blue, orange, pink and green trajectories denote the posterior mean
$\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$, matched isotropic mean
$\overline P_t^{\mathrm{iso}}$, and GREAT (sliding window), respectively.
The plotted centers are arithmetic means of the individual dB values.
Matching shaded bands are the empirical pointwise 25th–75th percentiles,
not confidence intervals for those means or posterior credible regions.
Only the abrupt case has the purple dashed direction-change marker at $t=129$.
Markers on method curves have been removed; no values, coordinates,
colors, means, quantiles or numerical tables change with this presentation.

Use the individual or combined LaTeX captions in
[solid_trajectory_captions.tex](solid_trajectory_captions.tex). The tabulated
Student 95% confidence intervals above still describe means of the per-simulation
32-report arithmetic averages. They have a different interpretation from
the displayed pointwise interquartile bands. The paired comparisons and mixed
method ordering remain the basis for conclusions about accuracy.
