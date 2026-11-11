# Contact prediction diagnostics

The additional notebook [contact_prediction_diagnostics.ipynb](../notebooks/contact_prediction_diagnostics.ipynb) separates velocity-prediction error from posterior orientation uncertainty. It reads the original stationary-plane and moving-sphere simulations. Existing notebooks, figure presets, numerical records, and main-paper PDFs remain unchanged.

## Physical model and prediction error

The point remains in bilateral contact with a surface. With known isotropic resistance $d=10\,\mathrm{N\,s/m}$, eliminating the constraint force gives $\dot q=P_{\mathcal X(q)}f/d$. In the measured sensor frame,

$$
v_t=R_t^\top f_t/d,\qquad u_t=P_{\mathcal X_t}v_t+e_t,
\qquad e_t\sim\mathcal N(0,\sigma^2I_3).
$$

The measured rotation supplies $G_t=R_t^\top R_{t-1}$. It accounts for the changing sensor coordinates. On the sphere, the physical tangent plane also changes as the point moves.

The conditional tracker uses measured velocities, known applied inputs, and measured frame rotations. GREAT uses measured velocities and frame rotations, but does not use the applied inputs. Its larger errors therefore reflect this information difference as well as its estimator. The posterior mean, mode, and isotropic mean comparisons share the same conditional observations, variances, forgetting factor, transport, and corrected parameter.

For three orthogonal requests of speed $v_0=0.02\,\mathrm{m/s}$, the reported error is

$$
\mathcal E_t(D)=\frac13\sum_{j=1}^3
 \|D v_t^{(j)}-P_{\mathcal X_t}v_t^{(j)}\|^2
 =\frac{v_0^2}{3}\|D-P_{\mathcal X_t}\|_F^2.
$$

The last identity also holds for the rotated orthonormal query basis used in the saved experiment. Multiplication by $10^6$ converts $(\mathrm{m/s})^2$ to $(\mathrm{mm/s})^2$; no additional coordinate average is applied.

The stationary case has $T=32$, $\sigma=0.01\,\mathrm{m/s}$, and $\beta=1$. The sphere has radius $0.05$ m, $T=192$, $\sigma=0.004\,\mathrm{m/s}$, measurement interval $0.05$ s, and $\beta=0.9$. Settings were selected using noisy held-out velocity-prediction errors in 64 independent validation simulations per condition. Every comparison uses all 256 saved test simulations; no settings are retuned.

The main figures use the actual saved measurement indices: $\{1,4,16,32\}$ for the stationary plane and $\{4,16,32,64,128,192\}$ for the sphere. Shading gives a pointwise Student 95% confidence interval for the Monte Carlo average. It is not a posterior prediction interval.

## Comparisons

The supplementary reference figure includes the regularized predictor

$$
D_t=\rho P_{\widehat{\mathcal X}_t}+(1-\rho)\frac23I_3,
$$

with $\rho$ selected on the independent validation simulations. The selected values are $0.6492223$ for the stationary concentrated-input case and $0.9280718$ for the sphere. This is a blend with the uniform-prior mean, rather than the posterior's matched isotropic mean. Both GREAT memory choices are retained. Comparisons that replace measured transport by $G_t=I_3$ are separately identified.

The following values are in $(\mathrm{mm/s})^2$. The stationary column is the error at $t=16$. The sphere column uses the original full-run record: each simulation averages all 48 action times $t=4,8,\ldots,192$, and the table then averages the 256 simulations. The six saved checkpoint values are not substituted for this full-run average.

| Predictor | Stationary, $t=16$: mean [95% CI] | Sphere, full run: mean [95% CI] |
|---|---:|---:|
| Posterior mean | 26.94 [23.36, 30.52] | 9.81 [8.51, 11.10] |
| Mode | 34.06 [28.18, 39.95] | 11.54 [9.77, 13.32] |
| Isotropic mean | 28.32 [24.29, 32.34] | 10.15 [8.75, 11.55] |
| Regularized mode | 33.05 [29.23, 36.87] | 11.17 [9.52, 12.82] |
| GREAT, sliding window | 74.67 [65.52, 83.82] | 67.56 [59.57, 75.56] |
| GREAT, exponential weighting | 70.57 [61.59, 79.54] | 78.44 [70.27, 86.62] |
| Posterior mean, $G_t=I_3$ | — | 12.46 [11.27, 13.65] |
| Mode, $G_t=I_3$ | — | 14.17 [12.60, 15.73] |

Paired comparisons use the same simulation. At stationary $t=16$, the posterior mean reduces error relative to the mode by $7.12$ with interval $[4.33,9.91]$, and relative to the isotropic mean by $1.37$ with interval $[0.73,2.02]$. The corresponding sphere full-run reductions are $1.73\,[1.22,2.25]$ and $0.34\,[0.23,0.46]$.

The mode comparison measures the benefit of posterior-mean velocity prediction. The matched isotropic comparison isolates the additional effect of retaining directional uncertainty. The transport comparisons show the benefit of using the measured sensor rotation. The GREAT comparisons also involve different available inputs, so they should not be presented as an uncertainty-only or universal algorithm ranking.

Dense action-time errors are retained for the saved example trajectory, but not for every simulation. The audit reconstructs its full 48-time errors for the posterior mean, mode, isotropic mean, and regularized mode. For all 256 simulations, it verifies the immutable full-run records and their paired summaries against the original producer.

## The moving-sphere posterior

The supplementary pair of spheres uses the original simulation 0 at the fixed corrected measurement index $t=96$. For plane orientations $P=I_3-nn^\top$, complement duality gives the normal-line law

$$
q_{-A_t^+}([n])=\exp\{n^\top(-A_t^+)n-\log Z_1(-A_t^+)\}.
$$

This density is relative to normalized invariant measure; it is not a probability assigned to a single normal. Antipodal normals describe the same plane. Both plots use the saved sensor coordinates and a common logarithmic color scale. The display does not rotate the mode toward the viewer. Signs only choose consistent representatives of the same normal lines.

At this snapshot the mode is $4.2613^\circ$ from the true normal line. The curvature eigenvalues of the negative log density at the mode are $230.9353$ and $277.6860$, a ratio of $1.2024$. Thus this moving snapshot has modest anisotropy. The retained stationary density figure provides the more pronounced directional example. The matched isotropic law has the same mode and expected squared chordal distance, $0.0079637814$; its two curvature eigenvalues are both $252.1533$.

The comparison law is normalized and its moments are evaluated separately. For any response-variance diagnostic, each law uses its own mean:

$$
r_q(v)=v^\top(M_q-M_q^2)v.
$$

Consequently, the retained stationary curves of $r_q(v(\theta))/v_0^2$ compare uncertainties under two different laws. They do not show two predictors' empirical errors under one shared law. At the moving snapshot's actual input, the full and isotropic response variances are $3.35960\times10^{-6}$ and $3.40723\times10^{-6}\,(\mathrm{m/s})^2$, respectively. These quantities are distinct from the Monte Carlo confidence intervals in the error plots.

The retained static information plot uses the label $p_z$. In the new notebook, $p_Z$ denotes the same joint conditional measurement law at the candidate plane $\mathcal X(Z)$, with the old coordinate vector $z=\operatorname{vec}(Z)$. This likelihood law is distinct from the orientation law $q_{-A_t^+}$. The old PDF remains byte-for-byte unchanged and has no inherited panel letter.

## Files and verification

New outputs are confined to `results/contact_prediction_diagnostics/` and these additional manuscript candidates:

- `results/main/contact/prediction_error_stationary.pdf`
- `results/main/contact/prediction_error_sphere.pdf`
- `results/supplementary/contact/prediction_error_references.pdf`
- `results/supplementary/contact/moving_sphere_posterior_spheres.pdf`

Matching PNG files accompany the PDFs. All figures expose their full Matplotlib code, use $(15,5)$-inch canvases, 300 dpi, and `text.usetex = True`.

Run the read-only audit with:

```bash
.venv/bin/python scripts/check_contact_prediction_diagnostics.py --report /tmp/contact-prediction-check.json
```

The checker independently recomputes all averages and paired intervals, verifies the SI-to-millimeter conversion, checks sensor-frame geometry and normal signs, and integrates both snapshot densities on a separate quadrature grid. It also checks actual PDF fonts, bounds, page sizes, raster resolution, matching exports, and the recorded preservation hashes for previous library files, notebooks, results, and plotting presets.
