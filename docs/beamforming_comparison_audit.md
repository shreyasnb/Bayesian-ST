# Beamforming comparison audit

These are additional comparisons of the existing beamforming observations.
Original raw records, selected settings, notebooks and exports are retained.
The new moving and stationary notebooks and outputs live separately from the
contact experiments. No measurement model or independent simulation cohort is
changed. Replaying the original seeds produces the same observations; such a
replay is not an additional independent repetition.

## Quantity and statistical unit

Every output power is evaluated using the same true generating covariance and
unit-gain minimum for all methods in a simulation:

$$
\ell_t=10\log_{10}r_t,\qquad
r_t=\frac{w_t^{\mathrm H}R_tw_t}{p_t^\star},\qquad
p_t^\star=(a^{\mathrm H}R_t^{-1}a)^{-1}.
$$

The frozen `true_ratio_db` and `true_ratio` fields are the respective quantities.
There are 256 independent final simulations per declared condition. For moving
interference, each simulation first contributes one time aggregate. Means and
two-sided Student 95% confidence intervals use these 256 aggregates and 255
degrees of freedom. Paired intervals first subtract values from the same
simulation. Reporting times and measurements inside a simulation are not
independent repetitions. The intervals are pointwise, not simultaneous ranking
guarantees.

The primary moving quantity remains

$$
L_{\mathrm{dB,all}}=\frac1{32}\sum_{j=1}^{32}\ell_{8j}.
$$

All three secondary definitions are retained alongside it:

| Machine fields | Per-simulation quantity | Reporting times |
|---|---|---|
| `mean_db/all` (primary) | arithmetic mean of individual dB scores | 8,16,...,256 (32) |
| `mean_db/t_ge_32` | arithmetic mean of individual dB scores | 32,40,...,256 (29) |
| `log_mean_ratio/all` | 10 log10 of the mean linear power ratio | 8,16,...,256 (32) |
| `log_mean_ratio/t_ge_32` | 10 log10 of the mean linear power ratio | 32,40,...,256 (29) |

The last definition reproduces the original 29-report `moving_episodes.csv`
quantity; it is not relabelled as the current primary average. Omitting the
first three reports excludes exactly 8,16,24. The nonlinear conversion is
performed within each simulation before any across-simulation summary. Jensen's
inequality implies the log-of-mean-ratio score is at least the mean-dB score
for each simulation; the audit checks that identity.

The ordering can change with the definition. In the abrupt case, the mode has
lower primary mean dB than the posterior mean, while omitting the first three
reports reverses their ordering. In the smooth case, their ordering reverses
when linear ratios are averaged before conversion to dB. These changes do not
select which quantity is reported: the all-32 mean-dB result stays primary.

## Original common-parameter moving comparisons

Both motion cases retain n=32, T=256, sigma^2=0.2, tau^2=1, uniform Bingham
initialization and identity transport. The mean, mode and matched isotropic
mean share beta=0.97, the observations, corrected posterior parameter and
variance information. EW GREAT uses the same exponential memory. The initial
arrival direction has squared overlap 1/2 with the desired steering vector.
Smooth motion is the original linear sweep from phi_0 to 3 phi_0; abrupt motion
changes after measurement 128, at measurement 129.

The original three-step GREAT references use their original measured-data
validation choices. Their two names are always distinguished. The entries
below are mean [95% CI], in dB.

| Motion | Posterior mean | Mode | Matched isotropic mean | GREAT (sliding window) | GREAT (exponential weighting) |
|---|---|---|---|---|---|
| Smooth | 0.329782 [0.322741, 0.336823] | 0.330671 [0.323259, 0.338084] | 0.331828 [0.324693, 0.338964] | 0.296236 [0.289991, 0.302481] | 0.352250 [0.344392, 0.360108] |
| Abrupt | 0.680180 [0.668593, 0.691766] | 0.664621 [0.652489, 0.676752] | 0.694762 [0.682895, 0.706628] | 0.693635 [0.684002, 0.703268] | 0.739070 [0.726870, 0.751269] |

The primary paired EW-GREAT-minus-posterior-mean differences are
0.022468 [0.019763,0.025173] dB for smooth motion and
0.058890 [0.053753,0.064027] dB for abrupt motion. These are finite-iteration
algorithm comparisons. They are interpreted together with the optimization
checks below, rather than as an uncertainty-only advantage over EW GREAT.
All 40 aggregate means and 32 paired aggregate intervals for the four averaging
choices are available in `moving/existing/` under the audit result root.
Sliding-window GREAT has the smallest primary mean among these five methods
for smooth motion; the exact mode has the smallest for abrupt motion. Excluding
the first three times reverses the abrupt mean-versus-mode ranking, while
averaging linear ratios before converting to dB reverses their smooth ranking.
Those secondary outcomes are retained alongside the declared primary outcome.

## Matching the exponential objective

With uniform initialization, define

$$
\eta_j=\frac{\tau_j^2}{\sigma_j^2(\sigma_j^2+\tau_j^2)},\qquad
C_t=\sum_{j=1}^t\beta^{t-j}\frac{\eta_j}{\eta_0}u_ju_j^{\mathrm H}.
$$

GREAT receives likelihood-scaled observations
$s_j=\sqrt{\eta_j/\eta_0}\,u_j$. Its data-column decay is
$\gamma=\sqrt\beta$, so covariance weights are $\gamma^{2(t-j)}=\beta^{t-j}$.
Consequently, the uncentered Bingham statistic is exactly $\eta_0 C_t$.
Trace centering and multiplication by a positive constant do not change its
modal eigenspace. At the fixed variances in these experiments, eta_j/eta_0=1;
the matching objective is the ordinary raw EW covariance objective.

For a unit rank-one basis x, the objective and exact minimum are

$$
J_t(x)=\operatorname{tr}C_t-x^{\mathrm H}C_tx,\qquad
J_t^\star=\operatorname{tr}C_t-\lambda_{\max}(C_t).
$$

The exact marginal Bingham mode is the converged reference for this objective.
The audit reports its Rayleigh gap, gap divided by
$\lambda_{\max}-\lambda_{\min}$, stationarity residual and chordal discrepancy
from that exact mode. Chordal distance and squared chordal distance are stored
separately:

$$
d_c^2=1-|x^{\mathrm H}v_{\max}|^2,\qquad
J_t(x)-J_t^\star\geq
(\lambda_{\max}-\lambda_2)d_c^2.
$$

This discrepancy measures numerical optimization error relative to a common
objective. It is not tracking error relative to the unknown generating
subspace. Likewise, optimizing EW residual does not ensure the lowest true
beamforming output power.

The iteration audit retains the original three steps and assesses the fixed
step size with 1,3,10,30 inner steps. Selection uses measured validation log
power across all 32 report times for moving interference, and only t=200 for
stationary data. All four stationary report times remain in the objective
diagnostics. The original validation criterion is also replayed independently
as a reproducibility check. No test covariance selects the number of steps.

Validation selects 30 steps for both moving conditions and 10,3,10 steps for
stationary n=32,64,128. The moving paired differences below use each simulation's
mean over all 32 times; positive values favour the exact mode. Every interval
is a 95% Student interval from 256 paired simulation aggregates.

| Condition | GREAT steps | GREAT minus exact-mode relative power (dB) |
|---|---:|---:|
| Smooth | 3 | 0.021579 [0.019432, 0.023726] |
| Smooth | 30 | -0.000051 [-0.000409, 0.000308] |
| Abrupt | 3 | 0.074449 [0.069988, 0.078910] |
| Abrupt | 30 | 0.003074 [0.002130, 0.004018] |

More steps remove much of the three-step difference without changing the
measurement memory. They do not guarantee convergence in every simulation.
For smooth motion, the mean chordal distance over all test reports falls from
0.059760 to 0.003050, but the 30-step maximum is 0.973679. For abrupt motion the
corresponding means are 0.083593 and 0.005174, and the maximum is 0.985307.
The maximum relative Rayleigh gaps after 30 steps are 0.081436 and 0.134642.
These transient outliers remain in the data and figures. The remaining abrupt
power difference also prevents calling the 30-step estimate an exact mode.

At stationary t=200, three steps are already close to the mode in n=32 and64.
The n=128 paired difference falls from 0.000334 [0.000119, 0.000549] dB at three
steps to 0.000010 [-0.000013, 0.000034] dB at ten steps. This does not make the
mode a true-power optimum: an approximate eigenvector can occasionally have
slightly lower true output power, and those negative differences are retained.

The isolated timings include every covariance/parameter update and the task
decisions at the retained reporting times. They exclude observation generation,
scoring and reference-objective diagnostics. After one validation-trajectory
warm-up, each workload uses seven randomized-order repetitions of the original
test simulation 0 with one BLAS thread. The entries are median milliseconds
in this execution environment, not new statistical simulations.

| Motion | n | Measurements | Task decisions | EW GREAT, 3 steps | Selected steps | Selected EW GREAT | Exact mode |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smooth | 32 | 256 | 32 | 78.9 | 30 | 719.1 | 10.4 |
| Abrupt | 32 | 256 | 32 | 81.5 | 30 | 696.1 | 8.8 |
| Stationary | 32 | 200 | 4 | 79.4 | 10 | 230.9 | 4.3 |
| Stationary | 64 | 200 | 4 | 77.8 | 3 | 77.8 | 12.8 |
| Stationary | 128 | 200 | 4 | 103.0 | 10 | 239.4 | 45.9 |

The stationary workload makes decisions at 25,50,100,200; it is not a
final-decision-only timing. The exact mode performs one covariance update per
measurement and an eigensolve only when a decision is requested. It is faster
than these GREAT implementations at the tested dimensions and report schedules.
These measurements neither imply an asymptotic advantage for eigensolves nor
measure the more expensive posterior-moment computation. Complete phase times,
repetitions, medians and 90th percentiles are retained in each `objective/timing/`
folder.

## Independent posterior-mean memory selection

The controlled common-beta comparisons remain intact. A separate search selects
beta for the posterior-mean action itself from the original ordered grid
{1,0.99,0.97,0.9}, using 64 original independent validation simulations.
The criterion is the arithmetic mean of measured log output power, with the
same report times and time aggregation as the stated primary outcome.
Selection receives only training observations, known scales, the desired
steering vector and measured holdout covariance matrices. Latent signatures,
true covariance and oracle powers do not reach the selector.

For the primary mean-dB objective, subtracting log p_t^star adds a common
method-independent term at each time. Thus oracle normalization is unnecessary
for validation. That equivalence does not extend to a log-of-mean-linear-ratios
criterion when p_t^star varies over time. The three secondary summaries are
therefore post-selection diagnostics of the primary-selected policy, not
separately optimized policy claims.

The independent search selects beta=0.97 for both moving conditions, and beta=1
for each stationary dimension. These are the original common values. The new
selection therefore provides no beta-grid improvement to claim. When beta
matches, all 256 original mean/mode/isotropic records are reused after strict
verification. Tiny copy-roundtrip differences below 1e-12 dB are numerical ties.
Baseline numerical replay uses the saved simulation 0 and the original source
and artifact hashes; it is not described as recomputing all original integrals.

## Stationary final-time comparisons

Every stationary run has exactly T=200 training measurements and is evaluated
after the correction using measurement 200. Original beta=1 results are
retained for n=32,64,128. The following are original mean [95% CI] dB values,
including EW GREAT:

| Sensors | Posterior mean | Mode | Matched isotropic mean | GREAT (sliding window) | GREAT (exponential weighting) |
|---|---|---|---|---|---|
| 32 | 0.118789 [0.115119, 0.122459] | 0.114388 [0.110875, 0.117901] | 0.118795 [0.115125, 0.122465] | 0.363181 [0.351389, 0.374973] | 0.114385 [0.110873, 0.117897] |
| 64 | 0.236885 [0.231411, 0.242358] | 0.222701 [0.217619, 0.227782] | 0.236928 [0.231452, 0.242405] | 0.349989 [0.341957, 0.358021] | 0.222710 [0.217631, 0.227789] |
| 128 | 0.481558 [0.473464, 0.489652] | 0.440108 [0.432650, 0.447566] | 0.481689 [0.473588, 0.489789] | 0.677332 [0.665888, 0.688776] | 0.440442 [0.432999, 0.447885] |

A separate final-time validation search uses only measured output power at
measurement 200. It retains the original step multipliers and 1/3-step choices,
adds an explicit full-history window of 200 measurements, and retains EW
beta=1 with its original dimension-based step scaling. The covariance reference
also has a separately labelled mean-dB loading search at t=200. All settings
are frozen before scoring 256 original test simulations per dimension.

The globally selected sliding window is 200 in every dimension. It coincides
with the separately retained restricted full-history control; these are equal
settings and outputs, not independent algorithms. Window 200 and EW beta=1 use
the same measurement covariance through t=200; different candidate step-size
scales can still produce different finite-iteration estimates.

| Sensors | Original sliding GREAT | Final-time-selected full-history GREAT | Original EW GREAT | Final-time-selected EW GREAT |
|---|---|---|---|---|
| 32 | 0.363181 [0.351389, 0.374973] | 0.114442 [0.110925, 0.117960] | 0.114385 [0.110873, 0.117897] | 0.114435 [0.110918, 0.117952] |
| 64 | 0.349989 [0.341957, 0.358021] | 0.222805 [0.217728, 0.227882] | 0.222710 [0.217631, 0.227789] | 0.222710 [0.217631, 0.227789] |
| 128 | 0.677332 [0.665888, 0.688776] | 0.440277 [0.432829, 0.447725] | 0.440442 [0.432999, 0.447885] | 0.440442 [0.432999, 0.447885] |

The original advantage over sliding-window GREAT is not an advantage over
every GREAT implementation. Full-history and EW references approach the exact
mode and have lower mean relative power than the posterior mean in these
stationary cohorts. Some validation-selected settings perform slightly worse
on the test set than their original settings; those outcomes are retained.
All 14 stationary methods/settings and their 39 paired intervals are exported,
including the ten untouched original comparisons.

## Interpretation

Three effects must be separated. Sliding versus exponential/full-history
weights change measurement memory. GREAT versus the exact mode at identical
weights tests finite numerical optimization. Mean versus mode versus matched
isotropic mean at identical A, scales and beta tests posterior summaries.
The independently selected mean is an additional policy-selection comparison,
not a replacement for the controlled one.

Posterior-mean MVDR minimizes posterior expected **linear** output power under
the unit-gain constraint. The requested repeated-simulation average-dB score
is a different loss. In addition, the stationary benchmark fixes the
true desired/interference overlap at 1/2; it is not an average over the uniform
orientation prior. Bayes optimality therefore does not imply dominance under
this benchmark. The exact mode similarly optimizes the residual objective,
not the true covariance power. These distinctions explain why a numerically
correct uncertainty calculation need not win every power comparison.

The audit supports the stated differences for the declared models, grids and
budgets. It does not establish globally best achievable beamforming power or
an advantage over every GREAT implementation.

## Files and checks

- Moving notebook: `notebooks/beamforming_moving_comparison_audit.ipynb`.
- Stationary notebook: `notebooks/beamforming_stationary_comparison_audit.ipynb`.
- Result root: `results/beamforming_comparison_audit/`, with separate `moving/`
  and `stationary/` records, figures, selections and timing evidence.
- Existing-data summaries: each case's `existing/` folder.
- Independent mean validation: each case's `mean_validation/` folder.
- EW optimization evidence: each case's `objective/` folder.
- Final-time/full-history stationary selection: `stationary/selection/`.

All new plotting cells contain editable Matplotlib code, use LaTeX typography,
15-by-5-inch figure controls and 300-dpi PDF/PNG exports. This comparison audit
does not rewrite an existing study. New comparison figures are kept in the
audit folders instead of overwriting manuscript exports. The separately
requested application presentation changes have an explicit exact-hash
authorization record; it does not relax protection of any raw data, selected
settings, simulation checkpoints or numerical tables.

```bash
.venv/bin/python scripts/check_beamforming_comparison_audit.py
```

The read-only checker independently recomputes intervals from raw records,
verifies primary and secondary aggregation definitions, checks measured-only
selection argmins and replay identity, validates objective/chordal quantities,
checks executed notebook and actual PDF properties, and verifies preservation
hashes. Isolated timings are separate computational repetitions, not additional
statistical simulations.
