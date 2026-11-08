# Draft-5 simulation redesign

This report is generated from verified saved scientific artifacts; it performs no simulation, posterior sampling, fitting or tuning. The supplied draft's simulations section did not define the experimental protocol.

## Execution and evidence status

| Item | Status | Scope |
| --- | --- | --- |
| geometry | Completed, source/artifact verified | Numerical and modeling qualifications remain separate |
| beamforming | Completed, source/artifact verified | Numerical and modeling qualifications remain separate |
| contact | Completed, source/artifact verified | Numerical and modeling qualifications remain separate |
| scaling | Completed, source/artifact verified | Numerical and modeling qualifications remain separate |
| General gate | Passed | Core/source fingerprints are checked before formal paper execution |
| Figures | Verified | LaTeX, 200 dpi, producers, inputs and exports |
| Notebooks | Separate execution handoff | Execution must be verified after this summary; no notebook is inspected to derive simulation claims |

## Directional geometry and task identities

The real normal-line parameters $A^{(1)}=\operatorname{diag}(0,-8,-80)$ and $A^{(2)}=\operatorname{diag}(0,-80,-8)$ have the same mode and the same entire centered chordal-distance distribution. The complementary contact plane uses parameter $-A$ and mean $I-\mathbb E[nn^T]$. The sphere displays the real projective density, antipodally symmetric; uniform density is one with respect to normalized invariant measure.

| Belief | $\rho_{0.05}$ | $r_{\mathrm{rms}}$ | Mean eigenvalue 1 | Mean eigenvalue 2 | Mean eigenvalue 3 | Numerically resolved |
| --- | --- | --- | --- | --- | --- | --- |
| $q_{A^{(1)}}$ | 0.5193 | 0.2731 | 0.9254 | 0.06829 | 0.006294 | True |
| $q_{A^{(2)}}$ | 0.5193 | 0.2731 | 0.9254 | 0.006294 | 0.06829 | True |
| $q_{\operatorname{diag}(0,-0.5,-5)}$ | 0.9958 | 0.7072 | 0.4999 | 0.3975 | 0.1026 | True |
| $q_{\operatorname{diag}(0,-8,-8)}$ | 0.6468 | 0.3714 | 0.8621 | 0.06897 | 0.06897 | True |

The swapped-pair distance-CDF error is 3.331e-16, while the mean-response risks and expected soft commands change by direction. Native-series and independent continuous-integration mean discrepancy is at most 9.95e-13; maximum response-risk identity error is 7.904e-13.

Equal posterior mass does not make an HPD set uniformly tighter than a ball. The table below bounds the normal/blocked response norm $\|nn^Tv\|$; it is not a bound on admissible-plane speed.

| Belief | Query | Ball upper bound | HPD upper bound |
| --- | --- | --- | --- |
| $q_{A^{(1)}}$ | 0 | 1 | 1 |
| $q_{A^{(1)}}$ | 1 | 0.5193 | 0.6283 |
| $q_{A^{(1)}}$ | 2 | 0.5193 | 0.1987 |
| $q_{A^{(2)}}$ | 0 | 1 | 1 |
| $q_{A^{(2)}}$ | 1 | 0.5193 | 0.1987 |
| $q_{A^{(2)}}$ | 2 | 0.5193 | 0.6283 |

The soft action is the draft's $v_t^B=[I+\lambda(I-\overline P_t^+)]^{-1}v_t^{\mathrm{des}}$. Expected cost is checked under the same supplied belief, including independent quadrature. This is a conditional deterministic calculation; selected zero-response/two-snapshot constructions and quadrature nodes are not independent random tracking trials or demonstrations of a closed-loop controller.

## Beamforming: independent training episodes

This is the native proper-complex marginal extension: $u=hz+e$, $z\sim\mathcal{CN}(0,\tau^2)$ and $e\sim\mathcal{CN}(0,sI)$. Its likelihood correction has no real-Gaussian factor one half. Estimators receive passive snapshots only; desired steering enters the distortionless decision. The uniform ambient-line prior is not a prior restricted to an arrival-angle curve.

The primary and higher-dimensional cohorts are $N=8$: 128 validation/256 test episodes per law/noise; $N=16$: 64 validation/128 test episodes per law/noise. All four horizons and both noise powers are retained for Haar signatures and ULA interferer separations $3^\circ,10^\circ,30^\circ$ from desired $0^\circ$.

Loaded covariance selects its loading continuously over the full convex-shrinkage interval, including the exact isotropic endpoint. Ledoit--Wolf adapts shrinkage to each observed covariance and has an explicit one-snapshot convention. GROUSE and the globally regularized modal covariance use independent validation. Their cost or information differences are not relabeled a Bingham-shape advantage.

Positive gain means comparator minus the posterior-mean loss; intervals are paired pointwise 95% Student intervals across independent episodes. The sign counts are exploratory, with no multiplicity correction. Crossing zero is inconclusive; only the structural one-snapshot control is an exact equality. Covariance-only comparators have no invented projector-response loss.

| $N$ | Endpoint | Comparator | Gain | Crosses zero | Loss | Exact tie | Unresolved | Undefined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | true_power | $P_{\widehat{\mathcal X}_t}$ | 26 | 6 | 0 | 0 | 0 | 0 |
| 8 | heldout_power | $P_{\widehat{\mathcal X}_t}$ | 25 | 7 | 0 | 0 | 0 | 0 |
| 8 | response_loss | $P_{\widehat{\mathcal X}_t}$ | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | posterior_power | $P_{\widehat{\mathcal X}_t}$ | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | true_power | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 8 | 12 | 4 | 8 | 0 | 0 |
| 8 | heldout_power | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 5 | 17 | 2 | 8 | 0 | 0 |
| 8 | response_loss | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 12 | 12 | 0 | 8 | 0 | 0 |
| 8 | posterior_power | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 24 | 0 | 0 | 8 | 0 | 0 |
| 8 | true_power | Regularized PCA | 6 | 16 | 10 | 0 | 0 | 0 |
| 8 | heldout_power | Regularized PCA | 6 | 17 | 9 | 0 | 0 | 0 |
| 8 | response_loss | Regularized PCA | 28 | 4 | 0 | 0 | 0 | 0 |
| 8 | posterior_power | Regularized PCA | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | true_power | Loaded covariance | 15 | 7 | 10 | 0 | 0 | 0 |
| 8 | heldout_power | Loaded covariance | 15 | 7 | 10 | 0 | 0 | 0 |
| 8 | response_loss | Loaded covariance | 0 | 0 | 0 | 0 | 0 | 32 |
| 8 | posterior_power | Loaded covariance | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | true_power | Ledoit--Wolf | 29 | 1 | 2 | 0 | 0 | 0 |
| 8 | heldout_power | Ledoit--Wolf | 28 | 2 | 2 | 0 | 0 | 0 |
| 8 | response_loss | Ledoit--Wolf | 0 | 0 | 0 | 0 | 0 | 32 |
| 8 | posterior_power | Ledoit--Wolf | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | true_power | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | heldout_power | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | response_loss | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | posterior_power | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 8 | true_power | Known covariance | 0 | 0 | 32 | 0 | 0 | 0 |
| 8 | heldout_power | Known covariance | 0 | 0 | 32 | 0 | 0 | 0 |
| 8 | response_loss | Known covariance | 0 | 0 | 32 | 0 | 0 | 0 |
| 8 | posterior_power | Known covariance | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | true_power | $P_{\widehat{\mathcal X}_t}$ | 24 | 4 | 4 | 0 | 0 | 0 |
| 16 | heldout_power | $P_{\widehat{\mathcal X}_t}$ | 24 | 4 | 4 | 0 | 0 | 0 |
| 16 | response_loss | $P_{\widehat{\mathcal X}_t}$ | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | posterior_power | $P_{\widehat{\mathcal X}_t}$ | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | true_power | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 9 | 11 | 4 | 8 | 0 | 0 |
| 16 | heldout_power | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 5 | 19 | 0 | 8 | 0 | 0 |
| 16 | response_loss | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 17 | 7 | 0 | 8 | 0 | 0 |
| 16 | posterior_power | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 24 | 0 | 0 | 8 | 0 | 0 |
| 16 | true_power | Regularized PCA | 4 | 15 | 13 | 0 | 0 | 0 |
| 16 | heldout_power | Regularized PCA | 1 | 21 | 10 | 0 | 0 | 0 |
| 16 | response_loss | Regularized PCA | 31 | 1 | 0 | 0 | 0 | 0 |
| 16 | posterior_power | Regularized PCA | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | true_power | Loaded covariance | 13 | 8 | 11 | 0 | 0 | 0 |
| 16 | heldout_power | Loaded covariance | 11 | 14 | 7 | 0 | 0 | 0 |
| 16 | response_loss | Loaded covariance | 0 | 0 | 0 | 0 | 0 | 32 |
| 16 | posterior_power | Loaded covariance | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | true_power | Ledoit--Wolf | 26 | 4 | 2 | 0 | 0 | 0 |
| 16 | heldout_power | Ledoit--Wolf | 26 | 5 | 1 | 0 | 0 | 0 |
| 16 | response_loss | Ledoit--Wolf | 0 | 0 | 0 | 0 | 0 | 32 |
| 16 | posterior_power | Ledoit--Wolf | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | true_power | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | heldout_power | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | response_loss | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | posterior_power | GROUSE | 32 | 0 | 0 | 0 | 0 | 0 |
| 16 | true_power | Known covariance | 0 | 0 | 32 | 0 | 0 | 0 |
| 16 | heldout_power | Known covariance | 0 | 3 | 29 | 0 | 0 | 0 |
| 16 | response_loss | Known covariance | 0 | 0 | 32 | 0 | 0 | 0 |
| 16 | posterior_power | Known covariance | 32 | 0 | 0 | 0 | 0 | 0 |

The generation law matters; every predeclared separation remains visible:

| $N$ | Law | $P_{\widehat{\mathcal X}_t}$ | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | Regularized PCA | Loaded covariance | Ledoit--Wolf | GROUSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | haar | 6 gains/0 losses in 8 cells | 1 gains/0 losses in 8 cells | 3 gains/0 losses in 8 cells | 7 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells |
| 8 | ula3 | 8 gains/0 losses in 8 cells | 4 gains/0 losses in 8 cells | 2 gains/2 losses in 8 cells | 3 gains/2 losses in 8 cells | 7 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells |
| 8 | ula10 | 4 gains/0 losses in 8 cells | 3 gains/0 losses in 8 cells | 1 gains/0 losses in 8 cells | 5 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells |
| 8 | ula30 | 8 gains/0 losses in 8 cells | 0 gains/4 losses in 8 cells | 0 gains/8 losses in 8 cells | 0 gains/8 losses in 8 cells | 6 gains/2 losses in 8 cells | 8 gains/0 losses in 8 cells |
| 16 | haar | 6 gains/0 losses in 8 cells | 3 gains/0 losses in 8 cells | 2 gains/0 losses in 8 cells | 5 gains/0 losses in 8 cells | 7 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells |
| 16 | ula3 | 2 gains/4 losses in 8 cells | 5 gains/0 losses in 8 cells | 0 gains/4 losses in 8 cells | 3 gains/3 losses in 8 cells | 6 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells |
| 16 | ula10 | 8 gains/0 losses in 8 cells | 1 gains/0 losses in 8 cells | 2 gains/1 losses in 8 cells | 5 gains/0 losses in 8 cells | 7 gains/0 losses in 8 cells | 8 gains/0 losses in 8 cells |
| 16 | ula30 | 8 gains/0 losses in 8 cells | 0 gains/4 losses in 8 cells | 0 gains/8 losses in 8 cells | 0 gains/8 losses in 8 cells | 6 gains/2 losses in 8 cells | 8 gains/0 losses in 8 cells |

### Designed posterior and uncertainty

The selected-data complex fixture uses a unitary transformation fixing the mode and swapping complementary axes. The two beliefs therefore share their spectrum, mode and every centered chordal-distance quantile, while their task-optimal weights can differ. This establishes that radius and mode do not determine the decision; it is not a claim that typical random training yields the fixture's advantage.

| Belief | Decision | Posterior expected power | Excess over its Bayes action |
| --- | --- | --- | --- |
| 1 | $\overline P_t^+$ | 0.7819 | 0 |
| 1 | $P_{\widehat{\mathcal X}_t}$ | 0.8716 | 0.08969 |
| 1 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 0.8053 | 0.02338 |
| 2 | $\overline P_t^+$ | 0.6681 | 0 |
| 2 | $P_{\widehat{\mathcal X}_t}$ | 0.678 | 0.009846 |
| 2 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 0.669 | 0.0008472 |

One-snapshot maximum $\|\overline P-\overline P_{\mathrm{iso}}\|_F$ is 7.712e-15; independent modal/PCA discrepancy is 3.934e-14. The mode alone duplicates covariance PCA under the same prior, powers and weights.

Native means: 0 failures. Evaluated posterior reports: 320, with 0 unresolved scalar reports and 0 unresolved state reports. These are bounded subsets with 16384 draws in each independent training/checking batch. Dense scan angles are display coordinates, not independent repetitions; the declared quantile-precision checks concern the fixed query directions.

Central scalar intervals do not have uniform fixed-state frequentist coverage. At an exactly aligned query, the true covariance power is the upper support endpoint, excluded by every finite continuous equal-tail interval. State-region mass and repeated-data state inclusion answer different questions. The sphere used for complex $N>2$ is an explicitly labeled fixed-phase three-coordinate density section, not a marginal or full eight-/sixteen-dimensional posterior.

## Contact-normal tracking

The plant is an ideal maintained bilateral contact with known isotropic resistance. Applied free velocity $v_t$, noisy realized velocity $u_t$ and measured wrist orientation are available to corresponding methods. Surface position/normal and true normal rotations are scoring information. The real conditional likelihood is $u_t=P_{\mathcal X_t}v_t+e_t$, with normal-line parameter $-A_t^+$ for the tangent-plane parameter $A_t^+$.

The static study uses 32 validation and 128 independent test datasets per equal-energy input program. Physical plane/sphere trajectories use 16 validation and 64 test trajectories per condition. Gaussian errors, query coordinates and reporting times within a trajectory are not extra independent replicates.

Both the Bingham mode and independent conditional projector least squares use the observed requests; they agree under the same statistic. Residual PCA uses $v-u$, and ridge regression uses both signals and shrinks toward the same known-rank isotropic map. The local Gaussian is computed from the same Bingham curvature; the recursive extended Kalman filter additionally changes propagation and has its own validation-selected covariance. A finite local result can remain outside its valid chart.

The next tables count only complete numerical/local-validity comparisons as resolved. Raw finite endpoint differences are retained separately, so a local-approximation failure is not silently deleted or converted into a validated Bingham win. Ridge response maps are scored for prediction, and their unsupported soft decision cost is explicitly undefined.

### Static input programs

| Endpoint | Comparator | Gain | Crosses zero | Loss | Exact tie | Unresolved | Undefined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| response_mse | $P_{\widehat{\mathcal X}_t}$ | 11 | 1 | 0 | 0 | 0 | 0 |
| response_mse | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 8 | 4 | 0 | 0 | 0 | 0 |
| response_mse | Regularized projector least squares | 8 | 4 | 0 | 0 | 0 | 0 |
| response_mse | Ridge regression | 12 | 0 | 0 | 0 | 0 | 0 |
| response_mse | PCA of residuals | 12 | 0 | 0 | 0 | 0 | 0 |
| response_mse | $\mathcal N(0,H_t^{-1})$ | 0 | 2 | 0 | 0 | 10 | 0 |
| cost_1 | $P_{\widehat{\mathcal X}_t}$ | 11 | 1 | 0 | 0 | 0 | 0 |
| cost_1 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 8 | 4 | 0 | 0 | 0 | 0 |
| cost_1 | Regularized projector least squares | 8 | 4 | 0 | 0 | 0 | 0 |
| cost_1 | Ridge regression | 0 | 0 | 0 | 0 | 0 | 12 |
| cost_1 | PCA of residuals | 12 | 0 | 0 | 0 | 0 | 0 |
| cost_1 | $\mathcal N(0,H_t^{-1})$ | 0 | 2 | 0 | 0 | 10 | 0 |
| cost_10 | $P_{\widehat{\mathcal X}_t}$ | 12 | 0 | 0 | 0 | 0 | 0 |
| cost_10 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 8 | 4 | 0 | 0 | 0 | 0 |
| cost_10 | Regularized projector least squares | 8 | 4 | 0 | 0 | 0 | 0 |
| cost_10 | Ridge regression | 0 | 0 | 0 | 0 | 0 | 12 |
| cost_10 | PCA of residuals | 12 | 0 | 0 | 0 | 0 | 0 |
| cost_10 | $\mathcal N(0,H_t^{-1})$ | 0 | 2 | 0 | 0 | 10 | 0 |

### Physical tracking

| Endpoint | Comparator | Gain | Crosses zero | Loss | Exact tie | Unresolved | Undefined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| response_mse | $P_{\widehat{\mathcal X}_t}$, same $A_t^+$ | 5 | 3 | 0 | 0 | 0 | 0 |
| response_mse | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 0 | 7 | 1 | 0 | 0 | 0 |
| response_mse | $P_{\widehat{\mathcal X}_t},\ G_t=I$ | 5 | 3 | 0 | 0 | 0 | 0 |
| response_mse | $\overline P_t^+,\ G_t=I$ | 5 | 3 | 0 | 0 | 0 | 0 |
| response_mse | $P_{\widehat{\mathcal X}_t}$ | 5 | 3 | 0 | 0 | 0 | 0 |
| response_mse | PCA of residuals | 8 | 0 | 0 | 0 | 0 | 0 |
| response_mse | Ridge regression | 8 | 0 | 0 | 0 | 0 | 0 |
| response_mse | Extended Kalman filter | 0 | 5 | 2 | 0 | 1 | 0 |
| cost_1 | $P_{\widehat{\mathcal X}_t}$, same $A_t^+$ | 4 | 4 | 0 | 0 | 0 | 0 |
| cost_1 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 1 | 6 | 1 | 0 | 0 | 0 |
| cost_1 | $P_{\widehat{\mathcal X}_t},\ G_t=I$ | 5 | 3 | 0 | 0 | 0 | 0 |
| cost_1 | $\overline P_t^+,\ G_t=I$ | 5 | 3 | 0 | 0 | 0 | 0 |
| cost_1 | $P_{\widehat{\mathcal X}_t}$ | 4 | 4 | 0 | 0 | 0 | 0 |
| cost_1 | PCA of residuals | 8 | 0 | 0 | 0 | 0 | 0 |
| cost_1 | Ridge regression | 0 | 0 | 0 | 0 | 0 | 8 |
| cost_1 | Extended Kalman filter | 0 | 5 | 2 | 0 | 1 | 0 |
| cost_10 | $P_{\widehat{\mathcal X}_t}$, same $A_t^+$ | 6 | 2 | 0 | 0 | 0 | 0 |
| cost_10 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 1 | 6 | 1 | 0 | 0 | 0 |
| cost_10 | $P_{\widehat{\mathcal X}_t},\ G_t=I$ | 6 | 2 | 0 | 0 | 0 | 0 |
| cost_10 | $\overline P_t^+,\ G_t=I$ | 5 | 3 | 0 | 0 | 0 | 0 |
| cost_10 | $P_{\widehat{\mathcal X}_t}$ | 6 | 2 | 0 | 0 | 0 | 0 |
| cost_10 | PCA of residuals | 8 | 0 | 0 | 0 | 0 | 0 |
| cost_10 | Ridge regression | 0 | 0 | 0 | 0 | 0 | 8 |
| cost_10 | Extended Kalman filter | 0 | 5 | 2 | 0 | 1 | 0 |

A real conditional singleton adds a generally indefinite matrix of rank at most two. It can already yield directional uncertainty, unlike the exactly axial one-snapshot passive marginal posterior. The following singleton comparison was fixed by the horizon grid; both equal-energy input programs have the same first command and paired noise, so these two cells are not independent replications:

| Input program | Mean-response gain over matched overlap $(\mathrm{mm}/\mathrm{s})^2$ | Pointwise 95% interval | Resolved pairs |
| --- | --- | --- | --- |
| balanced | 1.709 | [0.001832, 3.417] | 128/128 |
| unbalanced | 1.709 | [0.001832, 3.417] | 128/128 |

### Terminal and time-average response errors

All finite values remain in these descriptive means, including flagged local results. Response MSE units are $(\mathrm{mm}/\mathrm{s})^2$. Methods select their own settings on independent terminal response loss; time-average loss is a separate retained endpoint, not a universally optimized objective.

| Condition | Prediction | Trajectories | Finite terminal values | Terminal MSE | Time-average MSE | Numerically unresolved | Local-chart unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plane | $P_{\widehat{\mathcal X}_t}$, same $A_t^+$ | 64 | 64 | 0.1704 | unresolved | 0 | 0 |
| plane | $\overline P_t^+$ | 64 | 64 | 0.1704 | 1.37 | 0 | 0 |
| plane | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 64 | 64 | 0.1704 | 1.402 | 0 | 0 |
| plane | $P_{\widehat{\mathcal X}_t},\ G_t=I$ | 64 | 64 | 1.248 | 26.03 | 0 | 0 |
| plane | $\overline P_t^+,\ G_t=I$ | 64 | 64 | 1.246 | 25.77 | 0 | 0 |
| plane | $P_{\widehat{\mathcal X}_t}$ | 64 | 64 | 0.1704 | 1.575 | 0 | 0 |
| plane | PCA of residuals | 64 | 64 | 0.4213 | 3.304 | 0 | 0 |
| plane | Ridge regression | 64 | 64 | 0.8644 | 4.877 | 0 | 0 |
| plane | Extended Kalman filter | 64 | 64 | 0.176 | 1.389 | 0 | 0 |
| sphere | $P_{\widehat{\mathcal X}_t}$, same $A_t^+$ | 64 | 64 | 3.313 | unresolved | 0 | 0 |
| sphere | $\overline P_t^+$ | 64 | 64 | 3.29 | 8.313 | 0 | 0 |
| sphere | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 64 | 64 | 3.288 | 8.743 | 0 | 0 |
| sphere | $P_{\widehat{\mathcal X}_t},\ G_t=I$ | 64 | 64 | 6.35 | 10.94 | 0 | 0 |
| sphere | $\overline P_t^+,\ G_t=I$ | 64 | 64 | 6.249 | 8.876 | 0 | 0 |
| sphere | $P_{\widehat{\mathcal X}_t}$ | 64 | 64 | 3.313 | 10.31 | 0 | 0 |
| sphere | PCA of residuals | 64 | 64 | 4.485 | 16.86 | 0 | 0 |
| sphere | Ridge regression | 64 | 64 | 46.99 | 51.5 | 0 | 0 |
| sphere | Extended Kalman filter | 64 | 64 | 3.037 | 6.667 | 0 | 0 |

### Excitation, not merely command magnitude

Every command has the same norm across balanced and directionally concentrated programs. The local information depends on the actual subspace; a small input-Gram eigenvalue alone does not establish weak normal information.

| Input program | $t$ | Total input energy | Mean $\lambda_{\min}(F_t)$ | Mean $\lambda_{\max}(F_t)$ | Input rank |
| --- | --- | --- | --- | --- | --- |
| balanced | 1 | 0.0004 | 1.421 | 4 | 1 |
| balanced | 2 | 0.0008 | 4 | 6.637 | 2 |
| balanced | 4 | 0.0016 | 9.421 | 12 | 3 |
| balanced | 8 | 0.0032 | 20 | 22.64 | 3 |
| balanced | 16 | 0.0064 | 41.42 | 44 | 3 |
| balanced | 32 | 0.0128 | 84 | 86.64 | 3 |
| unbalanced | 1 | 0.0004 | 1.421 | 4 | 1 |
| unbalanced | 2 | 0.0008 | 2.814 | 7.96 | 2 |
| unbalanced | 4 | 0.0016 | 5.887 | 15.78 | 3 |
| unbalanced | 8 | 0.0032 | 11.8 | 31.53 | 3 |
| unbalanced | 16 | 0.0064 | 23.6 | 63.08 | 3 |
| unbalanced | 32 | 0.0128 | 47.22 | 126.1 | 3 |

### Fixed-time conditional modal certificate

The stationary certificate uses the residual dual $v-u=(nn^T)v-e$, which has the same sensor variance and the complementary modal normal. Known noise, deterministic inputs/transports and pointwise reporting times are required. Truth-dependent curvature/drift is an evaluation term. Invalid sufficient conditions remain invalid; they are not clipped to a radius or reported as successful confidence bounds.

bounds:

| t | Reports | Valid | Invalid | Violations | Mean valid bound |
| --- | --- | --- | --- | --- | --- |
| 1024 | 128 | 128 | 0 | 0 | 0.3204 |
| 4096 | 128 | 128 | 0 | 0 | 0.1601 |

dynamic_bounds:

| condition | t | Reports | Valid | Invalid | Violations | Mean valid bound |
| --- | --- | --- | --- | --- | --- | --- |
| plane | 8 | 128 | 0 | 128 | 0 | invalid |
| plane | 40 | 128 | 64 | 64 | 0 | 0.6788 |
| plane | 80 | 128 | 64 | 64 | 0 | 0.4637 |
| plane | 160 | 128 | 64 | 64 | 0 | 0.3277 |
| sphere | 8 | 128 | 0 | 128 | 0 | invalid |
| sphere | 40 | 128 | 0 | 128 | 0 | invalid |
| sphere | 80 | 128 | 0 | 128 | 0 | invalid |
| sphere | 160 | 128 | 0 | 128 | 0 | invalid |

### Higher-dimensional constraints

These are generalized single linear maintained constraints with isotropic resistance, not a claimed seven-joint robot with known Jacobians. Rank is $k=N-1$ and the native normal-line complement is used. Both dimension-normalized response-map error and normal chordal error are shown: a one-constraint task occupies a decreasing fraction of a larger space, so raw average response error alone would be misleading. The point-normal error for the mean and matched-overlap rows is that of their common modal plane, not a claim that a full-rank mean is itself a subspace estimate. More data are supplied as $t=4N$; this is not a fixed-sample dimension comparison.

| $N$ | $t$ | Prediction | Datasets | MSE $(\mathrm{mm}/\mathrm{s})^2$ | Normalized map error | Normal chordal error | Unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 32 | $\overline P_t^+$ | 32 | 16.65 | 0.3331 | 0.4079 | 0 |
| 8 | 32 | $P_{\widehat{\mathcal X}_t}$ | 32 | 18.31 | 0.3662 | 0.4079 | 0 |
| 8 | 32 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 32 | 16.51 | 0.3302 | 0.4079 | 0 |
| 8 | 32 | PCA of residuals | 32 | 52.54 | 1.051 | 0.702 | 0 |
| 16 | 64 | $\overline P_t^+$ | 32 | 18.1 | 0.724 | 0.6992 | 0 |
| 16 | 64 | $P_{\widehat{\mathcal X}_t}$ | 32 | 25.99 | 1.039 | 0.6992 | 0 |
| 16 | 64 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 32 | 18.56 | 0.7424 | 0.6992 | 0 |
| 16 | 64 | PCA of residuals | 32 | 42.76 | 1.71 | 0.922 | 0 |
| 32 | 128 | $\overline P_t^+$ | 32 | 10.84 | 0.8674 | 0.8192 | 0 |
| 32 | 128 | $P_{\widehat{\mathcal X}_t}$ | 32 | 17.22 | 1.377 | 0.8192 | 0 |
| 32 | 128 | $\overline P(\kappa P_{\widehat{\mathcal X}_t})$ | 32 | 11.01 | 0.881 | 0.8192 | 0 |
| 32 | 128 | PCA of residuals | 32 | 23.75 | 1.9 | 0.9738 | 0 |

### Numerical and modeling scope

```json
{
  "seconds": 123.9123755720002,
  "rows": {
    "static_metrics": 10752,
    "static_fisher": 1536,
    "dynamic_metrics": 4608,
    "dynamic_bounds": 1024,
    "higher_dimension": 384,
    "bounds": 256,
    "paired_static": 72,
    "paired_dynamic": 64,
    "story_checks": 32
  },
  "static_numerical_failures": 0,
  "static_local_gaussian_unresolved": 323,
  "dynamic_numerical_failures": 0,
  "dynamic_local_gaussian_unresolved": 7,
  "higher_numerical_failures": 0,
  "story_numerical_failures": 0,
  "bound_valid": 448,
  "bound_violations": 0,
  "maximum_modal_audit_error": 1.9895196601282805e-12,
  "maximum_decision_identity_error": 6.505213034913027e-19
}
```

Posterior means minimize squared response loss under the maintained distribution; they need not dominate every fixed physical path. The draft's expected command is a soft quadratic decision, not a guaranteed admissible velocity or a closed-loop controller. Wrist transport $G_t=R_t^TR_{t-1}$ removes a measured coordinate change, not the surface's physical curvature. Forgetting under a changing normal does not inherit stationary credible-set calibration. Accurate integration does not establish model correctness or local-Gaussian validity.

## Scaling, memory and output cadence

All dimensions $N=3,8,16,32,64,128$ retain the declared fixed spans 80/400 and separate 64-observation data targets. The real model uses $k=N-1$ through exact normal-line duality; the proper-complex model uses $k=1$. General-rank moments are not silently substituted. The paper cost profile uses 7 warmed, randomly ordered serial repetitions with BLAS limited to one thread. Repeated timing draws are not statistical Monte Carlo datasets.

The dense natural-parameter array uses exactly $8N^2$ bytes for real float64 and $16N^2$ for complex128. One conditional correction has rank at most two and a marginal correction rank one, while the stored dense update requires $O(N^2)$ operations. Dense optional transport, eigenspace extraction, positive-series moments and reports are separate work. A whole native mean already includes its eigensystem; a report includes its own mean and integration/sampling. Their nested clocks must not be added twice.

Endpoint stream comparisons assimilate the same 64 observations and output once. Rank-one modal/GROUSE beamforming uses the exact Woodbury action; it is not charged an unnecessary dense covariance inverse. The mean uses a dense solve. Conditional point and mean actions both receive the known requests. Fixed engineering settings measure implementation costs, not statistical accuracy rankings.

| Model | $N$ | Operation | Median wall ms | Median CPU ms | Resolved/attempted | CPU/wall |
| --- | --- | --- | --- | --- | --- | --- |
| conditional_real | 3 | endpoint_mean | 6.278 | 6.28 | 7/7 | 1 |
| conditional_real | 3 | endpoint_mode | 1.218 | 1.218 | 7/7 | 1 |
| conditional_real | 3 | report | 49.34 | 49.3 | 7/7 | 0.9999 |
| conditional_real | 8 | endpoint_mode | 1.335 | 1.336 | 7/7 | 1.001 |
| conditional_real | 8 | endpoint_mean | 7.952 | 7.954 | 7/7 | 1 |
| conditional_real | 16 | endpoint_mean | 9.18 | 9.182 | 7/7 | 1 |
| conditional_real | 16 | endpoint_mode | 1.412 | 1.412 | 7/7 | 1 |
| conditional_real | 32 | endpoint_mode | 1.887 | 1.887 | 7/7 | 1 |
| conditional_real | 32 | endpoint_mean | 12.56 | 12.56 | 7/7 | 1 |
| conditional_real | 64 | endpoint_mode | 4.373 | 4.374 | 7/7 | 1 |
| conditional_real | 64 | endpoint_mean | 19.18 | 19.18 | 7/7 | 1 |
| conditional_real | 128 | endpoint_mean | 37.84 | 37.85 | 7/7 | 1 |
| conditional_real | 128 | endpoint_mode | 13.98 | 13.98 | 7/7 | 1 |
| marginal_complex | 3 | endpoint_mean | 5.468 | 5.473 | 7/7 | 1.001 |
| marginal_complex | 3 | report | 25.53 | 25.53 | 7/7 | 1 |
| marginal_complex | 3 | endpoint_GROUSE | 2.617 | 2.62 | 7/7 | 1.001 |
| marginal_complex | 3 | endpoint_mode | 1.021 | 1.023 | 7/7 | 1.001 |
| marginal_complex | 3 | endpoint_SCM | 0.3879 | 0.3885 | 7/7 | 1.002 |
| marginal_complex | 8 | endpoint_mode | 1.166 | 1.167 | 7/7 | 1 |
| marginal_complex | 8 | endpoint_mean | 6.705 | 6.698 | 7/7 | 1 |
| marginal_complex | 8 | report | 44.74 | 44.74 | 7/7 | 1 |
| marginal_complex | 8 | endpoint_GROUSE | 2.527 | 2.529 | 7/7 | 1.001 |
| marginal_complex | 8 | endpoint_SCM | 0.2905 | 0.2908 | 7/7 | 1.001 |
| marginal_complex | 16 | endpoint_GROUSE | 2.531 | 2.532 | 7/7 | 1 |
| marginal_complex | 16 | endpoint_mode | 1.253 | 1.254 | 7/7 | 1.001 |
| marginal_complex | 16 | endpoint_SCM | 0.388 | 0.3883 | 7/7 | 1.001 |
| marginal_complex | 16 | report | 90.64 | 90.65 | 7/7 | 1 |
| marginal_complex | 16 | endpoint_mean | 6.662 | 6.653 | 7/7 | 1 |
| marginal_complex | 32 | report | 227.3 | 227.3 | 7/7 | 1 |
| marginal_complex | 32 | endpoint_GROUSE | 2.755 | 2.758 | 7/7 | 1.001 |
| marginal_complex | 32 | endpoint_mean | 9.819 | 9.822 | 7/7 | 1 |
| marginal_complex | 32 | endpoint_mode | 2.04 | 2.041 | 7/7 | 1.001 |
| marginal_complex | 32 | endpoint_SCM | 0.7208 | 0.7217 | 7/7 | 1.001 |
| marginal_complex | 64 | endpoint_SCM | 1.297 | 1.298 | 7/7 | 1.001 |
| marginal_complex | 64 | endpoint_GROUSE | 2.729 | 2.732 | 7/7 | 1.001 |
| marginal_complex | 64 | endpoint_mode | 4.757 | 4.759 | 7/7 | 1.001 |
| marginal_complex | 64 | endpoint_mean | 12.63 | 12.63 | 7/7 | 1 |
| marginal_complex | 128 | endpoint_GROUSE | 2.845 | 2.848 | 7/7 | 1 |
| marginal_complex | 128 | endpoint_mode | 16.13 | 16.13 | 7/7 | 1 |
| marginal_complex | 128 | endpoint_SCM | 3.751 | 3.752 | 7/7 | 1.001 |
| marginal_complex | 128 | endpoint_mean | 31.43 | 31.44 | 7/7 | 1 |

The generic real mean curve uses a common positive-series backend across dimensions; it is not the specialized three-dimensional contact controller's latency. The real $N=3$ report uses continuous sphere integration, and complex reports use two independent iid batches. Other report dimensions are explicitly not evaluated; a resolved mean is not a tail certificate. Temporary memory peaks are measured separately with Python/NumPy allocation tracking, excluding native LAPACK workspaces and total process RSS. Timing variability and host scheduling are not worst-case deadlines or proofs of empirical complexity exponents.

Retained timing attempts: 1421; unresolved: 0. See the full timing, allocation, capability and configuration tables for all 36 targets and all repeats.

## Interpretation

The draft's exact corrections and task identities are conditional mathematical statements. Random-training gains, geometry-specific losses, information imbalance, numerical limitations and measured compute cost are separate empirical findings. Broad or anisotropic distributions can matter to a task even when their modes agree, but a designed example does not predetermine a benchmark win. Strong point, shrinkage and local controls remain necessary. Historical beamforming/contact studies and their source certificates remain preserved under their original protocols.
