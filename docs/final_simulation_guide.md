# Final simulation implementation for Bayesian subspace tracking

Implement the remaining experiments and produce the figures needed for the main paper and supplementary. Use the supplied TeX files as the mathematical specification. This is an implementation task, including independent checks, experiments, plotting, and a final account of what the results support.

The repository is <https://github.com/shreyasnb/Bayesian-ST>. The inspected `test` head is `a91faabeb10a63b1d1c613d75cc32503cfc0cac6`, checked on 10 November 2026. Inspect the live repository before starting. Reuse correct implementations and cached scientific results when their data, source hashes, and assumptions still match. A previous passing audit does not establish agreement with the supplied, subsequently revised supplementary.

## 1. Working branch, scope, and outputs

- Work only on `test`. Fetch the current remote state and start from `origin/test`. If `test` already exists, continue it safely. Preserve uncommitted work, using a separate worktree if necessary. Do not modify `main` or `dev`, reset someone else's work, or force-push.
- Read `AGENTS.md`, the dependency files, the two example notebooks, the supplied main and supplementary, and the current simulation findings. Record the exact code commit and TeX hashes used. The consistent formulation uses ambient dimension `n`, noise variance `sigma2`, signal variance `tau2`, and the sharper tracking bound below. An older TeX file may instead use `N`, `s`, and the looser bound. Identify such differences explicitly rather than choosing formulas by equation number alone.
- Keep the two applications. Contact remains a real, low-dimensional physical example. Beamforming becomes the principal moderate-dimensional performance experiment.
- Preserve historical results and notebooks. Write all new paper figure PDFs into these two directories:

  - `results/beamforming/`
  - `results/contact/`

  Create `data/` and `tables/` inside each directory for numerical records and machine-readable summaries. Put optional preview PNGs in `preview/` inside the corresponding example directory. Do not scatter new final figure PDFs among the old audit directories.
- Add a portable command-line entry point for each stage: small verification run, pilot, frozen paper run, plotting from saved results, and final checks. Adapt existing runners instead of duplicating the scientific implementation. State the actual commands in the repository README.
- The plotting stage must read saved numerical results. Changing a camera, label, or color must not rerun simulation, resample a posterior, change tuning, or select a new trajectory.
- Update the notebooks to explain and display the final saved outputs in a coherent order. Keep scientific calculations reproducible from scripts outside the notebooks.
- Commit and push the completed changes to `test` after verification. Do not claim that an experiment completed if it was only implemented or started. Avoid committing large intermediate caches unnecessarily.

## 2. Starting point and remaining gaps

Inspect these existing components, resolving paths against the current checkout:

| Area | Existing components to inspect | Work still needed |
|---|---|---|
| Likelihoods and recursive correction | `lib/models.py`, `lib/filter.py`, current model tests | Reconcile singleton/block, real/complex, weighting, and transport with the supplied TeX |
| Information and curvature | `lib/guide_information.py`, `lib/guide_information_counts.py` | Preserve independent checks, add stationary inverse-Fisher comparisons with bias/locality diagnostics |
| Tracking | `lib/guide_tracking.py`, `lib/theory.py` | Export the sharper bound and exact aggregate conditional drift, then recompute validity and violation statistics |
| Beamforming | `lib/array_geometry_study.py`, `lib/beamforming_decisions.py`, existing moving-array study | Add task-performance cohorts at `n=32,64`, genuine GREAT comparison, and causal variance adaptation |
| Contact | `lib/contact_geometry_study.py`, `lib/contact_mechanics.py`, `lib/anisotropy_study.py` | Retain the physical assumptions, improve the 3D camera/axes, and compare common controls fairly |
| Moments and visualization | `lib/moments.py`, `lib/spectral_series.py`, `lib/distribution_geometry_visuals.py` | Distinguish mean, normalizer, second-moment, and tail accuracy. Validate any general-rank extension separately |
| Variance estimation | `lib/block_em.py`, `scripts/run_block_em_study.py` | Connect the fixed-boundary block fit to a genuinely causal tracker without repeated assimilation |
| Prediction and model mismatch | `lib/guide_reliability.py` and its runner | Evaluate the actual forgetting/adaptive recursion, rather than only stationary-block fits to drifting data |
| Computation | `lib/simulation_scaling.py` and its runner | Separate correction, actions, uncertainty reports, and EM at their actual output frequencies |

Existing evidence is mixed against strong controls. Preserve those findings. Current beamforming performance studies cover rank one at `n=8,16` and moving arrays at `n=6`. Larger dimensions through `128` have been used for numerical cost, not equivalent task-performance cohorts. The current native scaling backend covers complex rank one and real codimension one. General-rank Fisher checks do not establish general-rank moment scalability.

## 3. One observation versus a block of observations

Use precise labels: **one observation** and **block of L observations**. A vector `u_t` is not itself a rank-one matrix. Its marginal likelihood increment is rank at most one. The conditional increment can have rank two.

Let `W=[u_j]` and `V=[v_j]` have shape `n x L`. Use transpose for real data and conjugate transpose for complex data.

### 3.1 Exact likelihood corrections

For real observations with fixed variances,

\[
B^{\mathrm{marg}}(W)=\eta WW^\top,
\qquad \eta=\frac{\tau^2}{2\sigma^2(\sigma^2+\tau^2)},
\]
\[
B^{\mathrm{cond}}(W,V)
=\frac{WV^\top+VW^\top-VV^\top}{2\sigma^2}.
\]

For a block generated at one common subspace, with `beta=1` and no intervening transport, these equal the sum of the individual increments. Test this for the parameter, density ratios, mode projector when unique, mean projector, and predictive moments.

**Do not divide these increments by L.** Normalizing a covariance can preserve its leading eigenspace under a uniform prior, but it changes the Bingham distribution, its concentration, its evidence, and its posterior predictions.

For proper complex data use `E[e e^H]=sigma2 I` and `E[e e^T]=0`. Each real/imaginary noise component has variance `sigma2/2`. The complex marginal coefficient is

\[
\eta_{\mathbb C}=\frac{\tau^2}{\sigma^2(\sigma^2+\tau^2)}.
\]

The complex conditional increment has denominator `sigma2`, with conjugate transposes. Do not carry the real factor of one half into complex Gaussian likelihoods.

### 3.2 Exact aggregation of the discounted recursion

For `G_j=I`, a block ending at time `t` reproduces the singleton recursion only if

\[
A_t^+=\beta^L A_{t-L}^+
       +\sum_{j=t-L+1}^{t}\beta^{t-j}B_j.
\]

For fixed variances, form weighted blocks by multiplying column `j` of both `W` and `V` by `sqrt(beta**(t-j))`. One block step that discounts only once by `beta`, or uses an unweighted `WW^H`, defines a different tracker. If variances change within the block, use each observation's actual coefficient and variance.

For known transport, define `Phi(t,j)=G_t ... G_(j+1)`, with `Phi(t,t)=I`. Check

\[
A_t^+=\beta^L\Phi(t,t-L)A_{t-L}^+\Phi(t,t-L)^*
 +\sum_j\beta^{t-j}\Phi(t,j)B_j\Phi(t,j)^*.
\]

Here `*` means the appropriate transpose. Include noncommuting rotations in the tests. If rotations are learned from intermediate observations, retaining their recorded values permits an algebraic comparison but does not establish equivalence to a delayed algorithm that never formed those intermediate estimates.

Before any common-subspace block likelihood or EM fit, align known coordinate changes. Transform every response and recorded input to one chosen frame and express the boundary prior in that frame. In contact, a fixed world plane observed from changing wrist frames is not a fixed plane in the raw measurement coordinates. Do not let known frame motion appear as fitted sensor noise. Unknown physical drift remains a separate approximation.

### 3.3 What the block comparison must establish

Run three distinct comparisons:

1. **Exact aggregation:** same measurements, weights, variances, initial belief, and endpoint. Singleton and aggregated calculations should agree within declared numerical tolerances.
2. **Delayed output:** produce an action after every observation versus after each disjoint block. Compare at common reporting times and account separately for delay. Never score a block estimate before its final measurement arrives.
3. **Constant-subspace approximation:** compare different block lengths under controlled subspace motion. This tests the approximation made by treating a changing block as one state. It is different from the first algebraic check.

Use `L` in `{1,4,16,32}` initially. Document physical sampling intervals and memory. The boundary factor is `beta_block=beta_single**L`, but matching this factor alone does not supply the missing within-block weights.

A sliding window used by GREAT is an optimization dataset. Reusing that window in successive optimizations is legitimate. Repeatedly adding its full likelihood increment to an already updated Bingham parameter would count measurements repeatedly. Keep these two uses of a window distinct.

Export `block_updates.pdf` for each example. Show numerical agreement separately from the task-error/delay tradeoff. Compare total processing and action costs, not just a matrix multiplication. Include a nonuniform-prior fixture to detect accidental covariance normalization.

## 4. Benchmarks and information available to each method

### 4.1 Two common references across both applications

Use these in both main examples, with identical data, variance estimates, initial belief, transport, forgetting, and reporting times:

1. **Bingham mode:** replace the posterior mean projector by the leading-subspace projector from the same `A_t`. Apply the corresponding point-estimate task rule. This isolates the effect of retaining posterior moments.
2. **Isotropic uncertainty:** retain the same mode and match the posterior expected squared chordal distance about that mode, while removing directional differences within the leading and complementary subspaces.

For the second reference, construct

\[
M_{\mathrm{iso}}=\alpha\widehat P+\xi(I-\widehat P),\qquad
\alpha=\frac{\operatorname{tr}(\widehat P M)}{k},\qquad
\xi=\frac{k-k\alpha}{n-k},
\]

where `M=E[P|data]` and `P_hat` is the same mode. Check trace, admissible eigenvalues, and equality of mean overlap. This mean suffices for the task rules used here. If reporting the density or intervals of an isotropic Bingham distribution, actually fit its concentration parameter and validate that distribution. Do not invent its tails from its mean.

For rank-one marginal inference with a uniform prior, one observation produces an exactly axial posterior. The full and isotropic means must agree. This is an equality check, not an anisotropy demonstration.

With a uniform prior and fixed marginal variances, the modal estimate is also exponentially weighted PCA. Verify this equality. Label identical estimators once in performance plots rather than presenting them as independent algorithms.

### 4.2 GREAT

Implement a genuine GREAT reference using the primary paper, Algorithm 1 and Section IV-B/C:

<https://arxiv.org/abs/2412.09052v2>

The author implementation is linked at <https://gitlab.ethz.ch/asasfi/ST_for_sysID>. Verify accessibility and provenance before reusing it. The published real objective and gradient are

\[
F_t(X)=\|(I-XX^\top)W_t\|_F^2,\qquad
\operatorname{grad}F_t(X)=-2(I-XX^\top)W_tW_t^\top X.
\]

At every new observation, GREAT warm-starts from the preceding estimate and takes a finite number of Grassmann exponential-map gradient steps. Its window covariance admits an add-new/remove-old rank-two update. It is neither an eigensolve nor GROUSE under another name.

- Implement and validate the original sliding-window algorithm. Keep the existing GROUSE implementation under its own name if it is retained.
- Add a separately labeled exponentially weighted GREAT comparison when matching the Bingham recursion. Remark 9 weights data columns by `gamma**(t-j)`, so covariance weights are `gamma**(2*(t-j))`. To match `beta**(t-j)`, use **gamma=sqrt(beta)**.
- Match the actual covariance weights, initialization, and coordinate transport when comparing objectives. Tune the step size, window/memory, and gradient-step count on validation episodes. Report task performance and computation. Do not claim that finitely many gradient steps equal exact modal optimization.
- For a motion-compensated GREAT variant, express both the estimated subspace and historical measurements/covariance in the same current coordinates. Transporting only the estimate does not match the transported Bingham objective. Label this variant explicitly.
- The paper is real-valued. For native complex beamforming, implement and label **GREAT, complex extension**, using Hermitian projectors and complex Grassmann exponential steps. Verify the real restriction, directional derivatives, basis invariance, orthonormality, and descent. A real block embedding is not automatically the same algorithm.
- For contact, GREAT uses measured responses `u_t` but does not use recorded `v_t`. It is a relevant response-only reference when those responses sufficiently excite the admissible subspace. State this information difference. Its comparison cannot isolate the benefit of Bayesian uncertainty.
- If author code cannot be obtained, implement from the published algorithm and validate the resulting implementation. Do not replace GREAT with GROUSE or a generic optimizer and retain the GREAT label.

### 4.3 Application-specific references

**Beamforming:** include validation-tuned modal diagonal loading, loaded exponentially weighted sample covariance with the isotropic endpoint included, and the true interference covariance as an evaluation-only oracle. Retain Ledoit--Wolf shrinkage in supplementary. Apply the same distortionless constraint and desired steering vector throughout. Tune loading without test episodes or future snapshots.

The matched isotropic mean gives diagonally loaded modal beamforming weights, up to an irrelevant positive covariance scale. Verify this identity. Gains over the mode alone do not identify an advantage from anisotropy.

**Contact:** include an input-aware point estimate minimizing the same conditional projector residual, or its exact spectral solution, with validation-tuned regularization if used. Report its relationship to the conditional mode instead of duplicating identical methods. Retain the separately tuned local Gaussian/EKF reference, ridge, and response-only PCA in supplementary. A same-`A` local Gaussian approximation and a recursively propagated EKF are different comparisons. Keep chart-validity failures visible.

GeRoST is optional if an additional robust comparator is useful. Use the actual published algorithm and independently tune its radius. Apply the same response-only qualification in contact and validate and label any complex extension for beamforming. Do not make the main comparison crowded merely to include every available method.

For each estimator, record whether it uses snapshots only, recorded inputs, measured coordinate changes, fitted variances, or oracle quantities. Point estimators receive no fabricated posterior intervals.

## 5. Experimental protocol

- Use independent validation and test episodes, fixed seeds, paired data across methods, identical reporting times, and the same held-out queries or disturbance draws. Freeze configurations before the full test run.
- For new primary conditions, plan around 64 validation and 256 test episodes, subject to a documented pilot cost estimate. Existing smaller cohorts may be retained with their actual counts. Do not increase repetitions selectively until a favorable result appears.
- Average held-out snapshots and query losses within an episode. Episodes, not posterior draws, reporting times, or individual query components, are the units for confidence intervals.
- Predeclare the primary outcome for each study. Retain paired differences and 95% intervals, not only sign counts or p-values. Mark broad grids as exploratory unless multiple comparisons are controlled.
- Predeclare representative trajectory IDs, reporting times, and display inputs. Choose these without examining which case has the largest advantage. Do not hide losses, failed sufficient conditions, integration failures, or finite-iteration limits.
- Keep posterior expected loss, true-state loss, and independent held-out loss separate. Bayes optimality under a maintained posterior is not evidence of uniform dominance over repeated experiments.
- Include the same-mode and isotropic controls throughout. Use a small consistent subset in main-paper plots and retain the full comparisons in supplementary.

### 5.1 Beamforming

Use interference-only training with

\[
u_t=h_tc_t+e_t,\quad \|h_t\|=1,\quad
c_t\sim\mathcal{CN}(0,\tau^2),\quad
e_t\sim\mathcal{CN}(0,\sigma^2I_n).
\]

The desired steering vector `a` enters the subsequent beamforming decision. Neither `c_t` nor the interference direction is an estimator input. State that Gaussian coefficients and independent receiver noise are model assumptions.

1. Preserve the existing `n=8,16` cohorts. Add `n=32,64`, rank one, as new task-performance cohorts. Add `n=128` after numerical and runtime feasibility checks. Keep the unsupported case visible if it cannot be completed.
2. Use stationary training counts proportional to sensor count, initially `L/n` in `{1/4,1/2,1,2}`. Preserve a fixed-count sweep in supplementary. State the steering normalization, total interference power, noise variance, and resulting power convention.
3. Hold desired/interference overlap approximately fixed across dimensions, or control angular separation relative to array beamwidth. Also retain the original fixed-angle ULA conditions. Increasing aperture changes resolvability, and independent Haar directions become more nearly orthogonal as dimension increases.
4. Include stationary, smoothly moving, abruptly changing, sensor-noise-change, and interference-power-change conditions. Primary unknown-motion comparisons use identity transport. True rotations are reserved for explicitly labeled oracle checks.
5. For moving arrays, declare the action cadence in observations and physical time. Benchmark the actual pipeline at that cadence. Preserve the existing moving `n=6` study as a supplementary reference and add a moderate-dimensional moving cohort, initially `n=32`.
6. Measure independent held-out disturbance output power and true-covariance output power. A useful primary quantity is output power divided by the true-covariance oracle power. Use `10 log10` for power ratios and define the ratio in saved figure metadata. Also retain subspace chordal error.
7. Show at least one beampattern with desired and interfering directions marked, using a predeclared episode. A beampattern illustrates a decision. It is not an independent performance repetition.
8. If adding `k=2` or higher, first validate the required moment backend. The matched covariance requires equal variance in an orthonormal spatial basis. Arbitrary nonorthogonal steering vectors or unequal modal powers generally do not give this covariance and belong in model-mismatch tests. Do not silently change the likelihood to obtain a higher rank.

### 5.2 Contact

Keep the main example on real `Gr(2,3)`. The physical model is a maintained bilateral constraint, known isotropic resistance `d`, known applied force `f`, and measured velocity:

\[
d\dot q=f+J_c(q)^\top\lambda_c,\quad J_c(q)\dot q=0,
\quad u_t=P_{\mathcal X(q_t)}v_t+e_t,\quad v_t=f_t/d.
\]

The constraint Jacobian, true normal, and reaction are used to generate or score the experiment, not by the tracker. Use independent velocity sensor noise for the matched likelihood. Inertial and other tangential forces must be negligible or explicitly compensated.

- Preserve fixed-plane and moving plane/sphere experiments. Use equal-energy balanced and directionally concentrated requests. Report the resulting Fisher eigenvalues instead of calling one case merely `weak`.
- Show the physical surface, applied request, admissible response, and measured response. Include a noiseless independent mechanics check.
- Evaluate held-out projected-response MSE and the stated command cost. For a soft command penalty, use `v^T(I-M)v`, not `||(I-M)v||^2`. The task penalty is not a hard constraint-satisfaction guarantee.
- Average the fixed query set within each episode. Keep query vectors and the task penalty common across methods.
- A measured wrist rotation gives `G_t=R_t^T R_(t-1)`. It removes a coordinate change, not unknown surface curvature. Compare identity and measured-frame transport separately. Learned forecast rotations, if retained from older experiments, are a separate empirical variant.
- Use the sphere identity `||P_dot||_2=||q_dot||/R` to verify the physical drift calculation. Distinguish prescribed-path theorem checks from closed-loop motion driven by state-dependent actions.
- Higher-dimensional codimension-one tests may use normal-line duality. Describe them as generalized linear-constraint tests, not as a complete high-degree-of-freedom robot demonstration.

## 6. Mathematical checks and theory figures

Run exact algebra checks before expensive experiments. Where appropriate, start with relative/absolute tolerances near `1e-10` in double precision, record the scale used, and distinguish floating-point error from Monte Carlo error. Derivative and sampling checks need their own declared tolerances and uncertainty estimates.

### 6.1 Likelihood, invariance, and moments

Check the following without relying on a single implementation path:

- Direct Gaussian log likelihood versus its constant-plus-trace form, including determinants and all variance-dependent constants.
- Conditional increments can be indefinite. Do not project `A` onto the positive semidefinite cone. Remove scalar identity shifts only.
- Basis invariance, orthogonal/unitary equivariance, trace-shift invariance, and exact known-motion transport of parameter and mean.
- Mean symmetry/Hermitian symmetry, eigenvalues between zero and one, trace `k`, uniform mean `(k/n)I`, and line/complement duality. For real contact planes, the normal-line parameter is `-A_plane`.
- Mode/mean eigenspace agreement and minimum mean squared chordal loss. Track nonunique modal eigenspaces explicitly.
- Normalizer derivatives and second-moment identities against an independent reference at manageable dimensions.
- Posterior response-risk decomposition, full response covariance, marginal denoising, and predictive covariance. Full response covariance needs `E[P v v^* P]`; its trace alone can be computed from the first moment.
- Beamforming distortionless gain, first-order optimality, and the excess posterior expected-power identity. Test the isotropic endpoint and the isotropic/diagonal-loading equivalence.
- The contact command solution and its expected cost, including zero task penalty and uniform posterior limits.

Save scalar checks in tables. Do not create filler figures for identities that are completely conveyed by a numerical tolerance table.

If a higher-rank marginal study starts from a uniform prior with fewer than `k` independent observations, the modal subspace can be nonunique. Use a predeclared tie convention with a degeneracy flag, or a matched informative prior. Do not attribute arbitrary eigensolver tie choices to a benefit from posterior uncertainty. An inverse-curvature calculation also requires an isolated mode.

### 6.2 Fisher information, KL divergence, and identifiability

Reproduce the real marginal and conditional formulas

\[
\mathcal F_{\mathrm{marg}}
=\frac{L(\tau^2)^2}{\sigma^2(\sigma^2+\tau^2)}I_{k(n-k)},
\qquad
\mathcal F_{\mathrm{cond}}
=\frac{S_{11}\otimes I_{n-k}+I_k\otimes S_{22}}{\sigma^2}.
\]

For proper complex marginal data, use real coordinates of dimension `2*k*(n-k)` and twice the displayed scalar coefficient. Derive any additional complex conditional information from the complex Gaussian likelihood in real coordinates rather than copying a real matrix formula blindly.

- Compare the analytic matrix with independent score covariance and the expected negative likelihood Hessian over several noise levels and actual observation counts. Count measurements per dataset separately from the number of simulated datasets.
- Check local KL expansion along several tangent directions and finite-difference step sizes. Check the exact global KL identities in the supplementary before using finite differences.
- Check conditional positive definiteness using `lambda_min(S11)+lambda_min(S22)>0`, including singular input Grams and degenerate cases. Demonstrate the distinction from the global condition `rank(V)>=n-1` using explicit fixtures.
- Compare balanced and concentrated inputs at equal `tr(VV^T)`. Test the worst-subspace information bound and the feasible isotropic design, including the requirement `L>=n` for positive isotropic input energy.
- Under the same Gaussian input law, verify the expected conditional information and the information lost when those inputs are marginalized. Do not compare different data-generation laws.
- For measurements generated at a common fixed subspace under the evaluated likelihood, verify that deterministic weights give expected weighted likelihood curvature `sum(w_j F_j)`, whereas score covariance uses `sum(w_j**2 F_j)`. Historical measurements generated along a drifting path do not automatically obey these identities when evaluated at the current subspace.

Export `information.pdf` for each example. Use an analytic-versus-estimated eigenvalue plot, a directional information plot, and/or a compact relative-error map. Avoid displaying a full large matrix when its isotropic structure makes that uninformative.

### 6.3 Posterior curvature and local accuracy

Verify the invariant-density Hessian

\[
H_A=2(A_{11}\otimes I_{n-k}-I_k\otimes A_{22}),
\]

with entries `2*(lambda_i(A)-lambda_(k+a)(A))`. Each such eigenvalue occurs twice in real coordinates for a complex posterior. Compare matrices only in aligned tangent frames, or compare spectra.

- Vary concentration from broad to concentrated, with both equal and unequal relevant eigenvalue differences.
- Compare local empirical spread with `H_A^{-1}` only where the local approximation is appropriate. Keep chart/broad-distribution failures visible.
- Distinguish the invariant-density Hessian from the graph-coordinate density Hessian, which includes the volume term `n I` at the mode in the real chart. Do not claim exact moments of unbounded graph coordinates from a Laplace approximation.
- Add a stationary inverse-Fisher accuracy study with `beta=1`, known variances, and independent repetitions. Report tangent estimation bias and the fraction of estimates remaining in the declared local regime. Use the inverse Fisher matrix as a local reference. Do not assert a global intrinsic Cramér--Rao equality or use it as a posterior-covariance bound.
- For forgetting, show the distinction between posterior curvature and sampling spread. Any local weighted-estimator covariance approximation must use the squared weights as well.

Export `posterior_curvature.pdf`. Keep the stationary inverse-Fisher and concentration diagnostics in supplementary unless they produce a particularly clear compact main-paper panel.

### 6.4 Tracking bound and its information connection

Rebuild the certificate from the supplied supplementary. Do not reuse an older cached bound with unchanged labels.

For `G_j=I`, fixed true variances, deterministic subspace paths and conditional inputs, define

\[
w_j=\beta^{t-j},\quad m_t=\sum_jw_j,\quad b_t=\sum_jw_j^2,
\quad a_0=\min_c\|A_0-cI\|_2,
\]
\[
A_t^{\mathrm{ref}}=\sum_jw_j\bar B_j(\mathcal X_t),\quad
d_t=\left\|\sum_jw_j[\bar B_j(\mathcal X_j)-\bar B_j(\mathcal X_t)]\right\|_2,
\quad \varepsilon_t=\beta^ta_0+d_t+\nu_t.
\]

The sufficient condition is `epsilon_t < gamma_t/2`. The manuscript bound is

\[
d_c(\widehat{\mathcal X}_t,\mathcal X_t)
\leq\frac{\sqrt{\min(k,n-k)}\,\varepsilon_t}{\gamma_t-\varepsilon_t}.
\]

Compute the initialization radius as `a0=(lambda_max(A0)-lambda_min(A0))/2`. Trace-centering followed by the operator norm is generally only an upper bound on this minimum over scalar identity shifts. Include a nonuniform initial parameter in this independent check.

For the real marginal model,

\[
\gamma_t=\eta\tau^2m_t,\qquad
\nu_t=4\eta(\sigma^2+\tau^2)(\sqrt{b_th}+h),
\quad h=n\log9+\log(2/\delta).
\]

For the real conditional model, use `C_t=sum(w_j v_j v_j^T)`, `D_t=sum(w_j**2 v_j v_j^T)`,

\[
\gamma_t=\frac{\lambda_{\min}(C_{11,t})+\lambda_{\min}(C_{22,t})}{2\sigma^2},
\quad
\nu_t=(2\sqrt n+\sqrt{2\log(1/\delta)})
\sqrt{\lambda_{\max}(D_t)/\sigma^2}.
\]

Use the **norm of the aggregate conditional drift matrix** from the supplementary. The sum of separate norm bounds may be retained as an explicitly looser comparison. For proper complex marginal data use `eta_C`, `h_C=2*n*log(9)+log(2/delta)`, and the corresponding complex drift and gap. Do not use the real covering dimension for complex data.

Required checks and plots:

- Independently reconstruct initialization, deterministic drift, and centered measurement fluctuations, and check their sum against the realized parameter difference, allowing scalar identity shifts.
- Verify `2*gamma_t=lambda_min(sum(w_j F_j(X_t)))` in a common tangent frame at the current true subspace. This is a reference gap, not the eigengap of the realized posterior parameter. Save both with distinct names.
- Run stationary, smooth-drift, and abrupt-change studies across forgetting factors and nonuniform initial priors. Include insufficient excitation, zero signal, and fixed-memory regimes in which the condition never holds.
- Use predeclared fixed reporting times and independent episodes. Report validity counts before reporting violations. A failed sufficient condition is neither a theorem violation nor an eligible success.
- Show actual error, valid bounds, and the initialization/noise/drift contributions. Add a heatmap of validity versus drift and memory. Never clip a bound to the geometric ceiling and continue calling the clipped curve the theorem bound.
- Recompute all counts and exact binomial violation intervals. Zero observed violations still has a nonzero upper confidence limit. Fixed-time statements are not simultaneous trajectory guarantees.
- Do not use true drift, true normals, or oracle input information to tune the tracker. These quantities are for validation of the analysis.

Export `tracking_bound.pdf`. Prefer the low-dimensional matched cases for detailed theorem checks. Do not force an informative bound in the high-dimensional, short-data regime where its sufficient condition fails.

## 7. Variance estimation and causal adaptation

### 7.1 Fixed-boundary block EM

Use a prior formed before the entire fitted block, fixed throughout its EM iterations. Retain the Gaussian scalar likelihood terms. Compare the normalized block log evidence

\[
\ell(\theta)=\sum_jc_j(\theta)
 +\psi_k\left(A_{\mathrm{pre}}+\sum_jB_j(\theta)\right)
 -\psi_k(A_{\mathrm{pre}})
\]

with an independent numerical reference where feasible. This is not the historical recorded-prior objective.

For conditional data, include both terms in

\[
\sigma_{\mathrm{new}}^2=
\frac{\|W-MV\|_F^2+
\operatorname{tr}[(M-M^2)VV^\top]}{nL},
\]

then impose the declared positive variance floor. Compare with an explicitly labeled residual-only ablation, without assuming a universal improvement in every dataset.

For marginal data, define

\[
a=\frac{\operatorname{tr}(MWW^*)}{kL},\qquad
b=\frac{\operatorname{tr}((I-M)WW^*)}{(n-k)L}.
\]

Check the full constrained M-step, including the pooled isotropic solution when `a<b`, a floor above one or both component estimates, and all-zero data. The noise contributes within the subspace as well as perpendicular to it. Preserve the boundary `tau2=0`, where orientation is unidentifiable.

Use multiple positive starts and a zero-signal start. A zero initialization can remain trapped in the recorded uniform-prior setting. Do not claim that zero signal variance is generally an absorbing EM state with an arbitrary nonuniform prior. Record finite iteration limits, evidence increments, and numerical uncertainty. Exact EM ascent applies only with exact E/M steps for the same fixed block and prior.

Export `variance_estimation.pdf`: evidence change versus iteration, fitted versus true variances across starts, and the contribution of posterior response uncertainty. Preserve all floor/boundary checks in tables.

### 7.2 Causal tracker

Implement a simple primary scheme with completed, nonoverlapping past blocks:

1. Process each new observation once using the current variance estimates.
2. Save the belief immediately preceding each fitting block.
3. At the block boundary, fit its variances from that fixed preceding prior and its observations. Under motion this is an explicitly identified constant-subspace approximation.
4. Use the fitted variances for **subsequent** corrections. Keep the streaming subspace belief already obtained. Do not append the fitted block posterior to it.
5. If a separate variant replaces the state with a reconstructed block posterior, rebuild it once from the saved boundary prior and label that variant separately. It need not equal the weighted singleton recursion.

Score next-observation predictions before seeing the scored observation. Then assimilate it. Future data must not enter a fitted variance, forecast, or decision.

Compare true variances, fixed misspecified variances, and causally learned variances, using matched data and separate validation. Include noise-only, motion-only, and signal-power-only changes. Report learned variances alongside future task loss and predictive quality. A rise in fitted sensor variance after a motion change is not successful sensor-noise recovery.

Export `noise_adaptation.pdf` for both applications. The beamforming version is a main-paper candidate. Report the fitting interval, block length, number of EM iterations, warm starts, floor, and full computational cost. This is empirical Bayes inference. The fixed-true-variance tracking theorem does not automatically cover it, and it does not integrate variance uncertainty.

## 8. Predictive uncertainty and model mismatch

Use separate targets:

- State subspace or normal line.
- Clean projected response.
- Noisy next measurement.

State-region inclusion is useful for checking the inferred distribution. Clean-response and noisy-response diagnostics assess different downstream quantities. Do not use them interchangeably or turn this into a worst-case-region paper.

For each target, distinguish prior-predictive experiments, repeated fixed-state experiments, and empirical diagnostics under drift. Include the **actual causal forgetting/variance-learning recursion**. Existing stationary-block fits to drifting states are not its calibration study.

- Plot nominal versus achieved inclusion for several levels, with episode-level uncertainty. Show interval widths or scores beside inclusion so that trivially wide intervals are visible.
- Validate tails and quantiles independently of the mean. A resolved mean at `n=128` does not certify intervals or state regions at that dimension.
- Normalized predictive log scores require the normalizer and the appropriate real/complex likelihood constants. Compare the actual Bingham-mixture predictive density with a separately labeled Gaussian moment approximation on excluded observations.
- Check predicted total response risk against repeated actual squared response error, with common held-out input distributions. Full directional covariance requires second projector moments.
- If calibrating widths or temperatures, fit them on validation data and freeze them. Do not claim a matched-coverage advantage if the achieved coverage misses its target. Temperature calibration is a separate fitted operation, not exact Bayesian propagation.
- Retain unresolved numerical cases and provide their counts beside the statistical summaries.

Export `predictive_calibration.pdf` and `predictive_scores.pdf`.

Keep model-mismatch experiments separate from the matched likelihood:

| Beamforming | Contact |
|---|---|
| Desired-signal contamination in training | Known anisotropic resistance with correctly transformed measurements/noise |
| Additional unequal-power or nonorthogonal interference | Unknown/wrong resistance, friction and sticking |
| Correlated or non-isotropic sensor noise | Contact loss or unilateral behavior outside the bilateral model |
| Direction changes faster than the assumed memory | Velocity estimated by differencing noisy positions, with correlated errors |

Generate contact mismatch from the actual constrained mechanics. With positive-definite `D`, the transformed velocity and effective force are `D^(1/2) q_dot` and `D^(-1/2) f_eff`. Raw isotropic velocity noise becomes covariance `sigma2 D`, not isotropic noise. Check this transformation independently. Keep contact-loss and friction failures visible rather than describing increased posterior spread as a repair of incorrect physics.

Export `model_mismatch.pdf` for each example.

## 9. Scalability and numerical accuracy

Separate these operations and their output cadence:

1. Additive parameter correction and optional transport.
2. Modal eigensystem.
3. Mean projector.
4. Task action, including linear solves.
5. Second moments, predictive scores, intervals, or other reports.
6. Variance fitting.

Do not add nested timings twice. A native mean routine may already compute an eigensystem. Use efficient point-estimator actions, including rank-one inverse identities, rather than forcing a dense inverse on a comparator.

Retain the existing dimension grid through `128`, and vary spectral span/concentration as well as dimension. Add timings at the statistical states encountered in the new beamforming experiments. Distinguish synthetic fixed-spectrum integration tests from end-to-end random-data performance.

Report a warmed single-thread setting with hardware/software versions, several randomized repetitions, median and upper-quantile elapsed time, and memory measurements with their scope. Dense parameter storage is quadratic in `n`. State separately whether allocation tracking includes native linear-algebra workspaces and process memory. Host timing is not a worst-case deadline.

The old endpoint measurements assimilate 64 observations and produce one endpoint action. Add separate schedules for an action every observation, an action every block, and optional uncertainty reporting. Include causal variance estimation in full-cycle timing if used in the main experiment.

The native mean/normalizer backend supports rank one or codimension one. General-rank means may use MCMC, while normalized evidence and predictive scores need additional normalizer support. A higher-rank pilot requires independent-chain convergence, effective sample size for the actual reported functionals, agreement across initializations, and Monte Carlo uncertainty. Do not present that pilot as deterministic native scaling.

Check normalizer, mean, second moments, and quantiles at their actual requested precision. The native series supplies mean/first-derivative truncation control, but this does not by itself certify floating-point rounding, higher derivatives, or tails. Preserve broad and highly concentrated failures.

Export `computation_time.pdf` and `numerical_accuracy.pdf`. Add a compact task-loss versus computation panel to the beamforming main summary. Treat rank dependence as an additional supplementary output only after a valid backend is available.

## 10. Figure preparation

### 10.1 Shared conventions

- Retain LaTeX text and the notebook's serif style. Centralize plotting settings. Use 200 dpi for previews and rasterized layers. Export PDF as the primary format, with vector text and lines. PDF does not have one global dpi. Embed fonts and avoid rasterizing the whole figure merely to impose a dpi.
- **Do not add plot titles, figure titles, captions, or long explanatory annotations in Matplotlib.** The paper will supply captions. Short panel letters, axis labels, legends, colorbars, and meaningful event markers are sufficient.
- Use fixed, colorblind-distinguishable method colors across both examples, accompanied by line styles or markers. Keep the established continuous density palette where suitable. Use sequential scales for density and a zero-centered diverging scale for signed paired differences.
- Match the final paper widths and inspect plots at those widths. Avoid a large figure whose text becomes illegible when reduced. Keep labels, legends, and colorbars uncropped.
- Use informative comparisons: paired-loss scatter or difference distributions, heatmaps across memory/drift, uncertainty bands, and clear geometry. Use line plots when time or sample count is genuinely the quantity of interest. Do not convert every result into a line with error bars.
- Distinguish confidence intervals across independent episodes from posterior credible intervals in the figure metadata. They are different quantities even when both use 95%.
- Use ordinary labels with explicit quantities and units. Examples: `Number of observations`, `Number of sensors`, `Time (s)`, `Chordal distance`, `Response MSE ((mm/s)^2)`, `Output power / oracle power (dB)`, `Noise variance ((mm/s)^2)`, `Computation time (ms)`, `Nominal probability`, and `Observed inclusion`.
- Avoid labels such as `weak`, `tightness`, `confidence gain`, `information richness`, or unexplained audit IDs. Use `Balanced inputs` and `Concentrated input directions`, accompanied by the measured information where relevant. Preserve old configuration IDs only in machine-readable records.
- Do not collapse mixed results into a favorable ranking. Provide paired effects and intervals, with all tested conditions in saved tables.
- Write concise proposed LaTeX captions into the figure manifest or a separate caption-notes Markdown file, not into the PDF. Include the conditions, repetition count, interval meaning, and any oracle or approximation there.

### 10.2 True 3D orthographic views

The current `draw_density_sphere` already uses a true 3D unit sphere and orthographic projection. Its default `azim=0` projects two coordinate directions along the same screen direction. Native axes are hidden and only short custom stubs remain. Fix the camera and axis presentation rather than replacing the sphere with a 2D heatmap.

- Keep `projection='3d'`, `ax.set_proj_type('ortho')`, and `ax.set_box_aspect((1,1,1))`.
- Use one shared oblique camera, initially `elev=25, azim=-50`, and adjust it once if needed to keep all three axes clearly distinguishable. Reuse the camera across comparisons.
- Show three visible, noncollinear axes or an adequate 3D triad. Label normal coordinates `n_1,n_2,n_3`, distinguishing them from physical position axes. Do not rely on tiny axis stubs after `set_axis_off()`.
- Use restrained great circles or other genuine surface-depth cues. Keep equal limits, unit radius, and uncropped labels. Encode density by color, not by deforming the sphere's radius.
- Contact planes on `Gr(2,3)` are represented by unoriented unit normals. Use the antipodally symmetric normal-line density with parameter `-A_plane`. Do not imply that `n` and `-n` are distinct constraint planes.
- Use area-correct integration/sampling on the sphere. A rectangular latitude/longitude histogram is not automatically density with respect to invariant surface measure.
- Use the same color normalization across matched distributions and label whether the colorbar gives density or log density. Keep density normalization and displayed contours numerically consistent.
- For complex higher-dimensional subspaces, a sphere can represent a selected density section only. Identify that limitation in caption metadata. Do not label such a section as the full posterior or attach posterior mass to it without performing the necessary integration.

### 10.3 Show and quantify anisotropy

Create `results/contact/posterior_anisotropy.pdf` with:

1. An actual posterior from a predeclared stationary contact episode and its matched isotropic reference, with the same mode and mean squared chordal distance.
2. Shared-camera orthographic views showing unequal angular spread in different directions. Add unobtrusive equal-density contours if useful.
3. A directional response-risk comparison using the same beliefs and a fixed family of unit query vectors. This should connect the visible shape to a task quantity.

Also retain the analytic swapped-eigenvalue fixture from `lib/anisotropy_study.py`: equal mode and equal entire chordal-distance law, but different directional moments and response costs. Label this a constructed identity illustration, separately from data-generated performance.

Record relevant curvature eigenvalues and, where the mode is isolated, their largest/smallest ratio. This quantifies local anisotropy. It is not a calibration statistic. A plot that looks elliptical because of camera projection alone is insufficient.

For beamforming, use task-relevant response/power uncertainty or explicitly identified tangent sections. Verify whether the full posterior improves held-out power over the matched isotropic reference. If it does not, report that finding. Visible anisotropy and beneficial anisotropy are separate claims.

Suggested paper statement, to be used only after the new plots confirm it: “The posterior assigns unequal uncertainty to rotations of the admissible plane. The matched isotropic distribution has the same mode and mean squared chordal distance but gives different projected-response risks.” Add a claim of lower repeated-experiment task loss only where that comparison is supported.

## 11. Figure catalogue and paper mapping

Create the individual PDFs below in each example directory when the corresponding study applies. A scalar identity may instead be recorded in a table. Export combined paper summaries from the same saved arrays, without rerunning scientific calculations.

| PDF basename | Required content | Intended placement |
|---|---|---|
| `setup.pdf` | Array/steering and received snapshots, or contact surface and projected motion | Main, possibly combined with the task figure |
| `task_performance.pdf` | Held-out output power or response MSE, paired common controls, relevant dimensions/regimes | Main |
| `posterior_anisotropy.pdf` | Contact density and directional response risk, or correctly identified array sections/functional uncertainty | Contact main candidate, array supplementary unless particularly informative |
| `noise_adaptation.pdf` | Causal variance estimates with future task loss and predictive quality | Beamforming main candidate, contact supplementary |
| `information.pdf` | Information eigenvalues, independent verification, conditional excitation | One contact main panel if space permits, full versions supplementary |
| `posterior_curvature.pdf` | Hessian, local covariance, concentration and inverse-Fisher comparisons | Supplementary |
| `tracking_bound.pdf` | Sharp error envelope, contributions, and validity versus drift/memory | Supplementary, with a compact main result if informative |
| `block_updates.pdf` | Exact aggregation error and block-length/delay/task/computation tradeoff | Supplementary |
| `additional_comparisons.pdf` | GREAT, regularization, memory, and retained example-specific controls | Supplementary |
| `variance_estimation.pdf` | Fixed-boundary EM evidence, starts, floors, boundaries, residual correction | Supplementary |
| `predictive_calibration.pdf` | Separate state/clean/noisy targets, inclusion and width for actual recursion | Supplementary |
| `predictive_scores.pdf` | Excluded-data scores and predicted-versus-realized response risk | Supplementary |
| `model_mismatch.pdf` | Declared noise, motion, source, and physical mismatch conditions | Supplementary |
| `computation_time.pdf` | Actual cadence, operation breakdown, dimension/concentration, memory | Compact beamforming main panel, full versions supplementary |
| `numerical_accuracy.pdf` | Independent integration, moment, normalizer, and tail precision | Supplementary |

Produce `paper_summary.pdf` in each example directory from the selected main panels. Do not fit every result into it. Contact should explain the physics and directional uncertainty. Beamforming should show task performance at larger dimension, adaptation if supported, and the cost of producing those decisions.

The supplied supplementary contains three combined figure spaces. Cover all of them with these per-example bundles and their associated tables:

| Supplementary figure space | Per-example bundle | Contents |
|---|---|---|
| `fig:sup-information`, `fig:sup-tracking` | `supp_information_tracking.pdf` | Information, curvature, inverse-Fisher study, sharp tracking bound and validity |
| `fig:sup-beamforming`, `fig:sup-contact` | `supp_comparisons.pdf` | Full task controls, block/singleton comparison, memory/dimension variants, anisotropy ablations |
| `fig:sup-variance`, `fig:sup-reliability` | `supp_variance_prediction_cost.pdf` | EM, causal adaptation, predictive quality, mismatch, and actual computation costs |

Do not make these bundles illegible to force every condition into one page. Use representative panels for the paper and retain complete grids, numerical checks, and raw paired summaries in the repository. List every omitted condition and its location. Do not modify the main or supplementary prose automatically. Supply proposed caption text and exact figure paths for insertion.

## 12. Completion checks and final handoff

Before calling the implementation complete:

- [ ] Reproduce a small existing condition before changing the implementation.
- [ ] Verify both likelihoods, complex conventions, singleton/block weighting, transport, and mean/normalizer identities.
- [ ] Implement and validate genuine GREAT, with the complex extension and input-information qualifications stated explicitly.
- [ ] Use the two common references in both examples and retain strong application-specific controls.
- [ ] Complete the new moderate-dimensional beamforming task cohorts and retain the low-dimensional contact explanation.
- [ ] Replace stale tracking exports and recompute validity/violation counts from the actual manuscript expressions.
- [ ] Verify information, KL, curvature, the information/reference-gap identity, and local inverse-Fisher accuracy.
- [ ] Complete fixed-boundary EM checks and causal future-only variance adaptation without double counting.
- [ ] Evaluate the actual recursion's predictions, uncertainty, scores, and physical/model mismatch.
- [ ] Charge the actual action/report/learning schedule, with numerical precision and unresolved targets reported.
- [ ] Export title-free, LaTeX-styled PDF figures in `results/beamforming/` and `results/contact/`, with readable orthographic 3D views.
- [ ] Rebuild all final figures from saved data. Inspect PDFs visually at intended paper size for clipping, missing axes, hidden legends, and misleading color normalization.
- [ ] Create `results/beamforming/figure_manifest.json` and `results/contact/figure_manifest.json`. For each output record its experiment, formula/claim, producer, configuration, source/data hashes, seed set, repetition unit, numerical tolerance/status, proposed caption, and main/supplementary placement.
- [ ] Create a concise `completion_report.md` with completed experiments, unsupported claims, numerical/statistical failures, mixed comparisons, reproduction commands, and the final commit. Keep detailed validation records in the example folders.
- [ ] Push the verified code, documentation, selected numerical summaries, and final PDFs to `test`. Report the resulting commit.

Complete the mandatory work rather than stopping after producing scaffolding. If a calculation is unsupported, preserve the failed or unresolved record, identify the limitation precisely, and state which figure or claim remains incomplete. Do not substitute a visually plausible approximation or reuse an older experiment under a new caption.

### Primary references for implementation details

- GREAT: <https://arxiv.org/abs/2412.09052v2>, Algorithm 1, Sections IV-B/C, Remarks 5 and 9. DOI: <https://doi.org/10.1109/TAC.2025.3636986>.
- Matplotlib 3D axes and orthographic projection: <https://matplotlib.org/stable/api/toolkits/mplot3d/axes3d.html>.
- Matplotlib figure export: <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html>.
- Existing scientific evidence: `results/simulation_redesign/findings.md`, `results/implementation_audit/findings.md`, their configurations, numerical tables, and producer hashes at the recorded repository commit.
