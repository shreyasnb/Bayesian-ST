# Implement a unified Bayesian beamforming study on branch `test`

Work in `https://github.com/shreyasnb/Bayesian-ST`. Implement and execute the study below, inspect the resulting figures, commit the code, notebook, documentation and reproducible results, and push **only to a branch named `test`**. This is an implementation task, not a request for another plan or an unexecuted notebook.

The purpose is one coherent beamforming application connecting measurement information, modal tracking, posterior uncertainty, parameter inference and the resulting beamforming decision. Retain unfavorable comparisons and numerical limitations. Success means a correct, informative experiment, not a predetermined Bayesian performance advantage.

## 1. Branch isolation comes first

Before changing any tracked file:

1. Read applicable `AGENTS.md` instructions. Inspect repository status, remotes and existing branches. Preserve all unrelated or uncommitted work.
2. Fetch `origin`. Record the local and remote `main` and `dev` tips. Read the latest `origin/dev` as the base. The version reviewed when this prompt was prepared was `3b47f8d7765d125fff6a1bbe8c8e9d43a8627517`, but inspect the current version rather than assuming it is unchanged.
3. Create a separate worktree on a new local branch named exactly `test`, based on `origin/dev`. Do all edits and generated outputs there. If `test` already exists, inspect and resume it safely rather than resetting or overwriting it. If needed, incorporate `origin/dev` into `test` while preserving existing work.
4. Never commit on, reset, delete, force-update or push `main` or `dev`. Do not force-push any branch. Do not use `git push --all` or `--mirror`.
5. After implementation and verification, confirm the checked-out branch is `test`, commit only the intended changes, and push explicitly:

```bash
git push --set-upstream origin HEAD:refs/heads/test
```

Verify the remote `test` SHA matches the intended commit. Confirm your operations did not move `main` or `dev`; do not revert changes made independently by others. Report the branch URL and commit. Do not merge into another branch or open a PR unless separately requested. If authentication blocks the push, retain the completed local commit and report the exact blocker instead of claiming success.

## 2. Inspect and reuse the existing implementation

Read the current `notebooks/01_array.ipynb`, including its saved outputs, and inspect the relevant modules before editing. Likely sources include:

- `lib/models.py`, `lib/filter.py`, `lib/geometry.py`, `lib/theory.py`;
- `lib/moments.py`, `lib/spectral_series.py`, the current process-prediction and reference evaluators;
- `lib/array_story.py`, `lib/story_visuals.py`, `lib/decision_study.py`, `lib/decision_visuals.py`;
- `lib/plotting.py`, `lib/visuals.py`, `lib/reporting.py`, `lib/validation.py`;
- `benchmarks/case1/`, current test and cache conventions;
- `notebooks/00_theory.ipynb`, `02_field.ipynb` and the matched-process part of `03_contact.ipynb`.

Read `l4dc.tex` and `supplementary.tex` if supplied. If unavailable, the mathematical scope in this prompt is sufficient to proceed. Write the proposed new passive-model statements and proofs in a repository document rather than blocking on the manuscript. Use result names and stable labels instead of copying old equation numbers.

Preserve the old complex rank-three results and the separate real low-data study. Introduce a clearly named unified study with its own modules/scripts/results directory. Make the revised array notebook tell the new story first, retaining old comparisons as explicitly labeled supplementary material. Avoid overwriting historical result tables or silently invalidating their interpretation. Shared numerical changes require the relevant existing regression checks to pass. Do not suppress numerical validation or cache checks merely to execute a notebook; recompute affected targets or preserve a properly verified historical source manifest.

## 3. Use the natural passive beamforming model

The primary new application is a single interferer:

\[
u_t=h_tz_t+e_t,\quad \|h_t\|_2=1,\quad
z_t\sim\mathcal{CN}(0,q),\quad e_t\sim\mathcal{CN}(0,sI_n),
\qquad R_t=sI_n+qP_{\mathcal X_t},\quad \mathcal X_t=[h_t].
\]

Here `s` and `q` are variances/powers, not standard deviations. Repository `tau2` corresponds to `q`. Use proper complex Gaussians and the native complex Grassmann measure. Training snapshots contain interference plus sensor noise. The latent `z_t` and truth `h_t` are never supplied to an estimator. The desired unit steering signature `a_t` is supplied only for the decision, with distortionless constraint `w^H a_t=1`.

Use the same observation and decision model in three clearly distinguished regimes:

1. **Stationary matched statistical control:** a fixed Haar-distributed complex line, independent snapshots, known `s,q`, uniform Bingham prior, `beta=1`.
2. **Physical array:** a half-wavelength ULA, with
   `h_j(theta)=exp(1j*pi*j*sin(theta))/sqrt(n)`, `j=0,...,n-1`, and an independently specified desired arrival angle. Use stationary and deterministic moving-interferer paths. State angles in radians internally and degrees in figures.
3. **Matched stochastic tracking control:** the invariant Grassmann transition described below. It is a model for a changing spatial signature, not the same motion law as a single plane wave moving along the ULA steering curve.

A uniform prior on all complex lines is different from a distribution supported only on one arrival angle. State this distinction. Do not call a general ambient Bingham posterior an exact posterior over arrival angle. Distinguish arrival-angle error from the principal angle between two estimated subspaces.

Start physical tracking with `G_t=I`. Do not use true rotations in the practical estimator. An exact supplied-motion run may appear as a separately labeled oracle control. Any learned causal transport is an optional ablation, shared fairly with point comparators, and lies outside the deterministic-transport certificate unless separately analyzed.

Predeclare seeds, conditions, display trajectories, validation choices and evaluation times before running the main comparisons. Use a quick profile for debugging and a reproducible paper profile. A reasonable starting scale is `n=8`, with a separate `n=16` feasibility check, snapshot counts `1,2,4,8,16,32,64`, two noise regimes such as `q=1, s in {1,0.2}`, 256 independent test trajectories and 128 validation trajectories. Check numerical feasibility first and record any changes before scientific evaluation. Do not reduce dimensions or discard regimes after inspecting which method wins.

Long-horizon tracking certificates may need a separate larger-sample cohort. Evaluate expensive moments and tails only at predeclared reporting times. Do not request a full posterior report after every snapshot merely to make an animation.

## 4. Required result map

Implement or reuse an appropriate numerical counterpart for the following. Clearly distinguish an existing paper result, its new passive-model specialization, an approximation and an empirical finding.

| Paper result | Required numerical counterpart |
|---|---|
| Exact Bingham correction and transport | Normalized likelihood/update checks, complex constants, one assimilation per sample |
| Modal least-squares/EW-PCA connection | Matching passive Bingham/PCA modes under identical weights and transport |
| Fisher information | Derive and verify the passive complex information below; keep the original conditional result separate |
| Tracking bound | Derive and implement a valid passive covariance certificate; reuse conditional fixtures only with their original measurement model |
| Posterior response risk | Check and display the response-risk identity with clearly labeled derived spatial-projection queries |
| HPD/ball geometry and coverage | Derive the stationary passive invariance counterpart, then report calibrated/misspecified cases under their actual assumptions |
| Stochastic prediction and moment fitting | Check the exact predictive mean law and quantify the single-Bingham density approximation |
| Joint parameter inference and EM | Implement the appropriate passive hierarchy and distinguish it from recorded-prior EM and fixed parameters |
| Beamforming expected-power decision | Verify conditional optimality and evaluate actual repeated-training performance, including an anisotropy control |

The original conditional Fisher formula, global probe-rank identifiability result and conditional tracking theorem do **not** apply to passive snapshots by renaming the latent coefficients as known inputs. Their existing tests and plots remain valid in their own experiments. Use links or a short supplementary comparison, not an invented physical probe supplied to the array method.

## 5. Exact filtering and posterior decisions

For known fixed `s,q`, the complex correction is

\[
B_t=\frac{q}{s(s+q)}u_tu_t^H,\qquad A_t^+=A_t^-+B_t
\]

up to an identity shift. Preserve the trace-zero convention. There is no real-Gaussian factor `1/2`. With `A_0=0`, fixed parameters and identical transport/weights, its mode must match weighted covariance PCA. Check against an independent scatter-matrix implementation, allowing for nonunique eigenspaces at ties rather than hiding them.

Use the native rank-one spectral-series mean/normalizer backend where accurate. Check its actual numerical range, adaptive truncation, floating-point behavior and independent low-dimensional references. Its truncation guarantee does not establish tail or Hessian accuracy. Do not introduce a complex-to-unrestricted-real shortcut. Do not silently replace accurate means by an unchecked Laplace approximation at high concentration.

At known `s,q`, choose

\[
\overline R_t=sI+qM_t,\quad M_t=\mathbb E[P_{\mathcal X_t}\mid\text{data}],\qquad
w_t=\frac{\overline R_t^{-1}a_t}{a_t^H\overline R_t^{-1}a_t}.
\]

Implement the modal covariance, validation-tuned loaded sample covariance, matching EW-PCA, a tuned scalar-shrinkage control and the true-covariance oracle. Include a suitable existing geometric tracker where its information and tuning are comparable. Give tuned families comparable independent validation effort. Record boundary selections and expand a clearly inadequate family on validation only. The old three paired loading/forgetting candidates are not a broad loading search.

Verify unit gain and posterior expected-power optimality. For any feasible comparator `w`, the exact identity

\[
w^H\overline R_t w-w_t^H\overline R_t w_t
=(w-w_t)^H\overline R_t(w-w_t)\geq0
\]

is a useful check. Separately evaluate `w^H R_true w` and independent held-out snapshot power. Plot conditional posterior excess and realized performance as different quantities. Minimizing expected power is not the same as maximizing expected SINR or expected log-SINR.

### Required anisotropy control

For modal projector `P_hat`, rank `k=1` and mean `M`, form

\[
\rho=\frac{\operatorname{tr}(P_{\rm hat}M)-k^2/n}{k-k^2/n},\qquad
M_{\rm iso}=\rho P_{\rm hat}+(1-\rho)\frac{k}{n}I.
\]

Compare weights based on `sI+qM` and `sI+qM_iso`. This preserves the mode and expected squared chordal error from it. Do not fit this scalar to the test truth. Keep it distinct from a single scalar chosen globally on validation.

One passive snapshot with a uniform prior is a required negative control: its Bingham parameter has two eigenvalue levels, so `M == M_iso` and the two decisions agree. Gains over the mode at that horizon are not evidence of anisotropy. Use multiple snapshots for the nontrivial comparison, report `||M-M_iso||_F` and spectra, and retain ties or losses.

## 6. Derive and verify passive Fisher information

Use real local coordinates for a complex Grassmann tangent. With orthonormal `X` and `X_perp`, write the projector differential as

\[
\dot P=X_\perp ZX^H+XZ^HX_\perp^H,
\qquad Z\in\mathbb C^{(n-k)\times k}.
\]

For independent proper complex Gaussian snapshots with `R=sI+qP` and known `s,q`, the one-snapshot Fisher quadratic form should be

\[
\mathcal I_{\mathcal X}(Z,Z)
=\operatorname{tr}(R^{-1}q\dot P R^{-1}q\dot P)
=\frac{2q^2}{s(s+q)}\|Z\|_F^2.
\]

Derive this from the Gaussian score, document the real-coordinate convention and verify it independently. For `N` stationary snapshots multiply by `N`. The analogous real model has half this factor. Do not substitute the Hessian of a finite truncated normalizer for this measurement Fisher information.

Also pull this information back to the physical ULA arrival angle. Let

\[
g_\theta=(I-hh^H)\frac{\partial h}{\partial\theta},\qquad
\mathcal I_{\theta\theta}
=\frac{2Nq^2}{s(s+q)}\|g_\theta\|^2.
\]

For the normalized half-wavelength ULA and a radian angle,
`||g_theta||^2 = pi^2*cos(theta)^2*(n^2-1)/12`. Check the derivative and all degree/radian conversions independently. The ambient subspace Fisher information is isotropic; this physical-coordinate information changes with angle because the steering curve has a varying tangent. Do not claim anisotropic population information for the isotropic ambient model.

Validate using Monte Carlo score covariance and finite-difference expected likelihood curvature at predeclared states and SNRs. Include off-diagonal terms, not just a fitted scalar. A local covariance/CRB comparison is meaningful only with its regularity, bias and coordinate assumptions. A Bayesian finite-sample estimator is not required to sit above an unbiased CRB in every plotted case. Do not confuse posterior curvature, measurement Fisher information and effective sample size.

## 7. Implement an actual passive tracking certificate

Write a short derivation in `docs/beamforming_theory.md` before treating a curve as a confidence bound. For the additive discounted recursion at fixed parameters and a uniform initial prior, the modal passive tracker is weighted covariance PCA. Its perturbation is a centered quadratic Gaussian sample-covariance term, not the linear noise term in the paper's conditional theorem. The certificate below is for that recursion. It does not automatically apply to moment-projected stochastic prediction, learned parameters or integrated mixture posteriors.

A conservative explicit candidate is given here to make the task concrete. Verify its proof and constants before using it. Take a uniform initial prior, fixed known `s,q>0`, a deterministic true subspace path, independent proper complex snapshots and deterministic supplied transports. Identity transport is the primary practical case. At time `t`, let normalized weights `alpha_j>=0` sum to one, with transported observations `tilde u_j` and true transported projectors `tilde P_j`. Define

\[
\widehat R_t=\sum_j\alpha_j\widetilde u_j\widetilde u_j^H,
\quad R_t=sI+qP_{\mathcal X_t},\quad
d_t^{\rm cov}=q\left\|\sum_j\alpha_j(\widetilde P_j-P_{\mathcal X_t})\right\|_2.
\]

For fixed `0<delta<1`, set

\[
x=2n\log9+\log(2/\delta),\qquad
\eta_t=2(s+q)\left[\sqrt{2x\sum_j\alpha_j^2}
+x\max_j\alpha_j\right],\qquad
\varepsilon_t=d_t^{\rm cov}+\eta_t.
\]

The proposed justification is a weighted centered-exponential tail for each fixed unit complex vector, a `1/4` net of the complex sphere of size at most `9^(2n)`, and the Hermitian quadratic-form norm estimate. The covariance norms are at most `s+q`. This gives a fixed-time probability at least `1-delta` that the sample-covariance perturbation from `R_t` is at most `epsilon_t`. If `epsilon_t<q/2`, an eigenspace argument gives

\[
d_c(\widehat{\mathcal X}_t,\mathcal X_t)
\leq\sqrt{\min(k,n-k)}\frac{\varepsilon_t}{q-\varepsilon_t}.
\]

Use `k=1` in the primary experiment. Record invalid conditions as invalid, not as successful clipped certificates. A prior other than uniform introduces another deterministic term and requires separate handling. If you find a proof error, fix it or withhold the confidence claim; do not replace `eta_t` by the realized random error and call the result a high-probability bound.

This bound is intentionally conservative. As an initial nonvacuous stationary fixture, `n=4, k=1, s=0.2, q=1, N=4096, delta=0.05`, with equal weights and zero drift, gives approximately `eta=0.2570<0.5` and bound `0.3459`. Verify independently. Include a long-horizon cohort without calculating expensive moments at every step, as well as practically relevant shorter/drifting cases where this sufficient condition may fail. A sharper independently justified covariance bound may be added, with its source, assumptions and proof stated separately.

Display motion, retained information and noise together. For exponential weights, report `N_eff=1/sum(alpha_j^2)` and the weight sum used in the unnormalized Bingham parameter. Do not automatically equate this effective sample size with the Fisher information of a discounted pseudo-posterior: weighted score covariance and expected Hessian have different weight dependence.

For unit scan directions, the same modal event also controls

\[
\left|a(\theta)^H(\widehat R_{\rm mode}-R_t)a(\theta)\right|
\leq q\,d_c(\widehat{\mathcal X}_t,\mathcal X_t).
\]

This connects the tracking analysis to the array response. It does not automatically bound the posterior mean's error or certify credible coverage.

Evaluate certificates at predeclared fixed times across independent observation streams. If displaying a curve, distinguish pointwise guarantees from a union-bound simultaneous guarantee. Never plot an empirical quantile or a posterior radius as this theorem bound. Label the truth-dependent drift term as an evaluation quantity unavailable to the estimator.

## 8. Stochastic prediction and parameter inference

### A. Predict a distribution with an explicit transition

For the complex rank-one state, implement or adapt

\[
K_\kappa(\mathcal X\mid\mathcal Y)
\propto\exp\{\kappa\operatorname{tr}(P_{\mathcal X}G P_{\mathcal Y}G^H)\}.
\]

Use the complex normalizer. If
`M(kappa*P0)=a_kappa*P0+b_kappa*(I-P0)` and `rho=a_kappa-b_kappa`, the exact predictive mean is

\[
M_t^-=\rho_\kappa G M_{t-1}^+G^H+(1-\rho_\kappa)I/n.
\]

Check this mean law against an independent transition sampler and a low-dimensional reference. Fit a single Bingham density by moment matching. Record its inverse-moment residual, iteration limit and any failure. The predictive density is generally a mixture, so matching its first moment does not make the filter exact or certify its tails.

For complex rank one, `r=|h^H h_prev|^2` under the transition has density proportional to `exp(kappa*r)*(1-r)^(n-2)` on `[0,1]`; at `kappa=0` it is `Beta(1,n-1)`. This can supply an independent exact transition sampler with a uniform complex orthogonal direction. Verify the measure and limiting cases rather than porting the real Watson sampler unchanged.

Use a matched stochastic-signature run to assess process inference. Keep deterministic ULA motion as a physical mismatch/approximation test. A local Gaussian or same-parameter Laplace approximation is a useful control if correctly implemented. Do not reuse an EKF for `E[u|P]=0` as though it had the conditional model's nonzero-mean observation Jacobian. Evaluate a passive covariance likelihood when constructing that baseline.

### B. Infer sensor noise, interference power and process concentration

First implement unknown constant parameters; a changing parameter needs a stated change model or an explicitly approximate lagged method.

For passive data use covariance eigenvalues `a=s+q`, `b=s`, with `a>b>0`. Choose proper ordered inverse-gamma priors, document their scales and sensitivity, and implement the hierarchy in the paper's supplement. For proper complex data and a path `P_1,...,P_T`, the scalar conditionals have shape increments `k*T` and `(n-k)*T`, and scale increments

\[
\sum_t u_t^H P_tu_t,\qquad
\sum_t u_t^H(I-P_t)u_t,
\]

respectively, truncated to `a>b` and `0<b<a`. In the real model these counts and energies are halved. These priors are not equivalent to independent priors on `s,q`.

At fixed `a,b`, the complex measurement increment is `(1/b-1/a)*u_t*u_t^H`. An interior moving-state full conditional adds the previous and next process-center terms, with the appropriate supplied transports. For a stationary run use **one common projector** with parameter `A_init + sum_t B_t`, not independent per-observation states.

For a path beginning at state 1 with `T-1` transitions, infer

\[
p(\kappa\mid P_{1:T})\propto p(\kappa)
\exp\left\{\kappa\sum_{t=2}^T\operatorname{tr}(P_tG_tP_{t-1}G_t^H)
-(T-1)\psi_1(\kappa P_0)\right\}.
\]

Use proper priors and a stable scalar update. Alternate state and scalar updates, monitor mixing, and document full-path versus fixed-lag boundary handling. Conditional state distributions are Bingham; integrated state marginals are generally mixtures. Do not claim an exact single-`A_t` filtering recursion after integrating unknown parameters.

For decisions with uncertain powers, average the **joint** covariance:

\[
\overline R_t=\mathbb E[sI+qP_t\mid\text{data}],
\]

not a product of separate posterior means. Compare known-parameter controls, fixed validation choices, recorded-prior EM where applicable, and joint inference. Do not copy the conditional residual inverse-gamma update into a collapsed passive likelihood.

Recorded-prior EM has an ascent statement only for its fixed objective with accurate E-steps. Refitting the entire filtering history is a different objective. Preserve pre-correction priors, Gaussian constants and identity-shift corrections in predictive scores. Compare prospective scores before fitting/assimilation, without using future observations. Learning hyperparameters is distinct from updating `A` at fixed hyperparameters.

Use known `s,q` for the certified tracking comparison. Learned-parameter runs are separate unless their estimation errors are explicitly incorporated into a new bound. In a deterministic ULA run there is no “true kappa” unless that run was generated by the stated transition. Report process fit/predictive adequacy rather than inventing a ground-truth concentration.

## 9. Posterior risk, regions and calibration

Add derived spatial-projection queries `P v`, with independent unit queries shared by all methods. Label them as a posterior functional diagnostic, not an observed known-input training experiment. In the complex setting verify

\[
\mathbb E[\|Pv-b\|^2\mid\text{data}]
=v^H(M-M^2)v+\|Mv-b\|^2.
\]

Check the identity against independent posterior draws or an accurate low-dimensional integration. Plot directional risk and empirical loss binned by predicted risk, preserving independent trajectory units. The full response covariance needs second moments, whereas this total squared risk uses `M` alone.

For scanning power, distinguish pointwise scalar quantiles from extrema over a common state HPD region. Use central 50% and 95% scalar bands and nominal 95% state regions, as in the current notebook, with the probability interpretation labeled. State inclusion implies enclosure of all evaluated angular responses, but the converse fails. Neither object is automatically a frequentist confidence band. At uniform density use a valid tie convention if reporting exact-mass HPD regions. Report empirical coverage together with widths or a proper score, not coverage alone.

The paper's balanced-design coverage result assumes a real known-input experiment. Reuse its separate controlled validation or implement a small exact-reference fixture if absent. Do not label passive array coverage as a direct check of that proposition.

Derive the natural **stationary passive counterpart** as part of the unified analysis: at known fixed `s,q`, with a uniform subspace prior, `beta=1` and a fixed state, the data law is equivariant under `U -> Q U`, `P -> Q P Q^H`. An equivariant state region of exact posterior mass `1-alpha` has constant frequentist coverage over the Grassmannian by transitivity. Averaging that constant under the uniform prior gives `1-alpha`. Verify the argument, tie conventions and numerical mass accuracy independently, then test it across independent data sets at predeclared states. This concerns state inclusion, not arbitrary scalar response intervals. It does not automatically extend to estimated noise/power, forgetting or approximate dynamic prediction. The new passive result should be labeled as an invariance specialization, rather than presented as a new general coverage principle.

Keep any HPD-versus-ball worst-case leakage comparison secondary. Equal posterior mass does not imply one set gives tighter bounds for every direction. Such a comparison uses a fixed decision unless a different optimization is actually implemented. It must not be used as evidence that the posterior-mean beamformer itself requires regional optimization.

## 10. Plotting protocols and required visual story

Use the current notebook's plotting conventions through `lib.plotting` and compatible helpers:

```python
plt.rcParams['text.usetex'] == True
plt.rcParams['figure.dpi'] == 200
plt.rcParams['savefig.dpi'] == 200
```

Export every figure as a vector PDF and a 200-dpi PNG. Keep mathematical text in LaTeX, serif fonts, simple labels, restrained colors, readable legends, clean margins and consistent scales. Retain mean blue, mode orange and loaded covariance green where possible. Give the isotropized control a consistent distinct color and the oracle a neutral style. Install or repair the required TeX runtime if necessary; do not silently disable LaTeX.

Make figures interpretable: each should answer a question and identify observations, truth, posterior quantities and independent repeats. Prefer informative combinations of geometry, spectra, density contours, paired scatter, empirical distributions, annotated heatmaps and validity diagrams. A line with an error bar is acceptable where it is the right visual, but is not a substitute for designing the explanation. Avoid gratuitous 3D surfaces, dense dashboards, unexplained coordinates or rainbow styling. Use code-native vector diagrams for exact geometry.

Suggested figures, adapting layouts after inspecting actual data:

| Figure | Question and useful visual form |
|---|---|
| `01_setup` | ULA geometry, desired/interfering arrivals, observed snapshots, hidden spatial line and downstream weights. Include a compact covariance spectrum to explain rank one. |
| `02_information` | Analytic versus empirical score information; a normalized matrix/residual heatmap and coordinate-information view. Show physical angle dependence separately from isotropic ambient information. |
| `03_tracking` | Moving-interferer response or angular-power map, actual modal error, and a noise/drift budget diagram marking valid/invalid certificates. Include an empirical error-versus-bound panel at fixed reporting times. |
| `04_decisions` | Paired losses and complete SNR/snapshot gain maps, with full mean versus mode, matched isotropization, scalar shrinkage and loaded covariance. Show the one-snapshot equality control and remaining gains. |
| `05_uncertainty` | Same target with pointwise power quantiles and a state-region envelope, plus directional spread/risk and an inclusion-versus-width/score diagnostic. State precisely what is random in each panel. |
| `06_prediction_and_risk` | Posterior spatial-projection draws or density contours, mean/mode/truth and risk calibration. Label any dimensional projection; it is not a full-state credible region. |
| `07_parameter_inference` | Joint posterior uncertainty for sensor/process or parallel/perpendicular variances, learning across data amounts, prospective predictive scores and decision performance. Separate matched inference and physical mismatch. |
| `08_process_and_cost` | Exact first-moment prediction versus empirical/reference moments, density-projection discrepancy where measurable, and measured estimation/report cost. Keep approximation and sampling failures visible. |

Use a small main narrative, with detailed diagnostics in a notebook appendix. Do not fill every main figure with all methods. Preserve all regimes in the underlying tables and complete comparison panels. Select displayed trajectories before examining test outcomes and retain misses. Do not count steering-grid points or posterior draws as independent experimental repeats.

The first figure's setup should make clear what is measured and what is inferred. Distinguish a covariance scanning response from the actual weight beampattern `|w^H a(theta)|^2`. If plotting both, label them separately.

Use bounded, predeclared budgets for expensive tail/reference computations and show numerical status. Mean accuracy is not tail accuracy. If a high-dimensional uncertainty report remains unresolved, report that limitation and supply a correctly labeled low-dimensional reference; do not silently present a toy reference as the native array result.

## 11. Execution, validation and deliverables

Implement reusable numerical code, deterministic seeding, configuration profiles, checkpoint/resume and verified caches. Separate simulation from plotting, so a layout edit does not rerun the science. Avoid unbounded scans or arbitrary extensions after seeing test results. Record the generation law, priors, tuning, data count, code/configuration hashes, numerical budgets and machine/runtime for each output.

Use independent validation and test data. Form uncertainty intervals from paired independent trajectories, averaging within-trajectory probes first. Make numerical and Monte Carlo error distinguishable. Validate substantive new mathematics independently: normalized likelihoods, complex factors, Fisher covariance/derivatives, covariance-bound constants, eigenspace conditions, scalar posterior conditionals, predictive moment identities and expected-power optimality. Run the relevant existing test suite. Do not write tests that demand superiority over a comparator.

Run the quick profile through every implemented path, inspect all rendered figures, then execute the declared paper profile. Fix clipped labels, crowded legends, unreadable panels and misleading axes. Ensure notebook outputs correspond to the committed scripts/configurations. Keep an execution-status table: implemented, verified, approximate or blocked, with the precise reason for any incomplete numerical claim. Do not pass off cached old figures as output of a new protocol.

Deliver on branch `test`:

- The executed, revised array notebook and reusable modules/scripts.
- Separate reproducible results under a new directory such as `results/beamforming_unified/`, including PDFs, 200-dpi PNGs, paired metrics, parameters, numerical flags and configurations. Commit compact scientific summaries and figures; follow repository policy for large raw artifacts and provide a manifest and regeneration command for any excluded data.
- `docs/beamforming_theory.md`: passive Fisher derivation, tracking proof/assumptions, existing-result versus new-specialization distinctions, inference details and limitations. Cite established Gaussian information, covariance concentration and eigenspace perturbation results appropriately rather than implying they are automatically new contributions.
- `docs/beamforming_claims.md`: each paper claim mapped to code, plot, experiment assumptions and evidence status. Identify conditional claims still validated separately.
- A short results summary describing observed gains, ties, failures, computation and which figures belong in the main paper versus the supplement. Do not rewrite the manuscript or fabricate completed results.
- Relevant tests and exact commands to reproduce the quick and paper profiles.

Then commit and push only to `test` as instructed in Section 1. In the final response give the branch URL, commit SHA, checks executed, key result limitations and paths to the executed notebook and figures. Explicitly state whether `main` and `dev` were left untouched by your operations.

Useful primary references for the analysis:

- Complex Gaussian covariance Fisher information: Besson and Abramovich, https://arxiv.org/abs/1306.6415 .
- Gaussian sample covariance concentration: Koltchinskii and Lounici, https://arxiv.org/abs/1405.2468 . Its standard stationary result is not a substitute for deriving the weighted/time-varying assumptions above.
- Spectral-projector error analysis: Koltchinskii and Lounici, https://arxiv.org/abs/1504.07333 .

Prioritize the end-to-end connection: what the observations identify, how motion affects the subspace estimate, what the posterior adds beyond its mode, and whether that additional information improves the actual beamforming decision.
