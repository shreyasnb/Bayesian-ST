# Test whether process uncertainty and the full posterior add useful information

Implement and run a separate comparison in `shreyasnb/Bayesian-ST`, starting from the current `dev`. The reviewed head is `f09d3fecd3af11fff5af708c8d1d31a35db75ca4` (contact pilot). Check the head before working, record any later changes, and read repository instructions. Preserve existing experiments, saved results and manuscript files. Work in an isolated branch/worktree and put new results under `results/contact_uncertainty/`.

The existing contact pilot validates the mechanics and posterior integration, but its mean barely improves on its mode and tight-guide intervals under-cover. Determine whether a coherent process predictor improves useful uncertainty, and whether retaining a full Bingham distribution adds anything beyond a cheap local Gaussian method. A negative result is useful. Implement and run the comparison; do not stop at a plan or search for a more favourable application.

Read these sources first:

- `docs/contact_pilot_spec.md`, `results/contact_pilot/{findings.md,config.json,selected.json,summary.json}`;
- `lib/{contact_mechanics,contact_study,contact_posterior,process_prediction,prediction,filter,theory}.py`;
- `results/dev_prediction/process/{findings.md,reference_checks.json,coverage_summary.csv}`;
- `docs/dev_manuscript_notes.md` and the relevant numerical validation scripts.

## 1. Separate the questions

Test three claims separately:

1. Explicit process uncertainty improves calibration compared with validation-tuned discounting.
2. The global Bingham shape improves predictions or interval quality compared with a local Gaussian approximation using the same information.
3. Any improvement is large enough to justify its computation at the stated reporting frequency.

Do not attribute a gain over velocity-only PCA to Bayesian uncertainty, a gain over an unconstrained linear map entirely to posterior averaging, or a gain from a better process model entirely to the Bingham family. Keep known noise fixed in the primary comparisons. Do not add gain learning, rank selection or new sensing channels during this comparison.

## 2. Validate the process calculation on real Gr(2,3)

Extend the existing rank/codimension-one process calculation, rather than introducing an unrelated filter. For an orthogonal nominal motion G, use

\[
K_\kappa(\mathcal X\mid\mathcal Y)
\propto\exp\{\kappa\operatorname{tr}(P_{\mathcal X}G P_{\mathcal Y}G^\top)\}.
\]

Here a plane is represented by an unoriented unit normal n, with P=I-nnᵀ. For G=I, the transition is a normal-line Bingham law proportional to exp{κ(nᵀm)²}; the extra constant in the plane trace cancels. Respect the antipodal identification in sampling and integration.

Reuse the exact predictive first moment

\[
M^-=\rho_\kappa G M^+G^\top+(1-\rho_\kappa)\tfrac23 I_3,
\]

and fit a trace-zero Bingham parameter with this mean. The density projection is approximate even when the mean is accurate. Use the concentration-adapted contact mean/normalizer where the existing series cannot resolve a target; do not silently raise the series cap to handle spans near 80000. Record inverse residuals, backend errors and density/quantile errors separately. Treat κ=∞ as exact deterministic transport, avoiding unnecessary inversion.

First use a dimensionless matched-process control: n=3, k=2, uniform initial subspace, G=I, κ in {4,20}, T=24, s=0.12, and known coordinate inputs cycling e1,e2,e3. Use 200 independent trajectories per κ. This is a controlled process-validation experiment, not a mechanical contact simulation or a fixed-state coverage theorem.

Implement an independent full-density normal-line reference, with refinement checks on means, region mass and scalar CDFs. Choose the resolution on separate development seeds before evaluation. Avoid an unbudgeted dense all-pairs spherical grid: exploit the zonal kernel, adaptive integration or a suitable harmonic representation. If a full recursive reference is too costly for all trajectories, predeclare a fixed subset before testing and report its size and numerical limits. Never select reference targets by observed success or failure.

Check normalization, complement/rotation identities, the uniform and deterministic limits, and the measurement Jacobian. Small inverse-moment residuals alone do not certify tails. Retain unresolved targets. Do not proceed to a numerical superiority claim if the reference or the selected policy is unresolved.

## 3. Compare against inexpensive uncertainty methods

Include the following controls, using identical observations and independent validation:

- The existing weighted-projector-LS mode and its posterior mean, with independently tuned point-prediction memory.
- A discounted Bingham law with separate memory β and likelihood scale η: A_t=βA_(t-1)+ηB_t. This separates response lag from distribution spread. It is a generalized posterior control, not an exact stochastic prediction.
- The explicit-transition moment-matched Bingham filter.
- A local Laplace approximation to the **same Bingham parameter** at each requested report. In modal tangent coordinates its precision is 2(A11⊗I-I⊗A22). For a normal line with parameter C=-A_plane, this gives variances 1/[2(λ1(C)-λj(C))]. Compare its response moments/intervals with the resolved distribution, and count its computation separately. A singular cutoff is a failed local approximation or requires an explicitly reported broad fallback; do not insert a tiny variance floor that creates false confidence.
- A genuinely cheaper recursive tangent-Gaussian filter with the same assumed process and measurement model. This checks whether a useful process model and a 2×2 covariance already account for the benefit.

For the last control, at unit normal n and orthonormal tangent basis E, h(n,v)=v-n(nᵀv) has Jacobian

\[
J=-[(n^\top v)E+n(v^\top E)].
\]

Use a documented exponential-map/retraction update, covariance transport and stable covariance correction. Match the isotropic tangent process covariance to the kernel, rather than assigning a more favourable process variance to one method. In Gr(2,3), one option is Qκ=(Eκ[θ²]/2)I2, with θ=acos|nᵀm| and density proportional to sinθ exp{κcos²θ} on [0,π/2]. Label this as a local Gaussian process approximation. Explain initialization from observations: no true normals or geometry. A common short data-only warm-up is acceptable if its information and cost are counted and startup differences remain visible.

The same-parameter Laplace comparison isolates posterior shape/reporting. The recursive Gaussian control also assesses whether full moment propagation is worth its cost. Neither comparison should be replaced by a point estimator with no uncertainty.

## 4. Evaluate on the physical contact plant

Preserve the existing mechanics, force programs, d=10, v0=0.02, σ=0.001, dt=0.05, T=600 and reports at 60/180/600. Use plane, large sphere, small sphere and weak-input conditions, with anisotropic resistance as a separately labelled stress. Positions remain scoring records in this experiment. Do not claim superiority over position-based estimators.

The already inspected pilot test trajectories may be used for regression checks and exploratory reproduction. For the new comparison, freeze fresh validation/test roots **6245201/6245202**, with 16 validation and 64 test trajectories per condition. Record the complete configuration and selected policies before opening the new test results. One curved policy per method is shared across the three curved conditions; keep the stationary control separate.

Use G=I in the primary comparison to isolate uncertainty propagation. A causal mean-motion forecast, if subsequently tested, must be shared across matched controls and selected on validation only. Do not apply the deterministic-transformation tracking theorem to a noise-dependent forecast.

Suggested bounded grids:

- β: {1,.99,.98,.95,.9,.8,.6,.4,.2}. If .2 is selected at an open search boundary, expand once to {.1,.05} on validation, before freezing tests.
- η: {.125,.25,.5,1,2}, alongside β for the generalized posterior.
- Process dispersion: expected one-step chordal RMS {0,.0025,.005,.01,.02,.04,.08}; convert these to κ using the exact transition moment, rather than a small-angle formula. Zero is the deterministic limit. Apply the same validation boundary rule to every process method.

Select uncertainty policies on validation pre-correction predictive log score; select point-only policies on validation clean-response MSE. Also retain a validation interval-score comparison to reveal disagreements between observation prediction and clean-state uncertainty. Select one policy per declared scope, not one policy per hidden test condition. Count validation costs and candidate evaluations.

The physical sphere law is not the isotropic Kκ process. The physical test therefore assesses empirical robustness of that predictor; it cannot be described as a matched-process calibration guarantee. The Grassmann state describes instantaneous mobility and does not by itself encode all position, curvature or contact-mode dynamics.

## 5. Endpoints, diagnostics and one optional decision test

Primary condition/time: small sphere, t=600. Primary endpoint: nominal 95% **clean-response interval score**, averaged over the original nine queries and three components within each trajectory. For bounds l,u, target y and α=.05, use

\[
(u-l)+\frac{2}{\alpha}(l-y)1_{y<l}
       +\frac{2}{\alpha}(y-u)1_{y>u}.
\]

Predeclare primary paired comparisons of the process Bingham method against the scale-tuned discounted law and recursive Gaussian control. Use trajectory-level paired confidence intervals. Treat other comparisons as exploratory; queries, coordinates, times and posterior draws are not independent replications.

Also report:

- state-ball and HPD inclusion, clean component coverage and width, and clean vector MSE;
- pre-correction observation log score, including Gaussian constants and trace-shift restoration;
- posterior full-vector risk against empirical full-vector error;
- directional tangent errors versus posterior/Laplace widths, showing bias and dispersion separately;
- eigenvalue gaps and the difference between mean and modal response maps;
- update, prediction/inversion, mean, region, interval and decision costs at their actual frequencies. Nested timings must not be added twice.

Keep clean and noisy response intervals separate. If noisy intervals are evaluated, use an appropriate mixture CDF or independently checked approximation. Do not replace a non-Gaussian mixture by a Gaussian without recording that approximation. Report unresolved mean, region and interval calculations separately. Do not quietly clip local-Gaussian intervals to physical ranges.

The nine original query directions average to an isotropic response-map loss; they do not themselves demonstrate an uncertainty-dependent decision. If the primary comparison shows informative uncertainty beyond the cheap control, add just one fixed-time prediction/abstention test: with a predeclared velocity tolerance, predict only query components whose uncertainty permits that tolerance, otherwise request more data. Compare accepted predictions, tolerance violations and cost on fresh trajectories. Freeze the tolerance and decision rule on validation. This is a prediction decision, not a feasible contact controller or safety guarantee. Optional stopping/adaptive probing would require separate calibration and must not inherit the fixed-time theorem.

## 6. Return a concrete scope decision

Return reproducible source, frozen configurations and source hashes, numerical checks, trajectory-level tables, selected policies and one executed notebook. Use a small set of figures: coverage versus width/interval score, paired effects versus cost, and directional calibration. Preserve misses and numerical failures.

Write `results/contact_uncertainty/findings.md` answering:

1. Does explicit propagation improve useful uncertainty after tuning memory and spread fairly?
2. Does the full Bingham distribution improve anything material over the same-parameter Laplace and recursive Gaussian controls?
3. Are any gains confined to broad/ambiguous posteriors or also present in the physical tracking regimes?
4. What can be reported at 20 Hz, and what requires occasional reports?
5. Is an uncertainty-led paper supported, or is the defensible result a narrower identification/theory paper?

Distinguish a beneficial process model from a beneficial distribution family. Do not rewrite the manuscript, add untested claims or expand to other applications after an inconclusive result. If a component is numerically unresolved, state exactly which conclusions remain unsupported and return the completed controls and evidence.
