# Pilot: replacing the field example with constrained robot motion

Implement and run a small, separate experiment in [Bayesian-ST](https://github.com/shreyasnb/Bayesian-ST), starting from the latest `dev`. Reviewed commit: `bef07fe301d03ecf4bd714e0e15999514c24a2a2`. Read the repository instructions first. Preserve existing experiments, results and manuscript files. This pilot will inform a later decision about the paper's scope.

The question is whether physically generated contact data support useful response predictions or uncertainty beyond a tuned point estimator. A positive result is not required.

## 1. Generate motion from mechanics

Use a tool moving on a fixed surface, with a maintained bilateral contact/guide constraint, negligible inertia and known isotropic viscous resistance `d`. This is an idealized constrained mechanical simulation; it is not a general model of unilateral, frictional polishing.

At physical time `tau_t`, apply a known Cartesian force `F_t`. Solve

\[
\begin{bmatrix}dI_3&-\boldsymbol n(p_t)\\
\boldsymbol n(p_t)^\top&0\end{bmatrix}
\begin{bmatrix}\dot p_t\\\lambda_t\end{bmatrix}
=\begin{bmatrix}F_t\\0\end{bmatrix},
\qquad p_t\in\mathcal S,
\]

where `p_t` is the tool position, `S` is the surface and `n(p_t)` its unit normal. Integrate the constrained motion, controlling numerical surface drift. Start with a plane and spheres of different radii. Derive normals from geometry; do not prescribe rotations of a hidden subspace.

The filter receives

\[
v_t=F_t/d,\qquad u_t=\dot p_t+e_t,\qquad
e_t\sim\mathcal N(0,sI_3),\qquad
\mathcal X_t=T_{p_t}\mathcal S\in\mathrm{Gr}(2,3).
\]

The mechanics imply `dot p_t = (I - n_t n_t^T) v_t`. Check that identity independently against the constraint solver. Use instantaneous solver velocities for the matched observation model, rather than finite differences that silently change the noise law. Geometry and true normals are available only to the plant and scoring code.

Suggested scales: `d=10 N s/m`, free-velocity scale `v0=0.02 m/s`, `dt=0.05 s`, `T=600`. Check integration accuracy by halving `dt` over the same physical interval and observation schedule.

## 2. Keep the experiment bounded

Use 16 independent validation trajectories and 64 independent test trajectories per main condition. Pair observations across methods. Freeze seeds, geometry distributions, force programs, grids and endpoints before testing.

| Condition | Purpose |
|---|---|
| Plane, balanced probes, `beta=1` | Exact-model inference and coverage check |
| Curved guide, radii `0.20 m` and `0.05 m` | Geometry-driven changes of the tangent plane |
| Smaller-radius guide, weak excitation | Direction-dependent uncertainty |

For plane coverage, repeat a fixed non-axis-aligned normal over independent noise trajectories. Randomize curved-guide initial orientations independently of the force program. On the plane, cycle `v0*e1, v0*e2, v0*e3` and evaluate at complete cycles. For curved guides use a bounded, preassigned world-frame force program, for example

\[
v(\tau)=v_0[\cos(0.3\tau),\ \sin(0.3\tau),\ 0.5\sin(0.21\tau+0.4)]^\top.
\]

For weak excitation reduce the second and third components by a factor of five. Report actual surface travel, normal rotation and retained input excitation. Radius alone does not establish a useful tracking regime. If the generator needs adjustment, use validation only and record the change before freezing tests.

Use velocity-noise standard deviation `sigma=0.001 m/s`, with `s=sigma^2`, known to all methods. After the main pilot, add one exploratory stress with true resistance `D=d*diag(1,1.5,2)` while the estimator retains the scalar-resistance model. Solve the corresponding mechanical equations; label this as observation-model mismatch.

## 3. Reuse inference and fair controls

- Use `BinghamFilter(n=3, k=2, s=s, tau2=None)` and `update(u, v=v)`. Start with `A0=0`, identity prediction and no noise fitting.
- Tune one discount per method on the validation mixture, using `{1,.98,.95,.9,.8}`. Freeze it across curved test conditions. Keep the plane calibration control separate, at `beta=1`.
- Verify that the mode equals an independently calculated weighted projector least-squares solution under identical settings. This equality is a correctness check, not an accuracy comparison.
- Compare posterior-mean response prediction with the mode, independently tuned scalar shrinkage `D_rho = rho*P_hat + (1-rho)*(2/3)*I`, and an exponentially weighted ridge linear map from `v` to `u`. Tune shrinkage and ridge regularization on validation task loss, alongside memory.
- Include tuned EW PCA of measured velocities as a cheap control for the value of observing the applied forces.
- Record tool positions. If encoder positions are included as available measurements, add a causal local-plane estimate using the same noisy position stream. Do not claim practical superiority by withholding an available sensor channel from competing methods.

Reuse `lib/filter.py`, `lib/moments.py` and `lib/theory.py`. Avoid modifying the numerical core unless a documented defect requires it. A causal motion forecast is optional only after this first comparison; keep it a separate ablation with matched point and posterior controls.

## 4. Use deterministic posterior calculations

Here `k=n-1`. If `A` describes the tangent plane, its normal line has Bingham parameter `-A`:

\[
\bar P(A)=I_3-\mathbb E_{q_{-A},\,k=1}[nn^\top].
\]

Use the existing rank/codimension-one mean backend. Compute state regions and scalar response quantiles by deterministic sphere quadrature, exploiting this complement relation. Compare two integration resolutions and flag unresolved thresholds or moments. Do not introduce an MCMC budget issue into this pilot.

## 5. Evaluate responses, uncertainty and cost

At observations `60,180,600`, evaluate nine fixed queries of norm `v0`: coordinate axes and normalized pairwise sums/differences. These are hypothetical forces applied at the current contact position; they do not advance the trajectory. Truth is `y_t(w)=P_Xt*w`; the posterior prediction is `barP_t*w`.

Primary endpoint: terminal clean-response squared error, averaged over queries within each trajectory. Report paired gains against tuned shrinkage and ridge with 95% intervals across independent trajectories. Queries and time samples are not additional independent trials.

Also report:

- Tangent-subspace error; realized drift and excitation.
- Nominal 95% state-region coverage and componentwise clean-response interval coverage, together with widths. Separate any noisy-observation intervals.
- Whether `w^T*(barP-barP^2)*w` reflects empirical clean-response errors.
- Mean absolute scalar-response interval width, normalized by `v0`, to expose nearly uninformative intervals.
- Correction, posterior integration and decision costs at their actual reporting frequencies.

Balanced static coverage is an exact-model check. Curved motion with discounting is an approximate prediction experiment; assess its coverage empirically. Apply the existing modal tracking bound only where its assumptions and sufficient condition hold. It does not certify credible coverage or learned-motion forecasts.

## 6. Return evidence for the scope decision

Save new code and a reproducible runner separately, with results under `results/contact_pilot/`. Return configuration, trial-level tables, numerical checks, and a short findings file. Include three figures: physical setup/path with tangent planes; paired task-loss comparisons; coverage versus interval width. Retain unsuccessful conditions.

Conclude which outcome the pilot supports:

1. **Promising replacement:** the physical model checks pass and either a meaningful task-loss gain over tuned controls or informative, empirically calibrated uncertainty justifies the added computation.
2. **Physical example only:** geometry-based tracking works, but posterior predictions add little or drift uncertainty is unreliable. State exactly what remains supported.
3. **Unresolved or unsuitable:** the response law, inference numerics, sensing assumptions or runtime prevent a useful comparison.

Do not redefine the paper, replace its field results, or add process-noise machinery during this pilot. Recommend any next experiment only after reporting this bounded test.

Reference for the robotics setting: Lin, Howard and Vijayakumar, [Learning Null Space Projections, ICRA 2015](https://doi.org/10.1109/ICRA.2015.7139551). The force-balance realization and experiment above are the proposed test, not a reproduction of that paper.
