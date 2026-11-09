# Draft 5 notation and simulation scope

This map uses the theory and appendices of the supplied **Bayesian Subspace Tracking with Grassmann Bingham Distributions**, draft 5. It does not adopt an experiment protocol from the draft's application/simulation discussion. The simulations should test named mathematical claims rather than imply that every additional computational procedure is part of the paper.

## Quantities to use in figures

| Paper quantity | Meaning and figure label |
| --- | --- |
| \(N,k\), \(\mathcal X_t\in\mathrm{Gr}(k,N)\) | Ambient dimension, subspace rank, and state. Use \(N\) for dimension and \(L_t\) for a block's observation count. |
| \(P_{\mathcal X_t}\) | Orthogonal projector. It is the hidden state in simulations, never an estimator input. |
| \(q_A\), \(\psi_k(A)\) | Bingham density relative to normalized invariant measure and its log normalizer. |
| \(A_t^-\), \(A_t^+\) | Parameters before and after the current correction, stored in the trace-zero gauge. |
| \(\widehat{\mathcal X}_t\), \(P_{\widehat{\mathcal X}_t}\) | Modal subspace and its projector. For a single Bingham belief this is also the minimum mean squared chordal-distance estimate. |
| \(\bar P_t^+=\nabla\psi_k(A_t^+)\) | Mean projector matrix. It normally has full rank and is not a feasible rank-\(k\) projector. |
| \(\beta\) | Power-discount factor. It changes the retained evidence, not the sensor variance. |
| \(G_t\) | A supplied orthogonal coordinate/state transport. A learned or truth-derived rotation has additional assumptions and must not inherit the deterministic known-transport claim. |
| \(v_t,u_t,s\) | Known requested vector, measured response, and sensor-noise **variance** in the conditional likelihood. |
| \(\tau^2\) | Latent power in the marginal likelihood. It is not a standard deviation. |
| \(\bar P_t^+v_t^\star\), \(\mathcal R_t(v_t^\star)\) | Posterior mean clean response and total squared-response risk, with \(\mathcal R_t(v)=v^\top(\bar P_t^+-(\bar P_t^+)^2)v\). |
| \(v_t^B=[I+\lambda(I-\bar P_t^+)]^{-1}v_t^{\rm des}\) | The paper's soft expected-penalty decision. It is not a hard contact controller or a guaranteed feasible command. |
| \(r_{t,\rm rms}\), \(\rho_{t,\alpha}\), \(h_{t,\alpha}\) | Posterior RMS chordal distance, a centered credible-ball radius, and an HPD density-loss threshold. These are different summaries. |
| \(C_t,F_t,H_t\) | Design Gram matrix, measurement tangent Fisher information, and realized modal posterior curvature. Their meanings must remain distinct. |
| \(\gamma_t,\varepsilon_t(\delta),d_t^{\rm drift}\) | Current excitation separation, perturbation budget, and historical drift contribution in the fixed-time conditional modal bound. |

Use these quantities directly for variants: for example \(q_{\beta A_{t-1}^++B_t}\), \(q_{\beta G_tA_{t-1}^+G_t^\top+B_t}\), \(P_{\widehat{\mathcal X}_t}v_t^\star\), and \(\bar P_t^+v_t^\star\). A standard external algorithm can retain its established name, such as GROUSE or EKF. Avoid unnamed variants such as “tuned mode,” “calibrated mode,” or “weak probes.”

An axial Bingham control can be written \(q_{\kappa P_{\widehat{\mathcal X}_t}}\), using the paper's prior-concentration notation. Define whether \(\kappa\) matches RMS distance or one credible quantile. One scalar cannot in general match both. Concentration \(\kappa\), sensor variance \(s\), latent power \(\tau^2\), discount \(\beta\), and posterior mass \(1-\alpha\) are not interchangeable calibration parameters.

## What a sphere figure represents

For contact in \(\mathrm{Gr}(2,3)\), write the normal as a line \([n]\) and the admissible projector as \(I-nn^\top\). Complement duality gives the **normal** Bingham parameter \(-A_t^+\), and

\[
q_{A_t^+}(\mathcal X)\longleftrightarrow q_{-A_t^+}([n]),\qquad
\bar P_t^+=I-\mathbb E[nn^\top].
\]

Consequently a colored sphere is an exact antipodal representation of the full contact-normal state. Show both antipodal modes, use normalized surface/invariant measure consistently, and distinguish a density contour from a credible set of specified posterior mass. A local tangent Gaussian should be compared on the same state and loss; its coordinate ellipse is not automatically an invariant-density HPD region.

For native complex beamforming, \(\mathrm{Gr}_{\mathbb C}(1,2)=\mathbb{CP}^1\) has a Bloch-sphere representation, but a complex Bingham law there becomes a von Mises–Fisher law: its density is rotationally symmetric about its mode. It cannot demonstrate general directional anisotropy. For \(N>2\), a three-dimensional colored sphere can display a **section** \(h=U_3x\), \(x\in\mathbb S^2\), through three complex-orthonormal directions. Color \(q_A([h])/q_A([\widehat h])\), label it as a section, and show the complementary eigenvalue gaps separately. The section is not a normalized marginal density or the entire higher-dimensional state. A real \(\mathrm{Gr}(1,3)\) illustration is also legitimate if explicitly separated from the proper-complex array experiment.

## Directional uncertainty without changing the modal estimate

A coordinate permutation in the complement of a fixed mode preserves its entire centered chordal-distance distribution. For example, two finite parameters with the same modal eigenvector and the same eigenvalues, but swapped complementary directions, have identical centered ball radii at every mass. Their mean responses, total response risks and soft decisions can nevertheless differ along named vectors. This is a deterministic geometry comparison, not a tracking-performance win.

A matched-mass HPD region can be tighter in one direction and wider in another than a distance ball. Do not imply uniform HPD superiority. Likewise, an observed improvement over a point projector does not establish that anisotropy caused it: data-dependent scalar spread can explain regularization gains. Retain a same-mode/spread control and direct directional functional measurements.

For an empirical contact anisotropy study, use known requests that excite two tangent directions unequally while keeping total request energy fixed. Around the true plane \(\mathrm{span}(e_2,e_3)\), with normal \(e_1\), the two normal tangent-information eigenvalues are

\[
\frac{E_1+E_2}{s},\qquad \frac{E_1+E_3}{s},
\]

where \(E_i\) are the coordinate request energies. A small input-Gram eigenvalue alone is insufficient: request energy along the normal contributes to both local directions. A normal-directed request with a nearly zero measured motion is informative because the negative conditional increment retains the known request. A method receiving only measured motions has different information; any advantage over it is an observation-model advantage, not automatically an uncertainty-distribution advantage.

## Fair application and benchmark claims

The contact likelihood requires the physical response to be the closest admissible motion to a known request, or an equivalent isotropic-resistance/whitened model. Friction, anisotropic mobility and unilateral separation can violate that law. Simulate and score their actual mechanics if used as stress cases; do not silently replace them by projector truth. The posterior mean response is an expected velocity rather than a rank-constrained admissible motion.

A complete-vector marginal mode is weighted covariance PCA under the same weights, fixed parameters and initialization. A conditional-mode comparator can use the same spectral projector-regression objective and then coincide with the Bingham mode. These are important controls. GROUSE's missing-data capability is outside this paper's likelihood; comparisons here must use the complete vectors available to every method. GREAT-like projection-loss optimization, conditional projector regression, robust Grassmann-ball methods and posterior integration have different objectives. Objective equality does not establish recursion equality. Do not rename a custom optimizer as GREAT or GeRoST unless its published procedure is implemented.

For proper-complex passive arrays, explicitly state the extension from the real draft: use Hermitian projectors and the proper-complex Gaussian coefficient \(\tau^2/[s(s+\tau^2)]\), without the real factor \(1/2\). Conditional known-input Fisher information and the conditional tracking theorem do not transfer to passive snapshots by treating an unobserved coefficient as a known input. Preserve the correct passive counterparts separately.

The draft's main algorithm is additive correction with power discounting and optional supplied transport. The invariant stochastic kernels and full-path inverse-gamma/concentration Gibbs experiments in the older notebooks are additional models, not mechanisms analyzed by draft 5. They should be separate optional comparisons if retained. Draft 5 noise learning uses a fixed common-state block and one fixed boundary prior; every EM E-step reconstructs the block posterior from that same boundary. Frozen per-observation priors, repeated full-history refits, and a blockwise evidence objective are distinct procedures.

## Higher-dimensional and cost evidence

A codimension-one extension \(\mathrm{Gr}(N-1,N)\) can use the same rank-one complement machinery in \(N=8,16\). It is an ambient generalized-velocity constraint unless an actual robot Jacobian and resistance metric are supplied; avoid presenting arbitrary \(\mathbb R^8\) coordinates as a physical seven-joint robot. If known kinematics restrict the unknown covector to a smaller subspace, give that information to a structured point comparator as well.

Separate higher-dimensional evidence into full spectrum/risk summaries, explicitly labeled low-dimensional sections, and ambient-dimension benchmarks. Do not label a visualization of three selected coordinates as the complete posterior. General rank is more demanding than rank/codimension one and may need a separate bounded sampling study with retained diagnostic failures.

Report correction, eigenspace extraction, mean/normalizer evaluation, scalar uncertainty reporting and end-to-end requested outputs separately. The draft asserts \(O(N^2)\) storage and single-vector correction work; it does not assert that eigendecomposition, moments or credible sets cost \(O(N^2)\). Measure serial dedicated timings with fixed threads and report memory with the actual dtype. Fit slopes only over a range where the operation dominates overhead, and keep failed numerical targets instead of replacing them by unchecked approximations. Scheduling moments at selected reporting times is part of the stated algorithm and should be reflected fairly in comparator cost.

Independent trajectories are statistical replicates. Grid points, posterior samples, query directions and repeated times within one trajectory are not extra experimental units. Pair methods on each trajectory before forming uncertainty intervals. Keep broad-uncertainty cases where a global distribution can matter, concentrated cases where a Gaussian is competitive, and mismatch cases where credible mass fails to imply frequentist coverage. The tracking theorem is pointwise, truth-dependent and fixed-parameter; it is not an online certificate or a credible-coverage theorem.
