# Mathematical checks for the simulation redesign

The redesigned studies use the theory and appendices of draft 5. Their mathematical checks are separate from empirical performance, coverage and timing measurements. The general validation runner executes the existing theory notebook and the current regression suite before formal paper-profile simulation. Passing that software gate does not assert that a method wins or that a maintained belief is calibrated on every changing physical path.

## Independent deterministic geometry

`lib/anisotropy_study.py` and `tests/test_anisotropy_study.py` implement and independently check four real line distributions in three dimensions. The principal pair has parameters `diag(0,-8,-80)` and `diag(0,-80,-8)`. The other two are a broad anisotropic distribution and an axial distribution. All cases, queries, posterior mass and the soft-decision penalty `lambda=20` are fixed before evaluation.

The primary pair has the same mode, eigenvalues, and **entire centered chordal-distance distribution**, by a complementary-coordinate permutation. This is stronger than fitting one distance quantile. Direct continuous integrations in the two original angular coordinate orders check their CDFs. The native positive spectral series is checked against a separate nested continuous sphere integral for both the mean and log normalizer. A second credible-region reference uses the existing analytical radial/Dawson formula; direct continuous integration independently solves the ball and HPD mass equations. Existing concentration-adapted contact quadrature provides another mean comparison.

The quick profile resolves all four cases. Its maximum native mean discrepancy is below `1e-12`, the log-normalizer discrepancy below `3e-13`, and posterior-mass error below `5e-12`. The primary independent CDF difference is `2.3e-16`; both 95% radii are about `0.51934425`. The HPD density-loss threshold is about `3.15797356`. These are deterministic integration checks, not Monte Carlo tracking repetitions or frequentist confidence statements. Reported integration and series errors have their stated numerical scopes and are not rigorous floating-point enclosures.

The geometry artifacts store **normal-line** parameters and their `psi_1(A)`. For contact, the tangent-plane parameter is `-A` and its mean is `I-E[n n^T]`. The complement normalizer identity is `psi_2(-A)=-tr(A)+psi_1(A)`. A normal/line projection bound controls `||n n^T v||`, the blocked component; it must not be relabeled as a bound on admissible-plane speed. The complete response covariance is computed from projector functionals, and its trace agrees with `v^T(M-M^2)v`; the full covariance is not the matrix `M-M^2`.

The axial comparison matches mean modal overlap and RMS chordal distance, not all distance quantiles. The coordinate-swapped primary pair supplies the exact all-quantile control. These different meanings remain explicit in saved configuration and notebook notation.

## Soft decisions: expectation before optimization

For a maintained mean admissible projector `M`, the expected penalty is

\[
\|v-v^{\rm des}\|^2+\lambda v^T(I-M)v,
\qquad
v^B=[I+\lambda(I-M)]^{-1}v^{\rm des}.
\]

It is **linear** in `I-M`. Squaring `(I-M)` would omit projector uncertainty because `M` is generally not idempotent. An independent audit found that error in the development contact cost calculation; it was corrected before formal evaluation. `tests/test_contact_geometry_audit.py` now compares the expected cost with an explicit mixture of rank-two projectors, rather than merely comparing two implementations of the same mean formula. The test also checks the quadratic excess identity and the first-order condition for a nonprojector mean.

Soft decisions require a positive semidefinite contraction. The unconstrained ridge response map may have eigenvalues outside `[0,1]`, so it is evaluated for response prediction and excluded from the soft-admissibility objective. Its undefined decision quantities are retained as undefined rather than silently clipped into a projector or counted as numerical failures. A posterior mean response and a soft decision are not guaranteed hard-contact motions or controllers.

## Conditional normal duality, causality and supplied frames

For the contact tangent-plane model, `u=P_X v+e`. Its observed residual satisfies `v-u=P_(X-perp) v-e`. The normal-line likelihood therefore has the negative of the plane natural parameter, up to an identity shift, with the same sensor variance. The independent audit verifies this directly from residual likelihood matrices.

The audit also verifies that:

- changing future observations and requests cannot change earlier natural parameters;
- changing hidden normals, positions and scoring projectors leaves the inferred parameter path unchanged while changing the scored loss;
- applying noncommuting measured orthogonal frame changes gives exactly the transported version of a world-frame recursion;
- uniform and nonuniform codimension-one means in dimensions 3, 8 and 16 remain positive semidefinite contractions of trace `N-1`, with the correct normalizer gauge shift.

Only the current known request, noisy response and supplied frame transformation enter the estimators. Surface geometry is used by the mechanical plant and scoring. The simulated force/normal paths do not depend on sensor-noise realizations, so conditioning on the prescribed physical path is compatible with the deterministic-design tracking theorem. A fitted or true-normal-derived transport would require a different justification and is not used here.

## Printed conditional tracking bound

The existing shared `lib.theory.tracking_bound` helper computes a valid sharper variant: it retains the norm of the aggregate deterministic drift operator and an `epsilon/(gamma-epsilon)` eigenspace expression. These are not the exact displayed formulas in draft 5.

The new contact wrapper `paper_certificate` explicitly evaluates the draft's sum-of-norms `d_t^drift`, `epsilon_t(delta)` and printed `2 sqrt(k) epsilon_t/gamma_t` bound. The older aggregate quantities remain separately named diagnostics. Invalid sufficient conditions stay invalid with a missing bound; they are not clipped into successful certificates. The main normal certificate uses the residual's rank-one model, whose chordal error equals the complementary plane's chordal error. All guarantees are at fixed declared reporting times and use known fixed variance and evaluation truth.

## Passive complex array checks

The complex array extension uses Hermitian projectors and proper complex noise. Its marginal correction omits the real Gaussian factor `1/2`. No unobserved latent coefficient becomes a known conditional input. The modal correction path is independently accumulated observation by observation and compared with block covariance PCA. One passive snapshot remains a required axial negative control: the full mean equals its matched isotropization.

The known-zero-mean Ledoit–Wolf comparison computes the outer-product Frobenius variance directly. For `Y_j=u_j u_j^H` and `S=mean(Y_j)`, its numerator is

\[
\frac{1}{L^2}\sum_j\|Y_j-S\|_F^2
=\frac{\sum_j\|u_j\|^4-L\|S\|_F^2}{L^2}.
\]

This Hermitian identity applies to complex vectors and does not borrow a real-only fourth-moment formula. It is a covariance shrinkage estimate, not a finite-sample confidence theorem. The one-snapshot convention is stated explicitly. Continuous validation-selected sample-covariance loading includes the exact isotropic endpoint and rejects an unloaded singular covariance. Both edge intervals are refined in addition to internal grid basins, so an interior optimum close to a boundary is not missed merely because the sampled endpoint wins. The complex validation quadratic is checked against direct covariance solves.

Validation data, test data, held-out noise snapshots and independent query draws have separate deterministic streams. Every method shares the same test trajectory and held-out samples. The desired steering vector enters the decision rather than the snapshot likelihood. Population covariance and hidden directions enter validation scoring and test scoring, never posterior correction. A deliberately constructed equal-radius posterior fixture demonstrates a conditional decision mechanism and is explicitly not a random-training performance claim.

## Interpretation and remaining empirical questions

The Bingham mode is also its minimum mean squared chordal-distance estimate; retaining a distribution does not produce a different point optimum for that loss. Identical-likelihood point regression or covariance PCA can match the mode exactly. Any extra value must be demonstrated for posterior responses, expected decisions, directional regions or a stated uncertainty score. Conditional posterior optimality does not force a repeated-training advantage at every fixed state.

Sphere figures show the real contact-normal projective distribution exactly. A complex higher-dimensional sphere picture is a labeled real sphere section, not a complete state or a normalized marginal. Complex CP1 maps to an axisymmetric von Mises–Fisher law and cannot establish general anisotropy. General-rank tangent precision has the Kronecker-sum restriction described in draft 5; rank/codimension-one demonstrations do not remove it.

Higher-dimensional contact outcomes use `T=4N` balanced requests. Their coordinate-averaged response loss contains a `1/N` factor, so it is shown with a dimension-independent Frobenius or normal-angle measure; a decreasing coordinate average alone would not establish improved identification. The generalized-velocity examples are not presented as physical high-DOF robots with unmodeled known Jacobians.

The complete general-validation gate passed **209 tests**, and all four paper components now pass their current source/artifact checks. All four deterministic geometry cases resolve. The array study retains 3,072 independent trajectories with no native mean failures and no unresolved precision flags in its 320 uncertainty reports. Contact has no native numerical failures; its separate local-Gaussian chart diagnostics remain unresolved in 323 static and seven dynamic reports. Those local-chart flags are not relabeled as native integration failures or silently discarded. The printed conditional bound is valid in 448 reports, with no violations; invalid sufficient conditions remain visible. The scaling study retains all 36 cases and 1,421 operation records, with zero unresolved operations and seven measured repetitions per applicable target. These finite operating-range checks do not establish arbitrary-rank, arbitrary-concentration or worst-case runtime guarantees.

Empirical rankings, credible inclusion and timing distributions are reported in the source-verified findings and notebooks, including unfavorable comparisons. No test demands Bayesian superiority. Physical drift and model mismatch can invalidate credible-coverage interpretations even when every numerical calculation is resolved. Draft-5 block EM is described in the notation map; the legacy recorded-per-observation-prior procedure is a different objective and is not used as validation of fixed-block monotonicity in this redesign.
