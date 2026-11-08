# Draft-5 simulation protocol

This redesign uses the supplied draft's observation models, Algorithm 1, task identities and appendices. It does not copy an empirical simulations section. The main notebooks focus on the fixed-parameter conjugate tracker and supplied coordinate transport. Earlier stochastic-kernel and joint-parameter experiments remain archived additional studies, rather than being presented as part of Algorithm 1.

## Questions and geometry

The contact state is the admissible plane, with an **unoriented** normal line. Its density on the unit sphere is `q_{-A_t^+}([n])`; antipodal points represent the same state. Surface colors show density relative to normalized invariant measure. The sphere radius is fixed. Its radius is not a probability ordinate. Density floors, viewing coordinates and hidden rear lobes are recorded.

A complex array state in dimension eight or sixteen cannot be represented by a physical unit sphere. Its three-dimensional plots evaluate the actual density on a specified real-coefficient section `h=U_3 x`, `x in S^2`, through the mode and complementary eigenvectors. These are density sections, not marginal distributions, arrival-angle posteriors or representations of every state direction. The full spectrum accompanies them. Complex dimension two maps to an axially symmetric Bloch-sphere density and is a symmetry control, not an anisotropy example.

The deterministic pair in draft Section 4.1 has the same mode, spectrum and **every** centered chordal-distance quantile. Its directional risks and decisions differ. These calculations are kept separate from independent random-data comparisons. For contact, regional leakage bounds concern the inadmissible normal component; they are not bounds on the entire admissible velocity. Equal posterior mass does not make HPD regions uniformly tighter than balls.

## Beamforming

Training receives only proper-complex snapshots, with covariance `s I + tau2 P`. The desired unit signature enters the beamforming decision. The primary dimension is eight; a second independent cohort uses sixteen sensors. Both known-noise regimes are retained. The predeclared generation laws are Haar spatial signatures and stationary ULA interferers separated by 3, 10 or 30 degrees from the desired arrival. Sample counts include one, four, sixteen and sixty-four. The closest, intermediate and orthogonal geometries are all reported.

The posterior mean and mode are compared with a same-mode, same-spread axial control, independently validated point regularization, continuously validated sample-covariance loading, data-adaptive Ledoit--Wolf shrinkage, GROUSE and the true-covariance reference. Continuous covariance shrinkage includes the exact isotropic endpoint, rather than an arbitrary maximum loading. The complex Ledoit--Wolf extension uses Hermitian outer-product Frobenius variability; its single-observation convention is explicit. Validation and test streams are disjoint. The desired/interferer geometry does not select favorable test trajectories.

Conditional posterior expected-power optimality, actual true-covariance power and independent held-out power are different quantities. Derived spatial-projection losses are separate again. Gains against a point mode do not establish a benefit from directional anisotropy, since the mode may duplicate covariance PCA. The singleton equality control and matched-spread control remain mandatory. Numerical reports use bounded independent posterior batches at specified trajectories and times; accurate means do not certify credible tails.

## Contact-normal tracking

The maintained ideal contact satisfies the Euclidean closest-motion model with known isotropic resistance: `u=P_X v+e`. Known applied free velocity and measured velocity enter every force-aware comparator. Surface normals and positions enter generation and scoring only. Friction, unilateral activation and anisotropic mobility are not silently covered by this likelihood.

The stationary cold-start study varies input directions at identical energy and retains broad posterior regimes. Its Monte Carlo normals follow the stated Haar prior; fixed-normal visual examples are separately preassigned. Full mean, conditional spectral least squares, a same-spread control, validated regularization, rank-informed ridge regression and PCA of residuals are evaluated using their available observations. A local Gaussian with the **same** Bingham parameter isolates density shape. Numerical integration and validity of a local chart have separate flags.

The physical plane/sphere study supplies a known rotating wrist frame. Every corresponding method receives the same `G_t=R_t^T R_{t-1}`. This represents orientation measurements assumed exact, not oracle alignment of the unknown contact normal. A fixed-frame ablation retains the same data and parameters. The recursive Gaussian has data-only initialization and independently selected process spread. Its model differs from a same-parameter Gaussian approximation.

The expected soft decision is exactly draft (36), with penalty linear in `I-Pbar`. Squaring `I-Pbar` would be incorrect. It is an expected command penalty, not a guarantee that a command satisfies the unknown contact constraint. Unconstrained ridge maps are response predictors and are not inserted into that convex Bayes decision. Each dynamic method can choose its own memory on validation; such comparisons do not isolate posterior shape by themselves.

The modal certificate uses the equivalent normal-response model `v-u=(n n^T)v-e` and the printed draft's noise, sum-of-drift-norms budget and bound. The sharper legacy helper is not relabeled as the printed theorem. A long stationary cohort makes the sufficient condition testable without evaluating moments at every observation. Short or moving cases that fail it remain invalid. Guarantees are pointwise at prescribed times, use fixed known noise and deterministic inputs/transport, and do not certify credible-region coverage.

Higher-dimensional contact examples use one maintained linear constraint in generalized coordinates. They are not advertised as a physical high-DOF robot without a specified Jacobian. Normal chordal error and full response-map error accompany query-averaged loss: dividing loss by dimension can otherwise make one constraint appear artificially easier.

## Scaling and numerical limits

The dimension grid is 3, 8, 16, 32, 64 and 128. Fixed parameter spans and generated-data cases separate concentration from ambient dimension. Corrections, eigenspace extraction, moments, solves, full endpoint estimators and optional uncertainty reports are timed separately. Persistent storage is measured explicitly. Rank-one and codimension-one computations do not imply a general-rank moment backend.

Final timing runs are serial after the simulation worker pools finish, with one BLAS thread, warm-up and seven randomized repetitions. Wall and CPU clocks, dispersion, machine quota and numerical flags are recorded. Theoretical `O(N^2)` correction/storage and dense `O(N^3)` eigenanalysis are reference operation counts, not fitted universal latency laws. Endpoint comparisons give every estimator the same stream and reporting schedule; low-rank point decisions use their appropriate closed-form solve. Temporary tracked allocation excludes unmeasured native workspaces and total process memory.

## Validation and preservation

`lib/simulation_validation.py` checks the complete general-notebook certificate, current regression sources and every new scientific module/runner. New paper runs wait for this gate. Plotting reads source- and content-verified artifacts and performs no simulation, fitting or posterior sampling. Plot manifests bind all component manifests and presentation sources. Notebook execution is verified separately.

Both earlier executed notebooks, historical result hashes and the exact older numerical-certificate source trees are retained under `results/simulation_redesign/archive/`. They are historical evidence under their original producer versions. The old certificates are not bypassed to execute new work. No manuscript is edited. Fixed-block noise learning is distinct from the older frozen-per-observation-prior objective; these new performance comparisons hold noise parameters fixed and do not claim an EM or hyperparameter-learning advantage.
