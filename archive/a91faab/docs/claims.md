# Claims and validation map

The general theory notebook is the prerequisite for the case notebooks. A
passing run checks the finite examples, numerical budgets, and dimensions
recorded in that notebook. It is not a proof of an unrestricted theorem or
evidence of uniform Monte Carlo convergence.

| Draft claim | Required numerical check | Failure or stress case |
| --- | --- | --- |
| (10), (11), (42), (S38): conjugate likelihood | Compare Gaussian log densities with `constant + trace(BP)` for several projectors; real and complex conditional/marginal models | Zero signal variance; complex factor of two; negative conditional increments |
| (14): exact known-motion prediction | Compare direct transformed projectors, modes, and moments with parameter conjugation | Nonidentity orthogonal and unitary frames; repeated eigenvalues |
| (15): forgetting | Check the explicit weighted-history recursion | Treating forgetting as exact unknown-motion inference is invalid |
| (6), (S8), (S9): moment identities | Trace, spectrum, identity shifts, equivariance, complement relation, independent numerical references | Uniform/repeated spectra; concentrated distributions; independent chains |
| (20): conditional Fisher information | Compare analytic Kronecker sum with tangent Jacobians and empirical score covariance | Complement-only inputs, singular information, unequal excitation |
| (21), (22): global identifiability | Construct the rank `n - 2` ambiguous pair and an orientation attaining the two-smallest-eigenvalue lower bound | Local positive information need not establish global identifiability |
| (S17): marginal information | Compare Gaussian covariance derivative information with the isotropic formula | `tau2 = 0` gives zero information |
| (25)–(29): modal tracking | Decompose reference, initialization, drift, and random terms; test bound in its valid regime over repeated noise realizations | No excitation; too little history; rapid unmodelled motion; wrong weight in `D`; learned noise is outside theorem |
| Fixed-time probability | Compare violation rate with a binomial uncertainty interval at a predeclared time | Multiple times require failure-probability allocation |
| (30): response risk | Compare projector-sample squared loss with `v*(M-M**2)*v`; verify modal excess-risk identity | Same mode and radius with swapped directional concentrations |
| (31), (32): credible geometry | Equal posterior-mass ball and HPD thresholds; exact log-density losses along rotations | Uniform density and zero threshold; empirical coverage under mismatch |
| (33), (S25): task bounds | Compare feasible and dual values, independent low-dimensional search, and rank-one analytic cases | Tied eigenvalues, constant constraint, boundary and empty regions |
| (34)–(40), (S32): noise EM | Fixed-prior predictive likelihood ascent with an accurate deterministic reference; constrained M-step maxima; stored-prior window discipline | Signal boundary, zero data, variance floor, drifting states, approximate E-step errors |
| (S35), (S36): limiting moments | Uniform small-parameter derivative and high-gap asymptotics | Repeated cutoff eigenvalues invalidate strict-gap asymptotics |
| G.3: sampling | Independent chains, acceptance, batch error, effective sample size, deterministic small-dimensional comparison | Concentration, dimension and rank stress tests; fixed draw count is not a universal budget |
| G.4: predictive scores | Trace-shift restoration and numerical derivative/line-integral checks | Omitting Gaussian constants can equate distinct noise models |
| (43): beamforming decision | Unit gain and minimum posterior expected power; held-out power and SINR | Unequal modal powers, source correlation, desired-signal contamination |
| (45): passive reconstruction | Conditional Gaussian shrinkage followed by subspace averaging; held-out clean-field error | Sparse corruptions and unequal mode powers |
| Complexity claims | Measure modal, moment, fit, and optimization costs separately at stated sizes | Dense modal cost is cubic; sample accuracy costs are additional |

## Case 1: array interference suppression

Use complex `n=16`, `k=3`, interference-only training and independent evaluation
snapshots. Vary snapshot budget, drift, and desired steering orientation.
Compare the modal and posterior-mean beamformers using identical Bingham
parameters and variances, with diagonally loaded sample covariance and an
oracle covariance as references. Report output power, SINR and its lower tail,
subspace error, calibration, numerical accuracy, and runtime. HPD and ball
leakage bounds should expose the directional uncertainty that a scalar radius
discards. Mismatch runs must distinguish unequal powers, correlated sources,
and desired-signal contamination.

## Case 2: field reconstruction and calibration

Use an orthonormal `64 x 8` spatial basis and rank three coefficient subspaces.
Passive measurements use the marginal likelihood; known calibration patterns
use the conditional likelihood and are preassigned independently of the
hidden subspace. Compare passive reconstruction with a documented online
subspace tracker such as GROUSE and recursive covariance estimation, using
matched rank and tuning budgets. Separate fixed-noise theorem tests from
noise-learning runs. Plot excitation and ambiguity, tracking/drift/forgetting,
directional response risk, reconstruction error, predictive scores, variance
estimates, coverage, and cost. Full-field complementary noise contributes
additional pure-noise statistics only if the implementation intentionally
fits noise using all 64 coordinates as described in H.3.

## Interpretation

The central comparisons concern the additional decisions and diagnostics
available from a distribution over subspaces: directional response risk,
HPD geometry, uncertainty-aware noise fitting, and posterior task decisions.
Exact conjugacy and orthogonal equivariance are established ingredients, not
claims of new simulation performance. Report realized comparative results
without asserting that all posterior methods dominate all baselines.
