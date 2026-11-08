# Passive complex-array geometry

Profile: quick. Primary n=8 has 8 independent validation and 16 test trajectories per law/noise. The n=16 cohort has its own recorded denominator. No source from the draft simulations section defines this protocol.

The stationary ULA desired arrival is0deg and interferer separations are3,10,30deg; the Haar signature is the matched-prior control. Uniform ambient complex-line inference is not a posterior restricted to physical arrival angle. Known s,tau2 and beta1 are shared across point/mean decisions.

SCM diagonal loading is selected continuously via alpha in[0,1]; alpha1 is its exact isotropic endpoint. The separate Ledoit-Wolf reference adapts its shrinkage to each observed sample covariance. Its one-snapshot convention is explicitly alpha1. Global modal covariance regularization and per-posterior modal-overlap matching are different controls.

Methods receive identical passive snapshots; the desired steering is used only for decisions. Validation losses use known simulated covariances on independent episodes, never test truth for tuning. All methods share held-out measurements within a test trajectory.

Mean failures: 0; unresolved native reports: 0/80. Finite iid Monte Carlo precision is not a deterministic tail certificate. Scalar intervals at an exactly aligned ULA query can have zero fixed-state coverage at the support boundary; this does not contradict stationary state-region invariance.

The selected-data fixture is a structural calculation. Its two Bingham parameters have the same spectrum and mode, related by a unitary fixing the mode; hence every centered chordal-distance quantile agrees. Their posterior mean matrices and optimal expected-power weights differ. Synthetic two-snapshot blocks recover the parameters exactly up to the identity gauge. These are not typical random-training outcomes.

All true-power, held-out-power and derived response comparisons remain in paired.csv, including losses. The independent trajectory is the repetition unit. Response queries and scan-grid points are not extra repetitions. Methods supplying only a covariance do not acquire an invented projector-response prediction: their response_loss is undefined.

The scan bands use preassigned trial0, ULA10deg, s=.2. Densely displayed pointwise quantiles are Monte Carlo summaries from native iid draws; only the five fixed query directions carry the declared quantile-MCSE checks. Saved plot clouds use a predeclared subset of those draws.

## References

Ledoit and Wolf (2004), A well-conditioned estimator for large-dimensional covariance matrices, Journal of Multivariate Analysis93:365–411. The implementation uses the uncentered outer-product Frobenius form with Hermitian products for proper complex observations.

Balzano, Nowak and Recht (2010), Online identification and tracking of subspaces from highly incomplete information. GROUSE here uses complete observations and the same known-power decision covariance.

