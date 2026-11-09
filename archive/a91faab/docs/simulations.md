# Simulation protocol

The draft's experiments were proposed, rather than existing reproducible
results. These implementations supply explicit finite protocols and retain
unfavorable outcomes. No simulation is conditioned on one method winning.

## Array

The complex array has 16 channels and three interference modes. Initial modes
are the QR-orthonormalized ULA steering vectors at angles -0.4, 0.05, and 0.45
radians. Channel phase rotations generate a deterministic unitary path; half
of that motion is supplied to every tracker. Desired steering is a prescribed
mixture of one interference mode and a perpendicular direction, with squared
overlap 0.05, 0.25, or 0.85. It is normalized and supplied to beamformers.
This specifies subspace orientation experiments, rather than claiming that all
desired steering mixtures are physical single-angle ULA steering vectors.

Noise variance is 0.3 and nominal signal variance is 2. Training snapshots are
interference-only except for the explicitly labelled contamination scenario.
The three mismatch conditions change modal powers to [0.25,1,1.75], introduce
source correlation 0.55, or add desired-signal contamination of variance 1.5.
Their evaluation covariance is the true interference-plus-noise covariance.

Three predeclared candidates tune Bingham forgetting and loaded covariance
parameters using a separate seeded validation trajectory and independent
validation snapshots. Both mean and modal Bingham decisions then share the
selected parameter sequence. Population output power and 256 independent
held-out snapshots are evaluated at observation counts 12,36,72. SINR assumes
unit desired-signal power. The tenth-percentile statistic uses independent
trajectories at a fixed terminal time. Oracle covariance is not tuned.

Each selected posterior is sampled by two chains, one initialized at the mode
and one at an independent random subspace. Budgets increase from 6,000 retained
samples/2,500 burn-in sweeps to at most 24,000/8,000 per chain. Acceptance,
ESS, independent-chain R-hat, batch MCSE, actual budget and pass/fail status are
preserved. These are diagnostics, not convergence proofs. Ball and HPD mass
are 0.95 by sampled thresholds. Threshold errors and feasible/dual optimization
gaps are distinct quantities.

## Field

Eight low-frequency separable DCT-II spatial modes form an orthonormal
64-by-8 basis on an 8-by-8 grid. A Gaussian-QR rank-three initial subspace is
moved by a normalized Gaussian skew-symmetric generator at rate 0.018.
Half of this motion is supplied. Repetitions vary initial orientations, drift
directions and observations. Noise variance is 0.2 and nominal coefficient
variance is 1.5. Passive unequal powers and sparse three-sensor corruptions
are explicit model mismatch; the filter keeps its nominal likelihood.

Independent validation trajectories select exactly three candidates per method
for Bingham forgetting, EW PCA forgetting, rolling-window length, and GROUSE
step size. Validation uses known simulation truth, but the filters never do.
All reconstructions assimilate the current measurement and then estimate that
measurement's clean field. Wiener shrinkage uses the same nominal noise levels.
Squared field errors equal squared coefficient errors because the lift is
isometric. Normalized errors are also stored but can be unstable for low-energy
clean fields; the main plots use absolute squared error.

Known calibration patterns are preassigned coordinate cycles transported by
the supplied frame, with transported ranks 6,7,8 and equal norm 2. They are
independent of the hidden subspace. The physical state still has unmodelled
motion, so these runs are not a controlled common-subspace test of global
identifiability. The theory notebook performs that test separately. Fresh
response probes are deterministic and independent of the state.

Noise fits run separately for passive and conditional observations, using
window length eight and two EM iterations at three fitting times. Every
historical observation has its own frozen predicted prior. Scores are evaluated
before fitting and correction, retaining Gaussian constants. Prior Monte Carlo
log-mean-exp scores have finite-sample bias; importance ESS, sampler ESS and
independent-chain score disagreement are recorded. Fits use sampled E-steps
and their exact ascent is unverified. The conditional window diagnostic
separates mean residual and uncertainty at a common final-parameter E-step
target. This is not a claim that it equals the preceding EM update.

Only coefficient observations are fitted. The 56 complementary field noise
coordinates from Appendix H.3 are intentionally excluded, avoiding an unfair
noise-estimation comparison with methods receiving eight coordinates.

## Scope

Coverage is a repeated-observation statistic under each specified simulation
law, not a guarantee of the posterior approximation. Eight repetitions give
wide binomial intervals. Longer or repeated studies can be run by changing
the notebook configuration; cache hashes force resimulation when configuration
or implementation changes. Runtime values are machine-dependent. Modal,
moment, score, EM and regional optimization costs are preserved separately.

Field timing tables count cumulative tracker updates plus moment evaluations
for the posterior-mean estimator. They exclude the final matrix-vector
reconstruction; the zero-cost field oracle is omitted from logarithmic timing
plots. Array decision timings additionally include the final covariance solve.
The JSON configuration metadata should not be edited manually: change notebook
parameters and rerun instead. Table checksums protect the saved numeric tables,
not arbitrary manual edits to descriptive metadata.

Early array targets that fail the original 24,000-draw cap receive a separate
accuracy follow-up, beginning at 48,000 retained draws and 12,000 burn-in sweeps,
with caps of 96,000/24,000. New independent random streams re-evaluate the same
posterior parameters. `case1_accuracy.csv` records ESS, R-hat, MCSE, population
output-power sensitivity and credible-threshold changes. Original failures are
preserved. The notebook requires the extended checks to meet their stated
diagnostics; this does not convert those diagnostics into a convergence proof.

In the recorded array protocol, 22 of 168 selected posteriors miss the original
ESS threshold. These concentrate in early weak-information targets and in the
unequal-power/correlated-source conditions; they are not failures of conjugacy.
The longer-chain follow-up quantifies decision and threshold sensitivity rather
than silently changing the original budget. Runtime depends on spectral
anisotropy and chain mixing as well as ambient dimension.
