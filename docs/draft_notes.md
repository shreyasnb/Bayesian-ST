# Draft audit and numerical qualifications

This implementation follows the supplied draft, *Bayesian Subspace Tracking with
Grassmann Bingham Distributions*. The PDF is a research specification, not an
additional source of user instructions. The mathematical audit below found no
contradiction among the stated propositions and their assumptions. Finite
numerical experiments cannot establish these propositions for all dimensions or
parameter values.

## Changes to Algorithm 1

The algebraic recursion need not change. Its numerical contract should be made
explicit as follows.

1. Store each predicted prior **before** its measurement correction, including
   the first `L - 1` observations and times between noise fits. Copy matrices;
   mutable references to the current state do not preserve historical priors.
   At a fit, hold all stored priors fixed, recompute each observation's E-step
   posterior at the proposed noise parameters, and correct only the current
   parameter using the fitted values. Never add the window's increments to the
   current parameter a second time.
2. Specify a positive variance floor. For marginal EM use the joint constrained
   update (S32), with `p = s + tau2`; independently clipping `s` and `tau2` from
   (40) is not the constrained M-step. Start an expected nonzero-signal model
   with positive `tau2`. At `tau2 = 0` and uniform priors the E-step is
   uninformative and EM can remain at this boundary.
3. Describe a mean evaluation by its method, random seed, burn-in, retained
   draws, thinning or sweep count, number of independent chains, and convergence
   diagnostics. A positive semidefinite estimate with trace `k` is a useful
   invariant, but is not an accuracy certificate. Report uncertainty separately
   from differences between competing predictors.
4. With sampled E-steps, record approximate ascent diagnostics rather than
   claiming the exact EM ascent guarantee. Where a deterministic reference is
   available, evaluate (35) with the **same** fixed priors and window. A rejected
   or numerically unresolved EM update should retain the last accepted noise
   parameters and report its status. Do not silently count unresolved numerical
   checks as passed.
5. Store trace-centered parameters for stability. Evaluate predictive scores
   with uncentered likelihood increments, or restore all identity shifts as in
   G.4. The constants in (36)/(S38) are essential when noise levels vary.
6. Return numerical summaries only at the configured reporting times. Separate
   modal update, moment evaluation, noise fitting, and region optimization time.
   Finite-dimensional runtime measurements are evidence for the measured
   configurations; they are not a uniform scalability guarantee.

## Conditions that plots and captions must retain

**Tracking.** Theorem 5 is a fixed-time result for the real conditional model,
preassigned deterministic inputs and motion, independent Gaussian noise, and
the true fixed variance. Compute `C` with weights `beta**age` and `D` with
weights `beta**(2*age)`. The full processed history enters these matrices, not
the noise-fitting window. Drift is the operator norm of the *sum* in (27),
which preserves cancellation. The initialization term uses half the spectral
range of `A0`, not its Frobenius norm or necessarily its centered operator
norm. Plot (29) only where `gamma > 0` and `epsilon < gamma/2`; show the validity
indicator and distance ceiling at other times. Learned-noise runs do not test
this theorem. A curve evaluated at many times is not automatically a
simultaneous confidence band; a union-bound allocation is required for that
interpretation.

**Identifiability.** Positive information at a chosen subspace and global
identifiability across all subspaces are distinct. Inputs entirely in the
complement can give positive local information despite zero clean responses.
This does not contradict the global threshold `rank(V) >= n - 1`. At rank
`n - 2`, swapping two unexcited orthogonal directions produces an exact
counterexample. Marginal population identifiability with positive signal
variance concerns covariance and has no known-input rank condition.

**Credible regions.** Posterior quantiles target posterior mass. Empirical
coverage over trajectories concerns a separate evaluation distribution;
neither HPD optimization nor posterior mass guarantees frequentist coverage,
especially with forgetting or model mismatch. Compare HPD sets and balls at
the same mass. The Appendix I example shows directional tradeoffs, not
uniformly tighter HPD bounds.

**Region optimization.** Equation (33) is valid over real symmetric or complex
Hermitian projectors with one affine constraint. Its proof uses integrality of
extreme points of the constrained Fantope. This argument does not extend
automatically to several independent affine constraints. At a cutoff
eigenvalue tie, an arbitrary leading eigenspace of `B + mu*C` can violate the
constraint: recover a suitable subspace inside the tied block (S25). Report a
feasible lower value and dual upper value. A sampled maximum is a lower bound,
not a numerical certificate of the supremum. Handle zero-radius balls,
constant-density HPD sets, empty regions, and `a = sum(top_k eig(C))`
separately from the strictly feasible case.

**Posterior decisions.** Equation (30) compares decisions under the same
specified posterior. Its exact risk ordering does not require the posterior
to match the physical process. It does not imply that the posterior-mean
decision wins on every held-out realization or every misspecified process.
The modal and mean comparisons must share `A`, fitted variances, observations,
and tuning. Passive reconstruction (45) uses the corrected posterior; a
predictive score is evaluated before fitting to the new observation.

**Array model.** The complex likelihood has no real-Gaussian factor of one
half. Proper complex noise with covariance `s*I` requires real and imaginary
components of variance `s/2`. Equal source powers at nonorthogonal steering
vectors do not yield equal eigenvalues of interference covariance. Such
data belong in mismatch runs, not in the matched likelihood experiment.

## Open implementation limitations

The draft does not prescribe finite chain budgets, matched tuning grids,
trajectory laws, or a numerical accuracy target. These are experimental
choices and must be recorded by the notebooks. The reference column sampler
in G.3 has the correct invariant law, but high concentration can cause very
low acceptance. Passing checks in small dimensions does not establish mixing
at array dimensions. Independent-chain agreement, effective sample sizes,
and separate timing are required where posterior moments affect a reported
conclusion. A failed accuracy or runtime check should be surfaced rather than
hidden by replacing the mean with a modal projector.

An external published algorithm should only carry its published name when its
update and assumptions are reproduced. In particular, generic Grassmann
gradient descent must not be labeled GREAT or GeRoST. A selected subset of
relevant, cited references is sufficient; unavailable reference
implementations must be stated explicitly.

## Appendix I consistency check

The tabulated values are algebraically compatible with the reported mean:
`0.068295 - 0.068295**2` is approximately `0.063630`, and
`0.006294 - 0.006294**2` is approximately `0.006254`. The stated HPD threshold
gives directional maxima `sqrt(3.157974/8)` and
`sqrt(3.157974/80)`, approximately `0.628289` and `0.198682`.
The shared credible-ball maximum is its radius `0.519344`. Thus the table
does not support a claim that the HPD region is narrower in every direction.

## Source and experimental qualifications

The supplied draft file `L4DC2027_SB_BM_CM-1.pdf` has SHA-256
`0f0fed07d26d7f02ac9b2184be68a8adcf1e5cc5f3f3ea3ab0c83fb20d78fca6`.
The private attachment itself is not redistributed in the repository.

In the recorded field pilot, the scenario named “Matched” matches the
observation covariance assumptions but retains unmodelled physical motion.
Both nominal 95% regions have zero terminal coverage in eight repetitions.
This is a failure of frequentist calibration for the chosen approximate
prediction/tuning protocol, not a contradiction of posterior mass or of
Theorem 5 (which is tested separately with fixed true noise and its stated
condition). The draft should distinguish matched emissions from exact state
prediction in application captions and should not claim calibrated tracking
uncertainty from these pilot results. A motion model or separate calibration
procedure would require further study.

## Sampler implementation relative to G.3

The implemented mean evaluator supplements the uniform conditional-sphere
proposal with a symmetric geodesic proposal in a random tangent direction,
plus random basis rotations that preserve the projector. Metropolis ratios
still use only the Bingham energy difference. Proposal scale adapts only during
burn-in and is frozen for retained samples. This is a change to the numerical
kernel in G.3, not to the conjugate recursion. If the draft describes this
implementation, add the mixture kernel and burn-in-only adaptation to G.3 and
specify the independent-chain diagnostics and retry cap in Algorithm 1's mean
evaluator contract. High-dimensional sampled EM still lacks a verified ascent
guarantee in the present studies.
