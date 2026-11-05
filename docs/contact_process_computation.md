# Contact process computations

These additions specialize the existing rank/codimension-one machinery to the
real contact plane `Gr(2,3)`. They do not replace the general-rank implementation
with a rank-one approximation. The original library and frozen contact pilot
are preserved.

## Exact moment, approximate density

Write `P=I-nnᵀ`, with `n` and `-n` representing the same plane. The plane kernel
`exp(kappa tr(P_X G P_Y Gᵀ))` reduces to a Watson normal-line kernel because
`tr(P_X G P_Y Gᵀ)=1+(nᵀGm)²`. Its extra constant cancels in normalization.
If `a=E(cos² theta)`, `b=(1-a)/2` and `rho=a-b`, the exact predicted plane mean is

`M_minus = rho G M_plus Gᵀ + (1-rho) (2/3) I`.

`lib/contact_process_prediction.py` computes the transition coefficients with
stable one-dimensional formulas, caches them, and fits a trace-zero parameter
to this mean. The inverse fit uses two independent eigenvalue coordinates and
a covariance Hessian. It reuses warm starts and has explicit uniform,
deterministic and axisymmetric limits. The deterministic limit transports the
parameter directly and performs no inverse fit.

Radial analytic integrals and a concentration-adapted angular rule avoid the
old positive-series order cap on contact parameters with very large spans.
Refinement error and inverse residual are recorded separately. A converged
inverse with an unresolved moment rule is not accepted. The radial/azimuthal
specialization is three-dimensional; it cannot be silently reused for a
seven-dimensional general-rank problem.

An ambiguous first observation in physical validation exposed a finite-angular
resolution failure. The inverse now refits at bounded finer angular orders,
checks analytic branches the same way, and evaluates its original and final
objectives at resolved budgets. The tolerance is unchanged. The reproducible
[independent refinement audit](../results/contact_uncertainty/numerics/refinement/README.md)
uses unwarped quadrature, rotated ridges and difficult starting parameters;
it complements the regression tests. Earlier exclusions and intermediate
development failures remain archived rather than being removed from the record.

Fitting this moment does **not** match higher moments or make the predictive
mixture a Bingham density. Independent recursive full-density checks retain
total-variation, mean, region and scalar-CDF errors.

## Gaussian controls

`lib/contact_gaussian.py` uses the derivative of
`h(n,v)=v-n(nᵀv)` in a two-dimensional tangent basis `E`:

`J = -[(nᵀv) E + n(vᵀE)]`.

The recursive filter uses a Joseph covariance correction, an exponential-map
mean update and covariance transport into the new tangent basis. Its isotropic
process variance is `E(theta²)/2`, integrated under the same normal-line kernel.
This matches angular variance, not the full kernel. At broad dispersion its
mapped first moment can differ from the Watson moment; that difference is an
explicit approximation, not a more favorable process assumption.

Initialization uses six observations through the corresponding Bingham filter,
without geometry. Its information and cost are counted. The pre-correction
observation score is the true-likelihood Bingham mixture score during warm-up,
then the normalized linearized Gaussian score. It is not claimed to be the
exact nonlinear Gaussian-mixture likelihood.

The same-parameter Laplace control uses variances `1/(2 gap_j)` for the normal
parameter `-A`. A zero gap is a failed local approximation, rather than a tiny
inserted variance. For reporting, the mapped tangent-Gaussian law is integrated
using polar coordinates and exact periodic radial primitives. Intervals are
neither clipped to physical response ranges nor taken from a finite cloud of
sampled responses. Integration convergence and probability beyond the
projective cut locus are separate flags. A Gaussian local ellipse is not an
invariant-density Bingham HPD region.

## Independent full-density control

`lib/contact_spherical_reference.py` uses even real spherical harmonics and the
zonal kernel's Funk--Hecke multipliers. Prediction is diagonal in harmonic
degree and avoids a dense all-pairs spherical convolution. Updates multiply the
actual known-input likelihood before projection onto the harmonic basis.

Separate development streams choose degree 24/32 and polar report orders
128/192 before the 200-trajectory test streams are opened. The full recursive
reference is applied to the first 16 trajectories per transition regime.
Diagonal response queries need an axis-adapted marginal rule: a fixed azimuthal
grid can otherwise place artificial finite mass on an extremal response. The
implementation rotates the harmonic coefficients and integrates the regular
absolute polar coordinate. Earlier unresolved development tables remain saved.

These are numerical refinement diagnostics with explicit tolerances, not
rigorous error bounds for every possible density. A development pass does not
erase a failed held-out target.

## Scores and costs

Likelihood corrections always use the physical sensor variance. Pre-correction
scores include the Gaussian constants and restore trace shifts introduced by
parameter centering. Reporting `q_(A/T)` preserves the raw mode and changes only
the reported distribution; it is generalized-posterior calibration, not a
sensor-noise estimate.

Per-observation path wall time includes prediction, correction, moment and
score work. Practical process comparisons also count the auxiliary point
tracker that supplies their shared learned nominal rotation. Inverse time is
nested inside prediction, and Gaussian warm-up inside its path. Report time is
separate and occurs only at the declared report times. The experiment's
illustrative cloud never supplies a scored mean, interval or coverage region.
