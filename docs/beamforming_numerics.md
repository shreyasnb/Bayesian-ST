# Native complex-line computations

The unified array study uses a probability density proportional to
`exp(h^H A h)` on complex lines, with `h^H h=1`. The invariant measure is
normalized to total mass one. Global phase does not change the line or any
reported functional. None of the calculations replaces this law by an
unrestricted real Bingham distribution.

## Means and normalizers

`complex_moments` uses the existing positive spectral series with
`complex_data=True`. In eigen-coordinates, squared magnitudes have the
Dirichlet(1,...,1) law. The returned mean is the derivative of the finite
spectral approximation, including its identity-shift correction. Its adaptive
bound controls truncation in exact arithmetic; it does not enclose rounding,
credible-tail error or a Fisher-information calculation.

Complex dimension two has an analytic integral. With eigenvalue gap `d>=0`,
the leading squared coordinate has density proportional to `exp(d r)` on
`[0,1]`. Its mean is `1/(1-exp(-d))-1/d`; the uniform limit is `1/2`.
Stable small-gap series and exponential differences evaluate this identity.
The implementation checks it against the native spectral series and independent
Gauss--Legendre integration.

The declared order cap is 4096. Representative passive snapshots are checked
at dimensions 8 and 16. A span of 20000 is a deliberate budget-failure control:
no unchecked Laplace or Monte Carlo mean replaces an unresolved spectral mean.
The long-horizon modal certificate cohort therefore does not request posterior
means or tails at every observation.

## Exact independent sampling

Let `lambda_0` be the largest eigenvalue, `d_i=lambda_0-lambda_i`, and
`x_i=|h_i|^2`. In eigen-coordinates the target is a tilted uniform simplex law,
with density proportional to `exp(-sum_i d_i x_i)`.

Two exact rejection proposals are available:

1. Draw `x` uniformly from the simplex and accept with probability
   `exp(-sum_i d_i x_i)`.
2. Draw each nonleading `x_i` independently from the truncated exponential
   density proportional to `exp(-d_i x_i)` on `[0,1]`. Accept if their sum is
   at most one and set `x_0=1-sum_{i>0} x_i`.

The second accepted density is the same tilted simplex density. Write
`Z_i=integral_0^1 exp(-d_i x) dx`, with `Z_i=1` at `d_i=0`. The ratio of the
second proposal's acceptance probability to the first is

`1 / ((n-1)! product_{i>0} Z_i)`.

The target normalizer cancels. Selecting the better envelope therefore requires
only the eigenvalues, not an additional moment calculation. Independent uniform
phases, followed by the eigenvector transform, restore the native complex law.
The samples are independent rather than a Markov chain. The proposal budget is
finite; exhaustion is an explicit failure and never a successful partial sample.

Tests compare acceptance rates to independently evaluated normalizers, check
complex phase variation and scalar CDFs, and include tied spectra and exhausted
budgets. The eigendecomposition and proposal generation are included in timings.

## Transition measure and inverse moments

For concentration `kappa`, the transition overlap `r=|h^H h_previous|^2` has
density proportional to `exp(kappa r) (1-r)^(n-2)`. Equivalently `1-r` is a
Gamma(`n-1`, rate=`kappa`) variable truncated to `[0,1]`. At zero concentration,
`r` is Beta(`1,n-1`). Adding an independent uniform complex orthogonal direction
gives an exact transition draw. The deterministic limit returns the supplied
centre. This is not the real Watson radial distribution.

The scalar log normalizer is evaluated from

`kappa + log Gamma(n) + log P(n-1,kappa) - (n-1) log kappa`,

where `P` is the regularized lower incomplete gamma function. A hypergeometric
formula handles small concentration without cancellation. Scalar parameter
inference consequently avoids a full matrix-series evaluation at every scalar
proposal. Transition coefficients use the corresponding analytic mean.

The predictive mean law is exact. Fitting a single Bingham density to that mean
is an assumed-density projection and does not preserve the entire predictive
mixture. The inverse solver uses trace-zero eigenvalue coordinates and checks
its final forward moment, residual and objective. Its covariance preconditioner
uses the complex spectral identity

`E[x_i x_j]=(m_i-m_j)/(lambda_i-lambda_j)` for distinct eigenvalues.

Within an equal-eigenvalue block the conditional Dirichlet law gives diagonal
second moments twice the off-diagonal ones; row sums equal `m_i`. Near ties are
grouped only in the Newton preconditioner. A final checked forward evaluation,
not a Hessian assumption or scalar rescaling, decides convergence. Independent
Cartesian-simplex integration verifies these identities in dimension three.

## Regions, scalar quantiles and Monte Carlo error

`posterior_report` uses independent threshold-training and check batches.
Pointwise scalar columns are the .025, .25, .5, .75 and .975 quantiles.
Ball and HPD thresholds are trained separately. Their achieved posterior masses
are checked on the independent batch. Approximate batch quantile MCSE, relative
to the 95% interval width, is reported in addition to the mass discrepancy.
These diagnostics do not turn a finite empirical quantile into an exact-mass
region or a deterministic tail-accuracy certificate.

The quick engineering fixture uses 2048 draws per batch and retains two scalar
MCSE failures. Before scientific evaluation, the paper budget was fixed at
16384 draws in **each** independent batch. Exponential-like quadratic tails can
have relative endpoint MCSE approximately `1.7/sqrt(N)`, which explains this
increase. The change follows numerical accuracy, not a favorable method outcome.

In complex dimension two, ball and HPD energy thresholds are analytic. A separate
continuous radial/phase integral checks scalar power CDFs. Its outer variable is
the normalized radial CDF, so sharply concentrated mass cannot disappear between
physical-coordinate quadrature nodes. Aligned queries have an analytic CDF.
A hostile gap of one million exposed a missed boundary layer in the initial
physical-coordinate reference; the final regression checks this case and generic
high-concentration queries. Moderate concentrations are cross-checked against
independent Bloch-coordinate integration. The CP1 transition is a spherical
von Mises--Fisher kernel with concentration `kappa/2`, not a Watson kernel; the
full recursive reference is implemented separately.

At uniform density, the inequality `loss<=0` would include the whole space.
Instead, ties are resolved with a ball around an independent Haar centre. Its
squared radius is `mass^(1/(n-1))`, giving exact posterior mass under the uniform
law. This is a valid HPD tie subset and its randomization is unitary-equivariant
in distribution. The centre source/seed and the actual region constraint are
recorded, so downstream regional bounds do not accidentally optimize over the
whole Grassmannian.

The optional same-parameter local approximation uses complex tangent variances
`1/gap`, or `1/(2 gap)` in each real coordinate, mapped with the projective
exponential. A singular cutoff is rejected. Nonlocal probability beyond `pi/2`
is flagged rather than clipped; a local ellipsoid is not a global state HPD set.

The rejection proposal also has a declared efficiency limit. An independently
audited dimension-16 spectrum with seven equal leading eigenvalues at span 100
exhausts the default proposal cap. Its accepted partial draws remain unresolved.
This broad leading-subspace example is distinct from the ordinary unique-mode
array fixtures; no general efficient-tail guarantee follows from those fixtures.
