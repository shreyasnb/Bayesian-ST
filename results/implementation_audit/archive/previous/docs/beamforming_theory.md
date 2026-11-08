# Passive array model: derivations and scope

This document supplies the passive-model statements needed by the unified
beamforming study. The manuscript files were not supplied. Nothing here
renumbers or rewrites the manuscript. The Gaussian information identity,
eigenspace perturbation principle and invariance argument are established
ideas; the formulas below specialize them to this experiment. The explicit
weighted covariance bound is proved below rather than attributed to the
paper's different, real known-input theorem.

The numerical counterparts are in
[`lib/beamforming_theory.py`](../lib/beamforming_theory.py),
[`lib/beamforming_numerics.py`](../lib/beamforming_numerics.py), and the isolated
`results/beamforming_unified/{quick,paper}/` directories.

## Measurement and state

The observations are interference plus sensor noise:

\[
u_t=X_t z_t+e_t,\qquad z_t\sim\mathcal{CN}(0,qI_k),\quad
e_t\sim\mathcal{CN}(0,sI_n),\qquad
R_t=sI_n+qP_t,\quad P_t=X_tX_t^H.
\]

The primary study has complex rank one. Powers `s` and `q` are variances, not
standard deviations. A proper complex standard Gaussian has independent real
and imaginary parts of variance one half. Estimators receive only `u_t`;
neither the simulated `z_t` nor the true spatial signature is a known input.
The desired unit steering vector is supplied separately for a distortionless
beamforming decision. Derived queries `Pv` later in the experiment are
posterior functional checks, not additional training measurements.

The state is a native complex subspace. For rank one it is a line `[h]`, with
`h` and `exp(i phi) h` equivalent. Its uniform prior is Haar measure on
`Gr_C(1,n)`, not a prior on a single ULA arrival angle. An ambient Bingham
posterior is consequently not an exact posterior over that physical angle.
No unrestricted real Grassmann approximation is used.

## Exact conditional correction and the modal estimator

Since `P` is a projector,

\[
R^{-1}=s^{-1}I-\frac{q}{s(s+q)}P,\qquad
\det R=s^{n-k}(s+q)^k.
\]

Thus the normalized proper-complex log likelihood is

\[
\log p(u\mid P,s,q)=-n\log\pi-(n-k)\log s-k\log(s+q)
-\|u\|^2/s+\operatorname{tr}(B P),\quad
B=\frac{q}{s(s+q)}uu^H.
\]

The complex coefficient has no factor one half. With a Bingham prior
`q_A(P) ∝ exp(tr(AP))`, its correction is exactly `A+B`, up to an identity
shift. For rank `k`, centering a parameter by `cI` subtracts `kc` from its
log normalizer, because `tr(P)=k`; predictive scores must restore this shift.

For a uniform initialization, fixed `s,q`, deterministic transport and
additive discounting, the parameter is a positive scalar times the
transported weighted sample scatter, plus an irrelevant identity matrix.
Its leading eigenspace is exactly weighted covariance PCA. At a tied cutoff
the maximizing eigenspace need not be unique; the objective or admissible
eigenspace family is the appropriate check. This is not the conditional
known-input projector-regression identity from the contact example.

Exact likelihood correction does not establish that an approximate dynamic
prior is the true filtering distribution. Stochastic moment projection and
integrating uncertain powers introduce separate approximation or mixture
issues.

## Passive measurement Fisher information

Use orthonormal `X` and `Q=X_perp`. In real local coordinates, a complex
Grassmann tangent `Z` has projector differential

\[
D_Z=QZX^H+XZ^HQ^H.
\]

The coordinates used in code are
`[Re vec_F(Z), Im vec_F(Z)]`, of dimension `2k(n-k)`. Unit coordinate
derivatives have projector Frobenius norm `sqrt(2)`; another metric
normalization would rescale a displayed information matrix.

For a real coordinate `xi_i`, write `R_i=q D_i`. Differentiating the
normalized Gaussian log density gives

\[
S_i(u)=-\operatorname{tr}(R^{-1}R_i)
+u^H R^{-1}R_iR^{-1}u.
\]

Proper-complex Gaussian fourth moments give
`Cov(u^H C u,u^H D u)=tr(C R D R)`. Therefore

\[
\mathcal I_{ij}=\mathbb E[S_iS_j]
=\operatorname{tr}(R^{-1}R_iR^{-1}R_j).
\]

In the frame `[X,Q]`, `R=diag((s+q)I_k,sI_(n-k))`, while `D_Z` is
off-diagonal with blocks `Z^H,Z`. Direct block multiplication yields

\[
\boxed{\quad\mathcal I(Z,Z)=\frac{2q^2}{s(s+q)}\|Z\|_F^2.\quad}
\]

The matrix is a scalar identity in the stated real coordinates. For `N`
stationary independent snapshots it is multiplied by `N`. In a real
Gaussian model, the score carries a factor one half and the fourth-moment
covariance a factor two, producing half this quadratic form on real
`k(n-k)` coordinates.

This is a measurement information calculation. A finite-data Bingham
normalizer Hessian is a posterior sufficient-statistic covariance, not this
measurement Fisher matrix. Finite-sample posterior anisotropy does not imply
anisotropic population information in the isotropic passive model. A CRB
comparison additionally needs its regularity and local-unbiasedness
conditions; a Bayesian estimator's error is not required to exceed an
unbiased CRB in every finite experiment.

### Pullback to the physical arrival angle

For a normalized half-wavelength ULA,

\[
h_j(\theta)=n^{-1/2}e^{i\pi j\sin\theta},\qquad
h'_j=i\pi j\cos\theta\,h_j.
\]

The parallel phase derivative does not change the line. Subtracting it gives

\[
g_\theta=(I-hh^H)h',\quad
\|g_\theta\|^2=\pi^2\cos^2\theta\,
\frac1n\sum_{j=0}^{n-1}\left(j-\frac{n-1}{2}\right)^2
=\frac{\pi^2\cos^2\theta(n^2-1)}{12}.
\]

Consequently `I_theta_theta=2Nq^2 ||g_theta||^2/[s(s+q)]` for a radian
coordinate. Information for a degree coordinate is this value times
`(pi/180)^2`. Internal angles remain radians. The physical-angle information
varies because the steering curve's tangent changes with angle, even though
ambient subspace information is isotropic.

The independent checker evaluates full Monte Carlo score covariance matrices
(including off-diagonals), direct expected-likelihood finite-difference
curvature, and a finite-difference ULA derivative. Its score generation uses
known truth only for a mathematical verification, not as an estimator input.
Reported score-product errors are Monte Carlo diagnostics, separate from
finite-difference errors.

## Fixed-time passive modal tracking certificate

Assume fixed known positive powers, a uniform prior, a deterministic state
path, independent proper-complex snapshots, and deterministic supplied
unitary transports. Let `T_j` transport observation `j` into the reporting
frame at the fixed time. Normalize nonnegative weights to `sum alpha_j=1`.
Define

\[
\widehat R=\sum_j\alpha_j\widetilde u_j\widetilde u_j^H,\quad
\overline R=\mathbb E\widehat R=sI+q\sum_j\alpha_j\widetilde P_j,\quad
R_t=sI+qP_t,
\]

and `d_cov=||Rbar-R_t||_2`. This drift term uses true subspaces and is
available to evaluation code only.

### Weighted quadratic-Gaussian concentration

For any fixed complex unit vector `v`, the variables
`|v^H tilde_u_j|^2/c_j` are independent `Exp(1)`, where
`c_j=v^H(sI+q tilde_P_j)v <= s+q`. For `Y=E-1`, `E~Exp(1)`,

\[
\log\mathbb E e^{tY}=-t-\log(1-t)
\le\frac{t^2}{2(1-t)},\quad 0\le t<1,
\qquad
\log\mathbb E e^{-tY}=t-\log(1+t)\le t^2/2.
\]

Thus the centered weighted sum is sub-gamma with variance proxy
`sum alpha_j^2 c_j^2` and scale `max alpha_j c_j`. Its two-sided tail is at
most `2 exp(-x)` at

\[
(s+q)\left[\sqrt{2x\sum_j\alpha_j^2}+x\max_j\alpha_j\right].
\]

A one-quarter net of the complex unit sphere, viewed in real dimension
`2n`, has cardinality at most `9^(2n)`. For a Hermitian matrix `E`, if `w`
is a net point within one quarter of a unit `v`, then
`|v^H E v-w^H E w| <= (1/2)||E||_2`. Hence
`||E||_2 <= 2 max_net |w^H E w|`.

Taking a union bound with `x=2n log(9)+log(2/delta)` proves

\[
\Pr\{\|\widehat R-\overline R\|_2\le\eta\}\ge1-\delta,
\qquad
\boxed{\eta=2(s+q)\left[\sqrt{2x\sum_j\alpha_j^2}
+x\max_j\alpha_j\right].}
\]

This derivation allows the stated deterministic time-varying covariance
path; it does not invoke an iid stationary covariance result for a drifting
path. Sharper covariance concentration results exist, but no sharper
unproved constant is substituted in this experiment.

### Eigenspace and array-response implications

On that event `||Rhat-R_t||_2 <= epsilon=d_cov+eta`. If `epsilon<q/2`,
Weyl's inequalities give a strict leading eigengap. Let `Y` be the leading
sample eigenspace, and use `Q` complementary to the true `X`. The eigenvector
equation gives

\[
(Q^H Y)(\widehat\Lambda-sI)=Q^H(\widehat R-R_t)Y,
\qquad\lambda_{\min}(\widehat\Lambda-sI)\ge q-\epsilon.
\]

It follows that `||Q^H Y||_2 <= epsilon/(q-epsilon)`. Its rank is at most
`min(k,n-k)`, so

\[
\boxed{d_c(\widehat{\mathcal X},\mathcal X_t)
\le\sqrt{\min(k,n-k)}\frac{\epsilon}{q-\epsilon}.}
\]

Code records a failed sufficient condition as `valid=False,bound=NaN`.
It does not clip it to the universal chordal ceiling and count it as a
successful certificate. With `n=4,k=1,s=.2,q=1,N=4096,delta=.05`, equal
weights and zero drift, direct evaluation gives approximately
`eta=.2570`, `bound=.3459`, which is nonvacuous but deliberately conservative.

For every unit scan vector, on the same event,

\[
|a^H((sI+q\widehat P)-(sI+qP_t))a|
\le q\|\widehat P-P_t\|_2\le qd_c(\widehat{\mathcal X},\mathcal X_t).
\]

This is uniform over scan directions at that fixed time. Guarantees at
several times are pointwise unless their failure probabilities were
allocated in advance, for example `delta_total/m` at `m` reporting times.
Neither a realized error quantile nor a credible radius is this confidence
bound. The bound is for the modal additive recursion, not its posterior
mean, inferred powers, learned transport or stochastic moment projection.

For raw exponential weights `gamma_j=beta^(t-j)`, retain both
`W=sum gamma_j` and `N_eff=W^2/sum gamma_j^2`. The normalized covariance
certificate uses `alpha=gamma/W`, whereas the unnormalized Bingham
parameter uses `W`. Weighted expected likelihood curvature scales with
`sum gamma`, but weighted score covariance with `sum gamma^2`; their
different dependence prevents identifying `N_eff` with a discounted
pseudo-posterior's Fisher precision.

## Stationary passive coverage by invariance

This is a specialization of the standard invariance/Bayes averaging
principle, not a new general coverage principle. Assume a fixed state,
known `s,q`, uniform subspace prior, `beta=1`, and exact posterior
inference. Let `C(U)` be a unitarily equivariant state region of exact
posterior mass `1-alpha`.

For a unitary `Q`, the data law transforms as
`U|P -> Q U|Q P Q^H`, and its Bingham posterior transforms in the same way.
Equivariance gives equal frequentist state inclusion probabilities at
`P` and `QPQ^H`. Unitary transitivity on the Grassmannian makes this
probability a constant `c`. Averaging the constant over the uniform prior
and conditioning on observations gives

\[
c=\int \Pr_P\{P\in C(U)\}\,d\mu(P)
=\mathbb E_U[\Pr\{P\in C(U)\mid U\}]=1-\alpha.
\]

Thus the ideal state region has coverage `1-alpha` at **every fixed
state**, including a fixed ULA line. The prior need not equal a physical
angle distribution for this equivariant fixed-state conclusion. Arbitrary
scalar response intervals do not inherit this state-inclusion argument.

Randomization is allowed if the region construction is equivariant in law
and its auxiliary randomness is included in coverage. At a uniform density,
the entire Grassmannian is a density-level set but has mass one, not .95.
For a valid exact-mass tie rule, choose an independent auxiliary Haar line
as the center and use a chordal ball. For complex rank one,
`r=|h^H h_center|^2 ~ Beta(1,n-1)` and
`Pr(d_c^2<=x)=x^(n-1)`. A ball of squared radius
`(1-alpha)^(1/(n-1))` has the required mass. It is a valid selection inside
constant-density ties, equivariant after averaging the auxiliary center.
A fixed canonical center at empty data would break this equivariance.

Nonconstant rank-one Bingham density-loss levels are continuous under Haar
measure even when eigenvalues repeat, so ordinary HPD thresholds have no
positive-mass density tie except the uniform case. Numerical posterior
thresholds and Monte Carlo mass checks remain approximate: exact mathematical
mass is not certified by a small moment error. Coverage reports retain those
errors, independent stream counts and finite-repetition uncertainty.
Forgetting, approximate process prediction and estimating `s,q` are separate
experiments outside this fixed-parameter stationary statement.

## Process prediction on native complex lines

For `K_kappa(X|Y) ∝ exp(kappa tr(P_X G P_Y G^H))`, the conditional overlap
`r=|h_new^H G h_old|^2` has density proportional to
`exp(kappa r)(1-r)^(n-2)` on `[0,1]`. The invariant complex measure contributes
the power `n-2`; this is not the real Watson density. Its normalized
log normalizer is

\[
\psi_n(\kappa)=\log\,{}_1F_1(1;n;\kappa).
\]

For positive `kappa`, `1-r` is a Gamma(shape `n-1`, rate `kappa`) variable
truncated to `[0,1]`. At zero concentration, `r~Beta(1,n-1)`; the deterministic
limit is the supplied transported line. Uniform complex orthogonal directions
and phases complete an independent transition sampler.

Writing `a=psi_n'(kappa)`, `b=(1-a)/(n-1)` and `rho=a-b`, invariance gives
`E[P_new|P_old]=rho G P_old G^H+(1-rho)I/n`. Therefore

\[
M^-=\rho G M^+G^H+(1-\rho)I/n
\]

for any previous distribution. A subsequent inverse moment fit preserves
this first moment only. The predictive density is generally a mixture;
its projection into one Bingham density is not exact closure and does not
certify HPD masses or scalar tails. Independent transition sampling and a
low-dimensional full-density reference check this distinction. For complex
`n=2`, the Bloch-sphere overlap is `(1+x dot y)/2`, giving a von Mises--Fisher
kernel on that sphere, not the antipodal real-plane kernel.

The physical moving ULA follows a deterministic one-dimensional steering
curve rather than this invariant process. It has no ground-truth `kappa`.
A passive observation has `E[u|P]=0`; the contact example's nonzero-mean
measurement Jacobian cannot be reused as a passive EKF update.

## Unknown powers and process concentration

Use covariance eigenvalues `a=s+q` and `b=s`, ordered `a>b>0`. The implemented
prior is the globally normalized **joint** density

\[
p(a,b)\propto\operatorname{IG}(a;\alpha_a,\beta_a)
\operatorname{IG}(b;\alpha_b,\beta_b)1\{a>b\}.
\]

It is not equivalent to independent priors on `s,q`, nor to drawing `b`
from its untruncated marginal and then drawing `a` from a normalized
conditional truncation. The latter specification would add a
`b`-dependent normalization to the other conditional. For this joint prior,
its ordering probability is
`I_(beta_a/(beta_a+beta_b))(alpha_a,alpha_b)`, since
`beta_a/a` and `beta_b/b` are independent unit-rate Gamma variables before
ordering.

Given the state path, let
`E_parallel=sum u_t^H P_t u_t` and
`E_perp=sum u_t^H(I-P_t)u_t`. The proper-complex full conditionals are

\[
a\mid b,P,U\sim\operatorname{IG}(\alpha_a+kT,\beta_a+E_\parallel)
\text{ truncated to }(b,\infty),
\]
\[
b\mid a,P,U\sim\operatorname{IG}(\alpha_b+(n-k)T,\beta_b+E_\perp)
\text{ truncated to }(0,a).
\]

Real-model counts and energy increments are halved. At fixed `a,b`, the
measurement parameter is `(1/b-1/a) uu^H`. A stationary experiment uses
**one common projector** with parameter `A_init+sum B_t`. A moving-state
interior full conditional adds both neighboring terms:

\[
A_t=B_t+\kappa G_tP_{t-1}G_t^H
+\kappa G_{t+1}^H P_{t+1}G_{t+1}.
\]

Endpoints include only the existing neighbor; an initial nonuniform prior
would also enter state 1. A length-`T` path beginning at state 1 has `T-1`
transitions. Its concentration conditional is

\[
p(\kappa\mid P)\propto p(\kappa)
\exp\left(\kappa\sum_{t=2}^T\operatorname{tr}(P_tG_tP_{t-1}G_t^H)
-(T-1)\psi_n(\kappa)\right).
\]

The declared default uses IG shapes 3 and scales 3/1 for `a/b`, with a
scale-halving sensitivity control; `kappa~Gamma(shape=2,rate=.04)` has
mean 50, with rate .02 as a sensitivity control. Proper priors and the
ordering restriction are retained at every update. Full-path Gibbs draws
can remain correlated even when each state conditional is sampled exactly;
independent-chain and scalar mixing diagnostics are required. Fixed-lag
approximations, if used, must specify the lag boundary distribution and
transition count rather than presenting them as a full-path posterior.

Integrating powers and states produces a mixture, not an exact single-`A`
recursion. For decisions average joint samples of
`R=bI+(a-b)P`, not `E[b]I+E[a-b]E[P]`; the omitted covariance term is
generally nonzero.

For a stored prior parameter `A`, the pre-correction predictive log density
is the Gaussian constant plus `psi(A+B)-psi(A)`, with all identity shifts
restored. Recorded-prior EM freezes those individual priors while optimizing
its stated objective. Its ascent guarantee needs accurate E-steps for that
fixed objective. Recomputing all past priors under new parameters defines a
different objective; it does not inherit that guarantee. Prospective scores
are evaluated before current fitting and assimilation. Learned-parameter
runs do not inherit the known-power covariance certificate.

## Posterior risk and the beamforming decision

For a derived unit spatial query `v` and any action `b`, projector
idempotence gives the exact identity

\[
\mathbb E_q\|Pv-b\|^2
=v^H(M-M^2)v+\|Mv-b\|^2,\qquad M=\mathbb E_q[P].
\]

The scalar full-vector risk needs `M` only. The full response covariance
needs second projector moments. This posterior identity does not assert
frequentist calibration under a drifting or misspecified model.

For positive joint mean covariance `Rbar`, the distortionless expected-power
minimizer is `w_*=Rbar^(-1)a/(a^H Rbar^(-1)a)`. If `w^H a=1`, then
`(w-w_*)^H a=0` and `Rbar w_*` is a scalar multiple of `a`. Expanding the
quadratic form proves

\[
w^H\overline Rw-w_*^H\overline Rw_*
=(w-w_*)^H\overline R(w-w_*)\ge0.
\]

Conditional posterior excess is not repeated-training excess under the
true covariance, and minimizing expected power is not maximizing expected
SINR or expected log-SINR.

The matched isotropization `M_iso=rho P_mode+(1-rho)I/n`, with
`rho=(tr(P_mode M)-1/n)/(1-1/n)`, preserves mean modal overlap and expected
squared chordal error. It does not use test truth. With one passive snapshot
and a uniform prior, the parameter has only two eigenvalue levels, forcing
its mean to have the same axial form. Hence `M=M_iso`, and the two decisions
agree: any one-snapshot gain over the mode is not evidence of anisotropy.
Multiple snapshots can generate nontrivial empirical complement spectra
despite isotropic population information.

Pointwise scalar scanning quantiles, a state HPD envelope and the actual
weight beampattern are distinct quantities. A common state-region inclusion
implies enclosure of all evaluated scan responses; the converse fails.
Under the ideal stationary passive state-coverage assumptions proved above,
that same state-inclusion event also gives at least the stated frequentist
enclosure probability for the common-region envelope at one fixed time.
Outside those assumptions neither the envelope nor a collection of marginal
intervals automatically has a frequentist simultaneous confidence guarantee.
Marginal intervals do not inherit the common-state event. HPD-versus-ball
regional leakage comparisons use a fixed downstream decision unless a
separate robust optimization has actually been implemented.

## Primary references

- Besson and Abramovich, *On the Fisher information matrix for multivariate
  elliptically contoured distributions*, 2013:
  <https://arxiv.org/abs/1306.6415>. The complex Gaussian trace identity is
  specialized above with an explicit real-coordinate convention.
- Koltchinskii and Lounici, *Concentration inequalities and moment bounds for
  sample covariance operators*, 2014: <https://arxiv.org/abs/1405.2468>.
  The elementary weighted, time-varying net proof used here is given in full;
  an iid stationary result is not substituted for its assumptions.
- Koltchinskii and Lounici, *Asymptotics and concentration bounds for bilinear
  forms of spectral projectors of sample covariance*, 2015:
  <https://arxiv.org/abs/1504.07333>. The displayed eigenspace implication is
  also proved directly by its eigenvector equation.
