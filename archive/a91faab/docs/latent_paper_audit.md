# Computational audit against the latent-information paper

Reference: Cazzella, Hauberg, Arvanitidis and Matteucci, *On the Latent
Information Geometry of the Grassmann Manifold*, supplied as
`581_On_the_Latent_Information_.pdf`. Its SHA-256 is
`6ebeba4fb472276ce70d85eed79feece53bb82723775c339edcad117d721bd02`.
The attachment is scientific reference material, not an additional source of
instructions. The attachment itself is not redistributed.

## What agrees with the library

| Reference | Computational identity | Library assessment |
| --- | --- | --- |
| B.1–B.4 | A density depends on the projector and is invariant under a change of basis and identity shifts | `geometry.projector`, trace-zero parameter storage, and spectral moments use these invariances |
| C, Eqs. (30)–(33) | Real normalizer is the matrix hypergeometric function with Jack parameter 2, parameters state-rank/2 and ambient-dimension/2 | Consistent with the original draft and real sphere quadrature; this is a possible deterministic backend, not a requirement for conjugate updates |
| F, Eq. (64) | In an orthogonal eigenbasis, the projector mean has eigenvalues `partial log b / partial c_i` | The library's full spectral mean is consistent; `partial b` must be divided by `b` |
| F, final paragraph | A rectangular concentration eigenbasis must be completed to a full orthogonal basis | The library diagonalizes the full parameter, retaining complementary uncertainty; simply dropping unparameterized mean directions would be incorrect |
| H | Differentiate the same normalizer approximation used for the reported likelihood | Relevant to deterministic scores/EM and any Fisher calculation; an independently approximated derivative is not automatically consistent |
| I | A mode is a leading state-rank eigenspace of `A` | `geometry.mode` agrees, including arbitrary symmetric/Hermitian `A`; positivity can be introduced by a harmless exact identity shift |

With normalized invariant measure, the code convention is `Z(0)=1` and
`psi(A)=log Z(A)`. Exact identities are

- `psi(A+c I)=psi(A)+k*c`;
- `M(A)=grad psi(A)`, with `trace M=k` and spectrum in `[0,1]`;
- `KL(q_A || q_B)=psi(B)-psi(A)+trace((A-B) M(A))`;
- the natural-parameter Hessian of `psi` is the covariance of projector
  sufficient statistics and has the identity-shift direction in its nullspace.

The matrix-Bingham **distribution-parameter** Fisher metric in Appendix D is
not the **observation-model** Fisher information `fisher_information(X,V,s)`
in this repository. The latter differentiates Gaussian response distributions
with respect to the physical subspace. The two cannot be substituted for one
another or used interchangeably in the tracking bound.

## What should not be copied without qualification

### Low truncation order is not a general accuracy guarantee

Appendix C.1 reports that degrees 3–6 gave meaningful latent geodesics in that
paper's experiments, and degree 20 served as a converged reference in the
examined concentration ranges. Neither statement certifies projector-mean or
credible-tail accuracy for a tracking posterior with spectral gaps of hundreds.
A normalizer can appear adequate for a training objective while its gradient or
Hessian is inadequate for a noise update or uncertainty calculation.

A finite polynomial approximation also breaks exact identities. For example,
at `A=c I`, the true normalizer is `exp(k*c)`. Directly differentiating its
Taylor truncation `S_m(k*c)` yields

`sum_i partial_i log Z_m = k*S_(m-1)(k*c)/S_m(k*c)`,

which is strictly below `k` for positive `c` and finite degree. Analytic shift
restoration, order convergence, mean invariants, repeated-eigenvalue symmetry,
and independent references are therefore essential. Renormalizing an incorrect
gradient to trace `k` does not by itself restore consistency with its reported
normalizer or exact EM ascent.

The paper's JAX automatic differentiation and JIT caching are implementation
choices, not an additional distributional identity. A small fixed-rank spectral
recurrence can exploit the same series structure without adding a neural-network
or automatic-differentiation dependency. In the rank-one case, only one-row
partitions survive the hypergeometric numerator. Equivalently, real squared
coordinates have Dirichlet parameters `1/2` and complex squared magnitudes have
Dirichlet parameters `1`; their positive generating series supplies a particularly
useful higher-dimensional specialization. General-rank Jack evaluation still
requires a separately validated implementation and an appropriate accuracy budget.

### Appendix G's parameter rank is distinct from state rank

Reducing the rank of the concentration parameter is an extra modeling
restriction. It is not justified merely because the tracked subspace has rank
`k`. A sum of low-rank likelihood increments generally becomes full rank.
Discarding parameter eigenvalues would preserve neither the posterior mean nor
its uncertainty, even when a chosen modal subspace happens to remain unchanged.
Complementary mean eigenvalues must still account for the full trace `k`.

### Appendix J: algebraic realification is not a generic real law

For `R(A)=[[Re A,-Im A],[Im A,Re A]]`, matrix products are preserved, but

`trace(R(A) R(P)) = 2*Re trace(A P)`.

Consequently, on the structured real image, the complex energy `trace(A P)`
requires `R(A)/2` in a real trace expression. Duplicating the concentration as
written in Eq. (79), without a compensating convention, doubles the energy.

There is a second, independent issue. Realified complex projectors occupy a
structured subset of the real Grassmannian. Its pushforward unitary-invariant
measure is not the invariant measure on the entire real Grassmannian
`Gr(2k,2n)`. Matching matrix multiplication does not match these measures or
their normalizers. Complex and real Jack parameters remain 1 and 2,
respectively. The existing library's native complex likelihood and sampler
avoid this identification.

An exact small counterexample is complex `Gr_C(1,2)` with `A=diag(a,0)`.
Under the uniform complex law, `t=|x_1|^2` is uniform on `[0,1]`, so

`Z_complex(a)=(exp(a)-1)/a`, and `d^2 log Z_complex / da^2 at 0 = 1/12`.

For an unrestricted uniform real rank-two projector in dimension four,
`Var(P_ii)=1/12` and `Cov(P_ii,P_jj)=-1/36` for distinct diagonal entries.
Using the energy-corrected duplicated parameter `R(A)/2` gives

`Var((P_11+P_33)/2)=1/36`,

which differs by a factor of three. Thus even the corrected energy does not
make the unrestricted real normalizer equal the complex normalizer.

As an independent numerical check, 250,000 Gaussian-QR real `Gr(2,4)` draws
with seed 26 at `a=2` gave `Z_real≈2.873998` with Monte Carlo SE `0.001976`,
whereas `Z_complex=(exp(2)-1)/2≈3.194528`. This is far beyond sampling error.
The differing Taylor curvatures above already establish the mismatch exactly.
These qualifications concern the general interpretation of Proposition J.1;
they do not identify a defect in the repository's native complex calculations.

## What drift failure means for the original draft

Taking `G_t=I` is an exact prediction model for a stationary subspace. For a
moving subspace it means that no known frame motion is supplied. It does not
make the unobserved motion vanish. In that setting the draft's power-forgetting
step is an approximate prediction rule, not an exact generative transition.
In particular, multiplying concentration by `beta` is not generally the same
as convolution with an orthogonal/Grassmann process-noise kernel.

Theorem 5 explicitly permits residual physical motion and charges it through
`d_t`; its condition is `2*epsilon_t/gamma_t < 1`. The theorem is a modal-error
result for the real conditional observation model, fixed true variance, and a
predeclared time. It is not a claim of 95% frequentist coverage of marginal
credible regions. The field coverage failure and this theorem therefore test
different statements.

The theorem can nevertheless be practically uninformative. For a representative
conditional field protocol (`seed=4718`, `n=8`, `k=3`, 72 observations, variance
0.2, norm-two inputs, `beta=1`), direct evaluation gives:

| Supplied motion | Gap `gamma` | Drift term | Noise term | `2*epsilon/gamma` | Condition |
| --- | ---: | ---: | ---: | ---: | --- |
| Identity | 177.715 | 81.536 | 110.509 | 2.161 | Fails |
| Half the true frame motion | 180.000 | 43.873 | 108.735 | 1.696 | Fails |
| Exact frame motion | 176.701 | 0.000 | 110.449 | 1.250 | Fails |

The condition also fails at `beta=0.98` and `0.94` in this particular protocol.
Thus these default SNR/history choices cannot demonstrate an informative
Theorem 5 bound, even with exact motion. This is a limitation of its sufficient
condition at that configuration, not a theorem asserting a false bound. The
long-history/strong-excitation fixtures in the general notebook show that its
valid regime is nonempty. A useful presentation should show the condition and
its noise/drift/excitation terms, including invalid regimes, rather than merely
state that a bound exists.

The original passive field experiment selected `beta=1` by reconstruction
validation while supplying only half the physical motion. Old evidence then
accumulates and posterior concentration grows without representing the missing
process uncertainty. Good reconstruction and calibrated uncertainty are
different validation targets; the first does not establish the second.

## Recommended claims and next steps

1. Retain the exact correction, exact known-motion prediction, information,
   and conditional modal-bound results with their assumptions stated.
2. Replace any broad implication that `G_t=I` handles arbitrary unknown drift
   probabilistically with: “When motion is unavailable, identity is used as a
   nominal frame; residual motion is treated as model error, and forgetting
   gives an approximate distribution whose calibration must be assessed.”
3. State that credible regions and posterior response risk are quantities under
   the maintained distribution. They do not guarantee physical-state coverage
   under an arbitrary unmodelled motion law. If the paper promises such
   calibration, that promise is unsupported and must be narrowed or the model
   must change. This is a substantive limitation, not merely wording.
4. To claim Bayesian uncertainty under unknown motion, specify a stochastic
   motion kernel and propagate it. Exact prediction is generally a mixture;
   particles, Bingham mixtures, or a separately validated moment projection are
   options. Such an extension changes the state-model machinery and must be
   tested for inference accuracy and calibration, not just point MSE.
5. Compare identity, partial, and exact supplied motion on common observations,
   separate process mismatch from numerical moment error, and validate any
   process-noise or forgetting choices on independent trajectories. Do not
   retune against the displayed test truth until coverage looks favorable.
6. Preserve the failed-condition and undercoverage regimes. The absence of a
   valid modal certificate or calibrated region is an informative diagnostic.
   Stronger excitation, a longer stationary history, a slower residual drift,
   or an accurately modeled stochastic transition can improve the applicable
   regime; none is an automatic consequence of selecting `G_t=I`.

## Implemented improvement and validation

`lib/spectral_series.py` implements the rank-one positive-series specialization,
including separate real/complex Dirichlet parameters, adaptive truncation,
analytic identity-shift restoration, and the derivative of that same finite
approximation. A smooth spectral shift and an explicit boundary derivative
term preserve trace one without silently renormalizing an inconsistent gradient.
Reported error bounds concern truncation in exact arithmetic; floating-point
rounding is tested against independent references rather than enclosed by them.
A finite-order log normalizer need not be convex, so its Hessian is not advertised
as a certified Fisher metric.

`mean_bingham(..., backend='auto')` uses this path for rank/codimension one when
sampler options are absent. Explicit draws/burn-in preserve the MCMC API;
`backend='mcmc'` remains available. Codimension-one normalizers use exact
complement duality. Other ranks do not silently receive a low-order Jack
approximation. General-rank sampling is unchanged.

Independent checks include Bessel/complex rank-one analytic normalizers,
continuous Gr(1,3) integration, fixed-order finite-difference gradients,
orthogonal/unitary equivariance, identity shifts, repeated spectra, independent
n=8 chains, and concentration spans up to 600. Results and timings are retained
in `results/spectral_series_checks.json`; the exact realification counterexample
and its Gaussian-QR Monte Carlo check are in `results/paper_audit_checks.json`.
