# GREAT reference implementation

The shared implementation is [benchmarks/common/great.py](../benchmarks/common/great.py). It implements the measured-vector finite-gradient construction specified in the final simulation guide, citing GREAT, arXiv:2412.09052v2, Algorithm 1 and Sections IV-B/C; DOI 10.1109/TAC.2025.3636986. It is independently written and does not reuse or relabel GROUSE.

## Mathematical update

For an orthonormal representative X and an unnormalized covariance C of the optimization dataset, the real objective is

    F(X) = tr(C) - tr(X.T C X),
    grad F(X) = -2 (I - X X.T) C X.

Every measured vector updates C and warm-starts a declared finite number of Grassmann exponential-map steps from the preceding estimate. A tangent SVD evaluates the exponential; the implementation does not compute a covariance eigensystem. Finite steps do not generally reach the exact leading eigenspace.

The sliding-window update adds the new outer product and removes the oldest outer product. Exponentially weighted GREAT is a separate memory choice, C_t = beta C_(t-1) + u_t u_t*. Its data-column weight is gamma**age with gamma = sqrt(beta), so its covariance weights match beta**age. There is no hidden covariance/window normalization. A fixed gradient step size therefore has the units of the raw objective and is selected jointly with memory and the number of steps using independent validation episodes.

The separately labeled **GREAT, complex extension** replaces transposes by conjugate transposes and applies the complex Grassmann exponential in native complex coordinates. A real block embedding is not used. The real restriction is checked directly.

Supplied orthogonal/unitary coordinate changes transform the estimate, covariance and every stored sliding-window observation before adding/removing measurements. Transporting the estimate alone would optimize a different historical dataset. In primary unknown-motion beamforming this option is absent; measured wrist motion is an available coordinate input in the contact comparison.

## Inputs, cost and diagnostics

The class receives only measured u and optional supplied G. It does not receive hidden signal coefficients, true subspaces or contact requests v. It is consequently a response-only contact reference, unlike the known-input conditional estimator. Gains in that comparison do not isolate posterior uncertainty.

For rank k, each covariance correction is quadratic in ambient dimension, each gradient uses C X, and the exponential uses an n-by-k SVD. Sliding-window storage also retains the active observations. The optional objective monitor is disabled by default so the timed algorithm does not perform an additional final covariance product just for logging. When disabled, diagnostics report descent violations as unavailable, not zero. When enabled, increases remain recorded; there is no unreported line search.

## Verification

The twelve parameterized checks in [tests/test_great.py](../tests/test_great.py) cover:

- Direct residual-loss directional derivatives in real and complex coordinates.
- Agreement with an independent ambient skew-Hermitian matrix exponential.
- Basis invariance, orthonormality, rank-deficient/zero tangents and real restriction.
- Finite-step descent at a declared small step, while explicitly forbidding covariance eigensolves.
- Add-new/remove-old window covariance against independent reconstruction under noncommuting coordinate changes.
- Exact EW covariance weights against data-column gamma weights, including rejection of gamma = beta.
- Invalid fields, memory controls and coordinate transformations.

These are numerical/geometric checks of the implemented update, not a claim that every validation-grid step size descends or that finite steps equal modal optimization.

## Primary-source accessibility and provenance

The guide links the paper at https://arxiv.org/abs/2412.09052v2 and author code at https://gitlab.ethz.ch/asasfi/ST_for_sysID. Direct attempts to retrieve the paper PDF and author repository were made from the managed environment. The default sandbox could not reach its inherited proxy; approved retries outside the sandbox received CONNECT 403 from the managed network policy for both original hosts.

The author code has not been copied. Direct parity with its code and exact primary-PDF wording must remain unverified until an accessible source is supplied or those domains become available. The mathematical specification used here is the explicit objective, gradient, finite exponential steps, sliding update and EW weighting quoted in the supplied final simulation guide. Public allowed-host mirror checks and any subsequent provenance resolution are recorded in the final completion report; an inaccessible source is not reported as reviewed.
