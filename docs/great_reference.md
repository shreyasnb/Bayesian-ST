# GREAT implementation comparison

The comparison uses the user-designated [GeRoST repository](https://github.com/shreyasnb/gerost)
at commit `a5c47c07a3894fe21641a84537741afd1c8be604`:

- [`great.m`](https://github.com/shreyasnb/gerost/blob/a5c47c07a3894fe21641a84537741afd1c8be604/great.m)
- [`utils/gradf.m`](https://github.com/shreyasnb/gerost/blob/a5c47c07a3894fe21641a84537741afd1c8be604/utils/gradf.m)
- [`case2/stlibrary/algorithms/st/GREAT/run_alg.m`](https://github.com/shreyasnb/gerost/blob/a5c47c07a3894fe21641a84537741afd1c8be604/case2/stlibrary/algorithms/st/GREAT/run_alg.m)

The gradient, descent sign, raw sliding-window objective and Grassmann
exponential already agreed. The Python implementation now also waits until
at least `k` samples are present, supports the reference's adaptive step size,
and defaults to five inner steps.

| Reference control | Python control | Behavior |
| --- | --- | --- |
| `T` | `window` | Retain the latest measurements, including the current one |
| `K` | `inner_steps` | Five steps by default |
| `alpha` | `stepsize` | Positive fixed value or `None` for adaptive selection |
| `initialize(U0)` | `initial_basis` | Common orthonormal initial estimate; empty online history |

For adaptive selection, the canonical class uses
`1 / max(4 * sigma_max(W)**2, 1e-8)`. Python obtains the same largest squared
singular value from the raw covariance and rebuilds that covariance from the
active window to avoid cancellation when old high-energy samples leave.
The video adapter instead applies its `1e-8` floor before multiplying by four;
its behavior therefore differs at extremely small signal energy. Python follows
`great.m` for that boundary.

Python retains bounded history and conditional orthogonality repair. Manopt
reorthonormalizes every exponential output; the resulting basis signs or
rotations can differ while representing the same subspace. Comparisons use
subspace projectors. Exponential weighting and measured coordinate transport
remain explicit Python extensions. Proper-complex inputs use conjugate
transposes; this does not attribute a native-complex tracking guarantee to the
real reference.

## Verification

The unchanged MATLAB class and gradient were executed with GNU Octave 9.4.0
and Manopt commit `6c369671d8834aa28f7e222c7d89c5f216e6953d`, pinned by
the reference repository. Eleven cases cover fixed/adaptive/default controls,
multiple ranks, startup, short windows, window rollover, zero and tiny signals,
repeated spectra, and complex inputs. The largest observed absolute projector
difference was `5.55e-15`.

`tests/test_great_upstream.py` reads compact source-generated fixtures. Their
inputs, source hashes, runtime and regeneration procedure are recorded in
`tests/fixtures/gerost_great_provenance.json`. Normal tests need neither MATLAB
nor Octave.

The paper's explicit rank-one, fixed-step configurations remain unchanged.
One complete simulation per stationary dimension, moving condition and noise
experiment was rerun at the paper horizons; all saved powers, losses, scores
and variances checked for those simulations matched exactly. The full paper
cohorts were not rerun for this change.

This verifies the GitHub baseline selected for this comparison. The separate
original GitLab repository has not been inspected.
