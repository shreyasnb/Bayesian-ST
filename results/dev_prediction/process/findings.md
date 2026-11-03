# Explicit-process pilot findings

Commands: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_process_prediction.py` and `python -m pytest tests/test_process_prediction.py -q`.

This is a real Gr(1,2) assumed-density experiment, not general-rank process filtering. The transition parameter kappa is known in these simulations; it is not learned. The primary comparison is projected Bingham inference against full-density Fourier inference under the same process and observations. Identity and fixed beta=0.85 forgetting are diagnostic controls, not validation-selected application competitors. All variances are fixed at their true value 0.12 in the primary comparison.

Each of the two preselected matched-process regimes has 200 independent trajectories and terminal time 24. The four mismatch laws have 40 trajectories each and are exploratory. Initial states are uniform, sensor-frame input cycles are fixed independently of the state, and coverage intervals are pointwise Wilson intervals.

| Kappa | Method | Ball | HPD | Clean response | Noisy response | Mean density TV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Full-density reference | 0.950 | 0.955 | 0.960 | 0.955 | 0.0000 |
| 4 | Identity | 0.250 | 0.250 | 0.360 | 0.800 | 0.7695 |
| 4 | Power forgetting | 0.620 | 0.620 | 0.730 | 0.900 | 0.4855 |
| 4 | Projected transition | 0.950 | 0.950 | 0.965 | 0.955 | 0.0236 |
| 20 | Full-density reference | 0.965 | 0.960 | 0.965 | 0.970 | 0.0000 |
| 20 | Identity | 0.305 | 0.305 | 0.375 | 0.885 | 0.7036 |
| 20 | Power forgetting | 0.765 | 0.765 | 0.825 | 0.955 | 0.3193 |
| 20 | Projected transition | 0.960 | 0.960 | 0.970 | 0.965 | 0.0227 |

These finite results are consistent with nominal coverage for the projected law in the tested matched low-dimensional regimes. They do not prove general calibration, exact Bingham closure, or accuracy in arbitrary dimensions. Full-density and projected estimates retain their individual misses.

The independent fixed-prior reference has a first-moment error of 8.38e-16 and inverse-fit residual 4.16e-17, but its projected prediction differs from the actual mixture by TV 0.02833; after the same correction, TV is 0.02612. Thus exact moment matching is not exact density prediction. Grid refinement from 1024 to 4096 gives density-TV changes below 1e-6, and the region thresholds and masses are retained separately in reference_checks.json.

The trace-gauged inverse fit uses stable exact Bessel formulas on the real circle; native real/complex rank-one and codimension-one inverse fits are tested against the existing positive spectral series. General-rank inversion is explicitly rejected as unresolved. The deterministic limit and finite-kappa approach to it, uniform transition, unitary equivariance and independent real/complex transition draws are checked.

Under observation contamination, both the projected and full-density process filters have terminal ball/HPD coverage 0.80 across 40 exploratory trajectories. Correct process prediction does not repair a wrong likelihood. The deterministic slow/fast/turn outcomes remain in the same tables; point error and uncertainty are separate outcomes.

The separate 20-trajectory noise ablation uses complete eight-observation windows of frozen pre-correction priors and the existing exact-mean EM safeguard. It checks only the frozen-prior surrogate objective, not ascent of the approximate full state-space likelihood. Estimates remain variable; no noise-consistency claim follows.

Cost fields are cumulative at reporting times. Prediction includes moment evaluation and inversion; inversion_seconds is nested and must not be added again. Standalone moment timing is available only for the full-density reference. Other moment timings are missing rather than zero. Region integration is measured separately.

The complete bounded pilot took 55.83 seconds on x86_64. Every trajectory, miss, width, numerical discrepancy, innovation and fixed-window noise fit is retained in CSV tables.
