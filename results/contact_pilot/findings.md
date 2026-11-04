# Scope decision: Physical example only

The mechanics and inference numerics work, and the primary task has a modest gain over pooled scalar shrinkage. The improvement over its own mode is tiny and curved-guide credible uncertainty is unreliable. Keep this as a physical conditional-response illustration, pending a stronger sensing/dynamics comparison.

The physically generated force/velocity data fit the existing formulation under known isotropic resistance.
This supports a maintained bilateral-contact example. It does not establish frictional/unilateral contact
inference, a contact controller, safe velocity commands, or practical superiority over encoder-based estimators.

## Task-loss evidence

The primary condition is the balanced smaller-radius guide at observation 600. Each of 64 test trajectories
contributes one squared three-vector response error averaged over nine hypothetical queries.
Memory is shared across the curved validation mixture; scalar shrinkage is continuously optimized on that mixture.
Both primary references were specified before testing. Other pointwise intervals are exploratory.

| Primary reference | Reference MSE | Mean-prediction MSE | Relative reduction | Paired gain 95% interval |
| --- | ---: | ---: | ---: | --- |
| Mode | 1.08432e-06 | 1.08252e-06 | 0.166% | [1.55e-09, 2.059e-09] |
| Scalar shrinkage | 1.12555e-06 | 1.08252e-06 | 3.824% | [3.993e-08, 4.614e-08] |
| Ridge map | 0.000234283 | 1.08252e-06 | 99.538% | [0.0002243, 0.0002421] |

The 3.8% advantage over pooled shrinkage is supported in this frozen primary comparison.
Its roughly 0.17% advantage over its own mode is statistically detectable but small in practical terms.
The large ridge/PCA advantages mainly demonstrate projector structure and known-force information;
they do not isolate a benefit of maintaining a full distribution. Larger-radius and weak-input gains
over mode/shrinkage have intervals containing zero. In the anisotropic stress, mean prediction is worse
than tuned scalar shrinkage. All outcomes remain in [paired.csv](paired.csv).
The selected curved beta=0.8 is the lowest tested memory value. These are finite-family results,
not evidence that no better point-memory policy exists.

## Uncertainty: useful in the static control, unreliable under curvature

| Terminal condition | State-ball coverage | HPD coverage | Clean component coverage | Mean scalar width / v0 |
| --- | --- | --- | ---: | ---: |
| Plane | 63/64 | 63/64 | 95.0% | 0.0061 |
| Sphere, R=0.20 m | 64/64 | 64/64 | 98.5% | 0.1297 |
| Sphere, R=0.05 m | 42/64 | 23/64 | 66.4% | 0.0665 |
| Weak input program | 61/64 | 42/64 | 86.0% | 0.1569 |
| Anisotropic resistance | 0/64 | 0/64 | 6.3% | 0.0818 |

The static-plane state regions cover 63/64 trials; their Wilson intervals include 95%.
Mean clean component coverage is 95.0%. This is a clean matched-model control, not a drift guarantee.
Under small-radius motion, HPD coverage 23/64 and clean coverage 66.4% are far below nominal 95%.
The weak-input ball covers 61/64, but its HPD and clean-response summaries still under-cover.
A radius alone can conceal directional model error.

For a rank-two projector in three dimensions, each queried scalar component has full physical range
width v0. Normalized widths are therefore fractions of that range. The small-sphere interval width
is only about 6.6% of the possible range, yet it misses often: the issue is overconfidence, not
nearly uninformative intervals. No noisy-observation intervals are evaluated.
Coverage replicates are trajectories; 27 component indicators are averaged within each trajectory.
See [coverage.csv](coverage.csv) for Wilson/clustered intervals and [query_coverage.csv](query_coverage.csv) for individual components.

| Condition | Posterior full-vector risk | Empirical full-vector MSE | Empirical / posterior risk |
| --- | ---: | ---: | ---: |
| Plane | 3.331e-09 | 3.496e-09 | 1.05 |
| Sphere, R=0.20 m | 3.239e-06 | 2.167e-06 | 0.67 |
| Sphere, R=0.05 m | 4.188e-07 | 1.083e-06 | 2.58 |
| Weak input program | 6.986e-06 | 7.602e-06 | 1.09 |
| Anisotropic resistance | 8.158e-07 | 5.101e-05 | 62.52 |

Risk is informative under the stationary model, but it understates primary small-sphere error by
about 2.6 times and anisotropic physical-response error by about 62 times. Average weak-input risk
can be close while directional HPD/interval coverage is poor. Moments and tails answer different questions.

## Mechanics, numerics and the theorem

All 960 reports pass separate mean, state-region and response-interval resolution checks; no MCMC or noise learning is used.
The independent KKT/projector velocity discrepancy is at most 1.16e-17 m/s in matched conditions.
The full-horizon dt/2 checks change velocity by at most 7.24e-08 measurement-noise standard deviations.
The continuous-CDF quadrature extension passes independent pole/ridge references through concentration 1e6.
Existing core library files remain unchanged; this is a dimension-specific deterministic numerical extension.
389/768 applicable fixed-time modal certificates satisfy their sufficient condition, with 0 violations.
Invalid conditions remain in [bounds.csv](bounds.csv); these are not simultaneous-time guarantees.
Anisotropic resistance breaks the scalar response law and is excluded from the theorem.
Its response errors use actual independently solved KKT query velocities; geometric targets are retained separately.
Known noise and a correct numerical posterior cannot repair a wrong state or observation model.

## Computation and scope

In the primary case, cumulative correction time averages 0.089 s over 600 observations.
The three full posterior/interval reports together average 0.511 s; query decisions add 2.11e-05 s.
A single full report takes about 0.16 s, exceeding the 0.05 s observation step. This implementation does
not establish a 20 Hz full-interval controller. Mean-only evaluation is cheaper; clocks retain their actual frequencies.
These cloud wall-clock measurements are not dedicated-machine speed or asymptotic guarantees.

The pilot supports using the robot setting as a physical illustration of the conditional framework.
It does not yet justify a broad calibrated-contact claim or a large Bayesian decision advantage over its own mode.
The mean map E[P] is generally full rank: it predicts an expected response and is not one physically
admissible contact projector or a guaranteed feasible velocity command.

After this bounded test, the next practical comparison should include the same noisy encoder positions
for every applicable method, with a causal local-plane control. A later dynamics/observation model must
address curvature and uncertain resistance before stronger uncertainty claims. No such machinery is added here.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_contact.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_contact_posterior.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_contact_pilot.py --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_contact_mechanics.py
python scripts/summarize_contact_pilot.py
python scripts/make_contact_notebook.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/execute_notebook.py notebooks/03_contact.ipynb
python scripts/plot_contact_pilot.py
python scripts/write_contact_manifest.py
```

[Physical paths](figures/physical_setup.pdf), [paired task losses](figures/paired_losses.pdf), and
[coverage versus width](figures/coverage_width.pdf) are the main figures. The executed
[contact notebook](../../notebooks/03_contact.ipynb) contains the assumptions, interpretation and supplementary diagnostics.
The [pilot specification](../../docs/contact_pilot_spec.md) is preserved. Historical results and the field notebook are unchanged.
