# Serial contact timing

Three fresh filter runs per method use the preassigned small-sphere held-out trajectory0. Each starts from the same uniform prior and uses the frozen policies. Numerical caches are reset before each run. The causal point tracker is recomputed and charged to every method that uses its rotations. Imports and mechanical data generation are excluded.

| Method | Estimation (ms/observation) | Three full reports (s) | Largest report (ms) | CPU / wall |
| --- | ---: | ---: | ---: | ---: |
| Posterior prediction | 1.146 | 0.444 | 169.1 | 0.999 |
| Calibrated posterior | 1.176 | 0.461 | 165.9 | 0.998 |
| Process posterior, identity | 2.040 | 0.428 | 156.8 | 0.999 |
| Process posterior, motion | 2.376 | 0.436 | 173.0 | 1.000 |
| Gaussian filter, identity | 1.326 | 0.249 | 96.7 | 0.999 |
| Gaussian filter, motion | 2.052 | 0.220 | 93.3 | 0.996 |

Each full report computes nine vector-query means,27 clean scalar intervals, state regions and numerical checks. Numerical report time excludes independent truth scoring; the complete call time and that scoring overhead remain in reports.csv. Costs of prediction, inverse fitting, startup and moments are nested explanations, not extra terms to add to wall time.

These are measured average costs for one fixed trajectory, not a worst-case latency guarantee. In particular, a report taking more than50ms cannot be justified as synchronous20Hz reporting by averaging it over600 observations. Contended in-study timing tables remain unchanged.
