# Higher-dimensional numerical feasibility

This fixed-budget check does not implement or validate a seven-joint robot. The synthetic parameter spectra are concentration stress tests, not matched input-information experiments.

Gr(6,7) uses an exact mathematical complement to a rank-one normal law. Its deterministic series has a finite truncation budget; tail calculations still use independent Markov chains. The new three-dimensional radial/azimuthal integration cannot be reused unchanged in seven dimensions.

Gr(3,7) has twelve local coordinates. Its mean and tails use the actual general-rank sampler. General-rank process moment inversion remains unsupported. No rank-one shortcut or process scalability claim follows.

| Rank | Span | Sampling time (s) | Minimum scalar ESS | Tail MCSE / width | Diagnostics |
| --- | ---: | ---: | ---: | ---: | --- |
| 6 | 8 | 0.53 | 143.0 | 0.055 | Unresolved |
| 6 | 200 | 0.34 | 24.8 | 0.124 | Unresolved |
| 6 | 2000 | 0.45 | 29.9 | 0.138 | Unresolved |
| 6 | 80000 | 0.38 | 11.5 | 0.358 | Unresolved |
| 3 | 8 | 1.19 | 213.1 | 0.035 | Pass |
| 3 | 200 | 1.01 | 18.7 | 0.090 | Unresolved |
| 3 | 2000 | 1.01 | 18.4 | 0.095 | Unresolved |
| 3 | 80000 | 0.96 | 17.5 | 0.098 | Unresolved |

These diagnostics bound neither mixing bias nor all possible tail errors. A favorable result here would still require mechanically generated motion, known-kinematics baselines, isotropic-noise assumptions, shared noisy encoders, rank/constraint/joint-limit checks, and normalized input/Fisher comparisons before a physical arm study. The current study stops at numerical feasibility and makes no higher-dimensional calibration claim.

Known orientation constraints also reduce the unknown family: if the columns of B span ker(J_o), the joint-velocity space is B ker(n_x^T J_pos B). At a fixed configuration, the unknown surface normal n_x still has only two orientation coordinates. An unstructured Gr(3,7) benchmark therefore cannot treat all twelve Grassmann coordinates as independent unknown physical geometry when every method has access to the same known kinematics.
