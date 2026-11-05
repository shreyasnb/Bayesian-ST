# Contact prediction and uncertainty

This isolated study follows the practical mean-policy selection in `../mean/`. The executed walkthrough is `notebooks/03_contact.ipynb`; the tables here retain the detailed controls without placing every sweep in that walkthrough. Historical contact-pilot results are unchanged.

Each trajectory contains 600 observations of known force and noisy instantaneous velocity. True normals, positions and resistance enter simulation and scoring only. Sensor variance remains `1e-6 (m/s)^2` for every primary method. Plane policies are selected separately; large sphere, small sphere and forces mostly along one axis share a curved validation policy. Anisotropic resistance is a held-out model-mismatch stress and never selects policies.

| Method ID | Meaning |
| --- | --- |
| `raw` | Distribution maintained by the practical mean policy chosen in Stage 1. |
| `temperature` | The same raw recursion and modal plane, reporting `q_(A/T)` with constant temperature selected by validation clean-response interval score. This calibrates spread and does not estimate sensor noise. |
| `process_identity` | Bingham assumed-density filter with identity nominal motion and validation-selected transition concentration. |
| `process_motion` | The same filter using the exact causal rotations from the Stage-1 velocity predictor. |
| `gaussian_identity` | Recursive two-coordinate normal-line Gaussian filter with identity nominal motion. |
| `gaussian_motion` | The same Gaussian filter given exactly the same causal rotations as `process_motion`. |
| `*_laplace` | A report-only exponential-map local Gaussian approximation to the same Bingham parameter, isolating posterior shape. It has no measured all-step mean trajectory. |

Process and recursive-Gaussian concentration are selected independently by mean pre-correction predictive negative log density, including the true Gaussian noise constants. They receive the same candidate transition family. The Gaussian process covariance is the exact kernel angular second moment per coordinate, but its local process and linearized observation update remain approximations. Exact Bingham predictive first moments do not make the fitted predictive density exact. Learned rotations, fitted temperatures and physical sphere motion do not inherit the fixed-state/known-transformation coverage theorem.

All reported point controls use the same modal estimate as their associated method. `regularized_mse` evaluates `rho*P_hat*w`; `rho` is minimized continuously in `[0,1]` on validation. It is the original scalar-rescaling comparator, not a new mixture with an isotropic map. Its selection uses terminal clean-response MSE, so time-average results are secondary evaluations of that same frozen coefficient.

`preregistered.json` freezes sources, seeds, environment versions and Stage-1 selection. `selected.json` is written before any new uncertainty test tasks; `test_preregistered.json` also freezes the numerical gate and selected-file hash. `validation_trials.csv` and `tuning.csv` retain all candidates and one predeclared boundary extension. `selected_validation.csv` retains interval diagnostics for the NLL-selected process controls without using those diagnostics to change the selection.

The primary task is clean vector-response MSE over the original nine force directions. Queries form a tight frame, allowing an exactly equivalent response-map loss at every step. Queries, coordinates and repeated times are averaged within a trajectory, never counted as independent replicates. `paired.csv` uses paired trajectory-level intervals; primary and exploratory contrasts are marked explicitly. MSE in `(m/s)^2` is multiplied by `1e6` to obtain `(mm/s)^2`.

`metrics.csv` reports observations 60, 180 and 600; `components.csv` preserves their 27 scalar predictions. A report index `t` is one-based; its physical observation time is `(t-1)*0.05 s`. The preassigned trajectory-0 archives in `trajectories/` preserve all-step states and prediction losses. `story.csv` uses that same trajectory, query 1 (world y-axis) and a fixed 30-observation report schedule. These illustrations are not additional Monte Carlo replicates or a continuous-path coverage claim.

Numerical integration and local model validity have separate flags. `mean_ok`, `state_ok`, `interval_ok`, and `integration_numerical_ok` concern numerical resolution. A mapped Gaussian law can integrate accurately while `local_approximation_ok` is false because more than 1% of its unwrapped tangent mass crosses the projective cut locus. `numerical_ok` then remains false and the report remains visible. The Gaussian local ellipsoid is not an invariant-density Grassmann HPD region. Never combine those two region definitions in a single unnamed “HPD coverage” claim.

Cost columns are cumulative through the report time. `inverse_seconds` is nested in process prediction; deterministic transitions perform no inversion. Gaussian warm-up fits are included in prediction/update wall time, with their separate inverse timer unmeasured (`NaN`). `warmup_seconds` is nested, not additional. `filter_wall_seconds` includes estimation, predictive scores and maintained means. `end_to_end_seconds` additionally includes the external point tracker required to produce the common causal rotations (or the raw recursion, which was reused for temperature). Reporting costs are recorded separately at their actual frequencies; mean, region and interval timers are nested in `report_seconds`. Truth-based scoring time is not charged to a deployed estimator.
