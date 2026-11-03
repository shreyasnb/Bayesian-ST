# Application pilots

This directory contains a new experiment; historical application tables are unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_dev.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_application_ablation.py --workers 2
```

The default study freezes eight validation trajectories and sixteen independent
test trajectories. The primary conditions are the matched drifting identity-frame
array and passive field. Array inference is native complex n=16, rank 3, at 32
snapshots. Field inference uses n=8, rank 3, at 128 snapshots. The longer field
horizon distinguishes windows of 48, 96 and 192 observations. A window beyond
the available history saturates at the full-history estimator.

One policy per method and application is selected on matched-drift validation
data, then reused for static/mismatched and synthetic partial-motion controls.
All applicable methods receive the same motion information. Partial-motion
controls receive half the true motion; primary cases receive identity. Paired
robustness scenarios share trajectories and are not extra independent replicates.

Posterior means are tuned using their actual sampled task actions, rather than
using modal decisions as proxies. The array selection loss is independent
held-out output power; reconstruction selection uses terminal current-observation
squared error. SINR and its empirical lower tail are separate reported endpoints.
Mode decisions share the posterior-mean filter's selected parameter history;
EW PCA has separately tuned memory. Scalar shrinkage tunes both memory and
shrinkage. Loaded covariance starts from a Cartesian memory/loading grid.

A predeclared extra validation stage checks selected open grid boundaries before
final testing. Structural endpoints (beta=1 and rho=0 or 1), full-history window
saturation, and unresolved open tuning boundaries are distinguished. Remaining
unresolved boundaries prohibit a positive main-paper performance claim.

Two independent posterior chains start at 1,000 retained draws and 500 burn-in
sweeps each, expanding only within the frozen cap of 4,000 retained draws and
2,000 burn-in sweeps. Every capped diagnostic failure stays in the tables. A
favorable paired loss interval does not establish a supported gain if posterior
accuracy or selected-policy validation accuracy is unresolved. Sixteen test
trajectories form a bounded exploratory pilot, not broad superiority evidence.

Files:

- `predeclared_config.json`: seeds, endpoint, candidate families, numerical budgets,
  motion/data access, hardware, versions and numerical-source hashes.
- `validation_trials.csv`, `tuning.csv`: complete candidate losses, actual validation
  costs, selected settings, numerical flags and boundary status.
- `selected.json`: policies frozen before test jobs start.
- `metrics.csv`: one row per independent trajectory, condition and method.
- `paired.csv`: paired task-loss differences with pointwise exploratory Student
  intervals; no multiplicity-adjusted significance claim.
- `tails_cost.csv`: terminal loss tails, lower-tail SINR and cumulative time.
- `noise.csv`: separate real n=3, rank-one noise-learning ablation using the
  deterministic moment/normalizer backend and a frozen-window ascent safeguard.
- `config.json`: checksums and final gate association.

Cost columns separate cumulative prediction/correction, one terminal moment
evaluation and decision preparation/action. Every Bingham modal step is computed
inside the correction timer, matching the existing filter and EW-PCA reference.
Diagnostics and tuning costs are reported separately from final per-trajectory
inference. Dense regional optimization is not run here. Timings concern these
implementations and hardware, rather than generic asymptotic speed claims.

The noise ablation fixes the selected identity-frame forgetting rule, changes
only measurement-noise fitting, and retains exact observation-time priors.
Checked ascent concerns each frozen-window objective; it is not full state-space
likelihood ascent. A fitted variance can absorb missing dynamics and is not an
estimate of process dispersion.
