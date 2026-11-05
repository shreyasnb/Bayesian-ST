# Contact tracking study

This study starts from `f09d3fecd3af11fff5af708c8d1d31a35db75ca4`.
The supplied [tracking guide](contact_tracking_spec.md) extends the earlier
[uncertainty comparison](contact_uncertainty_spec.md). The later guide makes
clean query-velocity MSE the primary endpoint and asks for mean tracking before
uncertainty calculations. The earlier interval-score endpoint remains a
secondary, predeclared comparison. Neither guide is a request to change the
manuscript or manufacture an improvement.

The revised [contact notebook](../notebooks/03_contact.ipynb) follows one
preassigned trajectory through mechanics, identification, motion prediction and
uncertainty. It then checks whether that explanation survives independent
trajectories. Detailed search tables live in the results directories. The old
notebook is preserved byte-for-byte in
[`archive/03_contact_f09d3fe.ipynb`](../results/contact_tracking/archive/03_contact_f09d3fe.ipynb).
The separate attached notebook was inspected for its sequence and displayed
figure; none of its source code was used.

## What the robot observes

The plant solves a constrained viscous force balance. For isotropic resistance
`d I`, its instantaneous velocity is `P F/d`, where `P` is the contact tangent
projector. Every estimator receives the same known `F/d` and noisy velocity.
Positions and geometric normals are plant and scoring records only. This is an
ideal bilateral guide with known resistance. It does not model contact loss,
impacts, friction, unilateral constraints or a complete robot controller.

The condition called `weak` in machine-readable historical files means **forces
mostly along one axis**: the second and third components of the prescribed
force program are divided by five. A small input-Gram eigenvalue alone does
not establish weak information about the normal; local Fisher eigenvalues are
reported separately.

## Comparisons and information

1. Mode-only validation compares exponential memory, rolling likelihood windows
   and the existing causal geodesic velocity forecast. The noiseless ablation
   changes only measurement error, retaining the same mechanical trajectory,
   forces and update variance. True-normal alignment is an unavailable
   diagnostic, excluded from practical selection.
2. Only selected mean policies enter posterior reporting. Scaling `A` to `A/T`
   changes distribution spread and preserves its mode. Temperature is selected
   by clean-response interval score on validation truth; it is not a learned
   sensor variance.
3. Bingham process prediction and a recursive two-coordinate Gaussian use the
   same available nominal motion. Their process spread is selected separately
   by pre-correction predictive score. Both pay for the auxiliary point tracker
   that supplies a learned nominal rotation.
4. A local Gaussian approximation to the *same* Bingham parameter isolates
   distribution shape. A recursive Gaussian also changes propagation and the
   nonlinear measurement approximation; its comparison cannot isolate shape
   alone. A validation-fitted scalar multiple of each point projector checks
   whether simple regularization accounts for a prediction gain.

All physical comparisons keep sensor standard deviation `0.001 m/s`. The same
nine query forces form a tight frame, so their average loss is an isotropic
response-map error. Coordinates, queries and report times are not statistical
replicates: paired intervals use independent trajectories.

## Numerical interpretation

The real `Gr(2,3)` transition has an exact first-moment formula. Inverting that
moment into a Bingham parameter is an assumed-density approximation, not exact
closure of the filtering density. The independent spherical-harmonic control
uses a different, dimensionless matched process with known transition spread;
it is not the physical sphere's law.

Numerical integration flags, local-Gaussian validity, posterior mass and
repeated-data coverage are separate checks. In particular, a Gaussian
exponential-map distribution can be integrated accurately while placing too
much probability beyond its local coordinate chart. Such a report remains
flagged. A Gaussian local ellipsoid is not a Grassmann density HPD region.

Taking `G=I` is a choice of reference frame. The deterministic modal bound can
include nonzero residual state drift in that frame; its sufficient condition
may become uninformative. It does not promise credible-region coverage for
arbitrary drifting states. Learned rotations, fitted temperature and stochastic
moment projection do not inherit a fixed deterministic-frame theorem.

The mean `E[P]` minimizes posterior squared response-prediction loss and is
usually full rank. Its output is an expected velocity, not one feasible contact
projector or a guaranteed admissible command. The anisotropic-resistance stress
breaks the observation law itself and is scored against its actual KKT response.
Wider uncertainty or better memory does not repair that mismatch.

## Reproduction

Use the environment and TeX dependencies described in the repository README.
Limit BLAS threads when running several worker processes. The general notebook
is executed into an isolated validation directory; the historical notebook and
results are not overwritten. Its tests cover likelihoods, EM, numerical bounds,
sampling, scaling, and the newly added contact procedures. No example is
generated before this gate succeeds.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_contact_uncertainty.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_contact_tracking.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_contact_tracking_uncertainty.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_contact_matched.py --workers 4
python scripts/summarize_contact_tracking.py
python scripts/make_contact_tracking_notebook.py
OPENBLAS_NUM_THREADS=1 python scripts/execute_notebook.py notebooks/03_contact.ipynb
```

The runners freeze validation selections and source hashes before opening
held-out streams. Existing held-out outputs are protected against accidental
replacement. Use a new output directory for a new experiment. Frozen physical
data, paths and statistics are under `results/contact_tracking/`; matched-process
and numerical-backend results are under `results/contact_uncertainty/`.

On a checkout with the completed evidence, the mean runner deliberately refuses
to replace it. For an independent regeneration, pass `--output /tmp/contact-run/mean`
to the mean runner, `--output-dir /tmp/contact-run/uncertainty --mean-selected
/tmp/contact-run/mean/selected.json` to the uncertainty runner, and
`--output /tmp/contact-run/process` to the matched runner. The notebook generator
reads and verifies the published evidence by default.
