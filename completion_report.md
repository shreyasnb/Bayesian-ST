# Final simulation study

Working branch: `test`. Starting revision:
`a91faabeb10a63b1d1c613d75cc32503cfc0cac6`.
Scientific content revision: [`eee4c1e30882c5530eda8d25fe6afcc349f0ddf7`](https://github.com/shreyasnb/Bayesian-ST/commit/eee4c1e30882c5530eda8d25fe6afcc349f0ddf7).
A subsequent metadata-only commit records this identifier and its inventory.
The final pushed head is reported in the accompanying handoff.

## Specification and validation

The author confirmed **v5.tex** as the main manuscript;
**supplementary(14).tex** is the supplement. Exact file hashes and mathematical
conventions are in
[specification.json](results/beamforming/data/specification.json).
The [final guide](docs/final_simulation_guide.md) is preserved verbatim.
The manuscripts themselves were not edited, and their old simulation prose
was not used as evidence for the new cohorts.

The complete original general notebook passed with **315 regression tests**,
zero failures/errors/skips. Its fresh execution and current source gate are in
`results/beamforming/data/validation/`. The final scientific fingerprint is
`d3e5f91e4bc9ca18bf34d595fe6d2ed0137361dff835dccc72b552ad7bb5ec47`.
Before changing the implementation, fresh small beamforming and contact runs
reproduced 6,912 and 1,104 original numerical rows exactly, excluding clocks.
Proofs are in each application's `data/existing_condition_reproduction.json`.

The starting archive preserves 229 source/notebook files and verifies 2,358
historical result files left unchanged in place. The original general, field
and implementation-audit notebooks remain historical evidence; old validation
certificates have not been refreshed to imply new producers.

## Correctness changes

- Exact block corrections retain observation counts, per-observation scales,
  all within-block discount weights, the beta-to-the-block-length prior factor
  and known coordinate alignment. Covariance averaging is not a posterior
  correction.
- The current sharp modal bound uses the midpoint initialization radius and
  the norm after aggregating conditional drift. Reference information gaps
  are distinguished from realized posterior eigengaps.
- All new policy selection uses shared independent **measured** validation
  observations. Hidden normals, signatures, true drift and oracle losses are
  used only for evaluation. Missing or failed validation candidates cannot
  win by being scored on a surviving subset.
- Variance fitting uses completed disjoint blocks and one preceding fixed
  prior. Known coordinates are aligned before fitting. Fitted scales affect
  future corrections; fitting changes the already-streamed parameter by zero.
- The new friction stress test assigns the complete force returned by the
  existing mechanics helper. A regression checks the zero-friction limit.
- Calibration and mismatch cohorts remain separate in summaries and figures.
  A mean-accuracy flag is not represented as second-moment or tail validation.

## Theory and numerical outcomes

Each application completed 2,560 fresh fixed-time sharp-bound reports, with
128 independent noise episodes per cell. Beamforming had 384 eligible reports;
contact had 1,920. There were zero eligible violations. A zero-count cell with
128 eligible episodes has an exact pointwise 95% upper violation limit of
0.028408. These are fixed-time statements, not simultaneous trajectory
guarantees. Ineligible reports are not counted as successes.

Only the long stationary beamforming cases qualified; the new smooth,
abrupt, zero-signal and fixed-memory beamforming checks did not satisfy the
sufficient condition. Contact retained fast-drift and insufficient-excitation
failures. All initialization, drift and fluctuation decompositions matched
the independently reconstructed parameter to floating-point precision.

Algebra, global Gaussian KL, information-loss identities and all 28 beamforming
and 30 contact scalar checks passed. The local inverse-Fisher study used 256
episodes per cell. At beta=1 and 1,024 observations, sampling covariance/reference
ratios were 0.965/1.018 for beamforming and 0.991/1.014 for contact at the two
noise levels; local fractions were one. Short/broad cells retain locality and
bias diagnostics. Forgetting uses squared weights in the sampling reference;
inverse posterior curvature is not a repeated-data covariance bound.

Independent 128/256-resolution comparisons resolved the declared spectral
spans 8, 80 and 800. At span 40,000 the complex native series exceeded its
declared cap. The real native mean reported convergence but its independent
reference remained unresolved, so that comparison is unsupported rather than
evidence that the native mean is wrong. Mean, normalizer, second-moment and
quantile flags remain separate. Deterministic native dimension scaling applies
to rank one or codimension one; general-rank normalized inference is unsupported.

## Application and computation outcomes

Contact completed 64 validation and 256 test episodes per condition, with zero
numerical failures. Episode-average response MSE improved over the mode by
27.5%, 29.1%, 2.0% and 15.0% for balanced, concentrated, plane and sphere
conditions. Gains over matched isotropic uncertainty were 1.67%, 3.18%, 0.22%
and 3.38%; the plane interval includes zero. Isotropic was slightly better at
the final sphere checkpoint. Thus the gain from directional shape is smaller
than the total mean-versus-mode gain. Input-aware regularized comparisons and
all losing/inconclusive endpoints remain in `tables/paper/performance/paired.csv`.

Exact contact block aggregation agreed within 2.2e-12. Delayed actions incurred
separate losses. The reset uniform-prior sphere block likelihood improved from
L=1 to L=16, then worsened at L=32 as within-block normal drift increased;
the stationary plane improved steadily. These are distinct approximation and
delay effects, not an algebraic aggregation error.

Beamforming causal learning reduced mean held-out output power from 0.8832 to
0.7903 after a noise change, and from 0.5685 to 0.4087 after a power change.
Paired gain intervals exclude zero. It remained worse than supplied true
scales and incurred a small significant loss under smooth motion. These are
comparisons of complete validation-selected policies, including their memory
choices, rather than an isolated EM intervention.

At the fixed t=64 beam matched prior-predictive check (128 test episodes,
nominal 95%), the exact reference achieved 96.1% state-ball, 96.9%
clean-functional and 94.5% noisy-coordinate inclusion. Learning did not restore
exact calibration. Under unmodeled beam drift, geometric inclusion collapsed
even where noisy-observation inclusion remained near nominal. Fast-drift
response risk was severely underestimated. These are n=3 calibration fixtures,
not high-dimensional or joint response-vector coverage claims. All beam
prediction means, scores, fits and quantile checks resolved.

Beamforming completed 2,048 primary test episodes across the eight declared
cohorts and 128 block episodes. All 44,032 native means resolved at the declared
tolerance; 392 used the predeclared extended cap (maximum order 5,009 below
8,192). The mean lost to the mode in all 12 stationary cells. It beat matched
isotropization there, but usually by a practically tiny amount. Under smooth
and abrupt motion its mode gains were only 0.019 and 0.035 dB. Loaded-covariance
and GREAT comparisons were mixed. Posterior conditional optimality does not
imply repeated-fixed-state dominance under the chosen ULA geometry and prior.

Contact variance learning recovered the plane noise jump (final estimate
6.5e-5 versus true 6.4e-5), with a small observed-response gain because sensor
noise dominates that endpoint. Faster sphere motion instead inflated the
estimate to 1.4e-4 versus true 1.6e-5 and worsened task performance. Scalar
variance adaptation did not repair wrong resistance or contact loss.
Contact matched inclusion was near nominal for the exact reference. Learned
forgetting could be conservative, with broader regions; its strong geometric
inclusion under the tested drift contrasts with beta=1's failure, but establishes
no exact dynamic calibration law. Contact mixture-score intervals include zero
against the full-covariance Gaussian surrogate. Contact prediction targets met
their declared numerical tolerances. Prediction workers completed without failed
jobs in both applications.

Isolated timings used seven warmed randomized repetitions on one BLAS thread.
For the preassigned n=64, 64-observation endpoint task, median full-stream costs
were 19.9 ms (mean), 4.84 ms (mode), 1.60 ms (loaded covariance) and 22.2 ms
(GREAT window). For physical contact with actions every four observations,
the mean cost 23.1 ms versus the mode's 6.58 ms per 192-observation stream.
Actual causal mean/score/EM cycles cost 751 ms per 256 beam observations and
776 ms per 192 contact observations. These are single-data-set clocks, not
joint population loss/runtime guarantees or worst-case deadlines.

Reducing action frequency saves moment computations while retaining every
correction: n=128 complex full-stream cost fell from 1,073 ms for an action
every observation to 32.3 ms for one endpoint action. It is not a raw-block
matrix-product speedup. Persistent n=128 mean-parameter storage was 256 KiB
complex and 128 KiB real; allocation tracing excludes native workspaces/RSS.
All 2,674 measured cycles resolved without inference, fitting or report failures.
The high-concentration stress limits and 20 unsupported independent real
high-dimensional reference cases remain explicit in the accuracy tables.

## Interpretation and manuscript notes

Taking identity transport when motion is unknown is an allowed approximate
prediction. It does not make the true subspace stationary. Drift enters the
modal bound, and an ineligible bound is not a violation. Poor repeated-data
coverage under unmodeled drift limits the dynamic uncertainty claim; it does
not refute conjugacy, the Fisher identities or an eligible conditional bound.
The supplied main manuscript already makes this distinction.

The relevant contribution is a distribution's directional means, uncertainty,
predictions and variance fitting. The marginal mode shares a covariance-PCA
objective. Finite-gradient methods can cost less. Mean-versus-mode gains alone
cannot establish an anisotropy advantage; matched isotropic and independently
validated loading controls are required. Contact GREAT also lacks the applied
inputs available to the conditional estimator, so that comparison does not
isolate uncertainty.

If Algorithm 1 includes variance learning, make the saved preceding prior,
disjoint fitting blocks, coordinate alignment, future-only scale use and
learning cadence explicit. Its fixed-variance tracking theorem does not
directly apply to this empirical-Bayes recursion. A constant-state EM block
can mistake physical drift for noise. See
[the manuscript checks](docs/final_manuscript_checks.md).

GREAT implements finite Grassmann exponential steps and recursive covariance
updates, with independent real/complex geometric tests. The original paper
and ETH author-code hosts were denied by the managed network policy. Direct
primary-PDF/author-code parity remains unverified; no claim of having obtained
or executed that code is made. See [provenance](docs/great_reference.md).
All 16 beamforming GREAT policies selected the largest declared inner-step
budget; 14 selected the largest step multiplier. Static balanced contact also
reached its step/iteration limits. These are finite-budget comparisons, not
claims about converged GREAT or globally optimal tuning. The frozen grid was
not expanded after paper testing began.

## Reproduce and locate the figures

See [the exact stage commands](docs/final_simulations.md). The final read-only
handoff command is `python scripts/check_final_simulations.py`.
Each application directory contains the individual PDF catalogue, a compact
paper summary and three readable multipage supplementary bundles. Numerical
records, omitted conditions, captions, seed sets, precision flags and SHA
closures are recorded in its `data/`, `tables/` and `figure_manifest.json`.
Rendering reads saved outputs and never repeats simulation or tuning.

Clean-kernel execution passed for both notebooks (ten code cells and eight
hash-matched embedded figures each). The final read-only check verified all 24
stage/component manifests and the complete retained-file inventory. An
independent actual-PDF audit passed for all 38 files / 144 pages: seven-inch
widths, Type 1 fonts, empty PDF titles and no out-of-page text. Preview images
are 300 dpi. Regenerable checkpoint caches are excluded from Git.

## Native sphere restoration and manuscript exports

The mode-facing multicolor display was reverted: its rotation and white
contours made a correct density look like a flat target. The original native
coordinates, oblique camera and blue-to-pink palette are restored. Front-visible
coordinate great circles now provide a geometric depth cue. No density shading,
mode centering or change of posterior angular width is applied. The original
n=64, t=64 beamforming checkpoint is retained, and its section is correctly
identified as a restriction of the complex posterior rather than a physical
radiation pattern. Both notebooks explain these distinctions.

All active plotting defaults and exporters now use centralized 300 dpi. Final
paper pages are seven inches wide, with vector LaTeX text and curves and
2100-pixel-wide PNG previews. Full-resolution PNG/PDF links accompany notebook
images, since an inline preview is scaled to the reader's screen.

The PDF manuscript selection is under `results/main/` and
`results/supplementary/`, with separate application folders, captions, exact
page selectors and hash provenance. Main figures are single-page selections;
all remaining study pages are retained in the supplement. Complete original
application catalogues are preserved for the notebooks and audit trail.

Scientific outputs remain byte-identical to `ded5499`: 574 numerical
artifacts, 25 scientific manifests and 132 scientific/core source files.
The unchanged regression suite passed 313 tests and two subtests. Geometry
checks cover 20 beamforming and 464 contact laws, with native coordinates for
rendering and separately labeled auxiliary rotation stress checks. See
[reproduction and validation commands](docs/final_simulations.md).

Both current notebooks executed successfully in clean kernels. The independent
actual-export audit passed 38 canonical PDFs / 144 pages and 34 manuscript
selection PDFs / 74 pages, including all placed raster layers at 300 dpi,
embedded Type1 fonts, seven-inch pages and text within page bounds. All 74
canonical preview PNGs are 2100 pixels wide at 300 dpi. The extra draft PDFs
are page copies of the same evidence, not additional experimental results.
