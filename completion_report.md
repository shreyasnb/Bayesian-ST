> The fixed-T200 stationary revision supersedes only the earlier stationary
> dimension comparison in the historical sections below. See
> [its configuration and tables](docs/fixed_T200_beamforming.md). Other experiment
> records retain their original values.

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
number of observations between variance fits explicit. Its fixed-variance tracking theorem does not
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

## Plot notation, information geometry and manuscript exports

The current application notebooks use the implementation notebook's serif
LaTeX typography: 11-point axis text, 10-point ticks and legends, and 1.6-point
lines. More panel space preserves readability at a seven-inch PDF width. All
current previews and PDF raster layers use 300 dpi; text and curves remain
vector graphics. The sphere camera, native coordinates and saved density
values are unchanged. Only the color lookup is changed to a continuous
blue-purple-pink-yellow palette on the existing shared logarithmic scale.

Figure labels use the manuscript's matched isotropic mean,
$\overline P_t^{\mathrm{iso}}$. Directional-risk axes give their matrix
expression and query angle in radians. GREAT labels are kept concise, with
complex-extension provenance in the notebook text. Timing labels state time
to process 64 observations and describe the task-decision intervals directly.
Memory is retained on a separate supplementary page.

The main contact task panel shows absolute MSE and pointwise intervals for all
four same-data controls, including the blue posterior mean. The original
absolute paired MSE differences are retained on a supplementary page. A
posterior mean compared with itself has zero paired difference; no artificial
self confidence interval is introduced. All methods use common independent
episodes. The intervals of absolute means and paired differences have distinct
interpretations.

The contact information figure now shows exact likelihood distinguishability
of physical normal rotations using the original saved 16-input design. The
Gaussian KL and its local Fisher Hessian are checked against independent
Gaussian-mean differences. No new data, posterior fit, noise setting or
benchmark policy is introduced. The original Fisher parity validation remains
in the implementation notebook and saved numerical records.

Scientific outputs remain byte-identical to the frozen study: 574 numerical
artifacts, 25 scientific manifests and 132 scientific/core source files. Both
current notebooks and the complete PDF catalogue are regenerated from that
evidence. Main and supplementary insertion PDFs, external captions, page
selectors and source hashes remain under `results/main/`,
`results/supplementary/` and `results/draft_figure_manifest.json`. See
[the reproduction protocol](docs/final_simulations.md) and
[the typography comparison](docs/plotting_protocol.md).

The updated regression run passed 313 tests and two subtests. Both application
notebooks executed successfully in clean kernels. The actual-export audit
passed 38 canonical PDFs / 150 pages and 36 manuscript-selection PDFs / 77
pages. All raster layers and 77 preview PNGs are 300 dpi, with seven-inch
PDF pages, embedded Type1 fonts and text within page bounds. Actual PDF text
traces confirm the nominal 11/10-point typography. A rendered-text scan found
no occurrences of the removed timing terminology or bracketed extension labels.
The independent contact likelihood check agreed with direct Gaussian-mean KL
to 1.24e-14, and its numerical Hessian matched the saved Fisher matrix to a
relative error of 1.0e-10. These are display/identity checks on the frozen
experiment, not new statistical repetitions.

## Editable notebook figures

The five root notebooks now contain 49 complete Matplotlib figure cells: four
in theory, eight in beamforming, seventeen in field reconstruction, eight in
contact tracking and twelve in the implementation audit. Every figure cell
exposes its data, DPI, FIGSIZE, artists, labels, direct PDF save and plt.show.
No image-display facade or imported plotting routine hides those choices.
The numerical library and scientific generators remain unchanged.

All five notebooks executed successfully. The theory notebook retained its
original assertions and passed all 315 regression checks. New read-only data
helpers verify the saved numerical inputs; field/audit execution no longer
triggers historical simulations. Editable exports and fresh theory diagnostics
use separate local directories. All 3,781 previously tracked result files,
including published PDFs, remain byte-identical. The scientific gate and all
24 stage/component records remain valid. See the [notebook editability report](results/notebook_editability.json)
and [the editing workflow](docs/editable_notebooks.md).

## Complete notebook figure catalogue

The beamforming notebook appends 32 figures and contact appends 31. Together
they cover the 61 individual saved catalogue pages previously absent from the
notebooks and add two application tracking-error figures. The existing 25
beamforming cells and 26 contact cells are preserved exactly, including source,
outputs, execution counts and metadata. Only the added cells were executed.

Every addition exposes full Matplotlib code with a 15-by-5-inch canvas,
300 dpi, LaTeX and its own supplementary PDF save block. Application tracking
distinguishes point-subspace chordal distance from posterior-mean matrix error;
the separate theoretical plots retain their eligibility conditions. Numerical
artist arrays match the original catalogue plots, and the application tracking
intervals were independently checked against the same saved 256 trials per
checkpoint. No simulation, inference, benchmark selection or numerical record
was changed.


## Revised numerical-section figures

The latest presentation revision follows `numerical_section.tex` and the
attached plot instructions. Array comparisons now read the saved exact
covariance powers explicitly; no measured validation criterion or estimator
setting was changed. Variance plots use integer measurement indices and the
true change between 128 and 129, pre-correction estimates and the unchanged
normalized predictive log density. Contact curves retain the exact velocity
error, normalized response variance and native density geometry.

Fonts are sized for six-inch insertion. Tiled figure letters are omitted as
subsequently requested. All notebook PDF copies and the complete canonical
catalogue are regenerated; the six files referenced by the revised main
section occupy `results/main/`, with other figures retained in
`results/supplementary/`. Scientific data, validation gates and numerical check
cells remain unchanged. Details and verification commands are in
[the revised figure notes](docs/revised_numerical_figures.md).


## Fixed measurement total for stationary beamforming

The new stationary cohort uses exactly T=200 training measurements at each
of n=32,64,128, with common checkpoints 25,50,100,200 and primary evaluation
after the final correction. Parameter selection was repeated on 64 independent
validation simulations per dimension using the unchanged measured-power
criterion and candidate grids. Each dimension then completed 256 independent
final simulations with all ten methods sharing the same training stream.

All 768 final simulations resolved. The 3,072 native posterior means met the
original 1e-10 tolerance without cap extension. The fresh general-validation
certificate passed 319 tests. Original mixed performance records, contact,
variance, moving, theory and isolated timing data remain untouched; the current
stationary component has its own source hashes and validation certificate.

The main table contains n=64,T=200; the consolidated supplementary table contains
all three dimensions. Tables use arithmetic means of individual true_ratio_db
values and Student 95% intervals. Complete all-method and paired summaries are
retained as CSV/JSON. See [the study notes](docs/fixed_T200_beamforming.md).


## Independent moving-interference candidates

The stationary fixed-T200 results retain their main figure and n=64 table,
consolidated supplementary table and supplementary dimension curve. The
temporarily repurposed main PDF was preserved separately before the stationary
figure was restored. The independent moving-interference notebook uses the
matching original smooth and abrupt records at n=32, T=256, beta=0.97 with
covariance-based true_ratio_db. Its candidate exports have new filenames and
do not replace the stationary presentation. Details are in
[the independent moving study](docs/moving_beamforming.md).

Direction and noise events use t=129 and a single legend entry. The noise
step, pre-measurement estimates, exact power and mixture predictive densities
retain their existing numerical records; the new moving notebook never
writes the noise figure.

## Application plotting and separate comparison audit

The editable moving notebook now also exports a compact 7-by-3.5-inch
smooth/abrupt comparison with one shared legend. Its mean curves are unchanged;
all four methods have empirical pointwise 25th–75th percentile bands computed
from the 256 saved simulations. Student confidence intervals remain in the
numerical tables. Individual exports are retained. Ordinary 2D application
figures use four-sided borders and inward ticks; 3D spheres are unchanged.
See [the figure update and manuscript caption](docs/application_plotting_update.md).

The contact response-variance plot uses the saved e1/e2 input scan without
rescaling its unit-input risks, and the sphere prediction history includes
the existing identity-transport control. Existing contact settings and data
are preserved.

Two independent beamforming audit notebooks retain both GREAT variants,
all four requested averaging conventions, paired intervals, measured-only
validation, full-history stationary windows and isolated optimization-cost
measurements. These additional outputs neither replace the stationary study
nor modify the original moving records. The full-history stationary window
closes most of the earlier GREAT gap; more EW optimization steps substantially
reduce the moving optimization error, with rare transient errors retained.
The posterior mean has no uniform frequentist dB-performance advantage.
See [the complete comparison audit](docs/beamforming_comparison_audit.md).

## Solid trajectories and contact transport diagnostics

Application method trajectories and legend handles now use solid lines without
sampling markers. Standalone scientific points and dashed event indicators are
retained. Layouts, line widths, colours, existing uncertainty bands and original
numerical results are preserved; the identity-transport contact control now has
the separately requested dark gray colour. Copy-pastable, statistically verified
captions are in [the caption file](docs/solid_trajectory_captions.tex).

The original 256 sphere simulations were replayed with unchanged settings and
seeds to retain all 48 existing prediction indices. All 14,336 frozen checkpoint
and whole-run rows match the original tolerances; 24,576 native moment checks
pass. Original data are unchanged, with recovered records stored separately in
`results/contact_transport_diagnostics/`. The identity-control rise between
t=128 and 192 is widespread (94.5% of simulations); the dense history also
preserves its temporary advantage near t=160. Four supplementary diagnostics
check paired changes, projector transport, input excitation and posterior
curvature. The prescribed rotation schedule alone does not establish a causal
explanation. A short draft explanation is in
[contact_transport_explanation.tex](docs/contact_transport_explanation.tex).

The contact information figure keeps its exact divergence curves and adds a
thin connector at the actual values 0.2395700487 at 5 degrees and 0.2307692308
at 30 degrees, with no equality imposed on their heights.
