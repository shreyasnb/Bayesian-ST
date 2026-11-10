# Figures for the revised numerical section

The active beamforming and contact notebooks retain their cell order and
visible Matplotlib code. The revision uses the supplied `numerical_section.tex`
and `plot_updates_codex.txt`; it does not change the paper source, simulations,
estimator settings, benchmark selection criterion or frozen numerical tables.

## Quantities and units

- Array power uses `true_ratio_db` from per-simulation records, with
  $\ell_t=10\log_{10}(w_t^H R_t w_t/p_t^\star)$ and
  $p_t^\star=(a^H R_t^{-1}a)^{-1}$. The stationary plot uses $t=n$ and
  256 simulations; the old frozen measured-power summary is not reused.
- The variance experiment uses `output_oracle_db`, stored integer measurement
  indices and the pre-correction variance estimates. The true change is between
  measurements 128 and 129. Estimates completed after $t$ first apply at $t+1$.
  Variance shading is the middle 50%; power and log-density shading is a Student
  95% confidence interval for the Monte Carlo mean. The normalized predictive
  density remains the saved `prequential_score`.
- Contact velocity error is $\mathcal E_t(D_t)$ from the manuscript: one
  average over three squared Euclidean errors, followed by multiplication by
  $10^6$ to obtain $(\mathrm{mm}/\mathrm{s})^2$. Sphere values average all
  48 reporting times within each simulation before averaging simulations.
- Contact information uses lowercase $z$, physical angles in degrees and the
  unchanged exact likelihood divergence. The normalized response variance is
  $r_t(v(\theta))/v_0^2$; its saved unit-input curve requires no rescaling.
  Density geometry, camera, native coordinates, color map and logarithmic
  limits are retained. View padding accommodates the larger notebook labels.

At $n=t=64$, mean power values in dB are 0.763263643767 (posterior mean),
0.689371667838 (mode), 0.690951042886 (GREAT) and 1.077038079841
(regularized sample covariance). Time-averaged array variance-experiment
values are 0.434633, 1.056295 and 0.378372 dB for EM, fixed and true variances.

## Printed size and exports

Notebook cells expose width-aware label, tick and legend controls. On a
15-inch canvas the defaults are 25, 23 and 22 points, giving about 10, 9.2
and 8.8 points at six-inch insertion. DPI remains 300 for raster layers and
previews; PDF text and curves remain vector objects. Most canvases retain
15-by-5 inches; the four-panel array variance figure uses 15-by-7 inches.
Computation-phase bars use 15-by-7.5 inches so their two-line row labels remain separated.
Panel letters are omitted from tiled figures, following the later instruction
to use positional descriptions in captions.

`results/main/` contains exactly the six notebook PDF paths referenced in the
revised section. Other notebook figures and all complete canonical individual
PDFs are retained in `results/supplementary/`. Each `SAVE_PDF=True` block saves
its local export and copies those exact bytes to its visible `DRAFT_PATH`.

## Checks

```bash
python scripts/check_revised_numerical_figures.py
python scripts/check_notebook_bands.py
python scripts/check_manuscript_plot_exports.py
python scripts/check_final_simulations.py
```

These are read-only checks of numerical evidence and presentation. They do
not run a replacement simulation, tune parameters or fit a posterior.

## Changed sources

The primary edits are in `notebooks/01_array.ipynb`, `notebooks/03_contact.ipynb`,
`lib/notebook_beamforming_data.py`, `lib/notebook_contact_data.py`,
`lib/final_paper_visuals.py` and `lib/final_information_visuals.py`.
The inline plotting templates, notebook generators, export partition and
presentation checks were updated alongside them. Generated PDFs, previews,
captions and figure manifests are under `results/`. The theoretical notebook,
field notebook, implementation-audit notebook and numerical producer sources
are unchanged.

All requested main curves can be reconstructed from saved results. The sphere
contact time average uses its saved per-simulation aggregates; the raw dataset
does not retain every intermediate time for each simulation. The producer's
48-time aggregation and the basis-invariant velocity-error identity were
checked without regenerating data.
