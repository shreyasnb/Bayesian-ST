# Figures for the revised numerical section

The active beamforming and contact notebooks retain their existing cells and
visible Matplotlib code. The later fixed-$T=200$ instructions supersede the
stationary comparison in `numerical_section.tex` and `plot_updates_codex.txt`.
Only that stationary cohort was rerun; its original records remain available.
The paper source, unrelated simulations and benchmark selection criterion are
unchanged. A new cell displays the supplementary fixed-$T$ dimension curve.

## Quantities and units

- Array power uses `true_ratio_db` from per-simulation records, with
  $\ell_t=10\log_{10}(w_t^H R_t w_t/p_t^\star)$ and
  $p_t^\star=(a^H R_t^{-1}a)^{-1}$. The stationary table and supplementary dimension curve use fixed $t=T=200$
  for every $n$ and 256 simulations; the old frozen measured-power summary is not reused.
- The variance experiment uses `output_oracle_db`, stored integer measurement
  indices and the pre-correction variance estimates. The true step and the common event marker are drawn at measurement 129. Estimates completed after $t$ first apply at $t+1$.
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

At $n=64,T=200$, mean power values in dB are 0.2368846161 (posterior mean),
0.2227006028 (mode), 0.3499890378 (sliding-window GREAT) and 0.5763537874
(regularized sample covariance). The [fixed-horizon revision](fixed_T200_beamforming.md)
supersedes the earlier $t=n$ comparison while preserving its raw historical records. Time-averaged array variance-experiment
values are 0.434633, 1.056295 and 0.378372 dB for EM, fixed and true variances.

## Stationary presentation and optional moving figures

`results/main/beamforming/task_performance_output_power.pdf` retains the
stationary comparison at fixed $T=200$. Its main and supplementary tables use
the new values. The briefly repurposed abrupt-direction PDF and PNG were saved
under `results/beamforming_moving/figures/abrupt_previous_main.*` before restoring
the stationary export from its saved fixed-$T$ source.

The separate `notebooks/beamforming_moving_interference.ipynb` reads the
verified existing smooth and abrupt records without changing the application
notebooks or stationary evidence. Its two separately named main-folder PDFs
are optional candidates for review. They use exact `true_ratio_db` and
pointwise Student intervals across 256 simulations at every reporting time.
The abrupt direction event at $t=129$ has one legend entry; the smooth plot
has no event line. See [the moving study](moving_beamforming.md).

The noise figure retains its original numerical traces and labels. All four
panels use the same purple dashed 1.3-point event line at $t=129$ above the
shading, and the same horizontal limits. True noise changes at 129 independently
of reporting indices; scale estimates remain those available before $u_t$.
The temporal figures use the LaTeX horizontal label $t$. Their PNG previews
and PDF copies update from the visible SAVE_PDF blocks.

## Printed size and exports

Notebook cells expose width-aware label, tick and legend controls. On a
15-inch canvas the defaults are 25, 23 and 22 points, giving about 10, 9.2
and 8.8 points at six-inch insertion. DPI remains 300 for raster layers and
previews; PDF text and curves remain vector objects. Most canvases retain
15-by-5 inches; the four-panel array variance figure uses 15-by-7 inches.
Computation-phase bars use 15-by-7.5 inches so their two-line row labels remain separated.
Panel letters are omitted from tiled figures, following the later instruction
to use positional descriptions in captions.

`results/main/` retains the six application notebook PDF paths referenced in
the revised section and two separately named optional moving candidates.
Other notebook figures and all complete canonical individual
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

The current edits are in `notebooks/01_array.ipynb`,
`lib/final_beamforming.py`, `scripts/run_final_beamforming.py`,
`lib/notebook_beamforming_data.py` and `lib/final_paper_visuals.py`.
The inline plotting templates, notebook generators, export partition and
presentation checks were updated alongside them. Generated PDFs, previews,
captions and figure manifests are under `results/`. The theoretical notebook,
field notebook and implementation-audit notebook are unchanged. All contact
notebook cells and outputs are unchanged; its top-level source metadata was
refreshed for the shared input reader.

All requested main curves can be reconstructed from saved results. The sphere
contact time average uses its saved per-simulation aggregates; the raw dataset
does not retain every intermediate time for each simulation. The producer's
48-time aggregation and the basis-invariant velocity-error identity were
checked without regenerating data.
