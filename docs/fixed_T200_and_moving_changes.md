# Fixed-budget stationary results and independent moving candidates

The complete task-specific path list is in
[fixed_T200_and_moving_changed_files.txt](fixed_T200_and_moving_changed_files.txt).
`M` denotes a modified tracked file and `A` a new file, relative to
`dd399bacae3d7b37e31abb949b24740c1f07a4f3`.
The root artifact inventory supplies hashes for all retained study files.

## Stationary comparison

The primary comparison is after correction at measurement 200, for
$n=32,64,128$. There are 64 validation and 256 final simulations per dimension:
192 validation and 768 final simulations in total. All ten implemented methods
share the observations within each simulation. Saved checkpoints are
$t=25,50,100,200$. The physical model, seed construction, candidate grids,
measured-power selection criterion and numerical tolerances are retained.
The dedicated `--stationary-only` runner preserves the historical mixed
performance component.

The [main TeX table](../results/main/beamforming/stationary_T200_table.tex)
reports $n=64$. The [supplementary TeX table](../results/supplementary/beamforming/stationary_T200_table.tex)
reports all three dimensions. Full-precision all-method and paired CSV/JSON
exports are under
`results/beamforming/tables/paper/performance/stationary_T200/`.
See [the fixed-budget notes](fixed_T200_beamforming.md) for every numerical
entry and its interpretation.

The new stationary data replace the active dimension-scaled results in:

- `results/main/beamforming/task_performance_output_power.pdf`: retained
  stationary figure, now at fixed $T=200$.
- `results/beamforming/task_performance.pdf`: stationary page only; the existing
  moving comparison remains unchanged.
- `results/beamforming/setup.pdf` and `posterior_anisotropy.pdf`: the saved
  stationary state is now $n=64,t=200$.
- `results/beamforming/additional_comparisons.pdf`: stationary pages 1–3;
  unrelated pages retain their original PDF objects and previews.
- `results/beamforming/paper_summary.pdf` and the affected supplementary
  bundles/copies: only the corresponding stationary evidence changes.
- The corresponding inline notebook outputs, previews and supplementary copies.

The original raw stationary cohort, summaries, worker history and validation
certificate remain available under their original paths. The new cohort has
its own manifest, raw records, checkpoints and 319-test certificate. Historical
timing figures continue to use their original workload and reference data.

## Independent moving comparison

`notebooks/beamforming_moving_interference.ipynb` is separate from the five
existing notebooks. It reuses the matching original smooth and abrupt
experiments: $n=32,T=256$, $\sigma^2=0.2$, $\tau^2=1$, uniform initialization,
$G_t=I_n$ and $\beta=0.97$. Each condition retains its selected parameters,
64 validation simulations and 256 final simulations. No new moving simulation,
selection or posterior fit is performed.

The independent records, paired summaries, source PDFs, previews and manifests
are under `results/beamforming_moving/`. The two new optional main-folder
candidates are `moving_interference_smooth_output_power.pdf` and
`moving_interference_abrupt_output_power.pdf`. They do not select or replace
a manuscript block. The earlier repurposed abrupt PDF and PNG are preserved
as `results/beamforming_moving/figures/abrupt_previous_main.*`; the stationary
main export was restored from its saved fixed-$T$ source.

See [the moving notes](moving_beamforming.md) for all four method results,
paired intervals and the precise averaging definition. Every checkpoint has
its own pointwise Student 95% interval across 256 simulations.

## Noise presentation and verification

The existing noise figure retains its numerical arrays, parameters and labels.
Its horizontal axes are `$t$`, and all four panels mark measurement 129 using
one consistent dashed line and one common legend entry. The true noise step
is drawn at 129 independently of reporting times. Estimates are those available
before receiving $u_t$; a completed fit first applies at $t+1$.

PDFs retain vector LaTeX text, editable figure sizes and 300-dpi raster layers;
new full-resolution PNGs accompany the selected exports. No contact notebook
cell or output changes. The independent moving work preserves all five
existing notebooks, the stationary checkpoints and the stationary/noise main
PDFs byte for byte after restoration.

Detailed checks are saved in
`results/beamforming/data/validation/fixed_T200_revision_checks.json` and
`results/notebook_editability.json`. They cover the 319-test scientific gate,
raw-record summaries, paired intervals, causal variance timing, preserved
historical data, executed notebooks and actual PDF typography/resolution.
