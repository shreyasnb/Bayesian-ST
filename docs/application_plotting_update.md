# Application figure updates

The moving comparison is independent of the stationary fixed-$T=200$ experiment.
No existing measurements, estimates, selected settings, seeds, numerical
summaries, or comparison methods were changed to produce these figures.

## Moving beamforming

Editable source: `notebooks/beamforming_moving_interference.ipynb`.
The individual smooth and abrupt plotting cells and their exports remain
available. A separate editable cell exports the compact, 7 by 3.5 inch pair:

```tex
\includegraphics[width=\linewidth]{results/main/beamforming/moving_interference_comparison.pdf}
```

The solid curves use the existing method colors: blue for the posterior mean
$\overline P_t^+$, orange for the mode $P_{\widehat{\mathcal X}_t}$, pink for
$\overline P_t^{\mathrm{iso}}$, and green for GREAT (sliding window).
The purple dashed line at $t=129$ marks the interference-direction change
in the abrupt case. The individual and combined figures show arithmetic
means over the same 256 Monte Carlo simulations at the same 32 saved indices.
Their matching bands contain empirical pointwise 25th–75th percentiles.
Copy-pastable captions for all three layouts are in
[solid_trajectory_captions.tex](solid_trajectory_captions.tex).

The bands describe empirical performance variation across independent
simulations; they are not posterior credible regions or confidence intervals
for the mean. Their endpoints are computed directly from the 256 saved
`true_ratio_db` values for each method, condition, and reporting index.
The curves retain the arithmetic mean of these dB values. No times are added:
the reporting indices remain $8,16,\ldots,256$. Numerical summary tables retain
their original Student confidence intervals and paired comparisons.

Source PDF and 300 dpi preview copies are in `results/beamforming_moving/figures/`;
manuscript copies are in `results/main/beamforming/`. Retained individual names:

- `moving_interference_smooth_output_power.pdf`
- `moving_interference_abrupt_output_power.pdf`
- `moving_interference_comparison.pdf` (new)

All four method trajectories are solid and markerless, with their original
colors and mathematical notation. Scientific point markers and interval bars
retain their existing meaning. The shared legend contains the dashed
direction-change line once. The individual cells use the
same statistics and style controls. The earlier per-figure confidence-interval
presentation is replaced only for this moving-performance comparison.

## Four-sided borders

Ordinary 2D plots in `01_array.ipynb`, `03_contact.ipynb`, the moving notebook,
and `contact_prediction_diagnostics.ipynb` now have four visible spines and
inward ticks on all sides. Tick labels remain on the bottom and left.
Existing application figure dimensions, DPI and text controls remain editable.
The notebook generators emit the same border code. The two separate comparison
audit notebooks use this presentation too.

The existing 3D posterior-density drawing code, outputs, coordinates and density
normalization are retained exactly. The original complete PDF catalogues and
their historical certificates are retained. The notebook-generated individual
main/supplementary exports are refreshed; the separate stationary and moving
experiments remain separate.

## Contact plots

The response-variance scan uses the saved `query_family=0` inputs in
$\operatorname{span}(e_1,e_2)$, with the saved unit-input risks used directly as
$r_t(v(\theta))/v_0^2$. At $\theta=0$, the full/isotropic values are
0.0194622097/0.0884218390; at $\theta=\pi/2$, they are
0.1434670359/0.0982482450. The posterior-normal spheres are unchanged.

The moving-sphere prediction-history figure also contains the existing
identity-transport control, labelled "Posterior mean, $G_t=I_3$". The measured-transport posterior mean is blue, and the identity-transport
control is dark gray; both are solid and markerless. The revised history
uses all 48 original action indices after independent replay verification,
while the retained supplementary tracking figure uses its six stored checkpoints. Its whole-run difference uses `endpoint=episode_average`, not an average
over the sparse checkpoints: identity minus known transport is 2.65482464,
with paired 95% Student interval [2.09561102, 3.21403827] $(\mathrm{mm/s})^2$.
No new contact simulations or parameter selections were run.

## Verification and provenance

`results/application_plotting_update/` records the current PDF inspection,
border checks and exact before/after presentation-file hashes. Preservation
checks accept only those explicit presentation changes; original scientific
data, library implementations, settings and benchmark records remain protected.
`results/contact_plotting_addendum/` retains the separate contact checks and
the intermediate generator source used for its earlier stage.
`results/application_plotting_update/changed_paths.json` lists every created
and modified path relative to the starting `test` commit. The final test and
export summary is `results/application_plotting_update/verification.json`.

The separate beamforming comparison audits, including both GREAT variants,
are documented in `docs/beamforming_comparison_audit.md`; they do not replace
the four-method moving figure or any stationary manuscript result.


## Captions and statistical definitions

The [caption catalogue](solid_trajectory_captions.md) covers all eleven main
PDF exports and the two supplementary contact diagnostic figures. It records
method colors, actual plotted coordinates, simulation counts and band meanings;
the matching TeX snippets use only equation references already present in the
provided numerical section. The two moving layouts are alternative displays
of the same evidence.

Band meanings remain experiment-specific. The beamforming variance panels
show medians and empirical interquartile bands, whereas their power and log
density panels show means with Student 95% confidence intervals. Stationary
beamforming and contact prediction errors also use Student intervals for
Monte Carlo means. The controlled contact likelihood, normal-density and
response-variance plots evaluate a saved model deterministically and have no
Monte Carlo confidence bands. The numerical-section prose must use the actual
fixed $T=200$ stationary comparison and the current $e_1,e_2$ response-variance
scan, rather than its earlier $t=n$ and $e_2,e_3$ versions.

The new `results/solid_trajectory_update/caption_inventory.json` is a factual
read-only audit. It also lists the 87 application-notebook export routes and
38 preserved canonical documents, distinguishing individually checked
statistics from inherited supplementary metadata.
