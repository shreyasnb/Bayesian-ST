# Paper-width figure exports

The main numerical section now uses `results/main/contact/contact_comparison.pdf`
and `results/main/beamforming/moving_interference_comparison.pdf`. The contact
comparison places posterior velocity variance on the left and tracking error on
the right, with the prediction-error panel's four-entry legend centered in one
row. Individual basenames, saved inputs, simulation settings, methods,
statistics, notation and colours are unchanged.

| Figure basename | Canvas, inches | Tight PDF, inches | Current manuscript directory |
| --- | --- | --- | --- |
| `posterior_anisotropy_risk` | 7 × 2.1 | 7.000 × 2.100 | `results/main/contact/` |
| `prediction_error_sphere` | 7 × 2.4 | 7.000 × 2.371 | `results/main/contact/` |
| `contact_comparison` | 7 × 2.8 | 7.000 × 2.771 | `results/main/contact/` |
| `moving_interference_comparison` | 7 × 2.8 | 6.999 × 2.767 | `results/main/beamforming/` |
| `information_rotation` | 7 × 2.6 | 7.000 × 2.600 | `results/supplementary/contact/` |
| `posterior_anisotropy_spheres` | 7 × 3.1 | 7.000 × 3.100 | `results/supplementary/contact/` |
| `moving_sphere_posterior_spheres` | 7 × 3.2 | 6.898 × 3.171 | `results/supplementary/contact/` |
| `noise_adaptation_page1` (beamforming) | 7 × 4.5 | 6.961 × 4.437 | `results/supplementary/beamforming/` |

The separate smooth/abrupt beamforming exports retain 15 × 5-inch canvases;
tight export trims their PDF heights to approximately 4.887 inches. Existing
main-directory copies of supplementary figures are retained for compatibility.
All updated individual and tiled figures have 300-dpi PNG previews alongside
their PDFs, plus their existing notebook-local export copies.

## Re-export from saved results

Activate `.venv` and run only the setup/data-loading and selected plot cells:

- `notebooks/03_contact.ipynb`: setup, information rotation, and posterior
  anisotropy cells. The contour plot sharing the information cell is unchanged.
- `notebooks/contact_prediction_diagnostics.ipynb`: setup,
  `prediction-error-sphere` and `moving-sphere-posterior-spheres`.
- `notebooks/beamforming_moving_interference.ipynb`: setup/data loading and the
  three output-power cells.
- `notebooks/01_array.ipynb`: setup and the first noise-adaptation plot cell.

Their visible `FIGSIZE`, font and save controls are editable. Source generators
carry the same layouts. No simulation runner is required. Export the side-by-side
contact comparison directly from the same saved arrays with:

```bash
.venv/bin/python scripts/export_contact_comparison.py
```

The comparison exporter uses the installed scientific Python stack and TeX.
It reads the same saved unit-query risk arrays and contact-transport curves used
by the individual notebooks, retaining their trajectory coordinates, colours,
notation, and uncertainty band. It redraws only the combined layout, with vector
text and lines and a 300-dpi preview. Individual exports remain unchanged.

## Inspection

The exported PDFs were rendered at a 6.5-inch paper width for visual review.
Primary labels, ticks and legends print at approximately 9.25–9.86 pt. All text
fonts are embedded Type1; curves and borders remain vector objects. Both sphere
pairs have one common logarithmic colour bar, with labels inside their tight
page bounds. No TeX cropping is required. The obsolete risk `(c)` label is gone;
the information-rotation explanatory guide and highlighted scientific points
remain. Uncertainty bands, solid trajectories, and four-sided 2D borders are
preserved. Beamforming event lines are identified in the common legend.

`results/paper_figure_sizing_checks.json` records measured PDF dimensions,
printed font sizes, preview resolution, bounds and independent visual review.
The numerical-artifact preservation check found all 4,094 saved numeric files,
59 configuration JSONs and 77 scientific modules unchanged. Rendered-artist
comparisons against the original notebooks also verified trajectory coordinates,
band polygons, colours, line styles, event coordinates and explanatory guides.
Read-only arithmetic checks reconstructed the beamforming and contact summaries
from the original saved trials; no simulations were rerun.

The manuscript PDF audit supports tight bounding boxes and a `--figure` filter
for checking selected notebook exports. A full historical audit also expects
local exports for unrelated figures, which need not exist in a fresh checkout.
