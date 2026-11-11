# Plotting protocol

Current editable application figures use actual LaTeX with serif text,
`text.usetex=True`, and 300 dpi for PNG previews and raster layers in PDFs.
PDF text and ordinary curves remain vector objects: increasing dpi does not
increase their font size. Readability at final insertion depends on the
canvas width, text size and number of panels.

| Output | Canvas | Axis text / ticks / legend |
| --- | --- | --- |
| Ordinary editable application plots | 15 × 5 in by default | 25 / 23 / 22 pt |
| Beamforming variance-estimation four-panel figure | 15 × 7 in | 25 / 23 / 22 pt |
| Compact moving-beamforming comparison | 7 × 3.5 in | 12 / 11 / 10.5 pt |
| Preserved canonical catalogue | 7 in wide, figure-specific height | Retains its recorded production settings |

The application source generator scales the 25/23/22-point controls by
`FIGSIZE[0]/15` where an individual canvas has a different width. Existing
3D views retain their local controls, geometry and color normalization.
Historical notebooks and canonical PDFs are preserved separately; their
recorded settings are not rewritten to imply that they were regenerated.

The complete Matplotlib code is visible in each figure cell. It includes
local `DPI`, `FIGSIZE`, font, label and save controls, followed by the figure
constructor, artists, PDF save and `plt.show()`. Running a cell with
`SAVE_PDF=True` writes its local PDF and an identical copy to its declared
`results/main/` or `results/supplementary/` destination. Selected figures are
split without moving their position in the notebook; the paired contact
normal-density spheres remain side by side and their response-variance curve
is separate.

Ordinary 2D method trajectories are solid and markerless, retaining their
existing colors and mathematical method labels. This does not remove markers
from scientific point comparisons, confidence bars, scatter plots, coverage
estimates, or the modal/true-normal markers on density spheres. Dashed event
and reference lines retain their specific meaning. The purple vertical line
at $t=129$ denotes an interference-direction change in the moving beamforming
experiment and a noise-variance change in the separate variance-estimation
experiment. It does not denote a sensor fault. Current tiled manuscript
figures use positional descriptions in their captions instead of panel letters.

Ordinary 2D notebook plots have four visible borders, 0.9–0.95 pt wide, with
inward ticks on all four sides and tick labels on the bottom and left. The
3D density-sphere presentation and colorbars retain their separate controls.

Shaded regions have explicitly different statistical meanings:

| Figure family | Center | Shading or interval |
| --- | --- | --- |
| Moving beamforming output power | Arithmetic mean of individual dB values, 256 simulations | Pointwise empirical 25th–75th percentiles for all four methods |
| Beamforming variance estimates | Median across 256 simulations | Pointwise empirical 25th–75th percentiles for EM estimates |
| Beamforming noise-experiment power and log density | Arithmetic mean across 256 simulations | Pointwise Student 95% confidence intervals for all three methods |
| Stationary fixed-$T=200$ beamforming | Arithmetic mean across 256 simulations | Pointwise Student 95% confidence intervals for the posterior mean |
| Contact stationary/time-history errors | Arithmetic mean across 256 simulations | Pointwise Student 95% confidence intervals for the measured-transport posterior mean |
| Contact full-run error point comparisons | Mean of per-simulation averages, 256 simulations | Student 95% confidence intervals for every displayed method |
| Controlled contact KL, normal density and response variance | Deterministic evaluations of one design or saved posterior | No Monte Carlo confidence band |

The pointwise bands are not simultaneous bands, and none is automatically a
posterior credible region. Contact full-run comparisons average all 48 action
indices within each simulation. The independent sphere prediction history
uses those 48 original action evaluations after replay verification; the
retained supplementary tracking figure still uses its six stored checkpoints. The main controlled contact response-variance scan
uses $v/v_0=(\cos\theta,\sin\theta,0)$ and each law's own mean. Density
colorbars show explicitly normalized density on their recorded logarithmic
scale; colored display regions are not credible-region boundaries.

Numerical tables retain their original confidence intervals and paired
comparisons. No smoothing, jitter, interpolation, synthetic replications,
extra coordinates, inference, estimator tuning or benchmark changes are
introduced by a style render. Copy-pastable manuscript captions and exact
statistical definitions are in [solid_trajectory_captions.tex](solid_trajectory_captions.tex)
and [solid_trajectory_captions.md](solid_trajectory_captions.md).
