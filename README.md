# Bayesian subspace tracking

Implementation and simulations for **Bayesian Subspace Tracking with Grassmann Bingham Distributions**. The current `test` study follows the author's [final simulation guide](docs/final_simulation_guide.md), **v5.tex** as the main manuscript, and **supplementary(14).tex** as its supplement. The old numerical prose does not define the new experiments.

Start with the executed [beamforming notebook](notebooks/01_array.ipynb) or [contact notebook](notebooks/03_contact.ipynb). Each develops the observation model, information, posterior geometry, decisions, comparisons, variance learning and computational cost in order. Figures use LaTeX, 300 dpi previews and raster layers, vector PDFs, 11-point axes and 10-point ticks/legends. The published catalogue uses a nominal 7-inch width; editable cells retain the author’s canvases, with selected panels split into separate figures at the same notebook locations. Captions and detailed claims are outside the title-free PDFs. Manuscript selections are organized under [results/main/](results/main/) and [results/supplementary/](results/supplementary/), with page-level captions and provenance.

All five root notebooks now contain full editable Matplotlib code in their
figure cells. Edit `DPI`, `FIGSIZE`, labels, colors and PDF saves directly;
data-only helpers load the saved arrays. New exports go to
`results/notebook_exports/` so the published figures stay intact. See
[editing and running notebook figures](docs/editable_notebooks.md).

The [completion report](completion_report.md) records completed cohorts, mixed comparisons, numerical failures and unsupported claims. The [mathematical checks](docs/final_manuscript_checks.md) explain identity transport, the sharp modal bound, fixed-boundary EM and the limits of dynamic uncertainty claims.

## Results

| Evidence | Beamforming | Contact |
| --- | --- | --- |
| Paper summary | [PDF](results/beamforming/paper_summary.pdf) | [PDF](results/contact/paper_summary.pdf) |
| Figures and numerical records | [Directory](results/beamforming/) | [Directory](results/contact/) |
| Figure claims, sources and hashes | [Manifest](results/beamforming/figure_manifest.json) | [Manifest](results/contact/figure_manifest.json) |
| Proposed captions | [Notes](results/beamforming/caption_notes.md) | [Notes](results/contact/caption_notes.md) |

Beamforming task cohorts use proper-complex rank-one interference at 32, 64 and 128 sensors, with observation counts scaled to dimension. Moving studies use 32 sensors. Contact retains a physical three-dimensional maintained plane/sphere constraint with known isotropic resistance and measured wrist motion. Higher-dimensional real fixtures concern generalized-coordinate numerical cost, rather than a complete robot controller.

Both applications compare the posterior mean, the same-parameter mode and a matched isotropic control. GREAT uses finite Grassmann exponential steps and a recursively maintained raw covariance. Its native complex extension is explicitly labeled. Beamforming also retains independently validated modal/covariance loading, GROUSE and Ledoit--Wolf references. Contact GREAT receives measured responses only; conditional estimators also know applied inputs.

## Reproduce and verify

Use Python 3.12 and working TeX tools (`latex`, `dvipng`, `type1cm`, `type1ec`). Set up with `bash scripts/setup.sh`; activate `.venv` if that script created it. Every Matplotlib figure requires `text.usetex=True`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_final_simulations.py
```

This read-only check verifies the general gate, all stage/component producer and data hashes, historical preservation, the final PDFs/previews and executed notebooks. See the [reproduction protocol](docs/final_simulations.md) for the verify, pilot, frozen paper, isolated timing, rendering and notebook commands. Scientific calculations and rendering are separate; changing a camera does not repeat Monte Carlo experiments.

## Code

Reusable inference is in `lib/`. The final study adds:

- `final_theory.py`: exact discounted/transported blocks, sharp tracking certificates, local curvature and accuracy diagnostics.
- `final_beamforming.py` and `final_contact.py`: fixed paired cohorts and observed-data validation policies.
- `causal_variance.py` and `final_prediction.py`: future-only block variance learning and actual-recursion predictions.
- `final_computation.py` and `final_accuracy.py`: costs at stated task-decision intervals and independent high-precision integration.
- `final_paper_visuals.py`: render-only figures from saved verified arrays.

Shared GREAT is in `benchmarks/common/great.py`; application references remain in `benchmarks/case1/` and `benchmarks/case2/`. [GREAT provenance](docs/great_reference.md) distinguishes independent mathematical verification from unavailable direct author-code/PDF parity.

The original `BinghamFilter` supports arbitrary subspace rank for corrections. Deterministic native means/normalizers scale for rank one or codimension one. General-rank moment sampling requires its own convergence diagnostics; general-rank Fisher checks do not establish scalable normalized inference.

## Interpretation

A distribution adds directional moments, uncertainty, predictive scores, variance fitting and task-dependent decisions. It does not make a point estimate uniformly better or cheaper. With uniform initialization and matching fixed scales/memory, the marginal mode equals covariance PCA. Mean-versus-mode gains alone do not establish an anisotropy advantage; matched isotropic and validated loading controls address that question.

Identity transport is permitted when motion is unknown. Forgetting is then an approximate prediction, and posterior coverage need not hold under unmodeled drift. The fixed-variance modal bound has explicit excitation/drift conditions; its ineligible cases and adaptive-variance exclusions remain visible. See the completion report for the actual supported comparisons.

## Historical notebooks and evidence

| Notebook | Status |
| --- | --- |
| [00_theory](notebooks/00_theory.ipynb) | General validation with unchanged numerical checks and editable figures; fresh diagnostics use `results/notebook_runs/00_theory/` |
| [01_array](notebooks/01_array.ipynb) | Current beamforming walkthrough |
| [02_field](notebooks/02_field.ipynb) | Editable figures from verified historical field data, runnable in the current environment |
| [03_contact](notebooks/03_contact.ipynb) | Current contact walkthrough |
| [04_implementation_audit](notebooks/04_implementation_audit.ipynb) | Editable figures from verified historical implementation-audit data, runnable in the current environment |

The [starting-revision archive](archive/a91faab/preservation.json) preserves 229 original source/notebook files and binds 2,358 historical result files left in place. Original certificates have not been refreshed to claim new producers. Reproducing the archived simulations requires their matching commit/source tree; the five current root notebooks expose editable plotting code and run in the current environment. Earlier redesign, implementation-audit, contact and field evidence remains in its original directories. The `main` and `dev` branches are unchanged.
