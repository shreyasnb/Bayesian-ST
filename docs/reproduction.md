# Camera-ready reproduction

The root README is intentionally preserved from the earlier release. This file
describes the camera-ready script interface.

Use the existing checkout. CPython 3.12 and the pinned direct dependencies in
`requirements-lock.txt` reproduce the validated numerical stack. Install with:

```sh
bash scripts/setup.sh
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
```

Figures require LaTeX, `type1cm`, `type1ec`, and `dvipng`. On Debian/Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends texlive-latex-extra texlive-fonts-recommended cm-super dvipng poppler-utils
```

`pdfinfo` from Poppler checks PDF page counts and dimensions. The prepared cloud
image also supports the local TeX tools under `/workspace/.local/tex`; plotting
activates them when present. No notebooks, Jupyter packages, credentials, or
running services are required.

## Commands

From the checkout root:

```sh
python scripts/export_paper.py
python scripts/check_paper.py
python -m pytest -q
```

The first command redraws the five single-page figures and exports compact CSV
and TeX tables from the committed paper cohort; it does not simulate or retune.
The second independently checks the manuscript's rounded numerical claims,
paired intervals, stationary table, cohort sizes, and PDF layouts.

To regenerate the original simulations with frozen settings:

```sh
python scripts/run_contact.py --output results/rerun/contact/data --workers 4
python scripts/run_beamforming.py --output results/rerun/beamforming/data --workers 4
python scripts/export_paper.py --results results/rerun
python scripts/check_paper.py --results results/rerun
```

Or run all simulations and focused checks with one command:

```sh
python scripts/reproduce.py --workers 4
```

Reruns go into ignored `results/rerun/`, preserving the committed paper cohort.
Paths supplied to commands are resolved from the caller's working directory;
default data/output paths resolve from the script's location. Figures can be
redrawn after a rerun without rerunning the simulation.

## Frozen protocols and producers

The audited numerical source is `test` commit
`ec998a933fd693c2f5bdb62c6bf5654e9345853a`; the release starts from `main`
`57f520cbe860b73e13d71bae5c31391e2a009482`. Neither branch's history is rewritten.
The final supplied `l4dc(25).tex`, `supplementary(27).tex`, and
`Bayesian_Subspace_Tracking-11.pdf` define the retained experiments. Their TeX
build dependencies were not supplied as a complete paper source tree.

| Paper output | Numerical producer | Export |
| --- | --- | --- |
| Fig. 1, `results/contact/contact_comparison.pdf` | `final_contact` fixed-plane fixture and dense sphere transport | `paper_plots` |
| Fig. 2, `results/beamforming/moving_interference_comparison.pdf` | independent smooth/abrupt moving interference | `paper_plots` |
| Fig. 3, `results/contact/information_rotation.pdf` | fixed-plane conditional information | `paper_plots` |
| Fig. 4, `results/contact/posterior_anisotropy_spheres.pdf` | real normal Bingham quadrature | `paper_plots` |
| Fig. 5, `results/beamforming/noise_adaptation.pdf` | pre-correction prediction with future-only block EM | `paper_plots` |
| Stationary settings/results, moving averages | frozen per-trial beamforming records | beamforming table exporter |

Configurations under `configs/` retain exact seeds, reporting times, numerical
budgets, and unrounded selected baseline parameters. Data provenance records
identify original files and their hashes. `results/manifest.json` lists the
retained artifact checksums and figure specifications. Small source-generated
fixed-seed fixtures under `tests/fixtures/` test the transfer independently of
saved aggregate tables.

Contact keeps the 16-measurement fixed plane and 192-measurement dense sphere
cohort, with real moment resolution comparisons at `5e-8`. Beamforming keeps
stationary `T=200` at 32/64/128 sensors, independent moving interference at
32 sensors and `T=256`, and fixed-direction noise change at measurement 129.
Decision moments use `1e-10` and variance learning uses `1e-9`. Close/repeated
eigenvalues and the constrained variance-floor update remain checked.

The default commands use frozen selections from 64 separate validation
simulations. They do not repeat parameter selection or use hidden states for
tuning. Full cohorts contain 256 independent simulations. Time means precede
between-simulation means; differences are paired and intervals use Student's
`t`. Contact bands are pointwise confidence intervals for mean errors; moving
beamforming shades the middle 50% of values. Noise variance, power, and score
bands retain their separate definitions from the source.
