# Reproducing the paper examples

Use CPython 3.12 and install the pinned numerical dependencies:

```sh
bash scripts/setup.sh
source .venv/bin/activate
```

PDF rendering requires LaTeX, `type1cm`, `type1ec`, and `dvipng`. On Debian or
Ubuntu install `texlive-latex-extra texlive-fonts-recommended cm-super dvipng`.
The root README is retained from the earlier release; the commands below are
the current public interface.

## Redraw the frozen paper plots

```sh
python scripts/export_paper.py
```

Use `--example contact` or `--example beamforming` to export one example only.
This reads the committed simulation records and does not run simulations.

## Rerun the contact example

```sh
python scripts/run_contact.py --workers 4
python scripts/export_paper.py --results results/rerun --example contact
```

This reproduces the fixed-plane illustration and the 256-trial moving-contact
experiment, then writes `contact_comparison.pdf`, `information_rotation.pdf`,
and `posterior_anisotropy_spheres.pdf` under `results/rerun/contact/`.

## Rerun the beamforming example

```sh
python scripts/run_beamforming.py --workers 4
python scripts/export_paper.py --results results/rerun --example beamforming
```

This reproduces stationary beamforming at 32/64/128 sensors, smooth and abrupt
moving interference, and noise adaptation. It writes
`moving_interference_comparison.pdf` and `noise_adaptation.pdf` under
`results/rerun/beamforming/`. `--experiment stationary`, `moving`, or `noise`
runs a single part; both moving and noise records are needed for the two plots.

Both examples use the frozen seeds and selected parameters in `configs/`.
The default output location is ignored by Git and preserves the committed
paper results. For another run, choose a fresh `--output` data directory and
pass its enclosing results directory to the plot exporter. Existing simulation
files are never overwritten. Commands may run outside the checkout; relative
paths you supply are resolved from your current directory.

The numerical protocols use 256 independent trials with the original frozen
validation selections. Plotting retains pointwise mean confidence intervals
for contact, middle-50% bands for moving interference, and the distinct
variance, power and score bands in the noise experiment. Numerical accuracy
and EM validity guards remain in the simulation routines.

The code retains the numerical producers from `Bayesian-ST` commit
`ec998a933fd693c2f5bdb62c6bf5654e9345853a`. GREAT was compared against
[`gerost/great.m`](https://github.com/shreyasnb/gerost/blob/a5c47c07a3894fe21641a84537741afd1c8be604/great.m).
Frozen paper tables remain under `results/beamforming/tables/`.
