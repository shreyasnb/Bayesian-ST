# Geometry, inference and decisions

This is the separate draft-5 redesign of the beamforming and contact-normal notebooks on branch `test`, based on commit `2a8697c7e7fa27b3779f3abeaed32ca8b323a9ef`. Earlier results are preserved. See the [protocol](../../docs/simulation_redesign.md) and [notation map](../../docs/simulation_redesign_notation.md).

The four scientific components are `geometry`, `beamforming`, `contact` and `scaling`, each under `quick/` and `paper/`. Component manifests record configuration, seeds, source hashes, compact table hashes and numerical status. Quick data are engineering checks, not the paper comparison. Predeclared display examples do not replace the independent trajectory cohorts.

## Reproduce scientific outputs

Use the repository dependencies, working LaTeX/dvipng and the shared `lib.plotting` settings. All figures require `text.usetex=True`, 200-dpi display/export and PDF plus PNG output.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/validate_simulation_redesign.py
python scripts/run_anisotropy_study.py --profile quick
python scripts/run_array_geometry.py --profile quick --workers 2
python scripts/run_contact_geometry.py --profile quick --workers 2
python scripts/run_simulation_scaling.py --profile quick
python scripts/plot_geometry_studies.py --profile quick
python scripts/run_anisotropy_study.py --profile paper
python scripts/run_array_geometry.py --profile paper --workers 2
python scripts/run_contact_geometry.py --profile paper --workers 2
python scripts/run_simulation_scaling.py --profile paper
python scripts/plot_geometry_studies.py --profile paper
python scripts/summarize_simulation_redesign.py --profile paper
python scripts/make_geometry_notebooks.py
python scripts/execute_geometry_notebooks.py
python scripts/manifest_simulation_redesign.py
python scripts/check_simulation_redesign_handoff.py
```

Run final scaling serially after all other simulation processes finish. The general notebook is executed into `validation/`; original certificates and older result files remain unchanged. Scientific source changes invalidate the new gate. Completed caches verify their source/configuration/table hashes. Preserve an obsolete or damaged run and regenerate it into a fresh output directory; do not suppress validation checks or rewrite favorable subsets.

Full worker checkpoints are excluded from Git. Every numerical flag and compact paired metric is retained. Checkpoints and raw draws can be regenerated from the seeded runners and recorded configuration; clock/timestamp fields naturally vary. Archive files retain the previous executed notebooks and exact earlier validation sources. The handoff manifest lists all retained artifacts and excluded regenerable files.

The final findings and two short application interpretations are generated only after all four paper components and the final figure manifest verify. The main notebooks embed these interpretations and six figures each. Detailed paired outcomes, including losses and unresolved comparisons, remain in the linked [findings](findings.md). Changing a figure producer requires rerendering both profiles before the read-only handoff check; changing scientific sources requires rerunning the general gate and affected simulations.
