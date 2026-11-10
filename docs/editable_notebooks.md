# Editing notebook figures

All five notebooks in `notebooks/` contain their plotting code in the figure
cells. They no longer display pre-rendered images through `show()` or
`show_saved()`. The numerical library and published results are unchanged.

Run the setup cell, then edit and rerun the figure cell of interest. In the
theory notebook, first run the calculation cell that supplies that figure's
table. Each figure cell exposes:

- A data-loading line or a copy of an already calculated local table.
- `DPI` and `FIGSIZE`, passed directly to `plt.figure()` or `plt.subplots()`.
- The actual Matplotlib artists, labels, ticks, legends, colors and layout.
- A direct `fig.savefig(...)` command and `plt.show()`.

Shared `plt.rcParams` are visible in each notebook's setup cell. Defaults keep
LaTeX enabled, use 300 dpi, and follow the implementation notebook's serif
typography. These are editable defaults: there is no hidden function resizing
the figure or enforcing a particular dpi after the cell creates it.

The data readers in `lib/notebook_*_data.py` load and check saved numerical
arrays. They do not run a simulator, fit a posterior, or draw a figure. The
contact reader completes its existing numerical-module imports before the
notebook applies its visible plotting settings, so a later data load does not
reset those settings.

## PDF output

New PDFs are written under `results/notebook_exports/<notebook_name>/`. The
path and save call are visible in each figure cell; edit them or disable the
save if desired. These local exports are ignored by Git. The published PDFs
under `results/main/`, `results/supplementary/` and the application directories
retain their original files and provenance.

An ordinary Run All in `00_theory.ipynb` writes its new diagnostic CSVs and
validation certificate under `results/notebook_runs/00_theory/`. Its numerical
checks are unchanged; the separate output directory protects the frozen study.
The field and implementation notebooks read their historical saved data
without invoking stale simulation producers.

## Run and check

These commands execute the committed notebooks without rebuilding their cells:

```bash
python scripts/execute_editable_notebooks.py
python scripts/check_editable_notebooks.py --check-pdfs
python scripts/check_final_simulations.py
```

Select a single notebook with, for example,
`--notebook 03_contact.ipynb`. The optional `--check-pdfs` checks local exports;
omit it when inspecting a fresh checkout before running the notebooks.
Validation checks saved inputs and successful execution, rather than requiring
notebook images to match the published PNG bytes. Editing dpi, figsize or labels
does not require changing a figure manifest.

## Regenerate default cells

Regeneration deliberately resets the selected notebook's plotting source:

```bash
python scripts/make_editable_notebooks.py --notebook 03_contact
```

The application templates are ordinary Python source in
`scripts/notebook_plot_cells_beamforming.py` and
`scripts/notebook_plot_cells_contact.py`; their code is copied into the notebook,
not imported as a renderer at execution time. Historical field/audit templates
are source-only JSON under `scripts/notebook_templates/`.

The scientific `make_theory_notebook.py` generator remains unchanged. The
presentation-only `refactor_theory_plots.py` converts its output to editable
cells, and leaves an already editable theory notebook alone. Use the unified
builder above for the current workflow; older generators are historical tools.
