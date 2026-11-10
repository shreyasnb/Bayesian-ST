"""Expose the theory notebook's plotting cells without changing its checks.

The scientific notebook generator is part of the frozen numerical core and is
left intact. Run this presentation-only conversion after that generator if a
fresh theory notebook is needed. Already editable notebooks are left alone so
this command never overwrites a reader's plotting adjustments.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import nbformat


ROOT = Path(__file__).resolve().parents[1]


def refactor(path=ROOT/'notebooks/00_theory.ipynb'):
    path = Path(path)
    original = path.read_bytes()
    nb = nbformat.read(path, as_version=4)
    if nb.metadata.get('editable_plotting', {}).get('mode') == 'inline_matplotlib':
        return False
    intro = (
        '\n\nPlotting code is written out in the figure cells below. Each cell '
        'copies the preceding calculation into `plot_data`, then exposes '
        '`DPI`, `FIGSIZE`, every artist and label, the PDF path and '
        '`plt.show()`. Edit a plotting cell without rerunning the numerical '
        'study that precedes it. A normal Run All writes new validation '
        'records under `results/notebook_runs/00_theory/` and editable PDFs '
        'under `results/notebook_exports/00_theory/`; the frozen study and '
        'its original certificates are preserved.'
    )
    nb.cells[0].source += intro
    setup = nb.cells[1].source.replace('from lib.plotting import plt, save',
        'from lib.plotting import plt\nfrom IPython import get_ipython\n'
        'if get_ipython() is not None:\n'
        "    get_ipython().run_line_magic('matplotlib', 'inline')\n"
        "plt.rcParams.update({\n"
        "    'text.usetex': True, 'font.family': 'serif', 'font.size': 11,\n"
        "    'axes.labelsize': 11, 'axes.titlesize': 11, 'legend.fontsize': 10,\n"
        "    'xtick.labelsize': 10, 'ytick.labelsize': 10,\n"
        "    'lines.linewidth': 1.6, 'lines.markersize': 4,\n"
        "    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': None,\n"
        '})')
    assignment = "OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)"
    if setup.count(assignment) != 1:
        raise RuntimeError('Unexpected theory output assignment')
    setup = setup.replace(assignment,
        '# The existing isolated scientific validator recognizes this assignment.\n'
        + assignment + '\n'
        '# Its injected OUT is respected; an ordinary Run All uses private outputs.\n'
        "if OUT == ROOT/'results':\n"
        "    OUT=ROOT/'results/notebook_runs/00_theory'\n"
        'OUT.mkdir(parents=True,exist_ok=True)\n'
        "EDITABLE_OUTPUT=ROOT/'results/notebook_exports/00_theory'\n"
        'EDITABLE_OUTPUT.mkdir(parents=True,exist_ok=True)')
    nb.cells[1].source = setup
    specs = {
        'info': ('theory_information', (12., 4.5), 2),
        'track': ('theory_tracking', (15., 4.5), 3),
        'directional': ('theory_directional', (12., 4.5), 2),
        'scale': ('theory_scaling', (12., 4.5), 2),
    }
    converted, cells, numeric_hashes = [], [], {}
    for cell in nb.cells:
        if cell.cell_type != 'code' or 'fig,axs=plt.subplots' not in cell.source:
            cells.append(cell)
            continue
        numerical, artists = cell.source.split('fig,axs=plt.subplots', 1)
        key = next((name for name in specs if f'{name}.' in artists), None)
        if key is None:
            raise RuntimeError('Unrecognized theory plotting data')
        stem, figsize, count = specs[key]
        if key == 'track':
            # This data export belongs with its unchanged calculation, not the
            # plotting cell that readers may execute many times.
            export = "track.to_csv(OUT/'theory_tracking.csv',index=False)"
            if artists.count(export) != 1:
                raise RuntimeError('Unexpected tracking data export')
            artists = artists.replace(export, '')
            numerical += export + '\n'
        numerical = numerical.rstrip() + '\n'
        cell.source = numerical
        cell.metadata['editable_plotting_role'] = 'calculation'
        numeric_hashes[cell.id] = hashlib.sha256(numerical.encode()).hexdigest()
        cells.append(cell)
        cells.append(nbformat.v4.new_markdown_cell(
            f'### Editable plot: {stem.replace("theory_", "").replace("_", " ")}\n\n'
            f'The numerical table `{key}` was computed above. Only this cell '
            'needs to be rerun after editing the plot settings or labels.'))
        body = 'fig,axs=plt.subplots' + artists
        old_constructor = f'fig,axs=plt.subplots(1,{count},figsize=({int(figsize[0])},4.5))'
        new_constructor = (f'fig,axs=plt.subplots(1,{count},figsize=FIGSIZE, '
                           'dpi=DPI, layout="constrained")')
        if body.count(old_constructor) != 1:
            raise RuntimeError('Unexpected original figure constructor')
        body = body.replace(old_constructor, new_constructor)
        body = body.replace(f'{key}.', 'plot_data.')
        save = f"save(fig,OUT/'{stem}');plt.show()"
        if body.count(save) != 1:
            raise RuntimeError('Unexpected original figure save')
        body = body.replace(save,
            'fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches=None)\nplt.show()')
        source = (f'plot_data = {key}.copy()\n'
                  'DPI = 300\n'
                  f'FIGSIZE = {figsize!r}\n'
                  f'OUTPUT_PDF = EDITABLE_OUTPUT / "{stem}.pdf"\n\n'
                  + body.strip())
        plotted = nbformat.v4.new_code_cell(source)
        plotted.metadata.update(editable_plotting_role='plot', figure_id=stem)
        cells.append(plotted)
        converted.append(stem)
    if set(converted) != {value[0] for value in specs.values()}:
        raise RuntimeError('Not every theory figure was converted')
    nb.cells = cells
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    nb.metadata['editable_plotting'] = dict(
        schema=1, mode='inline_matplotlib', source_notebook='00_theory.ipynb',
        producer='scripts/refactor_theory_plots.py',
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_notebook_sha256=hashlib.sha256(original).hexdigest(),
        figure_ids=converted, numerical_cell_hashes=numeric_hashes,
        output_directory='results/notebook_exports/00_theory',
        validation_directory='results/notebook_runs/00_theory',
        scope='Original numerical checks; visible plotting code and private output paths only')
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return True


if __name__ == '__main__':
    print('Theory plotting cells converted.' if refactor() else
          'Theory plotting cells are already editable; no changes made.')
