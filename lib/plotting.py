"""Shared paper plots; TeX is a required runtime dependency.

PDF text and lines remain vector graphics. DPI controls notebook previews and
rasterized PDF layers; it does not rasterize an otherwise vector PDF.
"""
import os
from pathlib import Path
_local_tex = Path('/workspace/.local/tex')
if (_local_tex / 'usr/bin/dvipng').exists():
    os.environ['PATH'] = str(_local_tex / 'usr/bin') + os.pathsep + os.environ['PATH']
    os.environ.setdefault('TEXMFHOME', str(_local_tex / 'usr/share/texmf'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/bayesian-st-cache')
import matplotlib as mpl
DPI = 300
PAPER_WIDTH_IN = 7.0
mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'lines.linewidth': 1.35,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.figsize': (PAPER_WIDTH_IN, 3.2), 'figure.dpi': DPI,
    'savefig.bbox': None, 'savefig.dpi': DPI,
})
import matplotlib.pyplot as plt


def save(fig, path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(DPI)
    fig.savefig(path.with_suffix('.pdf'), dpi=DPI)
    fig.savefig(path.with_suffix('.png'), dpi=DPI)
