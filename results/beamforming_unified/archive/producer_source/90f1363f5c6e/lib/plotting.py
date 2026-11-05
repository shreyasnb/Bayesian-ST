"""Shared PMLR-sized plots; TeX is a required runtime dependency."""
import os
from pathlib import Path
_local_tex = Path('/workspace/.local/tex')
if (_local_tex / 'usr/bin/dvipng').exists():
    os.environ['PATH'] = str(_local_tex / 'usr/bin') + os.pathsep + os.environ['PATH']
    os.environ.setdefault('TEXMFHOME', str(_local_tex / 'usr/share/texmf'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
import matplotlib as mpl
mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'lines.linewidth': 1.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.figsize': (6.5, 2.5), 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})
import matplotlib.pyplot as plt


def save(fig, path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'))
