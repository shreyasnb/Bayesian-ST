"""Shared TeX plots; TeX is a required runtime dependency."""
import os
from pathlib import Path
_local_tex = Path('/workspace/.local/tex')
if (_local_tex / 'usr/bin/dvipng').exists():
    os.environ['PATH'] = str(_local_tex / 'usr/bin') + os.pathsep + os.environ['PATH']
    os.environ.setdefault('TEXMFHOME', str(_local_tex / 'usr/share/texmf'))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
import matplotlib as mpl
mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 1.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.figsize': (12, 5), 'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
})
import matplotlib.pyplot as plt


def save(fig, path):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'), dpi=200)
    fig.savefig(path.with_suffix('.png'), dpi=200)
