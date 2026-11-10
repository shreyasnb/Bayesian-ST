"""Source templates for directly editable supplementary notebook figures.

The returned Matplotlib statements are copied into the notebook. This module
does not render figures or produce numerical results.
"""
from __future__ import annotations

from textwrap import dedent


STYLE_SOURCE = r'''
LABELS = {
    'mean': r'$\overline P_t^+$',
    'mode': r'$P_{\widehat{\mathcal X}_t}$',
    'iso': r'$\overline P_t^{\mathrm{iso}}$',
    'isotropic': r'$\overline P_t^{\mathrm{iso}}$',
    'mode_loaded': 'Loaded modal beamformer',
    'modal_loading': 'Loaded modal beamformer',
    'covariance_loading': 'MVDR with covariance shrinkage',
    'scm': 'MVDR with covariance shrinkage',
    'scm_loaded': 'MVDR with covariance shrinkage',
    'ew_covariance': 'Sample covariance',
    'regularized': 'Regularized projector',
    'lw': 'Ledoit--Wolf', 'ledoit_wolf': 'Ledoit--Wolf',
    'grouse': 'GROUSE',
    'great': 'GREAT (sliding window)',
    'great_window': 'GREAT (sliding window)',
    'great_ew': 'GREAT (exponential weighting)',
    'oracle': 'Ideal MVDR',
    'known': 'True variances',
    'exact_known': r'True variances, $\beta=1$',
    'known_forgetting': 'True variances, forgetting',
    'misspecified': 'Fixed variances',
    'learned': 'Learned variances',
    'block_em': 'Block EM', 'residual': 'Residual fit',
    'ridge': 'Ridge regression', 'gaussian': 'Extended Kalman filter',
    'laplace': r'$\mathcal N(0,H_A^{-1})$',
    'residual_pca': 'PCA of responses',
}
COLORS = {
    'mean': '#0072B2', 'mode': '#D55E00', 'iso': '#CC79A7',
    'isotropic': '#CC79A7', 'great': '#009E73',
    'great_window': '#009E73', 'great_ew': '#56B4E9',
    'mode_loaded': '#8C564B', 'regularized': '#8C564B',
    'modal_loading': '#8C564B', 'covariance_loading': '#484848',
    'scm': '#484848', 'scm_loaded': '#484848', 'lw': '#8172B2',
    'ledoit_wolf': '#8172B2', 'grouse': '#B07C49', 'oracle': '#111111',
    'known': '#111111', 'exact_known': '#111111',
    'known_forgetting': '#8172B2', 'misspecified': '#999999',
    'learned': '#0072B2', 'block_em': '#0072B2',
    'residual': '#D55E00', 'ridge': '#B07C49', 'gaussian': '#009E73',
    'laplace': '#8172B2', 'residual_pca': '#B07C49',
}
MARKERS = {
    'mean': 'o', 'mode': 's', 'iso': '^', 'isotropic': '^',
    'great': 'D', 'great_window': 'D', 'great_ew': 'v',
    'mode_loaded': 'P', 'modal_loading': 'P',
    'scm': 'X', 'scm_loaded': 'X', 'covariance_loading': 'X',
    'oracle': '*', 'known': 's', 'exact_known': 's',
    'known_forgetting': 'D', 'misspecified': '^', 'learned': 'o',
}
METHODS = METHOD_LABELS = LABELS
STYLE = {key: (label, COLORS.get(key, '#484848'), MARKERS.get(key, 'o'))
         for key, label in LABELS.items()}
GREY, BLACK = '#70777D', '#222222'

def method_label(method, example=None):
    return LABELS.get(method, str(method).replace('_', ' '))

def color(method):
    return COLORS.get(method, '#484848')

def marker(method):
    return MARKERS.get(method, 'o')
'''


def plot_cell(example, stem, page, body, title=None, note=''):
    """Return one figure's complete source for insertion into a code cell."""
    filename = f'{stem}_supp_page{page}.pdf'
    source = f'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

evidence = SUPPLEMENTARY_EVIDENCE
example = EXAMPLE
DPI = 300
FIGSIZE = (15, 5)
SAVE_PDF = True
PDF_PATH = EDITABLE_OUTPUT / {filename!r}
DRAFT_SECTION = 'supplementary'
DRAFT_PATH = ROOT / 'results' / DRAFT_SECTION / {example!r} / PDF_PATH.name
'''
    source += '\n' + dedent(STYLE_SOURCE).strip() + '\n\n'
    source += dedent(body).strip() + '\n\n'
    source += '''if SAVE_PDF:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, dpi=DPI, bbox_inches=None)
    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())
plt.show()
'''
    return dict(example=example, stem=stem, page=page,
                title=title or stem.replace('_', ' ').capitalize(),
                note=note, source=source)
