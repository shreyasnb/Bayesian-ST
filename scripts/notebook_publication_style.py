"""Insert visible font controls into editable notebook plotting source.

This is a source generator, not a runtime renderer. Every generated rcParams
statement remains in the figure cell for direct editing.
"""
from __future__ import annotations
import re

CONTROLS = '''# Font sizes are chosen for a six-inch-wide insertion in the paper.
# The width factor preserves that printed size when FIGSIZE is changed.
LABEL_FONT_SIZE = 25. * FIGSIZE[0] / 15.
TICK_FONT_SIZE = 23. * FIGSIZE[0] / 15.
LEGEND_FONT_SIZE = 22. * FIGSIZE[0] / 15.
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif', 'font.size': LABEL_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE, 'axes.titlesize': LABEL_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE, 'ytick.labelsize': TICK_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'axes.labelpad': 8.,
    'figure.dpi': DPI, 'savefig.dpi': DPI,
})
'''


def apply_publication_style(source):
    """Return ordinary cell source with local, width-aware font controls."""
    if 'LABEL_FONT_SIZE =' not in source:
        source = re.sub(r'^(FIGSIZE\s*=\s*[^\n]+\n)',
                        lambda match: match.group(1) + CONTROLS, source,
                        flags=re.MULTILINE)
    def font(match):
        key, value = match.group(1), float(match.group(2))
        if value > 13.:
            return match.group(0)
        return key + '=' + ('TICK_FONT_SIZE' if key == 'labelsize' else
                             'LABEL_FONT_SIZE' if value >= 11. else 'LEGEND_FONT_SIZE')
    source = re.sub(r'\b(fontsize|labelsize)\s*=\s*([0-9]+(?:\.[0-9]*)?)', font, source)
    source = source.replace('labelpad=-2', 'labelpad=8').replace('pad=-2', 'pad=3')
    if '# Keep larger 3D labels inside the canvas.' not in source:
        finishing = '''
# Reserve space for TeX glyph extents at the larger printed font size.
_layout = fig.get_layout_engine()
if _layout is not None and hasattr(_layout, 'set'):
    _layout.set(h_pad=.18, w_pad=.16)
# Keep larger 3D labels inside the canvas. This changes only view padding.
for _plot_axes in fig.axes:
    if hasattr(_plot_axes, 'zaxis'):
        _plot_axes.set_box_aspect(_plot_axes.get_box_aspect(), zoom=.84)
# Tiled figures use positional descriptions in the manuscript captions.
if len([_axis for _axis in fig.axes if _axis.get_label() != '<colorbar>']) > 1:
    for _panel_axes in fig.axes:
        for _panel_text in list(_panel_axes.texts):
            if _panel_text.get_text() in ('(a)', '(b)', '(c)', '(d)', '(e)', '(f)'):
                _panel_text.remove()
'''
        source = source.replace('\nif SAVE_PDF:', '\n'+finishing+'\nif SAVE_PDF:')
    return source
