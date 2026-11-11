"""Insert visible font and border controls into editable plotting source.

This is a source generator, not a runtime renderer. Every generated rcParams
statement remains in the figure cell for direct editing.
"""
from __future__ import annotations
import ast
import re
from scripts.notebook_trajectory_style import apply_method_trajectory_style

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
    return apply_publication_borders(apply_method_trajectory_style(source))


BORDER_BEGIN = '# Publication borders: ordinary 2D axes only.'
BORDER_END = '# End publication borders.'


def publication_border_code(figure='fig'):
    """Return visible Matplotlib statements, never a runtime rendering helper."""
    return f'''{BORDER_BEGIN}
BORDER_WIDTH = .9  # Matplotlib points.
for _border_axes in {figure}.axes:
    if (_border_axes.name == '3d' or _border_axes.get_label() == '<colorbar>'
            or getattr(_border_axes, '_colorbar', None) is not None):
        continue
    for _border_side in ('left', 'right', 'bottom', 'top'):
        if _border_side in _border_axes.spines:
            _border_axes.spines[_border_side].set_visible(True)
            _border_axes.spines[_border_side].set_linewidth(BORDER_WIDTH)
    _border_axes.xaxis.set_label_position('bottom')
    _border_axes.yaxis.set_label_position('left')
    _border_axes.tick_params(axis='both', which='both', direction='in',
        width=BORDER_WIDTH, bottom=True, top=True, left=True, right=True,
        labelbottom=True, labeltop=False, labelleft=True, labelright=False)
{BORDER_END}
'''


def figure_blocks(source):
    """Split top-level plt.show blocks while retaining every source byte."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    stops = []
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == 'plt' and node.value.func.attr == 'show'):
            stops.append(node.end_lineno)
    if not stops:
        return [source]
    blocks = []
    start = 0
    for stop in stops:
        blocks.append(''.join(lines[start:stop]))
        start = stop
    if start < len(lines):
        blocks[-1] += ''.join(lines[start:])
    return blocks


def figure_kind(source):
    """Classify actual axes constructors, excluding automatically made colorbars."""
    counts = {'2d': 0, '3d': 0}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        if name not in ('subplots', 'subplot', 'add_subplot', 'add_axes', 'axes'):
            continue
        if name in ('subplot', 'axes') and not (
                isinstance(node.func.value, ast.Name) and node.func.value.id == 'plt'):
            continue
        projection = next((value.value for value in node.keywords
                           if value.arg == 'projection'), None)
        if projection is None:
            subplot_kw = next((item.value for item in node.keywords if item.arg == 'subplot_kw'), None)
            if isinstance(subplot_kw, ast.Dict):
                projection = next((value for key,value in zip(subplot_kw.keys,subplot_kw.values)
                                   if isinstance(key,ast.Constant) and key.value == 'projection'), None)
        if isinstance(projection, ast.Constant) and projection.value == '3d':
            counts['3d'] += 1
        else:
            counts['2d'] += 1
    return counts


def _replace_hidden_spines(source):
    """Replace only explicit spine visibility calls; keep all other statements."""
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'set_visible' or len(node.args) != 1:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value is not False:
            continue
        target = node.func.value
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Attribute):
            continue
        if target.value.attr != 'spines':
            continue
        value = node.args[0]
        replacements.append((value.lineno-1, value.col_offset, value.end_col_offset))
    for line, start, end in sorted(replacements, reverse=True):
        lines[line] = lines[line][:start] + 'True' + lines[line][end:]
    return ''.join(lines)


def apply_publication_borders(source):
    """Insert local border controls into 2D figure blocks; preserve pure 3D bytes."""
    changed = []
    for block in figure_blocks(source):
        if figure_kind(block)['2d'] == 0:
            changed.append(block)
            continue
        block = _replace_hidden_spines(block)
        if BORDER_BEGIN in block:
            changed.append(block)
            continue
        tree = ast.parse(block)
        lines = block.splitlines(keepends=True)
        save = next((node.lineno-1 for node in tree.body
                     if isinstance(node, ast.If)
                     and any(isinstance(value, ast.Name) and value.id == 'SAVE_PDF'
                             for value in ast.walk(node.test))), None)
        if save is None:
            raise ValueError('A 2D figure needs its visible SAVE_PDF block before plt.show()')
        # Existing templates use one explicitly named figure for each show.
        figure = 'fig'
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == 'plt' and call.func.attr in ('figure','subplots')):
                target = node.targets[0]
                figure = target.id if isinstance(target, ast.Name) else target.elts[0].id
                break
        lines[save:save] = ['\n' + publication_border_code(figure) + '\n']
        changed.append(''.join(lines))
    return ''.join(changed)
