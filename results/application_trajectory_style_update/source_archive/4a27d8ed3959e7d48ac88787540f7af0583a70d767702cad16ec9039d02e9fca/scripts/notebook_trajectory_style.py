"""Source-only controls for solid, marker-free empirical trajectories.

The returned statements stay visible in the notebook.  This module does not
import Matplotlib, load data, or render figures.  Coordinate expressions are
never rewritten; mathematical reference curves and isolated scientific points
retain their original styles.
"""
from __future__ import annotations

import ast
import re


_PLOT_METHODS = {'plot', 'step', 'semilogx', 'semilogy', 'loglog'}
_LINE_STYLE_KEYWORDS = {'linestyle', 'ls', 'dashes'}
_MARKER_KEYWORDS = {
    'marker', 'markevery', 'markersize', 'ms', 'markeredgecolor', 'mec',
    'markerfacecolor', 'mfc', 'markerfacecoloralt', 'markeredgewidth', 'mew',
    'fillstyle',
}
_STYLE_NAMES = re.compile(r'^(?:MARKERS?|MARKER_\w+|\w*marker_\w*|\w*stagger\w*)$', re.I)
_COLOR_CODES = set('bgrcmykw')
_MARKER_CODES = set('.,ov^<>12348spP*hH+xXDd|_')


def _is_line_call(node):
    return isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Attribute) and node.func.attr in _PLOT_METHODS
        or isinstance(node.func, ast.Name) and node.func.id == 'Line2D')


def _artist_name(call):
    return call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id


class _Source:
    def __init__(self, source):
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, line, column):
        # AST column offsets count UTF-8 bytes, unlike Python string slices.
        prefix = self.lines[line - 1].encode('utf-8')[:column].decode('utf-8')
        return self.starts[line - 1] + len(prefix)

    def span(self, node):
        return (self.offset(node.lineno, node.col_offset),
                self.offset(node.end_lineno, node.end_col_offset))

    def text(self, node):
        start, end = self.span(node)
        return self.source[start:end]


def _apply(source, edits):
    previous = len(source) + 1
    for start, end, replacement in sorted(set(edits), reverse=True):
        if end > previous:
            raise ValueError('Overlapping trajectory style source edits')
        source = source[:start] + replacement + source[end:]
        previous = start
    ast.parse(source)
    return source


def _parents(tree):
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _ancestors(node, parents):
    while node in parents:
        node = parents[node]
        yield node


def _format_info(node):
    """Recognize the small Matplotlib format grammar without importing it."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    fmt = node.value
    rest = fmt
    has_line = False
    for style in ('--', '-.', '-', ':'):
        if style in rest:
            rest = rest.replace(style, '', 1)
            has_line = True
            break
    color = ''
    if len(rest) >= 2 and rest[:1] == 'C' and rest[1:].isdigit():
        color, rest = rest, ''
    elif any(char in _COLOR_CODES for char in rest):
        color = next(char for char in rest if char in _COLOR_CODES)
        rest = rest.replace(color, '', 1)
    if len(rest) > 1 or (rest and rest not in _MARKER_CODES):
        return None
    return {'color': color, 'marker': rest, 'line': has_line}


def _figure_blocks(source):
    """Split at top-level shows, retaining all original bytes."""
    text = _Source(source)
    stops = [text.span(node)[1] for node in ast.parse(source).body
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Attribute)
             and isinstance(node.value.func.value, ast.Name)
             and node.value.func.value.id == 'plt' and node.value.func.attr == 'show']
    blocks, start = [], 0
    for stop in stops:
        # Include the show line's newline in its block.
        if source[stop:stop + 1] == '\n':
            stop += 1
        blocks.append(source[start:stop])
        start = stop
    if not blocks:
        return [source]
    blocks[-1] += source[start:]
    return blocks


def _axes_dimensions(tree):
    dimensions = set()
    names_3d = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {'subplots', 'subplot', 'add_subplot', 'add_axes', 'axes'}:
            continue
        projection = next((item.value for item in node.keywords if item.arg == 'projection'), None)
        subplot_kw = next((item.value for item in node.keywords if item.arg == 'subplot_kw'), None)
        if projection is None and isinstance(subplot_kw, ast.Dict):
            projection = next((value for key, value in zip(subplot_kw.keys, subplot_kw.values)
                               if isinstance(key, ast.Constant) and key.value == 'projection'), None)
        dimension = '3d' if isinstance(projection, ast.Constant) and projection.value == '3d' else '2d'
        dimensions.add(dimension)
        if dimension == '3d':
            # The known mixed templates use either a named sphere axis or a
            # list named spheres; their starred grid curves are protected too.
            for assignment in ast.walk(tree):
                if isinstance(assignment, ast.Assign) and node in list(ast.walk(assignment.value)):
                    names_3d.update(value.id for target in assignment.targets
                                    for value in ast.walk(target) if isinstance(value, ast.Name))
    return dimensions, names_3d


def _decision(call, source, parents, names_3d):
    """Classify only connected line artists, with explicit scientific exemptions."""
    keywords = {item.arg: item.value for item in call.keywords}
    if isinstance(call.func, ast.Name):
        # Explicit method trajectory legend handles follow the trajectory.
        # Point-only legends and geometric contour-level handles do not.
        style = keywords.get('linestyle', keywords.get('ls'))
        if isinstance(style, ast.Constant) and style.value in (None, '', ' ', 'None', 'none'):
            return 'point_or_geometry'
        controls = ' '.join(source.text(item.value) for item in call.keywords)
        if any(term in controls for term in ('METHOD_LABELS', 'LABELS[', 'STYLE[', 'method_label(',
                                             'COLORS[', 'color(method', 'marker(method')):
            return 'trajectory_legend'
        return 'point_or_geometry'
    if len(call.args) < 2 or any(isinstance(arg, ast.Starred) for arg in call.args):
        return 'point_or_geometry'
    receiver = source.text(call.func.value)
    if any(re.search(r'\b' + re.escape(name) + r'\b', receiver) for name in names_3d):
        return '3d_geometry'
    if any(isinstance(arg, (ast.List, ast.Tuple)) and not arg.elts for arg in call.args[:2]):
        return 'legend_proxy'
    if all(isinstance(arg, ast.Constant) for arg in call.args[:2]):
        return 'isolated_point'
    if ast.dump(call.args[0], include_attributes=False) == ast.dump(call.args[1], include_attributes=False):
        return 'identity_reference'
    fmt = _format_info(call.args[2]) if len(call.args) == 3 else None
    if fmt and fmt['marker'] and not fmt['line'] and not any(key in keywords for key in ('linestyle', 'ls')):
        return 'isolated_point'
    label_node = keywords.get('label')
    label = label_node.value if isinstance(label_node, ast.Constant) and isinstance(label_node.value, str) else ''
    if any(term in label.lower() for term in ('theorem bound', 'distance ceiling', 'true variance')):
        return 'mathematical_reference'
    y_source = source.text(call.args[1]).lower()
    if _artist_name(call) == 'step' and any(term in y_source for term in ('truth', 'true_schedule', 'true_noise')):
        return 'truth_reference'
    loops = [node for node in _ancestors(call, parents) if isinstance(node, ast.For)]
    for loop in loops:
        iterator = source.text(loop.iter)
        if any(term in iterator for term in ('kl_first', 'kl_second', 'spheres', 'sphere_axes')):
            return 'controlled_geometry'
        if ('load_constructed_geometry' in source.source and
                {'case', 'shade', 'style'}.issubset({node.id for node in ast.walk(loop.target) if isinstance(node, ast.Name)})):
            return 'controlled_geometry'
        if 'operatorname{tr}' in iterator or 'mathcal F^{-1}' in iterator:
            # One plotting call draws an empirical curve and an inverse-Fisher
            # reference.  Remove sampling markers but keep that reference dash.
            return 'mixed_empirical_reference'
    return 'trajectory'


def trajectory_style_report(source):
    """Describe the connected artists and their unchanged coordinate ASTs."""
    report = []
    line_offset = 0
    for block in _figure_blocks(source):
        text = _Source(block)
        tree = ast.parse(block)
        parents = _parents(tree)
        dimensions, names_3d = _axes_dimensions(tree)
        for call in sorted((node for node in ast.walk(tree) if _is_line_call(node)),
                           key=lambda node: (node.lineno, node.col_offset)):
            decision = ('3d_geometry' if dimensions == {'3d'} else
                        _decision(call, text, parents, names_3d))
            report.append({'line': line_offset + call.lineno, 'artist': _artist_name(call),
                           'receiver': text.text(call.func.value) if isinstance(call.func, ast.Attribute) else 'Line2D', 'decision': decision,
                           'coordinates': [ast.dump(arg, include_attributes=False) for arg in call.args[:2]]})
        line_offset += block.count('\n')
    return report


def _remove_keyword(text, call, keyword):
    start, end = text.span(keyword)
    # Prefer the preceding comma. Adjacent removed keywords then have disjoint
    # spans, including when the final keyword has no trailing comma.
    cursor = start - 1
    while cursor >= 0 and text.source[cursor].isspace():
        cursor -= 1
    if cursor >= 0 and text.source[cursor] == ',':
        return cursor, end, ''
    following = re.match(r'\s*,', text.source[end:])
    if following:
        return start, end + following.end(), ''
    raise ValueError('Cannot safely remove a trajectory style keyword')


def _transform_block(block):
    text = _Source(block)
    tree = ast.parse(block)
    parents = _parents(tree)
    dimensions, names_3d = _axes_dimensions(tree)
    if dimensions == {'3d'}:
        return block
    edits = []
    for call in ast.walk(tree):
        if not _is_line_call(call):
            continue
        decision = _decision(call, text, parents, names_3d)
        if decision not in {'trajectory', 'trajectory_legend', 'mixed_empirical_reference'}:
            continue
        for keyword in call.keywords:
            if keyword.arg in _MARKER_KEYWORDS or (keyword.arg == 'dashes' and decision != 'mixed_empirical_reference'):
                edits.append(_remove_keyword(text, call, keyword))
        fmt_node = call.args[2] if len(call.args) == 3 else None
        fmt = _format_info(fmt_node)
        style_keywords = [item for item in call.keywords if item.arg in {'linestyle', 'ls'}]
        if decision in {'trajectory', 'trajectory_legend'}:
            for keyword in style_keywords:
                if not (isinstance(keyword.value, ast.Constant) and keyword.value.value == '-'):
                    edits.append((*text.span(keyword.value), "'-'"))
            if fmt is not None:
                replacement = fmt['color'] + '-'
                if fmt_node.value != replacement:
                    edits.append((*text.span(fmt_node), repr(replacement)))
            # The default connected trajectory already has a solid line.  No
            # new keyword is needed; this preserves its global linewidth/color.
    return _apply(block, edits)


def _remove_statement(text, node):
    start, end = text.span(node)
    if node.col_offset == 0:
        start = text.starts[node.lineno - 1]
        if text.source[end:end + 1] == '\n':
            end += 1
    return start, end, ''


def _prune_marker_configuration(source):
    """Drop dead marker controls; retain controls used by scientific point plots."""
    for _ in range(6):
        tree = ast.parse(source)
        text = _Source(source)
        loads = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        edits = []
        for node in tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and _STYLE_NAMES.match(node.name) and node.name not in loads):
                edits.append(_remove_statement(text, node))
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if (name == 'STYLE' or _STYLE_NAMES.match(name)) and name not in loads:
                edits.append(_remove_statement(text, node))
        if not edits:
            break
        source = _apply(source, edits)

    # Main beamforming cells use STYLE=(label,color,marker).  Once the third
    # unpacked value is unused, remove it from both the map and its unpackings.
    tree = ast.parse(source)
    text = _Source(source)
    loads = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    style_maps = [node for node in tree.body if isinstance(node, ast.Assign)
                  and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == 'STYLE' and isinstance(node.value, ast.Dict)]
    unpackings = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and len(node.targets) == 1 and isinstance(node.targets[0], (ast.Tuple, ast.List))
                  and len(node.targets[0].elts) >= 3 and isinstance(node.value, ast.Subscript)
                  and isinstance(node.value.value, ast.Name) and node.value.value.id == 'STYLE']
    if style_maps and unpackings and all(isinstance(value, ast.Name) and value.id not in loads
                                         for node in unpackings for value in node.targets[0].elts[2:]):
        # Direct access to STYLE[method][2] means a surviving point plot still
        # needs these symbols, so its map must stay intact.
        third_access = any(isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript)
                           and isinstance(node.value.value, ast.Name) and node.value.value.id == 'STYLE'
                           and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)
                           and node.slice.value >= 2 for node in ast.walk(tree))
        if not third_access and all(isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) >= 3
                                    for mapping in style_maps for value in mapping.value.values):
            edits = []
            for mapping in style_maps:
                for value in mapping.value.values:
                    start = text.span(value.elts[1])[1]
                    end = text.span(value.elts[-1])[1]
                    edits.append((start, end, ''))
            for node in unpackings:
                target = node.targets[0]
                edits.append((text.span(target.elts[1])[1], text.span(target.elts[-1])[1], ''))
            source = _apply(source, edits)
    # A few diagnostic loops keep a marker/style as the final literal tuple
    # field. Once no artist uses it, remove that field rather than emitting an
    # unused marker or stagger control beside the live color/label controls.
    tree, text = ast.parse(source), _Source(source)
    loads = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.For) and isinstance(node.target, (ast.Tuple, ast.List))
                and isinstance(node.iter, (ast.Tuple, ast.List))):
            continue
        fields = node.target.elts
        remove = {index for index, field in enumerate(fields)
                  if isinstance(field, ast.Name) and field.id not in loads
                  and (field.id in {'m', 'symbol', 'style', 'offset'} or _STYLE_NAMES.match(field.id))}
        if not remove or len(remove) == len(fields):
            continue
        if not all(isinstance(row, (ast.Tuple, ast.List)) and len(row.elts) == len(fields) for row in node.iter.elts):
            continue
        def without_fields(sequence):
            kept = [text.text(value) for index, value in enumerate(sequence.elts) if index not in remove]
            inside = ', '.join(kept) + (',' if len(kept) == 1 else '')
            return ('(' + inside + ')') if isinstance(sequence, ast.Tuple) else ('[' + inside + ']')
        edits.append((*text.span(node.target), without_fields(node.target)))
        edits.extend((*text.span(row), without_fields(row)) for row in node.iter.elts)
    if edits:
        source = _apply(source, edits)
    return source


def apply_method_trajectory_style(source):
    """Return source with solid, marker-free empirical/method line histories.

    Pure 3D figure blocks and all coordinate expressions stay byte/AST exact.
    Isolated points, scatter, boxplot fliers, confidence intervals, reference
    identities/bounds/events, and controlled geometric curves are exempt.
    """
    blocks = _figure_blocks(source)
    transformed = [_transform_block(block) for block in blocks]
    # Shared controls can precede a sphere and then be reused by later 2D
    # figures in the same cell. Preserve that entire 3D source prefix exactly.
    dimensions = [_axes_dimensions(ast.parse(block))[0] for block in blocks]
    if not any('3d' in value for value in dimensions):
        result = _prune_marker_configuration(''.join(transformed))
    else:
        # A separately configured final 2D figure can discard its own dead
        # controls without touching the preceding sphere's code or namespace.
        if dimensions[-1] == {'2d'}:
            transformed[-1] = _prune_marker_configuration(transformed[-1])
        result = ''.join(transformed)
    assert_trajectory_array_parity(source, result)
    return result


def assert_trajectory_array_parity(before, after):
    """Require exact coordinate ASTs and non-style artist kwargs in order."""
    def signatures(source):
        signatures = []
        for node in sorted((value for value in ast.walk(ast.parse(source)) if _is_line_call(value)),
                           key=lambda value: (value.lineno, value.col_offset)):
            args = node.args[:2]
            if len(node.args) > 2 and _format_info(node.args[2]) is None:
                args = node.args
            keywords = [item for item in node.keywords
                        if item.arg not in _MARKER_KEYWORDS | _LINE_STYLE_KEYWORDS]
            signatures.append((ast.dump(node.func, include_attributes=False),
                               tuple(ast.dump(arg, include_attributes=False) for arg in args),
                               tuple(ast.dump(item, include_attributes=False) for item in keywords)))
        return signatures
    if signatures(before) != signatures(after):
        raise AssertionError('Trajectory coordinates or non-style artist options changed')
    return True
