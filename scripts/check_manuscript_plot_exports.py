"""Read-only audit of editable notebook and manuscript PDF exports.

The notebook is parsed as Python syntax; plotting cells are never executed.
Live PDF dimensions and raster resolution are checked against each visible
FIGSIZE/DPI declaration. Canonical curated copies are checked separately
against their manifest. PDF text and lines remain vector data, so dpi applies
only to embedded raster layers. Font readability is assessed using the primary
label/tick sizes at 6.5-inch insertion, not the smaller mathematical scripts.
Optional previews are rendered to the directory supplied with --previews.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError as error:
        raise SystemExit('This read-only PDF audit requires PyMuPDF.') from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.notebook_pdf_exports import declared_notebook_pdf_routes

MAIN_PANELS = {
    ('beamforming', 'task_performance_output_power.pdf'): [],
    ('beamforming', 'noise_adaptation_page1.pdf'): [],
    ('beamforming', 'moving_interference_smooth_output_power.pdf'): [],
    ('beamforming', 'moving_interference_abrupt_output_power.pdf'): [],
    ('beamforming', 'moving_interference_comparison.pdf'): [],
    ('contact', 'information_rotation.pdf'): [],
    ('contact', 'task_performance_page1.pdf'): [],
    ('contact', 'posterior_anisotropy_spheres.pdf'): [],
    ('contact', 'posterior_anisotropy_risk.pdf'): [],
}
POINT_TOLERANCE = .02
TEXT_TOLERANCE = .5
DPI_TOLERANCE = .15
FONT_TOLERANCE = .25
TEX_POINT_RATIO = 72. / 72.27


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return qualified(node.value) + '.' + node.attr
    return ''


def constant_value(node, values):
    """Evaluate only literals and simple arithmetic used by visible controls."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(constant_value(item, values) for item in node.elts)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Subscript):
        return constant_value(node.value, values)[constant_value(node.slice, values)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = constant_value(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        first = constant_value(node.left, values)
        second = constant_value(node.right, values)
        if isinstance(node.op, ast.Add):
            return first + second
        if isinstance(node.op, ast.Sub):
            return first - second
        if isinstance(node.op, ast.Mult):
            return first * second
        if isinstance(node.op, ast.Div):
            return first / second
    raise ValueError('Not a literal plotting control: ' + ast.dump(node))


def figure_declarations(source):
    """Return declarations in a cell, including cells containing two figures."""
    values, declarations = {}, []
    pdf_name = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == 'PDF_PATH' and isinstance(node.value, ast.BinOp):
                    pdf_name = ast.literal_eval(node.value.right)
                try:
                    values[target.id] = constant_value(node.value, values)
                except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
                    pass
        call = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
        if not isinstance(call, ast.Call) or qualified(call.func) not in ('plt.figure', 'plt.subplots'):
            continue
        kwargs = {item.arg: item.value for item in call.keywords}
        size = constant_value(kwargs['figsize'], values)
        dpi = constant_value(kwargs['dpi'], values)
        if not pdf_name:
            raise ValueError('Figure constructor has no PDF basename')
        if len(size) != 2 or any(not math.isfinite(float(item)) or item <= 0 for item in size):
            raise ValueError('FIGSIZE must contain two positive numbers')
        fonts = {name: float(values[name]) for name in
                 ('LABEL_FONT_SIZE', 'TICK_FONT_SIZE', 'LEGEND_FONT_SIZE') if name in values}
        from scripts.notebook_publication_style import figure_blocks
        block = next(block for block in figure_blocks(source) if pdf_name in block)
        tight_bbox = any(isinstance(item, ast.keyword) and item.arg == 'bbox_inches'
                         and isinstance(item.value, ast.Constant) and item.value.value == 'tight'
                         for item in ast.walk(ast.parse(block)))
        declarations.append(dict(filename=pdf_name, figure_size_in=list(size), dpi=float(dpi),
                                 primary_font_sizes_pt=fonts, tight_bbox=tight_bbox))
    return declarations


def notebook_declarations(root, routes):
    notebooks, figures = {}, {}
    for destination, route in routes.items():
        name = route['notebook']
        if name not in notebooks:
            notebooks[name] = json.loads((root / name).read_text())
        cell = notebooks[name]['cells'][route['cell']]
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        matches = [item for item in figure_declarations(source) if item['filename'] == route['filename']]
        if len(matches) != 1:
            raise ValueError(f'{name}, cell {route["cell"]}: ambiguous figure declaration for {destination}')
        figures[destination] = dict(**matches[0], notebook=name, cell=route['cell'],
                                   source_sha256=hashlib.sha256(source.encode()).hexdigest())
    return figures


def audit_pdf(path, expected_sizes, expected_dpi=300., primary_fonts=None,
              main_panel_policy=None, require_primary_sizes=False, preview=None,
              tight_bbox=False):
    """Read actual PDF vector text, embedded fonts, bounds and image transforms."""
    path = Path(path)
    digest = sha(path)
    errors, pages = [], []
    with fitz.open(path) as document:
        if len(document) != len(expected_sizes):
            errors.append('Page count differs from the source declaration')
        for page_index, page in enumerate(document):
            page_errors = []
            size = [page.rect.width / 72., page.rect.height / 72.]
            if page_index < len(expected_sizes):
                # Tight export trims the canvas margins; text bounds are checked below.
                tolerance = 12. if tight_bbox else POINT_TOLERANCE
                if any(abs(actual * 72. - expected * 72.) > tolerance
                       for actual, expected in zip(size, expected_sizes[page_index])):
                    page_errors.append('Page dimensions differ from the visible FIGSIZE or manifest')
            if any(abs(a-b) > POINT_TOLERANCE for a, b in zip(page.mediabox, page.cropbox)):
                page_errors.append('CropBox differs from MediaBox')
            if page.rotation:
                page_errors.append('Unexpected page rotation')
            traces = page.get_texttrace()
            outside = [dict(font=span['font'], bbox_pt=list(span['bbox'])) for span in traces
                       if (not all(math.isfinite(value) for value in span['bbox'])
                           or span['bbox'][0] < -TEXT_TOLERANCE
                           or span['bbox'][1] < -TEXT_TOLERANCE
                           or span['bbox'][2] > page.rect.width + TEXT_TOLERANCE
                           or span['bbox'][3] > page.rect.height + TEXT_TOLERANCE)]
            if outside:
                page_errors.append('Vector text extends beyond the page')
            if not traces:
                page_errors.append('No vector text is present')
            fonts = page.get_fonts(full=True)
            font_records = []
            for font in fonts:
                embedded = bool(font[0] and document.extract_font(font[0])[3])
                font_records.append(dict(name=font[3], type=font[2], embedded=embedded))
                if font[2] != 'Type1':
                    page_errors.append('Expected embedded Type1 LaTeX fonts, found ' + font[2])
                if not embedded:
                    page_errors.append('A PDF font is not embedded: ' + font[3])
            image_dpi = []
            for image in page.get_image_info(xrefs=True):
                a, b, c, d, _, _ = image['transform']
                placed = (math.hypot(a, b), math.hypot(c, d))
                if min(placed) <= 0:
                    page_errors.append('A raster layer has a degenerate transform')
                    continue
                resolution = [72. * image['width'] / placed[0], 72. * image['height'] / placed[1]]
                image_dpi.append(resolution)
                if any(abs(value - expected_dpi) > DPI_TOLERANCE for value in resolution):
                    page_errors.append('A raster layer differs from the declared dpi')
            sizes = Counter(round(float(span['size']), 4) for span in traces)
            readability = {}
            for name, declared in (primary_fonts or {}).items():
                target = declared * TEX_POINT_RATIO
                matched = [value for value in sizes if abs(value - target) <= FONT_TOLERANCE]
                actual = min(matched, key=lambda value: abs(value-target)) if matched else None
                at_six_inches = actual * 6.5 / size[0] if actual is not None else None
                readability[name] = dict(declared_pt=declared, actual_pt=actual,
                                         paper_width_in=6.5, paper_insertion_pt=at_six_inches)
                if require_primary_sizes and name != 'LEGEND_FONT_SIZE' and actual is None:
                    page_errors.append('Primary font size is absent from the actual PDF: ' + name)
                threshold = {'LABEL_FONT_SIZE': 9.5, 'TICK_FONT_SIZE': 8.7,
                             'LEGEND_FONT_SIZE': 8.4}[name]
                if at_six_inches is not None and at_six_inches < threshold:
                    page_errors.append('Primary text is too small at 6.5-inch insertion: ' + name)
            letters = re.findall(r'\([a-z]\)', page.get_text())
            if main_panel_policy is not None and letters != main_panel_policy:
                page_errors.append(f'Panel letters {letters!r} differ from {main_panel_policy!r}')
            if preview is not None:
                preview.parent.mkdir(parents=True, exist_ok=True)
                suffix = '' if len(document) == 1 else f'_page{page_index+1}'
                destination = preview.with_name(preview.stem+suffix+'.png')
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1., 1.))
                pixmap.set_dpi(72, 72)
                pixmap.save(destination)
            pages.append(dict(page=page_index+1, figure_size_in=size, vector_text_spans=len(traces),
                fonts=font_records, font_size_counts={str(key): value for key, value in sizes.items()},
                primary_font_readability=readability, raster_dpi=image_dpi,
                out_of_page_text=outside, panel_letters=letters, errors=page_errors))
            errors.extend(f'Page {page_index+1}: {error}' for error in page_errors)
    if sha(path) != digest:
        errors.append('PDF changed during the read-only audit')
    return dict(sha256=digest, pages=pages, errors=errors)


def run(root, previews=None, main_only=False, figure_names=None):
    root = Path(root).resolve()
    routes = declared_notebook_pdf_routes(root)
    if (root/'notebooks/beamforming_moving_interference.ipynb').is_file():
        from scripts.check_moving_beamforming import optional_moving_routes
        candidates = optional_moving_routes(root)
        if set(routes) & set(candidates):
            raise ValueError('An optional moving candidate replaces an existing figure')
        routes.update(candidates)
    declarations = notebook_declarations(root, routes)
    files, errors = {}, []
    matched = 0
    selected_routes = {name: route for name, route in routes.items()
                       if (not main_only or (route['example'], route['filename']) in MAIN_PANELS)
                       and (not figure_names or route['filename'] in figure_names)}
    if figure_names and set(figure_names) - {route['filename'] for route in selected_routes.values()}:
        raise ValueError('Requested figure has no selected notebook route')
    for destination, route in selected_routes.items():
        path = root / destination
        expected = declarations[destination]
        if expected['dpi'] != 300.:
            errors.append(destination + ': visible DPI is not 300')
        if not path.is_file():
            errors.append('Missing live export: ' + destination)
            continue
        local = root / route['local_pdf']
        if not local.is_file() or sha(local) != sha(path):
            errors.append(destination + ': local and manuscript PDF copies differ or local copy is missing')
        else:
            matched += 1
        panel_policy = MAIN_PANELS.get((route['example'], route['filename']))
        preview = (Path(previews) / route['example'] / (path.stem+'.png')
                   if previews is not None and panel_policy is not None else None)
        record = audit_pdf(path, [expected['figure_size_in']], expected_dpi=expected['dpi'],
            primary_fonts=expected['primary_font_sizes_pt'], main_panel_policy=panel_policy,
            require_primary_sizes=panel_policy is not None, preview=preview,
            tight_bbox=expected['tight_bbox'])
        record['source'] = expected
        record['local_pdf'] = route['local_pdf']
        files[destination] = record
        errors.extend(destination + ': ' + error for error in record['errors'])
    canonical, main_count, supplementary_count = {}, 0, 0
    manifest_path = root / 'results/draft_figure_manifest.json'
    if not main_only and not figure_names and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for destination, declaration in manifest['files'].items():
            path = root / destination
            if not path.is_file():
                errors.append('Missing canonical export: ' + destination)
                continue
            record = audit_pdf(path, [page['figure_size_in'] for page in declaration['pages']],
                               expected_dpi=float(manifest.get('dpi', 300.)))
            if sha(path) != declaration['sha256']:
                record['errors'].append('Canonical copy hash differs from its manifest')
            source = root / declaration['source_pdf']
            if not source.is_file() or sha(source) != declaration['source_pdf_sha256']:
                record['errors'].append('Canonical source hash differs from its manifest')
            canonical[destination] = record
            if declaration['section'] == 'main':
                main_count += 1
            else:
                supplementary_count += 1
            errors.extend(destination + ': ' + error for error in record['errors'])
    return dict(status='failed' if errors else 'passed',
        producer='scripts/check_manuscript_plot_exports.py', producer_sha256=sha(__file__),
        tool='PyMuPDF ' + fitz.VersionBind, live_exports_declared=len(routes),
        live_exports_checked=len(files), local_pdf_matches=matched,
        canonical_exports_checked=len(canonical), canonical_main=main_count,
        canonical_supplementary=supplementary_count, files=files, canonical_files=canonical,
        errors=errors, scope='Read-only Python AST and PDF inspection. No plotting code, fitting, '
            'simulation or numerical benchmark is executed. Primary text sizes are matched to '
            'actual PDF text and scaled to a 6.5-inch insertion; math subscripts are excluded '
            'from that assessment. Page-bound text is checked; artist overlap and graphical '
            'clarity require visual review. PDF vector text and lines have no dpi.',
        tolerances=dict(page_points=POINT_TOLERANCE, text_bounds_points=TEXT_TOLERANCE,
                        raster_dpi=DPI_TOLERANCE, primary_font_points=FONT_TOLERANCE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='Write the JSON audit to this path')
    parser.add_argument('--previews', type=Path, help='Render retained main and optional candidate previews to this directory')
    parser.add_argument('--main-only', action='store_true', help='Inspect retained main figures and optional moving candidates')
    parser.add_argument('--figure', action='append', help='Inspect only this declared PDF basename; repeat as needed')
    args = parser.parse_args()
    report = run(args.root, previews=args.previews, main_only=args.main_only,
                 figure_names=args.figure)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'live_exports_declared',
        'live_exports_checked', 'local_pdf_matches', 'canonical_exports_checked',
        'canonical_main', 'canonical_supplementary', 'errors')}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
