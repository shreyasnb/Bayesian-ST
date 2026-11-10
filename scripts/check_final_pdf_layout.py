"""Audit the actual final PDFs and previews, independently of their render code.

Run with a Python environment containing PyMuPDF and Pillow. The default is
read-only; ``--write`` refreshes both per-application audit reports only after
every selected application passes. This checks page-bound text, not whether
labels overlap one another; representative pages still need visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError as error:
        raise SystemExit('The optional PDF audit requires PyMuPDF and Pillow.') from error
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ('beamforming', 'contact')
WIDTH_IN = 7.
DPI = 300.
POINT_TOLERANCE = .01
TEXT_BOUND_TOLERANCE_PT = .5
DPI_TOLERANCE = .1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close(first, second, tolerance):
    return math.isfinite(first) and abs(first - second) <= tolerance


def audit_pdf(path, expected):
    """Inspect MediaBoxes, traced text, embedded fonts and placed PDF images."""
    path = Path(path)
    original_hash = sha(path)
    errors, outside, page_records = [], [], []
    font_types, font_cache = set(), {}
    image_count = 0
    resolutions = []
    with fitz.open(path) as document:
        title_empty = not (document.metadata.get('title') or '').strip()
        if not title_empty:
            errors.append('PDF metadata title is nonempty')
        if len(document) != len(expected['pages']):
            errors.append('Actual page count differs from figure_manifest.json')
        for index, page in enumerate(document, 1):
            media, crop = page.mediabox, page.cropbox
            if not _close(media.width, WIDTH_IN * 72, POINT_TOLERANCE):
                errors.append(f'Page {index}: MediaBox width is not seven inches')
            if any(not _close(a, b, POINT_TOLERANCE) for a, b in zip(media, crop)):
                errors.append(f'Page {index}: CropBox differs from MediaBox')
            if page.rotation != 0:
                errors.append(f'Page {index}: unexpected page rotation')
            if index <= len(expected['pages']):
                size = expected['pages'][index - 1]['figure_size_in']
                if any(not _close(actual, declared * 72, POINT_TOLERANCE)
                       for actual, declared in zip((media.width, media.height), size)):
                    errors.append(f'Page {index}: actual dimensions differ from declared figure size')
            text_spans = page.get_texttrace()
            for span in text_spans:
                bounds = tuple(float(value) for value in span['bbox'])
                x0, y0, x1, y1 = bounds
                if (not all(math.isfinite(value) for value in bounds)
                        or x0 < -TEXT_BOUND_TOLERANCE_PT
                        or y0 < -TEXT_BOUND_TOLERANCE_PT
                        or x1 > page.rect.width + TEXT_BOUND_TOLERANCE_PT
                        or y1 > page.rect.height + TEXT_BOUND_TOLERANCE_PT):
                    outside.append(dict(page=index, bbox_pt=list(bounds), font=span['font']))
            if not text_spans:
                errors.append(f'Page {index}: no vector text was found')
            for font in page.get_fonts(full=True):
                xref, _, font_type = font[:3]
                font_types.add(font_type)
                if font_type != 'Type1':
                    errors.append(f'Page {index}: unexpected font type {font_type}')
                if xref not in font_cache:
                    font_cache[xref] = bool(document.extract_font(xref)[3]) if xref else False
                if not font_cache[xref]:
                    errors.append(f'Page {index}: font {font[3]} is not embedded')
            images = page.get_image_info(xrefs=True)
            for image in images:
                a, b, c, d, _, _ = image['transform']
                width, height = math.hypot(a, b), math.hypot(c, d)
                if width <= 0 or height <= 0:
                    errors.append(f'Page {index}: degenerate image transform')
                    continue
                resolution = (72 * image['width'] / width, 72 * image['height'] / height)
                resolutions.extend(resolution)
                if any(not _close(value, DPI, DPI_TOLERANCE) for value in resolution):
                    errors.append(f'Page {index}: raster layer resolution is {resolution}, not {DPI:g} dpi')
            image_count += len(images)
            page_records.append(dict(page=index, media_box_pt=list(media),
                text_spans=len(text_spans), raster_layers=len(images)))
        page_count = len(document)
    if outside:
        errors.append('Traced text extends outside a page')
    if sha(path) != original_hash:
        errors.append('PDF changed while it was being audited')
    return dict(sha256=original_hash, pages=page_count, width_in=WIDTH_IN,
        pdf_title_empty=title_empty, font_types=sorted(font_types),
        all_fonts_embedded=all(font_cache.values()), raster_layers=image_count,
        raster_dpi_range=[min(resolutions), max(resolutions)] if resolutions else None,
        page_checks=page_records, out_of_page_text=outside, errors=errors)


def audit_example(root, example):
    """Check the complete declared PDF catalogue and every saved PNG preview."""
    root = Path(root)
    folder = root / 'results' / example
    manifest_path = folder / 'figure_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    errors = []
    required = dict(status='completed', stage='paper', example=example,
        text_usetex=True, titles=False, preview_dpi=int(DPI),
        rasterized_layer_dpi=int(DPI), paper_width_in=WIDTH_IN)
    for name, expected in required.items():
        if manifest.get(name) != expected:
            errors.append(f'Figure manifest {name} differs from {expected!r}')
    expected_files = set(manifest['figures'])
    actual_files = {path.name for path in folder.glob('*.pdf')}
    if expected_files != actual_files or len(expected_files) != 19:
        errors.append('Expected exactly the complete nineteen-PDF catalogue')
    files, preview_records = {}, {}
    font_types = set()
    outside = []
    for name, expected in manifest['figures'].items():
        path = folder / name
        if not path.is_file():
            errors.append('Missing PDF: ' + name)
            continue
        record = audit_pdf(path, expected)
        files[name] = record
        if record['sha256'] != manifest['files'].get(name):
            errors.append('PDF hash differs from figure manifest: ' + name)
        errors.extend(name + ': ' + error for error in record['errors'])
        font_types.update(record['font_types'])
        outside.extend(dict(file=name, **entry) for entry in record['out_of_page_text'])
        stem = Path(name).stem
        for index, page in enumerate(expected['pages'], 1):
            suffix = '' if index == 1 else f'_page{index}'
            preview = f'preview/{stem}{suffix}.png'
            # Bundles reuse individual previews rather than adding PNG copies.
            if not expected.get('constituent_outputs') and preview not in manifest['files']:
                errors.append('Missing declared page preview: ' + preview)
    for name, expected_hash in manifest['files'].items():
        if not name.endswith('.png'):
            continue
        path = folder / name
        if not path.is_file():
            errors.append('Missing preview: ' + name)
            continue
        if sha(path) != expected_hash:
            errors.append('Preview hash differs from figure manifest: ' + name)
        with Image.open(path) as image:
            resolution = image.info.get('dpi', ())
            if len(resolution) != 2 or any(not _close(value, DPI, DPI_TOLERANCE)
                                          for value in resolution):
                errors.append(f'Preview does not have {DPI:g}-dpi metadata: ' + name)
            if abs(image.width - WIDTH_IN * DPI) > 1:
                errors.append(f'Preview width is not seven inches at {DPI:g} dpi: ' + name)
            preview_records[name] = dict(sha256=expected_hash,
                size_px=list(image.size), dpi=list(resolution))
    if sha(manifest_path) != manifest_hash:
        errors.append('Figure manifest changed during the audit')
    return dict(status='failed' if errors else 'passed', example=example,
        tool='PyMuPDF ' + fitz.VersionBind,
        producer='scripts/check_final_pdf_layout.py', producer_sha256=sha(Path(__file__)),
        figure_manifest_sha256=manifest_hash, pdf_files=len(files),
        pdf_pages=sum(record['pages'] for record in files.values()),
        font_types=sorted(font_types), out_of_page_text=outside,
        raster_layers=sum(record['raster_layers'] for record in files.values()),
        preview_files=len(preview_records), files=files, previews=preview_records,
        errors=errors, tolerances=dict(page_points=POINT_TOLERANCE,
            text_bounds_points=TEXT_BOUND_TOLERANCE_PT, dpi=DPI_TOLERANCE),
        scope='Independent actual-PDF MediaBox/CropBox, page-count, metadata-title, '
            'embedded Type1 font, traced-text-bound and placed-raster-resolution audit; '
            'PNG resolution is checked from file metadata. PDF vector layers have no dpi. '
            'LaTeX and absence of in-panel titles are declared by the figure manifest; '
            'font size, overlapping labels and graphical clarity still require visual review.')


def audit_exports(root, example):
    """Inspect the curated main/supplementary pages without rasterizing them."""
    root = Path(root)
    manifest_path = root / 'results/draft_figure_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    selected = {name: entry for name, entry in manifest['files'].items()
                if entry['application'] == example}
    errors, files = [], {}
    if not selected:
        errors.append('No curated PDFs are declared for ' + example)
    if {entry['section'] for entry in selected.values()} != {'main', 'supplementary'}:
        errors.append('Both main and supplementary exports are required')
    for name, entry in selected.items():
        path = root / name
        if not path.is_file():
            errors.append('Missing curated PDF: ' + name)
            continue
        record = audit_pdf(path, entry)
        record['source_pdf'] = entry['source_pdf']
        record['source_pdf_sha256'] = entry['source_pdf_sha256']
        record['source_pages'] = entry['source_pages']
        files[name] = record
        if record['sha256'] != entry['sha256']:
            errors.append('Curated PDF differs from its manifest: ' + name)
        source = root / entry['source_pdf']
        if not source.is_file() or sha(source) != entry['source_pdf_sha256']:
            errors.append('Curated PDF source differs from its manifest: ' + name)
        errors.extend(name + ': ' + error for error in record['errors'])
    if sha(manifest_path) != manifest_hash:
        errors.append('Curated figure manifest changed during the audit')
    return dict(status='failed' if errors else 'passed',
        manifest='results/draft_figure_manifest.json', manifest_sha256=manifest_hash,
        pdf_files=len(files), pdf_pages=sum(record['pages'] for record in files.values()),
        files=files, errors=errors,
        scope='Actual page, font, text-bound and raster-resolution checks of the '
            'curated main/supplementary PDF copies and vector-preserving page extracts; '
            'source and exported PDF hashes are bound to the curated manifest.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=[*EXAMPLES, 'both'], default='both')
    parser.add_argument('--write', action='store_true', help='Refresh reports only if every selected application passes')
    parser.add_argument('--include-exports', action='store_true',
        help='Also audit curated PDFs declared in results/draft_figure_manifest.json')
    args = parser.parse_args()
    examples = EXAMPLES if args.example == 'both' else (args.example,)
    reports = {example: audit_example(ROOT, example) for example in examples}
    if args.include_exports:
        for example, report in reports.items():
            exports = audit_exports(ROOT, example)
            report['curated_exports'] = exports
            report['errors'].extend(exports['errors'])
            if exports['status'] != 'passed':
                report['status'] = 'failed'
    passed = all(report['status'] == 'passed' for report in reports.values())
    if args.write and passed:
        for example, report in reports.items():
            path = ROOT / 'results' / example / 'data/validation/pdf_layout_checks.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            temporary.replace(path)
    summary = {example: {name: report[name] for name in
        ('status', 'pdf_files', 'pdf_pages', 'font_types', 'raster_layers', 'preview_files', 'errors')}
        for example, report in reports.items()}
    if args.include_exports:
        for example in examples:
            summary[example]['curated_exports'] = {name: reports[example]['curated_exports'][name]
                for name in ('status', 'pdf_files', 'pdf_pages')}
    print(json.dumps(summary, indent=2))
    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
