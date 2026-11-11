"""Check supplementary page coverage, figure controls and append-only provenance."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_editable_notebooks import verify_notebook
from scripts.append_supplementary_notebooks import catalogue, NOTEBOOKS


def check(example, root=ROOT, original_baseline=None, enforce_defaults=False):
    root = Path(root).resolve()
    filename = NOTEBOOKS[example]
    notebook = json.loads((root/'notebooks'/filename).read_text())
    meta = notebook['metadata']['supplementary_appendix']
    start = meta['original_cell_count']
    if original_baseline is not None:
        original = original_baseline[filename]['cells']
        assert notebook['cells'][:start] == original, 'An original cell changed: '+filename
    covered, additions, default_matches = set(), [], []
    for index, cell in enumerate(notebook['cells']):
        info = cell.get('metadata', {})
        if info.get('editable_plotting_role') != 'plot':
            continue
        stem, page = info['figure_id'], int(info.get('source_page', 1))
        if info.get('source_catalogue_page', stem != 'tracking_error'):
            covered.add((stem, page))
        if index < start:
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        assigned = {target.id: node.value for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}
        matches = (ast.literal_eval(assigned['FIGSIZE']) == (15, 5)
            and ast.literal_eval(assigned['DPI']) == 300
            and ast.literal_eval(assigned['DRAFT_SECTION']) == 'supplementary')
        default_matches.append(matches)
        if enforce_defaults:
            assert matches, 'Requested supplementary defaults differ: '+str((stem, page))
        constructors = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'plt'
            and node.func.attr in ('figure', 'subplots')]
        assert len(constructors) == 1
        assert cell['execution_count'] is not None, 'Unexecuted appended figure'
        images = sum('image/png' in output.get('data', {}) for output in cell['outputs'])
        assert images == 1 and not any(output['output_type']=='error' for output in cell['outputs'])
        additions.append((stem, page))
    assert covered == catalogue(root, example), 'Missing saved catalogue pages: '+example
    assert len(additions) == meta['appended_figures']
    assert sum(stem == 'tracking_error' for stem, page in additions) == 1
    generic = verify_notebook(root, filename, check_pdfs=True)
    return dict(status='passed', notebook=filename, original_cells=start,
        original_cells_checked_against_baseline=original_baseline is not None,
        canonical_pages_covered=len(covered), added_figure_cells=len(additions),
        added_figsize_in=[15, 5], added_dpi=300,
        requested_defaults_match=all(default_matches),
        defaults_enforced=enforce_defaults,
        actual_application_tracking_error=True, rendered_figures=generic['rendered_figures'],
        pdf_exports=generic['pdf_headers_checked'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-baseline', type=Path,
        help='Optional pre-append JSON snapshot for checking the complete original cell objects.')
    parser.add_argument('--check-defaults', action='store_true',
        help='Also check the requested 15-by-5, 300-dpi supplementary defaults; styling remains editable.')
    args = parser.parse_args()
    original = json.loads(args.original_baseline.read_text()) if args.original_baseline else None
    print(json.dumps({example: check(example, original_baseline=original,
                                    enforce_defaults=args.check_defaults)
                     for example in NOTEBOOKS}, indent=2))


if __name__ == '__main__':
    main()
