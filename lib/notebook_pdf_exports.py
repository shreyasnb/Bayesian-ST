"""Read-only declarations and checks for notebook manuscript PDF copies.

Destinations come from the visible notebook save controls, without executing
notebook code or importing plotting libraries. The historical catalogue is
verified separately; callers can allow exactly these additional PDF paths.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


APPLICATION_NOTEBOOKS = {
    '01_array.ipynb': 'beamforming',
    '03_contact.ipynb': 'contact',
}
SECTIONS = {'main', 'supplementary'}


def _qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _qualified(node.value) + '.' + node.attr
    return ''


def _path_parts(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_parts(node.left) + _path_parts(node.right)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [('literal', node.value)]
    return [('variable', _qualified(node))]


def _relative_directory(value, location):
    if not isinstance(value, str):
        raise ValueError(location + ': missing local PDF output directory')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path.parts[:1] != ('results',):
        raise ValueError(location + ': PDF output directory must be inside results/')
    return path


def declared_notebook_pdf_routes(root):
    """Return manuscript-relative destination -> local PDF declaration.

Each figure must declare ``PDF_PATH``, ``DRAFT_SECTION`` and ``DRAFT_PATH``
before creating its figure. Section choices and PDF basenames remain editable;
the declarations must stay inside the current application directories.
"""
    root = Path(root).resolve()
    routes = {}
    for filename, example in APPLICATION_NOTEBOOKS.items():
        notebook = json.loads((root/'notebooks'/filename).read_text())
        metadata = notebook.get('metadata', {}).get('editable_plotting', {})
        local_directory = _relative_directory(metadata.get('output_directory'), filename)
        for index, cell in enumerate(notebook['cells']):
            if cell.get('cell_type') != 'code':
                continue
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
            tree = ast.parse(source)
            location = f'{filename}, cell {index}'
            pdf_name = section = None
            declared = False
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                    if 'PDF_PATH' in names:
                        parts = _path_parts(node.value)
                        if len(parts) != 2 or parts[0] != ('variable', 'EDITABLE_OUTPUT') or parts[1][0] != 'literal':
                            raise ValueError(location + ': expected EDITABLE_OUTPUT / PDF basename')
                        pdf_name = parts[1][1]
                        if (not pdf_name.endswith('.pdf') or Path(pdf_name).name != pdf_name
                                or '/' in pdf_name or '\\' in pdf_name or pdf_name == '.pdf'):
                            raise ValueError(location + ': expected a plain PDF basename')
                        section = None
                        declared = False
                    if 'DRAFT_SECTION' in names:
                        section = ast.literal_eval(node.value)
                        if not isinstance(section, str) or section not in SECTIONS:
                            raise ValueError(location + ': unknown manuscript section')
                    if 'DRAFT_PATH' in names:
                        expected = [('variable', 'ROOT'), ('literal', 'results'),
                            ('variable', 'DRAFT_SECTION'), ('literal', example),
                            ('variable', 'PDF_PATH.name')]
                        if _path_parts(node.value) != expected or pdf_name is None or section is None:
                            raise ValueError(location + ': inconsistent manuscript PDF path')
                        destination = (Path('results')/section/example/pdf_name).as_posix()
                        if destination in routes:
                            raise ValueError('Duplicate notebook PDF destination: ' + destination)
                        routes[destination] = dict(
                            notebook='notebooks/'+filename, cell=index, example=example,
                            section=section, filename=pdf_name,
                            local_pdf=(local_directory/pdf_name).as_posix())
                        declared = True
                value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
                if isinstance(value, ast.Call) and _qualified(value.func) in ('plt.figure', 'plt.subplots'):
                    if not declared:
                        raise ValueError(location + ': figure has no manuscript PDF declaration')
                    declared = False
            if declared:
                raise ValueError(location + ': manuscript PDF declaration has no figure')
    return routes


def verify_notebook_pdf_exports(root, require_all=False):
    """Check declared copies, allowing a checkout before its first notebook run.

Existing manuscript copies must be PDFs. When a corresponding local export is
available, the two files must match byte for byte. No figure size, label, DPI or
rendered-content hash is fixed here. Unknown additional PDFs are intentionally
not accepted by this helper: the caller compares the directory's actual files
against the historical catalogue plus the returned declared destinations.
"""
    root = Path(root).resolve()
    routes = declared_notebook_pdf_routes(root)
    files, missing = {}, []
    matched = 0
    for destination, declaration in routes.items():
        path = root/destination
        if not path.is_file():
            missing.append(destination)
            continue
        with path.open('rb') as handle:
            if handle.read(5) != b'%PDF-':
                raise ValueError('Notebook manuscript copy is not a PDF: ' + destination)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        local = root/declaration['local_pdf']
        checked = local.is_file()
        if checked:
            if digest != hashlib.sha256(local.read_bytes()).hexdigest():
                raise ValueError('Notebook manuscript copy differs from local PDF: ' + destination)
            matched += 1
        files[destination] = dict(sha256=digest, bytes=path.stat().st_size,
            local_pdf=declaration['local_pdf'], local_copy_checked=checked,
            example=declaration['example'], section=declaration['section'])
    if require_all and missing:
        raise ValueError('Run the notebook cells to save manuscript PDFs: ' + ', '.join(missing))
    return dict(status='verified', declared_count=len(routes), verified_copies=len(files),
        local_matches_checked=matched, missing=missing, routes=routes, files=files)
