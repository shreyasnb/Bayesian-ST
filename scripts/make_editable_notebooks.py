"""Generate editable notebooks without moving Matplotlib behind a renderer.

The cell sources for the two current applications are normal Python templates;
the historical field/audit walkthroughs use source-only JSON templates. The
scientific theory generator remains unchanged and has a presentation converter.
Generating a notebook explicitly resets that notebook's plotting defaults.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = ('00_theory','01_array','02_field','03_contact','04_implementation_audit')


def build_legacy(root, name):
    root = Path(root).resolve()
    if name not in ('02_field','04_implementation_audit'):
        raise ValueError('Unknown saved historical notebook: '+name)
    path = root/'scripts/notebook_templates'/(name+'.json')
    notebook = nbformat.reads(path.read_text(), as_version=4)
    meta = notebook.metadata['editable_plotting']
    mappings = [meta.get('input_hashes', {}),meta.get('data_hashes', {}),meta.get('loader_sources', {})]
    if meta.get('loader'):
        mappings.append({meta['loader']:meta['loader_sha256']})
    for mapping in mappings:
        for relative, expected in mapping.items():
            actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError('Saved notebook input changed: '+relative)
    nbformat.validate(notebook)
    return notebook


def generate(names=NAMES, root=ROOT):
    root = Path(root).resolve()
    for name in names:
        path = root/'notebooks'/(name+'.ipynb')
        if name == '00_theory':
            if not path.exists():
                runpy.run_path(str(root/'scripts/make_theory_notebook.py'), run_name='__main__')
            from scripts.refactor_theory_plots import refactor
            changed = refactor(path)
            print(('Converted ' if changed else 'Kept already editable ')+path.name)
            continue
        if name in ('01_array','03_contact'):
            from scripts.make_final_notebooks import build
            notebook = build(root, 'beamforming' if name=='01_array' else 'contact')
        else:
            notebook = build_legacy(root, name)
        nbformat.write(notebook, path)
        print('Wrote', path.name, 'with complete editable plotting cells')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', choices=NAMES, action='append')
    args = parser.parse_args()
    generate(args.notebook or NAMES)


if __name__ == '__main__':
    main()
