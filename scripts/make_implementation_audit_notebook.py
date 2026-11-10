"""Compatibility entry point for editable notebook generation.

Plotting statements are emitted directly into each notebook cell. The old
pre-rendered-image notebook generator is retained in Git history and archive.
"""
from pathlib import Path
import hashlib, json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.plotting import DPI
from scripts.make_editable_notebooks import generate

REQUIRED_IDS = {'B1', 'B2a', 'B2b', 'B3', 'B4', 'B5', 'B6', 'B7a', 'B7b',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6a', 'C6b', 'S1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_audit():
    from lib.implementation_validation import require_gate, verify_component
    require_gate(ROOT)
    base = ROOT/'results/implementation_audit'
    for name in ['information', 'information_counts', 'tracking', 'em', 'reliability']:
        verify_component(ROOT, 'paper', name)
    path = base/'paper/figures/manifest.json'
    figures = json.loads(path.read_text())
    if (figures.get('status') != 'completed' or figures.get('profile') != 'paper'
            or not REQUIRED_IDS.issubset(figures['figure_ids'])
            or figures.get('text_usetex') is not True or figures.get('figure_dpi') != DPI
            or figures.get('savefig_dpi') != DPI):
        raise RuntimeError('Complete the current paper guide figures before building this notebook')
    for mapping, folder in [('sources', ROOT), ('input_manifests', ROOT), ('files', path.parent)]:
        for name, expected in figures[mapping].items():
            if isinstance(expected, dict): expected = expected['sha256']
            if sha(folder/name) != expected: raise RuntimeError('Changed guide evidence: '+name)
    if figures['validation_sha256'] != sha(base/'validation/validation.json'):
        raise RuntimeError('Guide figures use a stale validation gate')
    if not (base/'findings.md').is_file() or not (base/'claim_manifest.json').is_file():
        raise RuntimeError('Write the completed findings and claim manifest before notebook generation')
    from scripts.summarize_implementation_audit import verify_report
    verify_report(ROOT)
    return figures


if __name__ == '__main__':
    generate(('04_implementation_audit',))
