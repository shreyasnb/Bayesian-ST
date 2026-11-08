"""Read-only verification of the published draft-5 simulation handoff."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/simulation_redesign'
sys.path.insert(0, str(ROOT))
import nbformat
from PIL import Image
from lib.simulation_validation import require_gate, verify_component, verify_legacy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify(mapping, folder=ROOT):
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            expected = expected['sha256']
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Changed or absent evidence: ' + str(path))


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'test':
        raise RuntimeError('Run the handoff check on the test branch')
    historical = verify_legacy(ROOT)
    for name in ('main', 'dev'):
        tip = subprocess.check_output(['git', 'rev-parse', name], cwd=ROOT, text=True).strip()
        if tip != historical['branch_tips'][name]:
            raise RuntimeError('Historical branch changed: ' + name)
    gate = require_gate(ROOT)
    if gate['tests'] != 209:
        raise RuntimeError('Unexpected general-validation test count')
    original = ROOT / 'notebooks/00_theory.ipynb'
    if sha(original) != gate['original_notebook_sha256']:
        raise RuntimeError('The original general notebook was changed')
    components = ('geometry', 'beamforming', 'contact', 'scaling')
    figure_count = 0
    for profile in ('quick', 'paper'):
        for component in components:
            verify_component(ROOT, profile, component)
        folder = BASE / profile / 'figures'
        figures = read(folder / 'manifest.json')
        if (figures['profile'] != profile or figures['text_usetex'] is not True
                or figures['figure_dpi'] != 200 or figures['savefig_dpi'] != 200):
            raise RuntimeError('Unexpected figure protocol: ' + profile)
        verify(figures['sources'])
        verify(figures['input_manifests'])
        verify(figures['files'], folder)
        if len(figures['files']) != 24 or any(len(figures['groups'][app]) != 6
                for app in ('beamforming', 'contact')):
            raise RuntimeError('Expected six main figures per application')
        for name in figures['files']:
            path = folder / name
            if path.suffix == '.png':
                with Image.open(path) as im:
                    if im.width < 1500 or im.height < 700:
                        raise RuntimeError('Figure is too small: ' + name)
                    if any(abs(value - 200) > .1 for value in im.info.get('dpi', (0, 0))):
                        raise RuntimeError('PNG is not 200 dpi: ' + name)
                    im.verify()
            elif path.suffix == '.pdf' and not path.read_bytes().startswith(b'%PDF-'):
                raise RuntimeError('Invalid PDF: ' + name)
        figure_count += len(figures['files'])
    summary = read(BASE / 'summary_manifest.json')
    if (summary['profile'] != 'paper' or summary['status'] != 'completed'
            or not summary['paper_evidence'] or not summary['figures_verified']
            or summary['pending']):
        raise RuntimeError('The final findings are not complete paper evidence')
    verify({summary['generator']['path']: summary['generator']['sha256']})
    verify(summary['source_hashes'])
    verify(summary['verified_input_hashes'])
    verify(summary['outputs'])
    if summary['gate_sha256'] != sha(BASE / 'validation/validation.json'):
        raise RuntimeError('Summary validation is stale')
    execution = read(BASE / 'execution.json')
    if execution['status'] != 'passed':
        raise RuntimeError('Notebook execution failed')
    verify(execution['sources'])
    if execution['figure_manifest_sha256'] != sha(BASE / 'paper/figures/manifest.json'):
        raise RuntimeError('Notebook figures are stale')
    if execution['summary_manifest_sha256'] != sha(BASE / 'summary_manifest.json'):
        raise RuntimeError('Notebook findings are stale')
    if execution['validation_sha256'] != sha(BASE / 'validation/validation.json'):
        raise RuntimeError('Notebook validation is stale')
    for name, record in execution['notebooks'].items():
        path = ROOT / name
        if sha(path) != record['sha256']:
            raise RuntimeError('Executed notebook changed: ' + name)
        nb = nbformat.read(path, as_version=4)
        cells = [c for c in nb.cells if c.cell_type == 'code']
        errors = [out for cell in cells for out in cell.outputs if out.output_type == 'error']
        images = sum('image/png' in out.get('data', {}) for cell in cells for out in cell.outputs)
        if (len(cells) != record['code_cells'] or errors or record['error_outputs']
                or any(c.execution_count is None for c in cells)
                or images != record['image_outputs'] or images != 6):
            raise RuntimeError('Incomplete saved notebook: ' + name)
    inventory = read(BASE / 'artifact_manifest.json')
    verify(inventory['sources'])
    verify(inventory['retained'])
    for name, record in inventory['excluded'].items():
        if (ROOT / name).exists():
            verify({name: record})
    print(f"Passed: {gate['tests']} tests; 8 verified study profiles; "
          f"{figure_count} PDF/PNG exports; 2 executed notebooks.")
    print(f"Preserved: {len(historical['unchanged_results'])} historical artifacts; "
          f"{len(historical['archive_files'])} archived notebook/source files; main and dev unchanged.")


if __name__ == '__main__':
    main()
