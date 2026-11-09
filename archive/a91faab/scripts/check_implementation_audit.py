"""Read-only final guide handoff checks, including archives and embedded images."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from lib.implementation_validation import (
    COMPONENTS, require_gate, verify_component, verify_mapping, verify_preservation, sha)
from scripts.summarize_implementation_audit import verify_report, verify_figures
from scripts.execute_implementation_audit import inspect

BASE = ROOT / 'results/implementation_audit'


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'test':
        raise RuntimeError('Expected test branch')
    archive = verify_preservation(ROOT)
    for name in ('main', 'dev'):
        current = subprocess.check_output(['git', 'rev-parse', name], cwd=ROOT, text=True).strip()
        if current != archive['branch_tips'][name]:
            raise RuntimeError('Historical branch changed: ' + name)
    gate = require_gate(ROOT)
    for profile in ('smoke', 'paper'):
        for component in COMPONENTS:
            verify_component(ROOT, profile, component)
        figures = verify_figures(ROOT, profile)
        for name in figures['files']:
            path = BASE / profile / 'figures' / name
            if path.suffix == '.png':
                with Image.open(path) as im:
                    if any(abs(value - 200) > .1 for value in im.info.get('dpi', (0, 0))):
                        raise RuntimeError('Incorrect PNG resolution: ' + name)
                    if im.width < 1500 or im.height < 700:
                        raise RuntimeError('Figure is too small: ' + name)
                    im.verify()
    verify_report(ROOT)
    execution = json.loads((BASE / 'execution.json').read_text())
    if execution['status'] != 'passed' or execution['tests'] != gate['tests']:
        raise RuntimeError('Current notebook execution is missing')
    verify_mapping(ROOT, execution['sources'])
    verify_mapping(ROOT, execution['input_hashes'])
    for name, expected in execution['notebooks'].items():
        folder = (BASE / 'paper/figures' if '04_' in name
                  else ROOT / 'results/simulation_redesign/paper/figures')
        if inspect(ROOT / name, folder) != expected:
            raise RuntimeError('Notebook execution or inline figures changed: ' + name)
    inventory = json.loads((BASE / 'artifact_manifest.json').read_text())
    for field in ('retained', 'sources', 'external_artifacts'):
        verify_mapping(ROOT, inventory[field])
    for name, record in inventory['excluded'].items():
        if (ROOT / name).exists():
            verify_mapping(ROOT, {name: record})
    # Review-time rejected/corrected outputs keep exact producer snapshots.
    for path in sorted((BASE / 'development').rglob('preservation.json')):
        meta = json.loads(path.read_text())
        mapping = meta.get('files', {})
        if mapping:
            verify_mapping(path.parent, mapping)
        for name, expected in meta.get('regenerable_files', {}).items():
            if (path.parent / name).exists():
                verify_mapping(path.parent, {name: expected})
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_simulation_redesign_handoff.py')],
                   cwd=ROOT, check=True)
    print(f"Passed implementation audit: {gate['tests']} tests; "
          f"{len(COMPONENTS)} smoke and paper components; all guide figure IDs; "
          f"three executed notebooks; {len(archive['files'])} preserved prior files.")


if __name__ == '__main__':
    main()
