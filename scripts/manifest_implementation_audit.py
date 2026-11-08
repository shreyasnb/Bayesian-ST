"""Inventory the final guide evidence without changing scientific artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.implementation_validation import science_sources, sha

BASE = ROOT / 'results/implementation_audit'


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'test':
        raise RuntimeError('The implementation audit belongs on test')
    retained, excluded = {}, {}
    for path in sorted(BASE.rglob('*')):
        if (not path.is_file() or path == BASE / 'artifact_manifest.json'
                or '__pycache__' in path.parts):
            continue
        collection = excluded if 'checkpoints' in path.parts or path.name == 'pytest.xml' else retained
        collection[str(path.relative_to(ROOT))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    names = set(science_sources(ROOT))
    for pattern in ('*implementation_audit*.py', 'run_guide_*.py', 'run_block_em_study.py'):
        names.update(str(path.relative_to(ROOT)) for path in (ROOT / 'scripts').glob(pattern))
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / 'tests').glob('test_guide_*.py'))
    names.update(['tests/test_block_em.py', 'lib/distribution_geometry_visuals.py',
        'lib/simulation_story_visuals.py', 'lib/plotting.py', 'README.md', '.gitignore',
        'scripts/plot_geometry_studies.py', 'scripts/make_geometry_notebooks.py',
        'scripts/execute_geometry_notebooks.py', 'scripts/summarize_simulation_redesign.py',
        'scripts/check_simulation_redesign_handoff.py', 'scripts/manifest_simulation_redesign.py',
        'notebooks/01_array.ipynb', 'notebooks/03_contact.ipynb',
        'notebooks/04_implementation_audit.ipynb', 'docs/implementation_guide.md',
        'docs/implementation_guide_audit.md'])
    meta = dict(branch=branch, source_commit='6e78cc9911e7e78452633f13dd947afa669c3470',
        created_utc=datetime.now(timezone.utc).isoformat(), retained=retained, excluded=excluded,
        sources={name: sha(ROOT / name) for name in sorted(names)},
        external_artifacts={'results/simulation_redesign/artifact_manifest.json':
            sha(ROOT / 'results/simulation_redesign/artifact_manifest.json')},
        scope='All prior archives, final raw tables, numerical flags, reviewed figures and '
              'certificates are retained. Only regenerable worker checkpoints and raw pytest XML '
              'are excluded. This inventory excludes itself.')
    (BASE / 'artifact_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(len(retained), 'retained artifacts;', len(excluded), 'regenerable worker files')


if __name__ == '__main__':
    main()
