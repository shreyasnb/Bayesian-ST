"""Content-addressed gate for the example notebooks."""
import hashlib
import json
from pathlib import Path

CORE = ['lib/geometry.py','lib/models.py','lib/filter.py','lib/em.py','lib/moments.py',
        'lib/theory.py','lib/reference.py','lib/validation.py','lib/studies.py',
        'scripts/make_theory_notebook.py','lib/spectral_series.py',
        'lib/prediction.py','lib/process_prediction.py','lib/prediction_study.py',
        'lib/application_ablation.py','scripts/validate_dev.py']


def core_hash(root):
    root=Path(root)
    paths=CORE+sorted(str(p.relative_to(root)) for directory in ['benchmarks','tests'] for p in (root/directory).rglob('*.py'))
    h=hashlib.sha256()
    for p in paths:
        h.update(p.encode()); h.update((root/p).read_bytes())
    return h.hexdigest()


def require_gate(root, gate_path=None):
    root=Path(root).resolve()
    path=root/'results/validation.json' if gate_path is None else Path(gate_path)
    if not path.is_absolute():path=root/path
    if not path.exists(): raise RuntimeError('Run notebooks/00_theory.ipynb successfully first')
    gate=json.loads(path.read_text())
    if gate.get('status')!='passed' or gate.get('core_hash')!=core_hash(root):
        raise RuntimeError('General validation is missing or stale; rerun 00_theory.ipynb')
    return gate
