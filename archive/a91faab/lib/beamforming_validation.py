"""Content-addressed validation and evidence checks for the passive study."""
from pathlib import Path
import hashlib
import json
from .validation import core_hash

ORIGINAL_SOURCES=(
    'lib/models.py','lib/filter.py','lib/geometry.py','lib/theory.py','lib/em.py',
    'lib/moments.py','lib/spectral_series.py','lib/process_prediction.py',
    'lib/reference.py','benchmarks/case1/baselines.py','benchmarks/case2/baselines.py')
PRESENTATION={'beamforming_visuals.py','beamforming_results.py','beamforming_history.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def science_sources(root):
    root=Path(root)
    names=list(ORIGINAL_SOURCES)
    names.extend(str(p.relative_to(root)) for p in (root/'lib').glob('beamforming_*.py')
                 if p.name not in PRESENTATION)
    # Runners specify scientific budgets, seed streams, CV and prospective fits.
    names.extend(str(p.relative_to(root)) for p in (root/'scripts').glob('run_beamforming_*.py'))
    names.append('scripts/validate_beamforming.py')
    return sorted(set(names))


def beamforming_hash(root):
    root=Path(root);digest=hashlib.sha256()
    for name in science_sources(root):
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def require_beamforming_gate(root):
    root=Path(root).resolve()
    p=root/'results/beamforming_unified/validation/validation.json'
    if not p.exists():raise RuntimeError('Run scripts/validate_beamforming.py successfully first')
    gate=json.loads(p.read_text())
    if (gate.get('status')!='passed' or gate.get('core_hash')!=core_hash(root)
        or gate.get('beamforming_hash')!=beamforming_hash(root)):
        raise RuntimeError('Passive beamforming validation is missing or stale')
    return gate


def verify_component(root,profile,component):
    """Verify completed scientific sources and saved tables before plotting.

    Component runners may retain their established manifest/config conventions;
    paths in `sources` are repository-relative and paths in `tables` are local
    to the component directory. Status is checked when a status file exists.
    """
    root=Path(root).resolve();directory=root/'results/beamforming_unified'/profile/component
    choices=[directory/'manifest.json',directory/'config.json']
    path=next((p for p in choices if p.exists()),None)
    if path is None:raise RuntimeError(f'Complete {profile}/{component} first')
    meta=json.loads(path.read_text())
    status=directory/'status.json'
    if status.exists():
        value=json.loads(status.read_text()).get('status')
        if value not in ('completed','passed','verified'):
            raise RuntimeError(f'Incomplete component {component}: {value}')
    source=meta.get('sources',{})
    if not source:raise RuntimeError(f'No source hashes in {path}')
    for name,digest in source.items():
        if isinstance(digest,dict):digest=digest.get('sha256')
        if not digest or sha(root/name)!=digest:raise RuntimeError(f'Stale source: {name}')
    tables=meta.get('tables',meta.get('files',{}))
    if not tables:raise RuntimeError(f'No artifact hashes in {path}')
    for name,digest in tables.items():
        if isinstance(digest,dict):digest=digest.get('sha256')
        local=directory/name
        if not local.exists():local=root/name
        if not digest or sha(local)!=digest:raise RuntimeError(f'Changed artifact: {local}')
    return meta
