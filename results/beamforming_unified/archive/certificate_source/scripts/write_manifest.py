"""Record source and artifact digests after all notebooks have completed."""
from pathlib import Path
import hashlib
import json
import nbformat
ROOT=Path(__file__).resolve().parents[1]
manifest={'files':{},'notebooks':{}}
for directory in ['lib','benchmarks','tests','scripts','notebooks','docs','results']:
    for p in sorted((ROOT/directory).rglob('*')):
        if not p.is_file() or any(part in p.parts for part in ['__pycache__','checkpoints','.ipynb_checkpoints']) or p.name in ['manifest.json','pytest.xml']:
            continue
        if p.suffix not in ['.py','.sh','.md','.json','.csv','.pdf','.png','.ipynb']:
            continue
        manifest['files'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['README.md','pyproject.toml','requirements-lock.txt','.github/workflows/tests.yml']:
    manifest['files'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
for p in (ROOT/'notebooks').glob('*.ipynb'):
    n=nbformat.read(p,as_version=4);nbformat.validate(n)
    code=[c for c in n.cells if c.cell_type=='code']
    assert code and all(c.execution_count is not None for c in code),p
    assert not any(o.output_type=='error' for c in code for o in c.outputs),p
    manifest['notebooks'][p.name]={'executed_cells':len(code),'errors':0}
(ROOT/'results/manifest.json').write_text(json.dumps(manifest,indent=2))
print(manifest['notebooks'])
