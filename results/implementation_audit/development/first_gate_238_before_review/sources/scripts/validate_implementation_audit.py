"""Execute the full general notebook and all regressions in a new directory."""
from pathlib import Path
import os
import sys
import json
import hashlib
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR', '/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.implementation_validation import science_hash, science_sources, sha, verify_preservation
from lib.simulation_validation import science_hash as redesign_hash


def main():
    verify_preservation(ROOT)
    out = ROOT / 'results/implementation_audit/validation'
    out.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'notebooks/00_theory.ipynb'
    saved = original.read_bytes()
    old_certificate = ROOT / 'results/simulation_redesign/validation/validation.json'
    old_saved = old_certificate.read_bytes()
    fingerprint = science_hash(ROOT)
    nb = nbformat.read(original, as_version=4)
    assignment = "OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)"
    matches = [c for c in nb.cells if c.cell_type == 'code' and assignment in c.source]
    if len(matches) != 1:
        raise RuntimeError('Review the isolated notebook output assignment')
    matches[0].source = matches[0].source.replace(assignment,
        f'OUT=Path({str(out)!r}); OUT.mkdir(parents=True,exist_ok=True)')
    for c in nb.cells:
        if c.cell_type == 'code':
            c.outputs = []
            c.execution_count = None
    certificate = out / 'validation.json'
    certificate.unlink(missing_ok=True)
    try:
        NotebookClient(nb, timeout=3600, kernel_name='python3',
            resources={'metadata': {'path': str(ROOT)}}).execute()
        if (original.read_bytes() != saved or science_hash(ROOT) != fingerprint
                or old_certificate.read_bytes() != old_saved):
            raise RuntimeError('Sources or original validation changed during the run')
        gate = json.loads(certificate.read_text())
        gate.update(science_hash=redesign_hash(ROOT), implementation_hash=fingerprint,
            sources={name: sha(ROOT / name) for name in science_sources(ROOT)},
            original_notebook_sha256=hashlib.sha256(saved).hexdigest(),
            predecessor_validation_sha256=hashlib.sha256(old_saved).hexdigest(),
            scope='Complete general notebook and full regression suite, including '
                  'the guide additions. Empirical outcomes, finite Monte Carlo '
                  'precision and missing manuscript-source checks are separate statuses.')
        certificate.write_text(json.dumps(gate, indent=2) + '\n')
        print(json.dumps(gate, indent=2), flush=True)
    except BaseException:
        certificate.unlink(missing_ok=True)
        raise
    finally:
        nbformat.write(nb, out / 'validation.ipynb')


if __name__ == '__main__':
    main()
