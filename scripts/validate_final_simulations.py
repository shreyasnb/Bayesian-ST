"""Run the complete general notebook and regressions without altering old gates."""
from pathlib import Path
import os
import sys
import json
import hashlib
import argparse
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/bayesian-st-jupyter')
os.environ.setdefault('IPYTHONDIR', '/tmp/bayesian-st-ipython')
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.final_validation import (
    science_hash, science_sources, verify_history, gate_path, current_gate_path, sha,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', choices=['stationary_T200'],
        help='Write a fresh revision certificate while preserving the original validation.')
    args = parser.parse_args()
    verify_history(ROOT)
    certificate = current_gate_path(ROOT) if args.revision else gate_path(ROOT)
    if certificate == gate_path(ROOT) and certificate.exists():
        previous = json.loads(certificate.read_text())
        if previous.get('final_science_hash') != science_hash(ROOT):
            raise RuntimeError('The original certificate is preserved; use --revision stationary_T200')
    out = certificate.parent
    out.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'notebooks/00_theory.ipynb'
    before = original.read_bytes()
    fingerprint = science_hash(ROOT)
    nb = nbformat.read(original, as_version=4)
    assignment = "OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)"
    matches = [cell for cell in nb.cells if cell.cell_type == 'code' and assignment in cell.source]
    if len(matches) != 1:
        raise RuntimeError('Review the original validation output assignment')
    matches[0].source = matches[0].source.replace(assignment,
        f'OUT=Path({str(out)!r}); OUT.mkdir(parents=True,exist_ok=True)')
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    certificate.unlink(missing_ok=True)
    try:
        NotebookClient(nb, timeout=3600, kernel_name='python3',
            resources={'metadata': {'path': str(ROOT)}}).execute()
        if original.read_bytes() != before or science_hash(ROOT) != fingerprint:
            raise RuntimeError('General notebook or scientific sources changed during validation')
        gate = json.loads(certificate.read_text())
        gate.update(final_science_hash=fingerprint,
            sources={name: sha(ROOT / name) for name in science_sources(ROOT)},
            original_notebook_sha256=hashlib.sha256(before).hexdigest(),
            revision=args.revision,
            scope='Complete original general notebook and all current regressions. '
                  'Final stage-specific scientific and statistical limitations are '
                  'retained independently of algebra/test success.')
        certificate.write_text(json.dumps(gate, indent=2) + '\n')
        print(json.dumps(gate, indent=2), flush=True)
    except BaseException:
        certificate.unlink(missing_ok=True)
        raise
    finally:
        nbformat.write(nb, out / 'general_validation.ipynb')


if __name__ == '__main__':
    main()
