"""Audit the reviewed baseline without rewriting any historical artifact.

The manifest is read from the immutable reviewed commit. Extension files are
not numerical dependencies of those saved results. Changes to existing sources
are listed separately; historical raw-table changes are always an error.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE='57f520cbe860b73e13d71bae5c31391e2a009482'
GUIDE_SHA='66ed53949e18f878b85a0f80567405dacd18af116d1108812aba3841eccad2d2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def committed(path):
    return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT)


def deterministic_checks():
    import numpy as np
    from lib.filter import BinghamFilter
    from lib.geometry import center,projector
    from lib.models import increment
    from benchmarks.case2 import EWCovariancePCA
    rng=np.random.default_rng(615702)
    records=[]
    for complex_data in (False,True):
        n,k=8,3
        f=BinghamFilter(n,k,.3,tau2=2.,beta=.93,complex_data=complex_data)
        ew=EWCovariancePCA(n,k,beta=.93)
        max_difference=0.;max_recursion=0.;max_prior=0.
        max_modal_score_difference=0.
        for t in range(32):
            V=rng.normal(size=(n,n));u=rng.normal(size=n)
            if complex_data:
                V=V+1j*rng.normal(size=(n,n));u=(u+1j*rng.normal(size=n))/np.sqrt(2)
            G=np.linalg.qr(V)[0]
            expected_prior=center(.93*G@f.A@G.conj().T)
            expected=center(expected_prior+increment(u,.3,2.,complex_data=complex_data))
            f.update(u,G=G);ew.transform(G);ew.update(u)
            # Before k generic observations, the top-k space is nonunique.
            # Compare modal objectives there, rather than arbitrary tied bases.
            if t+1>=k:
                max_difference=max(max_difference,float(np.linalg.norm(projector(f.basis)-projector(ew.basis))))
            max_modal_score_difference=max(max_modal_score_difference,
                float(abs(np.trace(f.A@(projector(f.basis)-projector(ew.basis))))))
            max_recursion=max(max_recursion,float(np.linalg.norm(f.A-expected)))
            max_prior=max(max_prior,float(np.linalg.norm(f.records[-1].prior-expected_prior)))
        assert max_difference<1e-8 and max_recursion<1e-8 and max_prior<1e-8 and max_modal_score_difference<1e-8
        records.append(dict(complex_data=complex_data,steps=32,
            maximum_ew_modal_projector_difference=max_difference,
            ew_projector_comparisons_start_at_observation=k,
            maximum_ew_modal_objective_difference=max_modal_score_difference,
            maximum_single_correction_parameter_difference=max_recursion,
            maximum_stored_prior_difference=max_prior))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/dev_prediction/baseline')
    parser.add_argument('--pytest-log',type=Path,default=Path('/tmp/dev-baseline-pytest.log'))
    parser.add_argument('--require-unchanged',action='store_true',
                        help='Require all reviewed source and artifact bytes to match (initial audit).')
    args=parser.parse_args()
    manifest=json.loads(committed('results/manifest.json'))
    files={};changed=[];missing=[];table_failures=[]
    for name,expected in manifest['files'].items():
        original=committed(name)
        assert sha(original)==expected,('Reviewed manifest disagrees with reviewed artifact',name)
        path=ROOT/name
        actual=sha(path.read_bytes()) if path.exists() else None
        files[name]=dict(reviewed_sha256=expected,current_sha256=actual,matches=actual==expected)
        if actual!=expected:
            changed.append(name)
            if actual is None:missing.append(name)
            if name.startswith('results/') and name.endswith('.csv'):table_failures.append(name)
    assert not table_failures,('Historical raw tables changed',table_failures)
    if args.require_unchanged:assert not changed,('Reviewed files changed before initial audit',changed)
    notebooks={}
    for name in ['00_theory','01_array','02_field']:
        notebook=json.loads(committed(f'notebooks/{name}.ipynb'))
        code=[c for c in notebook['cells'] if c['cell_type']=='code']
        assert all(c['execution_count'] is not None for c in code)
        assert not any(o.get('output_type')=='error' for c in code for o in c['outputs'])
        notebooks[name]=dict(executed_cells=len(code),errors=0,
                             sha256=sha(committed(f'notebooks/{name}.ipynb')))
    configs={}
    for name in manifest['files']:
        if name.startswith('results/') and name.endswith('_config.json'):
            metadata=json.loads(committed(name))
            configs[name]=dict(sha256=sha(committed(name)),config=metadata.get('config'),
                requested_config=metadata.get('requested_config'),source_hash=metadata.get('source_hash'),
                config_digest=metadata.get('config_digest'),tables=metadata.get('tables'),
                checks=metadata.get('checks'))
    pytest_log=args.pytest_log.read_text() if args.pytest_log.exists() else None
    passed=bool(pytest_log and re.search(r'45 passed,.*2 subtests passed',pytest_log))
    assert passed,'Run the original baseline pytest suite first and pass --pytest-log.'
    packages={name:importlib.metadata.version(name) for name in
              ['numpy','scipy','matplotlib','pandas','pytest','nbformat','nbclient']}
    result=dict(status='implemented and checked',reviewed_commit=BASE,
        inspected_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        guide=dict(filename='bayesian_st_implementation_checkpoints.pdf',sha256=GUIDE_SHA),
        versions=dict(python=platform.python_version(),packages=packages),
        baseline_gate=json.loads(committed('results/validation.json')),
        reviewed_manifest_sha256=sha(committed('results/manifest.json')),
        reviewed_files=len(files),files=files,changed_reviewed_paths=changed,missing_reviewed_paths=missing,
        historical_raw_tables_unchanged=True,notebooks=notebooks,configurations=configs,
        baseline_pytest=dict(passed=45,subtests_passed=2,log_sha256=sha(pytest_log.encode()),
                             command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q'),
        deterministic_checks=deterministic_checks(),
        interpretation=[
            'Original application comparisons supplied half the physical motion; identity-motion controls supplied none.',
            'Known-motion controls in drift_study are rotations of a static experiment, not independent replicates.',
            'Causal learned rotations depend on past noise, so the deterministic-transformation theorem does not automatically apply.',
            'Transition first-moment propagation can be exact while Bingham projection of the full density is approximate.',
            'Rolling likelihood sums assume fixed noise and identity motion; local stationarity is an approximation.',
            'Historical sampling flags and coverage failures remain unchanged; this audit is not a Monte Carlo rerun.'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2))
    (args.output/'pytest.log').write_text(pytest_log)
    (args.output/'README.md').write_text(
        '# Reviewed baseline\n\n'
        f'Reviewed commit: `{BASE}`. The immutable manifest verifies {len(files)} files.\n'
        'The original pytest run passed 45 tests and two subtests. Three saved notebooks\n'
        'have executed cells and no error outputs. Original raw-table hashes match.\n\n'
        '`audit.json` records versions, seeds, budgets, configuration and table hashes,\n'
        'the original core gate, deterministic recursion/prior/mode checks, and any\n'
        'subsequent differences in reviewed paths. New extension files do not make\n'
        'the original simulations dependencies retroactively.\n\n'
        'Reproduce the initial audit before edits with:\n\n'
        '```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q > /tmp/dev-baseline-pytest.log\n'
        'python scripts/audit_dev_baseline.py --require-unchanged\n```\n\n'
        'After extension changes, omit `--require-unchanged` to record those differences.\n'
        'Use the saved baseline pytest log explicitly when the suite has gained tests.\n')
    print(json.dumps(dict(status=result['status'],reviewed_files=len(files),
                         changed_reviewed_paths=changed,historical_raw_tables_unchanged=True,
                         notebooks=notebooks,deterministic_checks=result['deterministic_checks']),indent=2))


if __name__=='__main__':
    main()
