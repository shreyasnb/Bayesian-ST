# Reviewed baseline

Reviewed commit: `57f520cbe860b73e13d71bae5c31391e2a009482`. The immutable manifest verifies 163 files.
The original pytest run passed 45 tests and two subtests. Three saved notebooks
have executed cells and no error outputs. Original raw-table hashes match.

`audit.json` records versions, seeds, budgets, configuration and table hashes,
the original core gate, deterministic recursion/prior/mode checks, and any
subsequent differences in reviewed paths. New extension files do not make
the original simulations dependencies retroactively.

Reproduce the initial audit before edits with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q > /tmp/dev-baseline-pytest.log
python scripts/audit_dev_baseline.py --require-unchanged
```

After extension changes, omit `--require-unchanged` to record those differences.
Use the saved baseline pytest log explicitly when the suite has gained tests.
