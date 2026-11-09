#!/usr/bin/env bash
# Run from any directory. Override BAYESIAN_ST_PYTHON or BAYESIAN_ST_VENV if needed.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BAYESIAN_ST_PYTHON:-python3}"
venv_path="${BAYESIAN_ST_VENV:-${repo_root}/.venv}"
"${python_bin}" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use CPython 3.12 for the validated environment"'
"${python_bin}" -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install -r "${repo_root}/requirements-lock.txt"
"${venv_path}/bin/python" -m pip install --no-deps -e "${repo_root}"
"${venv_path}/bin/python" -m pip check
"${venv_path}/bin/python" -m ipykernel install --sys-prefix --name python3 --display-name 'Bayesian ST (Python 3.12)'
"${venv_path}/bin/python" -c 'from importlib.metadata import version; print("Numerical stack:", {p: version(p) for p in ["numpy", "scipy", "matplotlib", "mpmath", "nbformat", "nbclient"]})'
# TeX is intentionally not installed here: system package changes require the
# user's normal package-management permissions. See docs/reproducibility.md.
printf '%s\n' "Setup complete. Activate ${venv_path}/bin/activate and validate TeX before notebook execution."
