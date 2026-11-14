#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BAYESIAN_ST_PYTHON:-python3.12}"
venv_path="${BAYESIAN_ST_VENV:-${repo_root}/.venv}"
"${python_bin}" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use CPython 3.12 for the validated environment"'
"${python_bin}" -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install -r "${repo_root}/requirements-lock.txt"
"${venv_path}/bin/python" -m pip install --no-deps -e "${repo_root}"
"${venv_path}/bin/python" -m pip check
