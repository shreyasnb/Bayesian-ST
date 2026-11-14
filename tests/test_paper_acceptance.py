import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location(
    'paper_check', Path(__file__).resolve().parents[1]/'scripts/check_paper.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.mark.parametrize('trial_ids', [range(255), [*range(255), 254]])
def test_paper_grid_rejects_missing_or_duplicate_simulations(trial_ids):
    frame = pd.DataFrame({'trial': list(trial_ids), 'method': 'mean'})
    with pytest.raises(ValueError, match='simulation|Duplicate'):
        CHECK.grid(frame, ['method'], 1)


def test_paper_manifest_detects_changed_retained_data(tmp_path):
    data = tmp_path/'cohort.csv'
    data.write_text('loss\n1\n')
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps({'artifacts': {'cohort.csv': digest}}))
    assert CHECK.check_manifest(tmp_path) == 'passed'
    data.write_text('loss\n2\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        CHECK.check_manifest(tmp_path)


def test_paper_manifest_rejects_paths_outside_results(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'artifacts': {'../outside.csv': '0'*64}}))
    with pytest.raises(ValueError, match='escapes results'):
        CHECK.check_manifest(tmp_path)
