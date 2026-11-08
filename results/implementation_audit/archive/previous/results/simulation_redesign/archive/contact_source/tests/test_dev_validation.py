"""Certificate routing must preserve dependency checks for isolated outputs."""
import json
from pathlib import Path
import pytest
from lib import validation


def test_relative_repository_root_and_isolated_gate_reject_source_changes(tmp_path,monkeypatch):
    root=tmp_path/'repository';root.mkdir()
    (root/'numerical.py').write_text('VALUE = 1\n')
    monkeypatch.setattr(validation,'CORE',['numerical.py'])
    digest=validation.core_hash(root)
    original=root/'results/validation.json';original.parent.mkdir()
    original.write_text(json.dumps(dict(status='passed',core_hash=digest)))
    isolated=root/'results/dev_prediction/validation.json';isolated.parent.mkdir()
    isolated.write_text(original.read_text())
    monkeypatch.chdir(tmp_path)
    assert validation.require_gate(Path('repository'))['core_hash']==digest
    assert validation.require_gate(Path('repository'),gate_path='results/dev_prediction/validation.json')['core_hash']==digest
    (root/'numerical.py').write_text('VALUE = 2\n')
    with pytest.raises(RuntimeError,match='stale'):
        validation.require_gate(root,gate_path=isolated)
