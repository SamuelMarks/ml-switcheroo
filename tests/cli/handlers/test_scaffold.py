"""Unit tests for `handle_scaffold` (`ml_switcheroo.cli.handlers.scaffold`)."""

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.cli.handlers.scaffold import handle_scaffold


def test_handle_scaffold_writes_skeleton_and_skips_empty_clusters(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Verify `handle_scaffold` writes `<framework>_skeleton.json` and skips empty cluster entries.

  Args:
      tmp_path: Pytest temporary directory fixture.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.chdir(tmp_path)

  with patch("ml_switcheroo.cli.handlers.scaffold.ConsensusEngine") as mock_engine_cls:
    mock_engine: MagicMock = mock_engine_cls.return_value
    mock_engine.cluster.return_value = {"add": ["mock.add"], "empty_op": []}

    handle_scaffold(Namespace(framework="mock_fw"))
    mock_engine.ingest.assert_called_once()
    mock_engine.cluster.assert_called_once_with(threshold=0.8)

  out_file = tmp_path / "mock_fw_skeleton.json"
  assert out_file.exists()
  data = json.loads(out_file.read_text(encoding="utf-8"))
  assert data["framework"] == "mock_fw"
  assert data["mappings"] == [{"operation": "add", "api": "mock.add"}]
