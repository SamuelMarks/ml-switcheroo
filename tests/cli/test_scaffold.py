"""Test module."""

import json
from argparse import Namespace
from unittest.mock import patch
from ml_switcheroo.cli.handlers.scaffold import handle_scaffold


@patch("ml_switcheroo.cli.handlers.scaffold.ConsensusEngine")
def test_handle_scaffold(mock_engine_class, tmp_path, monkeypatch):
  """Test element."""
  monkeypatch.chdir(tmp_path)

  mock_engine = mock_engine_class.return_value
  mock_engine.cluster.return_value = {"add": ["mock.add"]}

  args = Namespace(framework="mock_fw")
  handle_scaffold(args)

  mock_engine.ingest.assert_called_once()
  mock_engine.cluster.assert_called_once_with(threshold=0.8)

  out_file = tmp_path / "mock_fw_skeleton.json"
  assert out_file.exists()

  data = json.loads(out_file.read_text())
  assert data["framework"] == "mock_fw"
  assert len(data["mappings"]) == 1
  assert data["mappings"][0] == {"operation": "add", "api": "mock.add"}
