"""Test module."""

import yaml
from unittest.mock import patch

from ml_switcheroo.cli.handlers.define import handle_define


def test_handle_define_not_found(tmp_path):
  """Test element."""
  res = handle_define(tmp_path / "missing.yaml")
  assert res == 1


@patch("ml_switcheroo.cli.handlers.define.resolve_semantics_dir")
@patch("ml_switcheroo.cli.handlers.define.shutil.copy2")
def test_handle_define_success(mock_copy, mock_resolve, tmp_path):
  """Test element."""
  # Setup mock semantics dir
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  mock_resolve.return_value = sem_dir

  # Create valid ODL yaml
  in_file = tmp_path / "op.yaml"
  data = {"operation": "TestOp", "tier": "neural_net", "description": "Test", "variants": {}}
  with open(in_file, "w") as f:
    yaml.dump(data, f)

  res = handle_define(in_file)

  assert res == 0
  mock_copy.assert_called_once_with(in_file, sem_dir / "odl" / "TestOp.yaml")


def test_handle_define_invalid_schema(tmp_path):
  """Test element."""
  in_file = tmp_path / "op.yaml"
  data = {"invalid": "data"}  # missing 'operation'
  with open(in_file, "w") as f:
    yaml.dump(data, f)

  res = handle_define(in_file)
  assert res == 1
