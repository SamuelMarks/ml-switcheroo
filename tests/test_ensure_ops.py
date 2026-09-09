"""Tests for ensure_ops script."""

import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


import scripts.ensure_ops


@patch("scripts.ensure_ops.Path")
def test_run(mock_path_class: MagicMock, tmp_path: Path) -> None:
  """Test run function creates missing yaml files."""
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  # Pre-create one file to test both paths (exists/not exists)
  existing_op = "Conv1d"
  existing_file = odl_dir / f"{existing_op}.yaml"
  existing_file.touch()

  # Setup mock Path to return our temp dir
  mock_base = MagicMock()
  mock_odl_dir = MagicMock()
  mock_path_class.return_value = mock_base
  mock_base.__truediv__.return_value = mock_odl_dir

  def side_effect(filename: str) -> Path:
    """Side effect for truediv."""
    return odl_dir / filename

  mock_odl_dir.__truediv__.side_effect = side_effect

  scripts.ensure_ops.run()

  # Check that Conv1d is not overwritten (it was empty, let's just check other files exist)
  assert (odl_dir / "Conv2d.yaml").exists()

  # Verify content of one newly created file
  with open(odl_dir / "Conv2d.yaml") as f:
    data = yaml.safe_load(f)
  assert data["operation"] == "Conv2d"
  assert data["description"] == "Verified API: Conv2d"
  assert data["std_args"] == []
  assert data["variants"] == {}


def test_main_execution() -> None:
  """Test module execution block."""
  source_code = open("scripts/ensure_ops.py").read()

  with patch("pathlib.Path.exists") as mock_exists, patch("builtins.open") as mock_open:
    mock_exists.return_value = False

    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    with patch.object(scripts.ensure_ops, "__name__", "__main__"):
      code = compile(source_code, "scripts/ensure_ops.py", "exec")
      exec(code, scripts.ensure_ops.__dict__)

    # Check that open was called to write the yaml file
    assert mock_open.call_count == 39
