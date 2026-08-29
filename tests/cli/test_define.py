"""Test module."""

import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from ml_switcheroo.cli.handlers.define import handle_define


def test_handle_define_not_found(tmp_path: Path) -> None:
  """Docstring."""
  res: int = handle_define(tmp_path / "missing.yaml")
  assert res == 1


@patch("ml_switcheroo.cli.handlers.define.resolve_semantics_dir")
@patch("ml_switcheroo.cli.handlers.define.shutil.copy2")
def test_handle_define_success(mock_copy: MagicMock, mock_resolve: MagicMock, tmp_path: Path) -> None:
  """Docstring."""
  # Setup mock semantics dir
  sem_dir: Path = tmp_path / "semantics"
  sem_dir.mkdir()
  mock_resolve.return_value = sem_dir

  # Create valid ODL yaml
  in_file: Path = tmp_path / "op.yaml"
  data: dict[str, typing.Union[str, dict[str, str]]] = {
    "operation": "TestOp",
    "tier": "neural_net",
    "description": "Test",
    "variants": {},
  }
  with open(in_file, "w") as f:
    yaml.dump(data, f)

  res: int = handle_define(in_file)

  assert res == 0
  mock_copy.assert_called_once_with(in_file, sem_dir / "odl" / "TestOp.yaml")


def test_handle_define_invalid_schema(tmp_path: Path) -> None:
  """Docstring."""
  in_file: Path = tmp_path / "op.yaml"
  data: dict[str, str] = {"invalid": "data"}  # missing 'operation'
  with open(in_file, "w") as f:
    yaml.dump(data, f)

  res: int = handle_define(in_file)
  assert res == 1
