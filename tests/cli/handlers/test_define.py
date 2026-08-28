"""Docstring."""

from unittest.mock import patch
from pathlib import Path
from ml_switcheroo.cli.handlers.define import handle_define


def test_handle_define_not_found(tmp_path: Path) -> None:
  """Docstring."""
  assert handle_define(tmp_path / "not_found.yaml") == 1


def test_handle_define_invalid_yaml(tmp_path: Path) -> None:
  """Docstring."""
  f: Path = tmp_path / "def.yaml"
  f.write_text("invalid: yaml: :")
  assert handle_define(f) == 1


def test_handle_define_invalid_schema(tmp_path: Path) -> None:
  """Docstring."""
  f: Path = tmp_path / "def.yaml"
  f.write_text("id: valid_yaml_but_missing_fields")
  assert handle_define(f) == 1


def test_handle_define_success(tmp_path: Path) -> None:
  """Docstring."""
  f: Path = tmp_path / "def.yaml"
  f.write_text("id: my_op\ntier: math\ndescription: op")
  with patch("ml_switcheroo.cli.handlers.define.OperationDef"):
    with patch("ml_switcheroo.cli.handlers.define.resolve_semantics_dir", return_value=tmp_path):
      with patch("shutil.copy2") as mock_copy:
        assert handle_define(f) == 0
        mock_copy.assert_called_once()
