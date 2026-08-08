"""Tests for scripts/validate_odl_json.py."""

import json
import sys
from pathlib import Path
from unittest import mock
import pytest

# Add scripts directory to sys.path to import it
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir.resolve()))

import validate_odl_json  # noqa: E402


def test_validate_file_success(tmp_path: Path):
  """Tests validate_file returns True for a valid file."""
  valid_file = tmp_path / "valid.json"
  valid_data = {"version": 1, "ops": [{"name": "test_op", "args": [{"name": "x", "type": "tensor"}]}]}
  valid_file.write_text(json.dumps(valid_data), encoding="utf-8")

  # Assuming SemanticsFile schema accepts this. We may need to mock it.
  with mock.patch("validate_odl_json.SemanticsFile.model_validate") as mock_validate:
    assert validate_odl_json.validate_file(valid_file) is True
    mock_validate.assert_called_once_with(valid_data)


def test_validate_file_failure(tmp_path: Path):
  """Tests validate_file returns False when validation fails."""
  invalid_file = tmp_path / "invalid.json"
  invalid_file.write_text("invalid json", encoding="utf-8")

  assert validate_odl_json.validate_file(invalid_file) is False


def test_main_success(tmp_path: Path):
  """Tests main execution with successful validation."""
  valid_file = tmp_path / "semantics" / "valid.json"
  valid_file.parent.mkdir(parents=True)
  valid_file.write_text("{}", encoding="utf-8")

  test_args = ["validate_odl_json.py", str(valid_file)]
  with mock.patch.object(sys, "argv", test_args):
    with mock.patch("validate_odl_json.validate_file", return_value=True) as mock_val:
      validate_odl_json.main()
      mock_val.assert_called_once_with(Path(str(valid_file)))


def test_main_failure(tmp_path: Path):
  """Tests main execution with failed validation."""
  invalid_file = tmp_path / "semantics" / "invalid.json"
  invalid_file.parent.mkdir(parents=True)
  invalid_file.write_text("{}", encoding="utf-8")

  test_args = ["validate_odl_json.py", str(invalid_file)]
  with mock.patch.object(sys, "argv", test_args):
    with mock.patch("validate_odl_json.validate_file", return_value=False):
      with pytest.raises(SystemExit) as exc_info:
        validate_odl_json.main()
      assert exc_info.value.code == 1


def test_main_ignore_non_semantics(tmp_path: Path):
  """Tests main ignores files not in semantics directory or not JSON."""
  other_file = tmp_path / "other" / "file.json"
  other_file.parent.mkdir(parents=True)
  txt_file = tmp_path / "semantics" / "file.txt"
  txt_file.parent.mkdir(parents=True)

  test_args = ["validate_odl_json.py", str(other_file), str(txt_file)]
  with mock.patch.object(sys, "argv", test_args):
    with mock.patch("validate_odl_json.validate_file") as mock_val:
      validate_odl_json.main()
      mock_val.assert_not_called()


def test_import_error(monkeypatch):
  """Tests behavior when import fails."""
  import sys
  import importlib

  monkeypatch.setitem(sys.modules, "ml_switcheroo.semantics.schema", None)
  with pytest.raises(SystemExit) as exc_info:
    importlib.reload(validate_odl_json)
  assert exc_info.value.code == 1


def test_main_block(monkeypatch):
  """Tests main block execution."""
  import runpy

  monkeypatch.setattr(sys, "argv", ["validate_odl_json.py"])
  runpy.run_path(str(scripts_dir / "validate_odl_json.py"), run_name="__main__")
