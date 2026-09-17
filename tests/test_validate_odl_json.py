"""Unit tests for scripts/validate_odl_json.py."""

from pathlib import Path
import runpy
from unittest.mock import patch

import pytest

from scripts.validate_odl_json import main, validate_file


def test_validate_file_success(tmp_path: Path) -> None:
  """Test validate_file on a valid semantics JSON file.

  Args:
      tmp_path: Temporary directory fixture.
  """
  valid_file = tmp_path / "valid.json"
  valid_file.write_text('{"__constants__": []}', encoding="utf-8")
  assert validate_file(valid_file) is True


def test_validate_file_invalid_json(tmp_path: Path) -> None:
  """Test validate_file on malformed JSON content.

  Args:
      tmp_path: Temporary directory fixture.
  """
  invalid_file = tmp_path / "invalid.json"
  invalid_file.write_text("{broken json", encoding="utf-8")
  assert validate_file(invalid_file) is False


def test_validate_file_schema_error(tmp_path: Path) -> None:
  """Test validate_file on JSON violating the schema.

  Args:
      tmp_path: Temporary directory fixture.
  """
  bad_schema_file = tmp_path / "bad_schema.json"
  bad_schema_file.write_text("[1, 2, 3]", encoding="utf-8")
  assert validate_file(bad_schema_file) is False


def test_main_with_explicit_args_success(tmp_path: Path) -> None:
  """Test main CLI with explicit file arguments that succeed.

  Args:
      tmp_path: Temporary directory fixture.
  """
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  f1 = sem_dir / "valid.json"
  f1.write_text('{"__constants__": []}', encoding="utf-8")
  ignored_file = tmp_path / "ignored.txt"
  ignored_file.write_text("ignored", encoding="utf-8")
  non_semantics = tmp_path / "other.json"
  non_semantics.write_text("{}", encoding="utf-8")

  exit_code = main([str(f1), str(ignored_file), str(non_semantics)])
  assert exit_code == 0


def test_main_with_explicit_args_failure(tmp_path: Path) -> None:
  """Test main CLI with explicit file arguments that fail validation.

  Args:
      tmp_path: Temporary directory fixture.
  """
  sem_dir = tmp_path / "semantics"
  sem_dir.mkdir()
  f1 = sem_dir / "bad.json"
  f1.write_text("{invalid", encoding="utf-8")

  exit_code = main([str(f1)])
  assert exit_code == 1


def test_main_default_scan_success() -> None:
  """Test main CLI default scanning without arguments on repository semantics."""
  exit_code = main([])
  assert exit_code == 0


def test_main_default_argv_none(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main CLI with argv=None using monkeypatched sys.argv.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr("sys.argv", ["validate_odl_json.py"])
  exit_code = main(None)
  assert exit_code == 0


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test __main__ module execution block.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  import sys

  src_resolved = str(Path("src").resolve())
  removed = [p for p in list(sys.path) if "src" in p or Path(p).resolve() == Path(src_resolved).resolve()]
  for p in removed:
    sys.path.remove(p)

  monkeypatch.setattr("sys.argv", ["validate_odl_json.py"])
  sys.modules.pop("scripts.validate_odl_json", None)
  try:
    with patch("scripts.validate_odl_json.main", return_value=0):
      with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("scripts.validate_odl_json", run_name="__main__")
      assert excinfo.value.code == 0
  finally:
    for p in removed:
      if p not in sys.path:
        sys.path.insert(0, p)
