"""Test suite for the Loader module."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir.schema.ghost import StandardMap

from ml_switcheroo.frameworks.loader import (
  _resolve_resource_file,
  clear_definition_cache,
  get_definitions_path,
  load_definitions,
)


def test_load_definitions_file_not_found() -> None:
  """Loads definitions file not found."""
  clear_definition_cache()
  assert load_definitions("non_existent_framework") == {}


def test_load_definitions_json_error(tmp_path: Path) -> None:
  """Loads definitions JSON correctly handling an error."""
  clear_definition_cache()
  bad_json: Path = tmp_path / "bad.json"
  bad_json.write_text("invalid json")

  class MockFiles:
    """Mock file resource for testing."""

    def joinpath(self, path: Any) -> Path:
      """Return bad json path."""
      return bad_json

  with patch("importlib.resources.files", return_value=MockFiles()):
    assert load_definitions("bad") == {}


def test_load_definitions_success(tmp_path: Path) -> None:
  """Loads definitions successfully."""
  clear_definition_cache()
  good_json: Path = tmp_path / "good.json"
  good_json.write_text(json.dumps({"Add": {"api": "add"}}))

  class MockFiles:
    """Mock file resource for testing."""

    def joinpath(self, path: Any) -> Path:
      """Return good json path."""
      return good_json

  with patch("importlib.resources.files", return_value=MockFiles()):
    defs: dict[str, StandardMap] = load_definitions("good")
    assert "Add" in defs
    assert isinstance(defs["Add"], StandardMap)
    assert defs["Add"].api == "add"


def test_load_definitions_resolve_exception() -> None:
  """Tests load_definitions when _resolve_resource_file raises an exception."""
  clear_definition_cache()
  with patch("ml_switcheroo.frameworks.loader._resolve_resource_file", side_effect=RuntimeError("Boom")):
    assert load_definitions("error_fw") == {}


def test_resolve_resource_file_fallback_packages() -> None:
  """Tests _resolve_resource_file when first package raises exception."""
  mock_fp = MagicMock()
  mock_fp.is_file.return_value = True

  def mock_files(pkg: str) -> MagicMock:
    if pkg == "ml_ecosystem_snapshots.snapshots":
      raise ModuleNotFoundError()
    res = MagicMock()
    res.joinpath.return_value = mock_fp
    return res

  with patch("importlib.resources.files", side_effect=mock_files):
    res = _resolve_resource_file("test_fw")
    assert res == mock_fp


def test_resolve_resource_file_all_fail() -> None:
  """Tests _resolve_resource_file fallback when both in loop fail."""
  call_count = 0

  def mock_files(pkg: str) -> MagicMock:
    nonlocal call_count
    call_count += 1
    if call_count <= 2:
      raise RuntimeError()
    res = MagicMock()
    res.joinpath.return_value = MagicMock()
    return res

  with patch("importlib.resources.files", side_effect=mock_files):
    res = _resolve_resource_file("test_fw")
    assert res is not None


def test_resolve_resource_file_outer_fallback_exception() -> None:
  """Tests _resolve_resource_file final fallback when ecosystem fallback raises."""
  call_count = 0

  def mock_files(pkg: str) -> MagicMock:
    nonlocal call_count
    call_count += 1
    if call_count <= 3:
      raise RuntimeError()
    res = MagicMock()
    res.joinpath.return_value = "final_fp"
    return res

  with patch("importlib.resources.files", side_effect=mock_files):
    res = _resolve_resource_file("test_fw")
    assert res == "final_fp"


def test_get_definitions_path() -> None:
  """Gets definitions path."""
  path: Path = get_definitions_path("test_fw")
  assert path.name == "test_fw.json"


def test_get_definitions_path_exception() -> None:
  """Test get_definitions_path fallback on exception."""
  with patch("importlib.resources.files", side_effect=Exception("Failed")):
    path: Path = get_definitions_path("test_fw")
    assert path == Path("test_fw.json")


def test_get_definitions_path_fallback_second_package() -> None:
  """Test get_definitions_path when second package in loop has existing file."""
  mock_fp = MagicMock()
  mock_fp.is_file.return_value = True

  def mock_files(pkg: str) -> MagicMock:
    if pkg == "ml_ecosystem_snapshots.snapshots":
      res = MagicMock()
      res.joinpath.return_value.is_file.return_value = False
      return res
    res = MagicMock()
    res.joinpath.return_value = mock_fp
    return res

  with patch("importlib.resources.files", side_effect=mock_files):
    path = get_definitions_path("test_fw")
    assert path == Path(str(mock_fp))


def test_get_definitions_path_fallback_after_loop() -> None:
  """Test get_definitions_path fallback after loop when second package raises."""
  call_count = 0

  def mock_files(pkg: str) -> MagicMock:
    nonlocal call_count
    call_count += 1
    if call_count <= 2:
      res = MagicMock()
      res.joinpath.return_value.is_file.return_value = False
      return res
    if call_count == 3:
      raise RuntimeError()
    res = MagicMock()
    res.joinpath.return_value = "final_path"
    return res

  with patch("importlib.resources.files", side_effect=mock_files):
    path = get_definitions_path("test_fw")
    assert path == Path("final_path")
