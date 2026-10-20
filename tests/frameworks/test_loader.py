"""Test suite for the Loader module."""

import json
from pathlib import Path
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
  """Loads definitions JSON correctly handling an error.

  Args:
      tmp_path: Test argument or fixture.
  """
  clear_definition_cache()
  bad_json: Path = tmp_path / "bad.json"
  bad_json.write_text("invalid json")

  class MockFiles:
    """Mock file resource for testing."""

    def joinpath(self, path: str) -> Path:
      """Return bad json path.

      Returns:
          Computed test result or fixture value.
      """
      return bad_json

  with patch("importlib.resources.files", return_value=MockFiles()):
    assert load_definitions("bad") == {}


def test_load_definitions_success(tmp_path: Path) -> None:
  """Loads definitions successfully.

  Args:
      tmp_path: Test argument or fixture.
  """
  clear_definition_cache()
  good_json: Path = tmp_path / "good.json"
  good_json.write_text(json.dumps({"Add": {"api": "add"}}))

  class MockFiles:
    """Mock file resource for testing."""

    def joinpath(self, path: str) -> Path:
      """Return good json path.

      Returns:
          Computed test result or fixture value.
      """
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


def test_resolve_resource_file() -> None:
  """Tests _resolve_resource_file resolves from ml_ecosystem_snapshots.snapshots."""
  mock_fp = MagicMock()
  mock_res = MagicMock()
  mock_res.joinpath.return_value = mock_fp

  with patch("importlib.resources.files", return_value=mock_res):
    res = _resolve_resource_file("test_fw")
    assert res == mock_fp


def test_get_definitions_path() -> None:
  """Gets definitions path."""
  path: Path = get_definitions_path("test_fw")
  assert path.name == "test_fw.json"


def test_get_definitions_path_exception() -> None:
  """Test get_definitions_path fallback on exception."""
  with patch("importlib.resources.files", side_effect=Exception("Failed")):
    path: Path = get_definitions_path("test_fw")
    assert path == Path("test_fw.json")
