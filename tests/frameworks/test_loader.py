"""Test suite for the Loader module."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ml_switcheroo_ir.schema.ghost import StandardMap

from ml_switcheroo.frameworks.loader import (
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


def test_get_definitions_path() -> None:
  """Gets definitions path."""
  path: Path = get_definitions_path("test_fw")
  assert path.name == "test_fw.json"


def test_get_definitions_path_exception() -> None:
  """Test get_definitions_path fallback on exception."""
  with patch("importlib.resources.files", side_effect=Exception("Failed")):
    path: Path = get_definitions_path("test_fw")
    assert path == Path("test_fw.json")
