"""Test suite for the Loader module."""

import json
from pathlib import Path
from unittest.mock import patch
from ml_switcheroo.frameworks.loader import (
  load_definitions,
  clear_definition_cache,
  get_definitions_path,
  DEFINITIONS_DIR,
)
from ml_switcheroo_ir.schema.ghost import StandardMap


def test_load_definitions_file_not_found() -> None:
  """Loads definitions file not found."""
  clear_definition_cache()
  assert load_definitions("non_existent_framework") == {}


def test_load_definitions_json_error(tmp_path: Path) -> None:
  """Loads definitions JSON correctly handling an error."""
  clear_definition_cache()
  bad_json: Path = tmp_path / "bad.json"
  bad_json.write_text("invalid json")
  with patch("ml_switcheroo.frameworks.loader.DEFINITIONS_DIR", tmp_path):
    assert load_definitions("bad") == {}


def test_load_definitions_success(tmp_path: Path) -> None:
  """Loads definitions successfully."""
  clear_definition_cache()
  good_json: Path = tmp_path / "good.json"
  good_json.write_text(json.dumps({"Add": {"api": "add"}}))
  with patch("ml_switcheroo.frameworks.loader.DEFINITIONS_DIR", tmp_path):
    defs: dict[str, StandardMap] = load_definitions("good")
    assert "Add" in defs
    assert isinstance(defs["Add"], StandardMap)
    assert defs["Add"].api == "add"


def test_get_definitions_path() -> None:
  """Gets definitions path."""
  path: Path = get_definitions_path("test_fw")
  assert path == DEFINITIONS_DIR / "test_fw.json"
