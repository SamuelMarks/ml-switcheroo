import json
from pathlib import Path
from unittest.mock import patch
from ml_switcheroo.frameworks.loader import (
  load_definitions,
  clear_definition_cache,
  get_definitions_path,
  DEFINITIONS_DIR,
)
from ml_switcheroo.frameworks.base import StandardMap


def test_load_definitions_file_not_found():
  clear_definition_cache()
  assert load_definitions("non_existent_framework") == {}


def test_load_definitions_json_error(tmp_path):
  clear_definition_cache()
  bad_json = tmp_path / "bad.json"
  bad_json.write_text("invalid json")

  with patch("ml_switcheroo.frameworks.loader.DEFINITIONS_DIR", tmp_path):
    assert load_definitions("bad") == {}


def test_load_definitions_success(tmp_path):
  clear_definition_cache()
  good_json = tmp_path / "good.json"
  good_json.write_text(json.dumps({"Add": {"api": "add"}}))

  with patch("ml_switcheroo.frameworks.loader.DEFINITIONS_DIR", tmp_path):
    defs = load_definitions("good")
    assert "Add" in defs
    assert isinstance(defs["Add"], StandardMap)
    assert defs["Add"].api == "add"


def test_get_definitions_path():
  path = get_definitions_path("test_fw")
  assert path == DEFINITIONS_DIR / "test_fw.json"
