"""Test extra array api reader."""

import ast
from unittest.mock import patch
from ml_switcheroo.importers.array_api_reader import ArrayApiSpecImporter


def test_array_api_reader_relative_to_value_error(tmp_path):
  """Test element."""
  reader = ArrayApiSpecImporter()

  file_path = tmp_path / "test.py"
  file_path.write_text("def foo(): pass")

  with patch("pathlib.Path.relative_to", side_effect=ValueError):
    res = reader._parse_stubs([file_path], tmp_path)

  assert "foo" in res
  assert res["foo"]["from"] == "test.py"


def test_array_api_reader_subscript_no_slice():
  """Test element."""
  reader = ArrayApiSpecImporter()
  # Mock Subscript without slice
  node = ast.Subscript(value=ast.Name(id="List"))
  if hasattr(node, "slice"):
    del node.slice
  assert reader._parse_annotation(node) == "List"
