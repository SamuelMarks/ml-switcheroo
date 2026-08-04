"""Test suite for the Onnx Reader module."""

import pytest
from pathlib import Path
from ml_switcheroo.importers.future_onnx_reader import OnnxSpecImporter


@pytest.fixture
def importer() -> OnnxSpecImporter:
  """Provides a mock importer for testing."""
  return OnnxSpecImporter()


def test_parse_file_not_found(importer: OnnxSpecImporter, tmp_path: Path) -> None:
  """Parses file not found."""
  missing_file = tmp_path / "missing.md"
  assert importer.parse_file(missing_file) == {}


def test_parse_file_found(importer: OnnxSpecImporter, tmp_path: Path) -> None:
  """Parses file found."""
  md_file = tmp_path / "Operators.md"
  md_file.write_text('### <a name="Add"></a>\n**Add**\nDesc')
  result = importer.parse_file(md_file)
  assert "Add" in result


def test_parse_markdown_duplicate_op(importer: OnnxSpecImporter, tmp_path: Path) -> None:
  """Parses markdown duplicate op."""
  md_file = tmp_path / "ops.md"
  md_file.write_text('### <a name="Add"></a>\n**Add**\nThis is v1\n### <a name="Add"></a>\n**Add**\nThis is v2\n')
  result = importer._parse_markdown(md_file)
  assert len(result) == 1
  assert "This is v1" in result["Add"]["description"]


def test_parse_markdown_args(importer: OnnxSpecImporter, tmp_path: Path) -> None:
  """Parses markdown arguments and types."""
  md_file = tmp_path / "ops.md"
  md_file.write_text(
    '### <a name="Add"></a>\n**Add**\nSummary\n#### Inputs\n<dl><dt><tt>a</tt> : T</dt><dd>description</dd><dt>b</dt><dd>no type</dd><dt>c : int</dt><dd>no tags but has type</dd></dl>'
  )
  result = importer._parse_markdown(md_file)
  assert result["Add"]["std_args"] == [("a", "Tensor"), ("b", "Any"), ("c", "int")]

  md_file.write_text('### <a name="Conv"></a>\n#### Attributes\n<dl><dt><b>dilations</b> : list of ints</dt></dl>')
  result = importer._parse_markdown(md_file)
  assert result["Conv"]["std_args"] == [("dilations", "List[int]")]


def test_map_onnx_type(importer: OnnxSpecImporter) -> None:
  """Maps onnx type."""
  assert importer._map_onnx_type("list of ints") == "List[int]"
  assert importer._map_onnx_type("list of floats") == "List[float]"
  assert importer._map_onnx_type("list of strings") == "List[str]"
  assert importer._map_onnx_type("ints") == "List[int]"
  assert importer._map_onnx_type("floats") == "List[float]"
  assert importer._map_onnx_type("string") == "str"
  assert importer._map_onnx_type("bool") == "bool"
  assert importer._map_onnx_type("float") == "float"
  assert importer._map_onnx_type("int") == "int"
  assert importer._map_onnx_type("T") == "Tensor"
  assert importer._map_onnx_type("tensor(float)") == "float"
  assert importer._map_onnx_type("tensor") == "Tensor"
  assert importer._map_onnx_type("Unknown") == "Any"
