import pytest
from pathlib import Path
from unittest.mock import MagicMock

from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter


@pytest.fixture
def importer():
  return OnnxSpecImporter()


def test_parse_file_not_found(importer, tmp_path):
  missing_file = tmp_path / "missing.md"
  assert importer.parse_file(missing_file) == {}


def test_parse_file_found(importer, tmp_path):
  md_file = tmp_path / "Operators.md"
  md_file.write_text('### <a name="Add"></a>\n**Add**\nDesc')
  result = importer.parse_file(md_file)
  assert "Add" in result


def test_parse_markdown_duplicate_op(importer, tmp_path):
  md_file = tmp_path / "ops.md"
  md_file.write_text('### <a name="Add"></a>\n**Add**\nThis is v1\n### <a name="Add"></a>\n**Add**\nThis is v2\n')
  result = importer._parse_markdown(md_file)
  assert len(result) == 1
  assert "This is v1" in result["Add"]["description"]


def test_extract_summary(importer):
  text = """
<a name="Add"></a>
**Add**
This is a summary
spanning multiple lines.

#### Inputs
Some input details.
"""
  summary = importer._extract_summary(text)
  assert summary == "This is a summary spanning multiple lines."

  long_text = "A" * 400
  long_summary = importer._extract_summary(long_text)
  assert len(long_summary) == 303
  assert long_summary.endswith("...")


def test_extract_section_keys(importer):
  text = """
#### Inputs
<dl>
<dt><tt>x</tt> : T</dt>
<dt><b>y</b></dt>
<dt>z:list of ints</dt>
<dt><tt>alpha</tt> : float</dt>
<dt>beta : Any</dt>
</dl>
#### Constraints
"""
  args = importer._extract_section_keys(text, "Inputs")
  assert args == [("x", "Tensor"), ("y", "Any"), ("z", "List[int]"), ("alpha", "float"), ("beta", "Any")]

  # No header found
  assert importer._extract_section_keys(text, "Attributes") == []


def test_map_onnx_type(importer):
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
