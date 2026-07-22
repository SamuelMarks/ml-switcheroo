"""Test suite for the Onnx Reader Attrs module."""

import pytest
from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter

ONNX_MOCK_CONTENT = '\n## Operator Definitions\n\n### <a name="Conv"></a> **Conv**\n\nThe standard convolution layer.\n\n#### Inputs\n\n<dl>\n<dt><tt>X</tt> : T</dt>\n<dd>Input data tensor...</dd>\n<dt><tt>W</tt> : T</dt>\n<dd>Weight tensor...</dd>\n</dl>\n\n#### Attributes\n\n<dl>\n<dt><tt>auto_pad</tt> : string</dt>\n<dd>Padding strategy...</dd>\n<dt>dilations : list of ints</dt>\n<dd>Dilation value...</dd>\n<dt><tt>strides</tt> : ints</dt>\n<dd>Stride value along each spatial axis.</dd>\n<dt><tt>group</tt> : int</dt>\n<dd>Number of groups.</dd>\n<dt>legacy_attr</dt>\n<dd>Some attribute with no type defined.</dd>\n</dl>\n\n---\n\n### <a name="Relu"></a> **Relu**\n\nRectified Linear Unit.\n\n#### Inputs\n\n<dl>\n<dt>X : T</dt>\n</dl>\n'


@pytest.fixture
def importer():
  """Provides a mock importer for testing."""
  return OnnxSpecImporter()


@pytest.fixture
def mock_spec_file(tmp_path):
  """Provides a mock spec file for testing."""
  fpath = tmp_path / "Operators.md"
  fpath.write_text(ONNX_MOCK_CONTENT, encoding="utf-8")
  return fpath


def test_extract_attributes_integration(importer, mock_spec_file):
  """Extracts attributes integration."""
  semantics = importer.parse_file(mock_spec_file)
  assert "Conv" in semantics
  conv_def = semantics["Conv"]
  args = conv_def["std_args"]
  assert isinstance(args, list)
  assert len(args) > 0
  assert isinstance(args[0], tuple)
  arg_map = dict(args)
  assert arg_map["X"] == "Tensor"
  assert arg_map["W"] == "Tensor"
  assert arg_map["auto_pad"] == "str"
  assert arg_map["dilations"] == "List[int]"
  assert arg_map["strides"] == "List[int]"
  assert arg_map["group"] == "int"
  assert arg_map["legacy_attr"] == "Any"


def test_extract_section_logic_tuples(importer):
  """Extracts section logic tuples."""
  text = "\n#### Attributes\n<dl>\n<dt><tt>kernel_shape</tt> : ints</dt>\n<dt>simple_attr</dt>\n</dl>\n#### Inputs\n"
  args = importer._extract_section_keys(text, "Attributes")
  assert len(args) == 2
  assert args[0] == ("kernel_shape", "List[int]")
  assert args[1] == ("simple_attr", "Any")


def test_no_attributes_section(importer, mock_spec_file):
  """Verifies the behavior of no attributes section."""
  semantics = importer.parse_file(mock_spec_file)
  assert "Relu" in semantics
  args = semantics["Relu"]["std_args"]
  assert args == [("X", "Tensor")]


def test_html_tag_cleaning_and_mapping(importer):
  """Verifies the behavior of HTML tag cleaning and mapping."""
  text = "\n#### Inputs\n<dt>clean : int</dt>\n<dt><tt>in_tags</tt> : float</dt>\n<dt><tt>bold_name</tt> : <b>bool</b></dt>\n<dt>spaced_out :   string   </dt>\n<dt>no_type</dt>\n<dt>list_type : list of floats</dt>\n<dt>t_alias : T</dt>\n"
  args = importer._extract_section_keys(text, "Inputs")
  arg_map = dict(args)
  assert arg_map["clean"] == "int"
  assert arg_map["in_tags"] == "float"
  assert arg_map["bold_name"] == "bool"
  assert arg_map["spaced_out"] == "str"
  assert arg_map["no_type"] == "Any"
  assert arg_map["list_type"] == "List[float]"
  assert arg_map["t_alias"] == "Tensor"


def test_missing_file_returns_empty(importer, tmp_path):
  """Verifies the behavior of missing file returns empty."""
  res = importer.parse_file(tmp_path / "ghost.md")
  assert res == {}
