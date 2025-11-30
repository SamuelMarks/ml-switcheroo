"""
Tests for Extended ONNX Spec Reader (Attributes & Type Support).

Verifies that:
1.  The reader parses `#### Attributes` in addition to Inputs.
2.  HTML cleanup logic handles `<tt>`, `<dt>`, and type separators correctly.
3.  Arguments from both sections are merged into `std_args`.
4.  Type strings are extracted and mapped to Fuzzer-friendly hints.
"""

import pytest
from ml_switcheroo.importers.onnx_reader import OnnxSpecImporter

# --- Mock Data ---

ONNX_MOCK_CONTENT = """
## Operator Definitions

### <a name="Conv"></a> **Conv**

The standard convolution layer.

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input data tensor...</dd>
<dt><tt>W</tt> : T</dt>
<dd>Weight tensor...</dd>
</dl>

#### Attributes

<dl>
<dt><tt>auto_pad</tt> : string</dt>
<dd>Padding strategy...</dd>
<dt>dilations : list of ints</dt>
<dd>Dilation value...</dd>
<dt><tt>strides</tt> : ints</dt>
<dd>Stride value along each spatial axis.</dd>
<dt><tt>group</tt> : int</dt>
<dd>Number of groups.</dd>
<dt>legacy_attr</dt>
<dd>Some attribute with no type defined.</dd>
</dl>

---

### <a name="Relu"></a> **Relu**

Rectified Linear Unit.

#### Inputs

<dl>
<dt>X : T</dt>
</dl>
"""


@pytest.fixture
def importer():
  """Returns an instance of the OnnxSpecImporter."""
  return OnnxSpecImporter()


@pytest.fixture
def mock_spec_file(tmp_path):
  """Creates a temporary .md file with mock ONNX content."""
  fpath = tmp_path / "Operators.md"
  fpath.write_text(ONNX_MOCK_CONTENT, encoding="utf-8")
  return fpath


def test_extract_attributes_integration(importer, mock_spec_file):
  """
  Scenario: Parse a file containing Inputs and Attributes.
  Expectation: `std_args` contains both tensor inputs and config attributes with types.
  """
  semantics = importer.parse_file(mock_spec_file)

  assert "Conv" in semantics
  conv_def = semantics["Conv"]
  args = conv_def["std_args"]

  # Structure check: args should be list of tuples (name, type)
  assert isinstance(args, list)
  assert len(args) > 0
  assert isinstance(args[0], tuple)

  # Convert to dict for easier assertions
  arg_map = dict(args)

  # Check Inputs types
  assert arg_map["X"] == "Tensor"
  assert arg_map["W"] == "Tensor"

  # Check Attributes types
  assert arg_map["auto_pad"] == "str"
  assert arg_map["dilations"] == "List[int]"
  assert arg_map["strides"] == "List[int]"  # "ints" mapping
  assert arg_map["group"] == "int"
  assert arg_map["legacy_attr"] == "Any"  # Fallback


def test_extract_section_logic_tuples(importer):
  """
  Unit test to verify the extraction helper returns tuples.
  """
  text = """
#### Attributes
<dl>
<dt><tt>kernel_shape</tt> : ints</dt>
<dt>simple_attr</dt>
</dl>
#### Inputs
"""
  # Should stop before Inputs
  args = importer._extract_section_keys(text, "Attributes")

  assert len(args) == 2
  assert args[0] == ("kernel_shape", "List[int]")
  assert args[1] == ("simple_attr", "Any")


def test_no_attributes_section(importer, mock_spec_file):
  """
  Scenario: Operator has Inputs but no Attributes (e.g., Relu).
  Expectation: `std_args` contains inputs only, no crashes.
  """
  semantics = importer.parse_file(mock_spec_file)

  assert "Relu" in semantics
  args = semantics["Relu"]["std_args"]

  # Verify input parsing
  assert args == [("X", "Tensor")]


def test_html_tag_cleaning_and_mapping(importer):
  """
  Verify different HTML variances seen in type definitions.
  """
  text = """
#### Inputs
<dt>clean : int</dt>
<dt><tt>in_tags</tt> : float</dt>
<dt><tt>bold_name</tt> : <b>bool</b></dt>
<dt>spaced_out :   string   </dt>
<dt>no_type</dt>
<dt>list_type : list of floats</dt>
<dt>t_alias : T</dt>
"""
  args = importer._extract_section_keys(text, "Inputs")
  arg_map = dict(args)

  assert arg_map["clean"] == "int"
  assert arg_map["in_tags"] == "float"
  # Mapping for bool matches 'bool' inside <b>bool</b>
  assert arg_map["bold_name"] == "bool"
  assert arg_map["spaced_out"] == "str"
  assert arg_map["no_type"] == "Any"
  assert arg_map["list_type"] == "List[float]"
  assert arg_map["t_alias"] == "Tensor"


def test_missing_file_returns_empty(importer, tmp_path):
  """
  Scenario: File does not exist.
  Expectation: Empty dict, log error.
  """
  res = importer.parse_file(tmp_path / "ghost.md")
  assert res == {}
