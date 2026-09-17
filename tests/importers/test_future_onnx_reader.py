"""Test module."""

from pathlib import Path
from unittest.mock import patch

from ml_switcheroo.importers.future_onnx_reader import OnnxSpecImporter


def test_onnx_reader_missing_file(tmp_path: Path) -> None:
  """Docstring."""
  importer = OnnxSpecImporter()
  res = importer.parse_file(tmp_path / "missing.md")
  assert res == {}


def test_onnx_reader_parse_markdown(tmp_path: Path) -> None:
  """Docstring."""
  importer = OnnxSpecImporter()

  md_content = """
## some header

### <a name="Abs"></a><a name="abs">**Abs**</a>

#### Summary

*Abs* computes the absolute value.

#### Inputs

<dl>
<dt><tt>X</tt> : T</dt>
<dd>Input tensor</dd>
<dt>Y: list of ints</dt>
<dd>Other</dd>
<dt>Z</dt>
<dd>No type</dd>
<dt>W: </dt>
<dd>Empty type</dd>
</dl>

#### Attributes

<dl>
<dt><b>alpha</b> : float</dt>
<dd>An attribute</dd>
<dt>beta</dt>
<dd>Another attribute</dd>
</dl>

### <a name="Add"></a><a name="add">**Add**</a>

#### Summary

Adds two tensors.
    """
  md_file = tmp_path / "Operators.md"
  md_file.write_text(md_content)

  res = importer.parse_file(md_file)

  assert "Abs" in res
  assert res["Abs"]["description"] == "*Abs* computes the absolute value."
  assert res["Abs"]["std_args"] == [
    ("X", "Tensor"),
    ("Y", "List[int]"),
    ("Z", "Any"),
    ("W", "Any"),
    ("alpha", "float"),
    ("beta", "Any"),
  ]

  assert "Add" in res
  assert res["Add"]["description"] == "Adds two tensors."
  assert res["Add"]["std_args"] == []


def test_map_onnx_type() -> None:
  """Docstring."""
  importer = OnnxSpecImporter()

  # Lists
  assert importer._map_onnx_type("list of ints") == "List[int]"
  assert importer._map_onnx_type("list of floats") == "List[float]"
  assert importer._map_onnx_type("list of strings") == "List[str]"
  assert importer._map_onnx_type("ints") == "List[int]"
  assert importer._map_onnx_type("floats") == "List[float]"

  # Primitives
  assert importer._map_onnx_type("string") == "str"
  assert importer._map_onnx_type("str") == "str"
  assert importer._map_onnx_type("bool") == "bool"
  assert importer._map_onnx_type("float") == "float"
  assert importer._map_onnx_type("int") == "int"

  # Tensors
  assert importer._map_onnx_type("tensor") == "Tensor"
  assert importer._map_onnx_type("T") == "Tensor"

  # Fallback
  assert importer._map_onnx_type("unknown") == "Any"


def test_long_summary(tmp_path: Path) -> None:
  """Docstring."""
  importer = OnnxSpecImporter()
  long_desc = "A" * 310

  md_content = f"""
### <a name="Long"></a><a name="long">**Long**</a>
#### Summary
{long_desc}
    """
  md_file = tmp_path / "Operators2.md"
  md_file.write_text(md_content)

  res = importer.parse_file(md_file)
  assert "Long" in res
  assert len(res["Long"]["description"]) == 303  # 300 + "..."


def test_onnx_reader_extra_branches(tmp_path: Path) -> None:
  """Docstring."""
  importer = OnnxSpecImporter()
  md_content = """
### <a name="Dup"></a>
#### Summary
First paragraph.

Second paragraph.

### <a name="Dup"></a>
#### Summary
Another summary.
    """
  md_file = tmp_path / "Extra.md"
  md_file.write_text(md_content)
  res = importer.parse_file(md_file)
  assert "Dup" in res


def test_onnx_reader_all_missing_branches(tmp_path: Path) -> None:
  """Test remaining branches in future onnx reader."""
  importer = OnnxSpecImporter()
  md_content = """
### HeaderWithoutAnchor

### <a name="OpTest"></a>**OpTest**

#### Summary

OpTest

Detailed summary for OpTest.

#### OtherSection
Some other section text.

#### Inputs
Some normal paragraph before dl.
<dl>
<dt><tt>NoColon</tt></dt>
<dd>No colon description</dd>
<dt></dt>
<dd>Empty dt</dd>
</dl>
"""
  md_file = tmp_path / "TestBranches.md"
  md_file.write_text(md_content)
  res = importer.parse_file(md_file)
  assert "OpTest" in res
  assert res["OpTest"]["description"] == "Detailed summary for OpTest."
  assert res["OpTest"]["std_args"] == [("NoColon", "Any")]

  # Test synthetic tokens ending on h3 and h4, and op without _raw_summary
  import inspect
  from unittest.mock import MagicMock
  from markdown_it.token import Token

  dummy_file = tmp_path / "dummy.md"
  dummy_file.write_text("dummy")

  t_h3 = Token("heading_open", "h3", 1)
  t_h4 = Token("heading_open", "h4", 1)

  with MagicMock() as mock_md:
    mock_md.parse.return_value = [t_h3, t_h4]
    with patch("markdown_it.MarkdownIt", return_value=mock_md):
      parsed = importer._parse_markdown(dummy_file)
      assert parsed == {}

  # Trigger op without _raw_summary via inspect in _map_onnx_type
  def pop_raw_summary(raw: str) -> str:
    """Mock mapping function that clears _raw_summary from frame semantics.

    Args:
        raw: The raw type string to map.

    Returns:
        Mapped type string fallback.
    """
    frame = inspect.currentframe()
    while frame:
      if "semantics" in frame.f_locals:
        for op in frame.f_locals["semantics"].values():
          op.pop("_raw_summary", None)
        break
      frame = frame.f_back
    return "Any"

  with patch.object(importer, "_map_onnx_type", side_effect=pop_raw_summary):
    res2 = importer.parse_file(md_file)
    assert "OpTest" in res2
    assert res2["OpTest"]["description"] == ""
