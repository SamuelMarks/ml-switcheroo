"""Test suite for the Sass Reader module."""

import pytest
from ml_switcheroo.importers.sass_reader import SassHtmlParser, SassSpecImporter


@pytest.fixture
def parser():
  """Provides a mock parser for testing."""
  return SassHtmlParser()


@pytest.fixture
def importer():
  """Provides a mock importer for testing."""
  return SassSpecImporter()


def test_sass_html_parser(parser):
  """Verifies the behavior of SASS HTML parser."""
  html = "\n    <table>\n        <tbody>\n            <tr><td>Opcode Header</td><td>Description Header</td></tr>\n            <tr><td>FADD</td><td>FP32 Add</td></tr>\n            <tr><td>IADD3</td><td>Integer Addition</td></tr>\n            <tr><td>invalid</td><td>some desc</td></tr>\n            <tr><td>ONLYONE</td></tr>\n            <tr><td>UPPER M N</td><td>Desc with space</td></tr>\n        </tbody>\n    </table>\n    "
  parser.feed(html)
  ops = parser.extracted_ops
  assert ("FADD", "FP32 Add") in ops
  assert ("IADD3", "Integer Addition") in ops
  assert not any((op == "Opcode Header" for (op, _) in ops))
  assert not any((op == "invalid" for (op, _) in ops))
  assert not any((op == "ONLYONE" for (op, _) in ops))
  assert not any((op == "UPPER M N" for (op, _) in ops))


def test_parse_file_not_found(importer, tmp_path):
  """Parses file not found."""
  assert importer.parse_file(tmp_path / "missing.html") == {}


def test_parse_file_utf8(importer, tmp_path):
  """Parses file utf8."""
  html_file = tmp_path / "doc.html"
  html_file.write_text("<table><tbody><tr><td>FADD</td><td>FP32 Add</td></tr></tbody></table>", encoding="utf-8")
  result = importer.parse_file(html_file)
  assert "Add" in result
  assert result["Add"]["api"] == "FADD"


def test_parse_file_fallback_encoding(importer, tmp_path):
  """Parses file fallback encoding."""
  html_file = tmp_path / "doc.html"
  html_file.write_bytes(b"<table><tbody><tr><td>FMUL</td><td>FP32 Multiply \xff</td></tr></tbody></table>")
  result = importer.parse_file(html_file)
  assert "Mul" in result
  assert result["Mul"]["api"] == "FMUL"


def test_conflict_resolution(importer, tmp_path):
  """Verifies the behavior of conflict resolution."""
  html_file = tmp_path / "doc.html"
  html = "\n    <table>\n        <tbody>\n            <tr><td>DADD</td><td>Integer Addition</td></tr>\n            <tr><td>FADD</td><td>FP32 Add</td></tr>\n            <tr><td>HADD2</td><td>Some other text Addition</td></tr>\n        </tbody>\n    </table>\n    "
  html_file.write_text(html)
  result = importer.parse_file(html_file)
  assert "Add" in result
  assert result["Add"]["api"] == "FADD"


def test_infer_abstract_op(importer):
  """Infers abstract op."""
  assert importer._infer_abstract_op("FADD", "Unknown") == "Add"
  assert importer._infer_abstract_op("FMUL", "Unknown") == "Mul"
  assert importer._infer_abstract_op("IADD3", "Unknown") == "Add3"
  assert importer._infer_abstract_op("IABS", "Unknown") == "Abs"
  assert importer._infer_abstract_op("SOME_OP", "FP32 Subtract") == "Sub"
  assert importer._infer_abstract_op("SOME_OP", "Integer Multiply") == "Mul"
  assert importer._infer_abstract_op("SOME_OP", "FP32 Minimum") == "Min"
  assert importer._infer_abstract_op("SOME_OP", "FP32 Maximum") == "Max"
  assert importer._infer_abstract_op("SOME_OP", "Convert Integer to FP32") == "CastFloat"
  assert importer._infer_abstract_op("SOME_OP", "Convert FP32 to Integer") == "CastInt"
  assert importer._infer_abstract_op("SOME_OP", "Absolute Value") == "Abs"
  assert importer._infer_abstract_op("SOME_OP", "Logic Operation") == "BitwiseOp"
  assert importer._infer_abstract_op("SOME_OP", "Fused Multiply and Add") == "FusedMultiplyAdd"
  assert importer._infer_abstract_op("MNMX", "FP32 Minimum/Maximum") == "MinMax"
  assert importer._infer_abstract_op("UNKNOWN", "Something else") is None
