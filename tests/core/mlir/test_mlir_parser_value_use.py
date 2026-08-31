"""Tests for parsing MLIR value uses with suffixes."""

from ml_switcheroo.core.mlir.parser import MlirParser


def test_parse_value_use_suffix() -> None:
  """Test parsing a value use with a suffix index (e.g. %0#1)."""
  code = '"dialect.op"(%0#1, %1#2) : (i32, i32) -> ()'
  parser = MlirParser(code)
  module = parser.parse()
  op = module.body.operations[0]

  assert op.operands[0].name == "%0"
  assert op.operands[0].use_index == 1
  assert op.operands[1].name == "%1"
  assert op.operands[1].use_index == 2

  # Should also roundtrip correctly
  # Generic format output removes the trailing trivia logic for simple tests
