"""Tests for parsing MLIR locations."""

from ml_switcheroo.core.mlir.parser import MlirParser


def test_parse_trailing_location() -> None:
  """Test parsing a trailing location after an operation."""
  code = 'dialect.op() loc("foo.mlir")'
  parser = MlirParser(code)
  module = parser.parse()
  op = module.body.operations[0]

  assert op.location == '"foo.mlir"'
