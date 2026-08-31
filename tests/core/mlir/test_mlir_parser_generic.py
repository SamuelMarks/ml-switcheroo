"""Tests for parsing generic MLIR operations."""

from ml_switcheroo.core.mlir.parser import MlirParser


def test_parse_generic_operation() -> None:
  """Test parsing a generic operation with properties, attributes, and successors."""
  code = """
"dialect.op"(%0, %1) [^bb1] <{prop = 1}> {attr = 2} : (i32, i32) -> i32
"""
  parser = MlirParser(code)
  module = parser.parse()
  op = module.body.operations[0]

  assert op.is_generic is True
  assert op.name == "dialect.op"
  assert len(op.operands) == 2
  assert op.operands[0].name == "%0"
  assert op.operands[0].type_node.body == "i32"
  assert op.operands[1].name == "%1"
  assert op.operands[1].type_node.body == "i32"

  assert len(op.successors) == 1
  assert op.successors[0] == "^bb1"

  assert len(op.properties) == 1
  assert op.properties[0].name == "prop"
  assert op.properties[0].value == "1"

  assert len(op.attributes) == 1
  assert op.attributes[0].name == "attr"
  assert op.attributes[0].value == "2"

  assert len(op.result_types) == 1
  assert op.result_types[0].body == "i32"
