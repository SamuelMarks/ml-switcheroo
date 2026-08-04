"""Test module."""

import pytest
from ml_switcheroo.core.mlir.parser import MlirParser


def test_mlir_parser_invalid_token():
  """Test for test_mlir_parser_invalid_token."""
  with pytest.raises(ValueError, match="Unexpected"):
    MlirParser("~").parse()


def test_mlir_parser_sym_id():
  """Test for test_mlir_parser_sym_id."""
  code = "func.func @main() { return }"
  parser = MlirParser(code)
  module = parser.parse()
  assert module.body.operations[0].name == "func.func"
  assert module.body.operations[0].name_trivia[-1].text == "@main"


def test_mlir_parser_array_attr():
  """Test for test_mlir_parser_array_attr."""
  code = "sw.op {arr = [1, 2]}"
  parser = MlirParser(code)
  module = parser.parse()
  assert module.body.operations[0].attributes[0].value == ["1", "2"]


def test_mlir_parser_empty():
  """Test for test_mlir_parser_empty."""
  parser = MlirParser("   ")
  module = parser.parse()
  assert len(module.body.operations) == 0


def test_mlir_parser_op_tail_region():
  """Test for test_mlir_parser_op_tail_region."""
  code = "sw.op { ^bb0: }"
  parser = MlirParser(code)
  module = parser.parse()
  assert len(module.body.operations[0].regions) == 1


def test_mlir_parser_branch_coverage():
  """Test for test_mlir_parser_branch_coverage."""
  from ml_switcheroo.core.mlir.parser import MlirTransformer

  transformer = MlirTransformer()
  # operation with all None children (193->198)
  op = transformer.operation([None, None])
  assert op.name == ""

  # operation with an empty list child (229->239)
  op = transformer.operation([[]])
  assert op.name == ""
