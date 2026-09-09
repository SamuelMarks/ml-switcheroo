"""Test extra corner cases for the MLIR parser."""

from typing import List, Any
from lark import Token, Tree
from ml_switcheroo.core.mlir.parser import MlirTransformer, OperationNode


def test_mlir_transformer_operation_extras() -> None:
  """Test corner cases inside MlirTransformer.operation method."""
  transformer = MlirTransformer()

  children: List[Any] = [
    Tree("operands", [Tree("operand_list", [Tree("operand", [Token("VAL_ID", "%0")])])]),
    Tree("op_tail", []),
    Tree("function_type", [Token("TYPE", "i32")]),
    Tree("successor_list", [Tree("successor", [Token("OTHER_TOKEN", "dummy")])]),
  ]
  op: OperationNode = transformer.operation(children)
  assert len(op.operands) == 1
  assert op.operands[0].name == "%0"

  children2: List[Any] = [
    Tree("operands", []),
    Tree(
      "function_type",
      [
        Tree("unknown_data", []),
        Tree("type_list_parens", [Token("OTHER", "dummy"), Token("TYPE", "f32")]),
        Token("ARROW", "->"),
        Tree("unknown_data", []),
        Tree("type_list_parens", [Token("OTHER", "dummy"), Token("TYPE", "f32")]),
      ],
    ),
  ]
  op2: OperationNode = transformer.operation(children2)
  assert len(op2.result_types) == 1
  assert op2.result_types[0].body == "f32"

  op3: OperationNode = transformer.operation([])
  assert op3.name == ""


def test_mlir_transformer_op_result_list_extra() -> None:
  """Test corner case with unexpected nodes in op_result_list."""
  transformer = MlirTransformer()
  children: List[Any] = [Tree("op_result_list", [Tree("op_result", [Token("VAL_ID", "%1")]), Tree("unknown_node", [])])]
  op: OperationNode = transformer.operation(children)
  assert len(op.results) == 1
  assert op.results[0].name == "%1"
