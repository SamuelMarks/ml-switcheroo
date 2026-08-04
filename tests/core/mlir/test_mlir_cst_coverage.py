"""Tests for the MLIR Concrete Syntax Tree (CST) components, covering additional branch paths.

This module contains unit tests to ensure coverage of corner cases, edge branches,
and custom behavior within the MLIR CST node classes, such as ValueNode, OperationNode,
and StableHloConstantOp. It focuses on testing token generation, trivia handling,
parentheses omission for operations, custom op tail string rendering, and specialized
stablehlo constant operations.
"""

from ml_switcheroo.core.mlir.cst import TypeNode, ValueNode, AttributeNode, OperationNode, StableHloConstantOp
from ml_switcheroo.core.cst.base import Trivia


def test_value_node_with_type():
  """Verify that a ValueNode with an associated TypeNode and colon trivia is correctly formatted.

  This test constructs a ValueNode representing "%0", assigns a TypeNode of type "i32",
  adds colon trivia (a whitespace), and asserts that its string representation
  serializes exactly to "%0 :i32".

  Args:
      None.

  Returns:
      None.
  """
  v = ValueNode(name="%0", type_node=TypeNode(body="i32"), colon_trivia=[Trivia(" ")])
  assert v.to_text() == "%0 :i32"


def test_operation_node_name_trivia_str():
  """Verify that providing a string for name_trivia in OperationNode is parsed into Trivia objects.

  This test instantiates an OperationNode passing a string with spaces for the name_trivia
  argument, ensuring that it is internally converted to a list of Trivia objects containing
  the correct spacing text.

  Args:
      None.

  Returns:
      None.
  """
  # line 146
  op = OperationNode(name="foo", name_trivia=" ")  # type: ignore
  assert isinstance(op.name_trivia, list)
  assert op.name_trivia[0].text == " "


def test_operation_node_name_trivia_none():
  """Verify that providing None for name_trivia in OperationNode defaults to an empty list.

  This test instantiates an OperationNode with name_trivia set to None and ensures
  that the name_trivia attribute is initialized as an empty list.

  Args:
      None.

  Returns:
      None.
  """
  # line 148
  op = OperationNode(name="foo", name_trivia=None)  # type: ignore
  assert op.name_trivia == []


def test_operation_node_no_parens():
  """Verify that an OperationNode with has_parens set to False does not format its operands with parentheses.

  This test constructs an OperationNode with multiple ValueNode operands but has_parens set to
  False. It verifies that the serialized text represents a flat sequence of operands
  separated by commas, e.g., "foo %0, %1".

  Args:
      None.

  Returns:
      None.
  """
  # lines 170-171, 185
  op = OperationNode(name="foo", operands=[ValueNode(name="%0"), ValueNode(name="%1")], has_parens=False)
  assert op.to_text() == "foo %0, %1"


def test_operation_node_op_tail():
  """Verify that OperationNode correctly formats its tail string and tail trivia.

  This test constructs an OperationNode specifying result_types, a custom op_tail_str (" -> "),
  and op_tail_trivia, verifying that the node properly serializes the result types
  and tail formatting.

  Args:
      None.

  Returns:
      None.
  """
  # lines 197, 204
  op = OperationNode(name="foo", result_types=[TypeNode(body="i32")], op_tail_str=" -> ", op_tail_trivia=[Trivia(" ")])
  assert op.to_text() == "foo  -> i32"


def test_operation_node_multiple_result_types():
  """Verify that OperationNode correctly formats multiple result types using the specified op_tail_str.

  This test constructs multiple OperationNode instances with multiple result types and different
  tail strings ("-> " and " : "). It ensures that the resulting serialization wraps the types
  with parentheses or lists them separated by commas based on the tail string format.

  Args:
      None.

  Returns:
      None.
  """
  # lines 211, 215
  op1 = OperationNode(name="foo", result_types=[TypeNode(body="i32"), TypeNode(body="f32")], op_tail_str="-> ")
  assert op1.to_text() == "foo-> (i32, f32)"

  op2 = OperationNode(name="foo", result_types=[TypeNode(body="i32"), TypeNode(body="f32")], op_tail_str=" : ")
  assert op2.to_text() == "foo : i32, f32"


def test_stablehlo_constant_op():
  """Verify the formatting of StableHloConstantOp across various configurations of attributes and types.

  This test constructs three different configurations of StableHloConstantOp:
  1. An operation with an attribute but no result types or name trivia.
  2. An operation with a result, name trivia, multiple types, leading/trailing trivia, and attributes.
  3. An operation with a single result type and attribute.
  It asserts that each configuration serializes to the correct MLIR text format.

  Args:
      None.

  Returns:
      None.
  """
  # lines 236-269
  # no result types, no name trivia
  op1 = StableHloConstantOp(name="stablehlo.constant", attributes=[AttributeNode(name="value", value="dense<1.0>")])
  assert op1.to_text() == "stablehlo.constant dense<1.0>"

  # with results, name trivia, multiple types, and trailing/leading trivia
  op2 = StableHloConstantOp(
    name="stablehlo.constant",
    results=[ValueNode(name="%0")],
    name_trivia=[Trivia("  ")],
    attributes=[AttributeNode(name="value", value="dense<1.0>")],
    result_types=[TypeNode(body="tensor<f32>"), TypeNode(body="tensor<i32>")],
  )
  op2.leading_trivia = [Trivia("  ")]
  op2.trailing_trivia = [Trivia("\n")]
  assert op2.to_text() == "  %0 = stablehlo.constant  dense<1.0> : (tensor<f32>, tensor<i32>)\n"

  # with exactly one result type
  op3 = StableHloConstantOp(
    name="stablehlo.constant",
    attributes=[AttributeNode(name="value", value="dense<1.0>")],
    result_types=[TypeNode(body="tensor<f32>")],
  )
  assert op3.to_text() == "stablehlo.constant dense<1.0> : tensor<f32>"
