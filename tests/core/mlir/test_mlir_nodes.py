"""Test suite for the Mlir Nodes module."""

from ml_switcheroo.core.mlir.nodes import (
  ValueNode,
  TypeNode,
  AttributeNode,
  OperationNode,
  BlockNode,
  RegionNode,
  ModuleNode,
  TriviaNode,
)


def test_trivia_rendering():
  """Verifies the behavior of trivia rendering."""
  t1 = TriviaNode("\n")
  t2 = TriviaNode("// comment")
  assert t1.to_text() == "\n"
  assert t2.to_text() == "// comment"


def test_value_and_type_rendering():
  """Verifies the behavior of value and type rendering."""
  v = ValueNode(name="%0")
  t = TypeNode(body="f32")
  assert v.to_text() == "%0"
  assert t.to_text() == "f32"


def test_attribute_rendering():
  """Verifies the behavior of attribute rendering."""
  a1 = AttributeNode(name="val", value="10")
  assert a1.to_text() == "val = 10"
  a2 = AttributeNode(name="metrics", value="dense<0>", type_annotation="tensor<1xi32>")
  assert a2.to_text() == "metrics = dense<0> : tensor<1xi32>"


def test_operation_with_name_trivia():
  """Verifies OperationNode with name_trivia."""
  op = OperationNode(
    name="foo",
    name_trivia=[TriviaNode("  ")],
    result_types=[TypeNode(body="f32")],
    trailing_trivia=[TriviaNode(" // tail")],
  )
  assert op.to_text() == "foo   : f32 // tail\n"


def test_block_node_leading_trivia_and_empty():
  """Verifies BlockNode with leading trivia and empty label edge cases."""
  blk = BlockNode(label="^bb1", leading_trivia=[TriviaNode("  ")])
  assert blk.to_text() == "  ^bb1:\n"


def test_operation_simple():
  """Verifies the behavior of operation simple."""
  op = OperationNode(
    name="arith.addf",
    results=[ValueNode(name="%0")],
    operands=[ValueNode(name="%a"), ValueNode(name="%b")],
    result_types=[TypeNode(body="f32")],
  )
  txt = op.to_text()
  assert txt.strip() == "%0 = arith.addf (%a, %b) : f32"


def test_operation_with_attributes_and_trivia():
  """Verifies the behavior of operation with attributes and trivia."""
  op = OperationNode(
    name='"sw.op"',
    results=[ValueNode(name="%sum")],
    attributes=[AttributeNode(name="name", value='"add"')],
    leading_trivia=[TriviaNode("\n"), TriviaNode("// Compute sum\n"), TriviaNode("    ")],
  )
  txt = op.to_text()
  expected_start = '\n// Compute sum\n    %sum = "sw.op" {name = "add"}'
  assert txt.startswith(expected_start)


def test_block_structure():
  """Verifies the behavior of block structure."""
  op = OperationNode(name="op", results=[ValueNode(name="%0")])
  blk = BlockNode(label="^bb0", arguments=[(ValueNode(name="%arg0"), TypeNode(body="i32"))], operations=[op])
  txt = blk.to_text()
  assert "^bb0(%arg0: i32):" in txt
  assert "%0 = op" in txt


def test_region_nesting():
  """Verifies the behavior of region nesting."""
  op_yield = OperationNode(name="yield")
  blk = BlockNode(label="^true", operations=[op_yield])
  region = RegionNode(blocks=[blk])
  op_if = OperationNode(name="scf.if", operands=[ValueNode(name="%cond")], regions=[region])
  txt = op_if.to_text()
  assert "scf.if (%cond) {" in txt
  assert "^true:" in txt
  assert "yield" in txt
  assert "}" in txt


def test_module_node():
  """Verifies the behavior of module node."""
  op = OperationNode(name="func.return")
  blk = BlockNode(label="", operations=[op])
  mod = ModuleNode(body=blk)
  assert "func.return" in mod.to_text()


def test_multiple_results_and_types():
  """Verifies the behavior of multiple results and types."""
  op = OperationNode(
    name="op",
    results=[ValueNode(name="%0"), ValueNode(name="%1")],
    result_types=[TypeNode(body="i32"), TypeNode(body="f32")],
  )
  txt = op.to_text()
  assert "%0, %1 = op" in txt
  assert ": (i32, f32)" in txt


def test_trailing_trivia():
  """Verifies the behavior of trailing trivia on various nodes."""
  t = TriviaNode(" ")

  v = ValueNode(name="%0", trailing_trivia=[t])
  assert v.to_text() == "%0 "

  typ = TypeNode(body="f32", trailing_trivia=[t])
  assert typ.to_text() == "f32 "

  attr_list = AttributeNode(name="array", value=["1", "2"], trailing_trivia=[t])
  assert attr_list.to_text() == "array = [1, 2] "

  blk = BlockNode(label="^bb", trailing_trivia=[t])
  assert blk.to_text() == "^bb: "

  reg = RegionNode(trailing_trivia=[t])
  assert reg.to_text() == "{} "

  mod = ModuleNode(body=BlockNode(label=""), trailing_trivia=[t])
  assert mod.to_text() == " "


def test_stablehloconstantop():
  """Test method."""
  from ml_switcheroo.core.mlir.nodes import StableHloConstantOp

  op = StableHloConstantOp(
    name="stablehlo.constant",
    results=[ValueNode(name="%0")],
    attributes=[AttributeNode(name="value", value="dense<1.0>")],
    result_types=[TypeNode(body="tensor<f32>")],
    name_trivia=[TriviaNode(" ")],
    leading_trivia=[TriviaNode("  ")],
    trailing_trivia=[TriviaNode(" // end")],
  )
  txt = op.to_text()
  assert txt == "  %0 = stablehlo.constant dense<1.0> : tensor<f32> // end\n"

  op2 = StableHloConstantOp(
    name="stablehlo.constant",
    attributes=[AttributeNode(name="value", value="dense<1.0>")],
    result_types=[TypeNode(body="f32"), TypeNode(body="i32")],
  )
  assert op2.to_text() == "stablehlo.constant dense<1.0> : (f32, i32)\n"
