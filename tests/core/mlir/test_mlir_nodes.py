"""Test suite for the Mlir Nodes module."""

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  ValueNode,
  TypeNode,
  AttributeNode,
  OperationNode,
  BlockNode,
  RegionNode,
  ModuleNode,
)


def test_trivia_rendering():
  """Verifies the behavior of trivia rendering."""
  t1 = Trivia("\n")
  t2 = Trivia("// comment")
  assert t1.text == "\n"
  assert t2.text == "// comment"


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
    leading_trivia=[Trivia("\n"), Trivia("// Compute sum\n"), Trivia("    ")],
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
