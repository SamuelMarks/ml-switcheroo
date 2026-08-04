"""Test suite for the Nodes Gap module."""

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import (
  BlockNode,
  AttributeNode,
  OperationNode,
  ValueNode,
  TypeNode,
  RegionNode,
)


def test_block_node_leading_trivia():
  """Verifies the behavior of block node leading trivia."""
  blk = BlockNode(label="^bb0", leading_trivia=[Trivia("\n")])
  txt = blk.to_text()
  assert "\n" in txt


def test_nodes_attribute_list():
  """Verifies the behavior of nodes attribute list."""
  attr = AttributeNode(name="foo", value=["1", "2"])
  assert attr.to_text() == "foo = [1, 2]"


def test_nodes_operation_space():
  """Verifies the behavior of nodes operation space."""
  op = OperationNode(
    name="sw.op", operands=[ValueNode(name="%0")], attributes=[AttributeNode(name="a", value="1")], name_trivia=[]
  )
  txt = op.to_text()
  assert "sw.op (%0) {a = 1}" in txt


def test_block_node_with_args():
  """Verifies the behavior of block node with arguments."""
  blk = BlockNode(label="^bb0", arguments=[(ValueNode(name="%0"), TypeNode(body="i32"))])
  txt = blk.to_text()
  assert "%0: i32" in txt


def test_operation_results():
  """Verifies the behavior of operation results."""
  op = OperationNode(name="sw.op", results=[ValueNode(name="%0"), ValueNode(name="%1")])
  txt = op.to_text()
  assert "%0, %1 = sw.op" in txt


def test_operation_name_trivia():
  """Verifies the behavior of operation name trivia."""
  op = OperationNode(name="sw.op", name_trivia=[Trivia("   ")])
  txt = op.to_text()
  assert "sw.op   " in txt


def test_operation_regions():
  """Verifies the behavior of operation regions."""
  blk = BlockNode(label="^bb0")
  reg = RegionNode(blocks=[blk])
  op = OperationNode(name="sw.op", regions=[reg])
  txt = op.to_text()
  assert "{" in txt


def test_operation_types():
  """Verifies the behavior of operation types."""
  op = OperationNode(name="sw.op", result_types=[TypeNode(body="i32")])
  txt = op.to_text()
  assert ": i32" in txt
  op2 = OperationNode(name="sw.op", result_types=[TypeNode(body="i32"), TypeNode(body="f32")])
  txt2 = op2.to_text()
  assert ": (i32, f32)" in txt2
  op3 = OperationNode(name="sw.op", name_trivia=[Trivia(" ")], result_types=[TypeNode(body="i32")])
  txt3 = op3.to_text()
  assert ": i32" in txt3


def test_operation_trailing_trivia():
  """Verifies the behavior of operation trailing trivia."""
  op = OperationNode(name="sw.op", trailing_trivia=[Trivia("\n")])
  assert "\n" in op.to_text()
