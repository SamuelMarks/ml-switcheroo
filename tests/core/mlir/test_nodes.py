"""Test suite for the Nodes Gap module."""

import typing

from ml_switcheroo.core.cst.base import Trivia
from ml_switcheroo.core.mlir.cst import AttributeNode, BlockNode, OperationNode, RegionNode, TypeNode, ValueNode


def test_block_node_leading_trivia() -> None:
  """Verifies the behavior of block node leading trivia."""
  blk: typing.Any = BlockNode(label="^bb0", leading_trivia=[Trivia("\n")])
  txt: str = blk.to_text()
  assert "\n" in txt


def test_nodes_attribute_list() -> None:
  """Verifies the behavior of nodes attribute list."""
  attr: typing.Any = AttributeNode(name="foo", value=["1", "2"])
  assert attr.to_text() == "foo = [1, 2]"


def test_nodes_operation_space() -> None:
  """Verifies the behavior of nodes operation space."""
  op: typing.Any = OperationNode(
    name="sw.op", operands=[ValueNode(name="%0")], attributes=[AttributeNode(name="a", value="1")], name_trivia=[]
  )
  txt: str = op.to_text()
  assert "sw.op (%0) {a = 1}" in txt


def test_block_node_with_args() -> None:
  """Verifies the behavior of block node with arguments."""
  blk: typing.Any = BlockNode(label="^bb0", arguments=[(ValueNode(name="%0"), TypeNode(body="i32"))])
  txt: str = blk.to_text()
  assert "%0: i32" in txt


def test_operation_results() -> None:
  """Verifies the behavior of operation results."""
  op: typing.Any = OperationNode(name="sw.op", results=[ValueNode(name="%0"), ValueNode(name="%1")])
  txt: str = op.to_text()
  assert "%0, %1 = sw.op" in txt


def test_operation_name_trivia() -> None:
  """Verifies the behavior of operation name trivia."""
  op: typing.Any = OperationNode(name="sw.op", name_trivia=[Trivia("   ")])
  txt: str = op.to_text()
  assert "sw.op   " in txt


def test_operation_regions() -> None:
  """Verifies the behavior of operation regions."""
  blk: typing.Any = BlockNode(label="^bb0")
  reg: typing.Any = RegionNode(blocks=[blk])
  op: typing.Any = OperationNode(name="sw.op", regions=[reg])
  txt: str = op.to_text()
  assert "{" in txt


def test_operation_types() -> None:
  """Verifies the behavior of operation types."""
  op: typing.Any = OperationNode(name="sw.op", result_types=[TypeNode(body="i32")])
  txt: str = op.to_text()
  assert ": i32" in txt
  op2: typing.Any = OperationNode(name="sw.op", result_types=[TypeNode(body="i32"), TypeNode(body="f32")])
  txt2: str = op2.to_text()
  assert ": (i32, f32)" in txt2
  op3: typing.Any = OperationNode(name="sw.op", name_trivia=[Trivia(" ")], result_types=[TypeNode(body="i32")])
  txt3: str = op3.to_text()
  assert ": i32" in txt3


def test_operation_trailing_trivia() -> None:
  """Verifies the behavior of operation trailing trivia."""
  op: typing.Any = OperationNode(name="sw.op", trailing_trivia=[Trivia("\n")])
  assert "\n" in op.to_text()


def test_operation_node_to_text_missing_branches() -> None:
  """Hit the missing branches in OperationNode.to_text()."""
  from ml_switcheroo.core.mlir.cst import AttributeNode, OperationNode

  # 198: is_generic
  op1: typing.Any = OperationNode(name="my.op", is_generic=True, operands=[])
  assert '"my.op"' in op1.to_text()

  # 217-219: successors
  op2: typing.Any = OperationNode(name="br", successors=["^bb1", "^bb2"], operands=[])
  assert "[^bb1, ^bb2]" in op2.to_text()

  # 223-227: properties
  prop: typing.Any = AttributeNode(name="operand_segment_sizes", value="array<i32: 1, 0>")
  op3: typing.Any = OperationNode(name="test.op", properties=[prop], operands=[])
  assert "<{operand_segment_sizes = array<i32: 1, 0>}>" in op3.to_text()


def test_operation_node_to_text_trailing_space() -> None:
  """Hit the branches where parts[-1] already ends with space."""
  op = OperationNode(
    name="my.op", name_trivia=[Trivia(" ")], successors=["^bb1"], properties=[AttributeNode(name="x", value="1")]
  )
  text = op.to_text()
  assert "^bb1" in text


def test_operation_node_to_text_trailing_space_2() -> None:
  """Hit the branches where parts[-1] already ends with space for properties."""
  op = OperationNode(name="my.op", name_trivia=[Trivia(" ")], properties=[AttributeNode(name="x", value="1")])
  text = op.to_text()
  assert "x = 1" in text
