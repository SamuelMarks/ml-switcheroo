import pytest
from ml_switcheroo.core.mlir.nodes import BlockNode, TriviaNode
from ml_switcheroo.core.mlir.tokens import TokenKind


def test_block_node_leading_trivia():
  blk = BlockNode(label="^bb0", leading_trivia=[TriviaNode(TokenKind.NEWLINE, "\n")])
  txt = blk.to_text()
  assert "\n" in txt


from ml_switcheroo.core.mlir.nodes import AttributeNode, OperationNode, ValueNode


def test_nodes_attribute_list():
  attr = AttributeNode("foo", ["1", "2"])
  assert attr.to_text() == "foo = [1, 2]"


def test_nodes_operation_space():
  op = OperationNode(name="sw.op", operands=[ValueNode("%0")], attributes=[AttributeNode("a", "1")], name_trivia=[])
  txt = op.to_text()
  assert "sw.op (%0) {a = 1}" in txt


from ml_switcheroo.core.mlir.nodes import TypeNode, RegionNode


def test_block_node_with_args():
  blk = BlockNode(label="^bb0", arguments=[(ValueNode("%0"), TypeNode("i32"))])
  txt = blk.to_text()
  assert "%0: i32" in txt


def test_operation_results():
  op = OperationNode(name="sw.op", results=[ValueNode("%0"), ValueNode("%1")])
  txt = op.to_text()
  assert "%0, %1 = sw.op" in txt


def test_operation_name_trivia():
  op = OperationNode(name="sw.op", name_trivia=[TriviaNode(TokenKind.WHITESPACE, "   ")])
  txt = op.to_text()
  assert "sw.opWHITESPACE" in txt


def test_operation_regions():
  blk = BlockNode(label="^bb0")
  reg = RegionNode(blocks=[blk])
  op = OperationNode(name="sw.op", regions=[reg])
  txt = op.to_text()
  assert "{" in txt


def test_operation_types():
  op = OperationNode(name="sw.op", result_types=[TypeNode("i32")])
  txt = op.to_text()
  assert ": i32" in txt

  op2 = OperationNode(name="sw.op", result_types=[TypeNode("i32"), TypeNode("f32")])
  txt2 = op2.to_text()
  assert ": (i32, f32)" in txt2

  op3 = OperationNode(name="sw.op", name_trivia=[TriviaNode(TokenKind.WHITESPACE, " ")], result_types=[TypeNode("i32")])
  txt3 = op3.to_text()
  assert ": i32" in txt3


def test_operation_trailing_trivia():
  op = OperationNode(name="sw.op", trailing_trivia=[TriviaNode(TokenKind.NEWLINE, "\n")])
  assert "\n" in op.to_text()
