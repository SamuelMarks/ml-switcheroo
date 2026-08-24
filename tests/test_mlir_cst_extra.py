"""Test module."""

from ml_switcheroo.core.mlir.cst import OperationNode, ValueNode, Trivia


def test_operation_node_no_space_if_trivia():
  """Test element."""
  op = OperationNode(
    name="test.op",
    operands=[ValueNode(name="%0", leading_trivia=[Trivia(" ")])],
    has_parens=False,
  )
  text = op.to_text()
  assert text == "test.op %0"
