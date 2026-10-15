"""Test suite for mlir cst."""

from ml_switcheroo.core.mlir.cst import OperationNode, Trivia, ValueNode


def test_operation_node_no_space_if_trivia() -> None:
  """Verify operation node no space if trivia behavior."""
  op: OperationNode = OperationNode(
    name="test.op",
    operands=[ValueNode(name="%0", leading_trivia=[Trivia(" ")])],
    has_parens=False,
  )
  text: str = op.to_text()
  assert text == "test.op %0"
