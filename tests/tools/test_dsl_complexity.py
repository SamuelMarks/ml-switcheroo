"""Test suite for the Dsl Complexity module."""

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef


def test_complexity_field_storage() -> None:
  """Verify `OperationDef` handles complexity field storage as expected."""
  op: OperationDef = OperationDef(
    operation="MatMul",
    description="Matrix Multiplication",
    std_args=[],
    variants={"torch": FrameworkVariant(api="mm")},
    complexity="O(N^3)",
  )
  assert op.complexity == "O(N^3)"


def test_complexity_default_none() -> None:
  """Verify `OperationDef` handles complexity default none as expected."""
  op: OperationDef = OperationDef(operation="Add", description="Addition", std_args=[], variants={})
  assert op.complexity is None
