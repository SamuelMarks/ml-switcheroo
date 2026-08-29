"""Test suite for the Dsl Complexity module."""

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef


def test_complexity_field_storage() -> None:
  """Verifies the behavior of complexity field storage."""
  op: OperationDef = OperationDef(
    operation="MatMul",
    description="Matrix Multiplication",
    std_args=[],
    variants={"torch": FrameworkVariant(api="mm")},
    complexity="O(N^3)",
  )
  assert getattr(op, "complexity") == "O(N^3)"


def test_complexity_default_none() -> None:
  """Verifies the behavior of complexity default none."""
  op: OperationDef = OperationDef(operation="Add", description="Addition", std_args=[], variants={})
  assert getattr(op, "complexity") is None
