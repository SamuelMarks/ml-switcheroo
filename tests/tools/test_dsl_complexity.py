"""Test suite for the Dsl Complexity module."""

from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant


def test_complexity_field_storage():
  """Verifies the behavior of complexity field storage."""
  op = OperationDef(
    operation="MatMul",
    description="Matrix Multiplication",
    std_args=[],
    variants={"torch": FrameworkVariant(api="mm")},
    complexity="O(N^3)",
  )
  assert op.complexity == "O(N^3)"


def test_complexity_default_none():
  """Verifies the behavior of complexity default none."""
  op = OperationDef(operation="Add", description="Addition", std_args=[], variants={})
  assert op.complexity is None
