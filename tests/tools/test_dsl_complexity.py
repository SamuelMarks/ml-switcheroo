"""
Tests for ODL Schema Extension: Cost Complexity Metadata.
Corresponds to Limitation #19 in the Architectural roadmap.
"""

from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant


def test_complexity_field_storage():
  """
  Verify 'complexity' field stores string values.
  """
  op = OperationDef(
    operation="MatMul",
    description="Matrix Multiplication",
    std_args=[],
    variants={"torch": FrameworkVariant(api="mm")},
    complexity="O(N^3)",
  )
  assert op.complexity == "O(N^3)"


def test_complexity_default_none():
  """
  Verify default complexity is None.
  """
  op = OperationDef(operation="Add", description="Addition", std_args=[], variants={})
  assert op.complexity is None
