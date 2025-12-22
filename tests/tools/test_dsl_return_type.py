"""
Tests for ODL Schema Extension: Return Type Specification.
Corresponds to Limitation #3 in the Architectural roadmap.
"""

from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant


def test_op_return_type_default():
  """
  Verify that return_type defaults to "Any".
  """
  op = OperationDef(
    operation="DefaultOp",
    description="Op with no return spec",
    std_args=[],
    variants={"torch": FrameworkVariant(api="foo")},
  )
  assert op.return_type == "Any"


def test_op_return_type_explicit():
  """
  Verify that return_type can be set explicitly (e.g. 'bool').
  """
  op = OperationDef(
    operation="IsNan",
    description="Checks for NaNs",
    std_args=[],
    variants={"torch": FrameworkVariant(api="isnan")},
    return_type="bool",
  )
  assert op.return_type == "bool"


def test_op_return_type_complex():
  """
  Verify complex type hints strings are accepted.
  """
  op = OperationDef(
    operation="TopK",
    description="Returns values and indices",
    std_args=[],
    variants={},
    return_type="Tuple[Tensor, Tensor]",
  )
  assert op.return_type == "Tuple[Tensor, Tensor]"
