"""Test suite for the Dsl Return Type module."""

from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant


def test_op_return_type_default():
  """Verifies the behavior of op return type default."""
  op = OperationDef(
    operation="DefaultOp",
    description="Op with no return spec",
    std_args=[],
    variants={"torch": FrameworkVariant(api="foo")},
  )
  assert op.return_type == "Any"


def test_op_return_type_explicit():
  """Verifies the behavior of op return type explicit."""
  op = OperationDef(
    operation="IsNan",
    description="Checks for NaNs",
    std_args=[],
    variants={"torch": FrameworkVariant(api="isnan")},
    return_type="bool",
  )
  assert op.return_type == "bool"


def test_op_return_type_complex():
  """Verifies the behavior of op return type complex."""
  op = OperationDef(
    operation="TopK",
    description="Returns values and indices",
    std_args=[],
    variants={},
    return_type="Tuple[Tensor, Tensor]",
  )
  assert op.return_type == "Tuple[Tensor, Tensor]"
