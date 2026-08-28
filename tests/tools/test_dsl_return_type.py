"""Test suite for the Dsl Return Type module."""

from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant


def test_op_return_type_default() -> None:
  """Verifies the behavior of op return type default."""
  op: OperationDef = OperationDef(
    operation="DefaultOp",
    description="Op with no return spec",
    std_args=[],
    variants={"torch": FrameworkVariant(api="foo")},
  )
  assert getattr(op, "return_type") == "Any"


def test_op_return_type_explicit() -> None:
  """Verifies the behavior of op return type explicit."""
  op: OperationDef = OperationDef(
    operation="IsNan",
    description="Checks for NaNs",
    std_args=[],
    variants={"torch": FrameworkVariant(api="isnan")},
    return_type="bool",
  )
  assert getattr(op, "return_type") == "bool"


def test_op_return_type_complex() -> None:
  """Verifies the behavior of op return type complex."""
  op: OperationDef = OperationDef(
    operation="TopK",
    description="Returns values and indices",
    std_args=[],
    variants={},
    return_type="Tuple[Tensor, Tensor]",
  )
  assert getattr(op, "return_type") == "Tuple[Tensor, Tensor]"
