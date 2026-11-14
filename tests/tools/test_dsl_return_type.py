"""Test suite for the Dsl Return Type module."""

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef


def test_op_return_type_default() -> None:
  """Verify `OperationDef` handles operation return type default as expected."""
  op: OperationDef = OperationDef(
    operation="DefaultOp",
    description="Op with no return spec",
    std_args=[],
    variants={"torch": FrameworkVariant(api="foo")},
  )
  assert op.return_type == "Any"


def test_op_return_type_explicit() -> None:
  """Verify `OperationDef` handles operation return type explicit as expected."""
  op: OperationDef = OperationDef(
    operation="IsNan",
    description="Checks for NaNs",
    std_args=[],
    variants={"torch": FrameworkVariant(api="isnan")},
    return_type="bool",
  )
  assert op.return_type == "bool"


def test_op_return_type_complex() -> None:
  """Verify `OperationDef` handles operation return type complex as expected."""
  op: OperationDef = OperationDef(
    operation="TopK",
    description="Returns values and indices",
    std_args=[],
    variants={},
    return_type="Tuple[Tensor, Tensor]",
  )
  assert op.return_type == "Tuple[Tensor, Tensor]"
