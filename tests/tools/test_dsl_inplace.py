"""Test suite for the Dsl Inplace module."""

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef


def test_inplace_flag_defaults_false() -> None:
  """Verify `OperationDef` handles inplace flag defaults false as expected."""
  op: OperationDef = OperationDef(
    operation="Add", description="Standard addition", std_args=[], variants={"torch": FrameworkVariant(api="torch.add")}
  )
  assert op.is_inplace is False


def test_inplace_flag_explicit() -> None:
  """Verify `OperationDef` handles inplace flag explicit as expected."""
  op: OperationDef = OperationDef(
    operation="Add_",
    description="In-place addition",
    std_args=[],
    variants={"torch": FrameworkVariant(api="torch.add_")},
    is_inplace=True,
  )
  assert op.is_inplace is True
