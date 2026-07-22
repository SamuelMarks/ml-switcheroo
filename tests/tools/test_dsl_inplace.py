"""Test suite for the Dsl Inplace module."""

from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant


def test_inplace_flag_defaults_false():
  """Verifies the behavior of inplace flag defaults false."""
  op = OperationDef(
    operation="Add", description="Standard addition", std_args=[], variants={"torch": FrameworkVariant(api="torch.add")}
  )
  assert op.is_inplace is False


def test_inplace_flag_explicit():
  """Verifies the behavior of inplace flag explicit."""
  op = OperationDef(
    operation="Add_",
    description="In-place addition",
    std_args=[],
    variants={"torch": FrameworkVariant(api="torch.add_")},
    is_inplace=True,
  )
  assert op.is_inplace is True
