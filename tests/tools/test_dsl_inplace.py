"""
Tests for ODL Schema Extension: In-Place Semantics.
Corresponds to Limitation #6 in the Architectural roadmap.
"""

from ml_switcheroo.core.dsl import OperationDef, FrameworkVariant


def test_inplace_flag_defaults_false():
  """
  Verify 'is_inplace' defaults to False for standard operations.
  """
  op = OperationDef(
    operation="Add", description="Standard addition", std_args=[], variants={"torch": FrameworkVariant(api="torch.add")}
  )
  assert op.is_inplace is False


def test_inplace_flag_explicit():
  """
  Verify 'is_inplace' can be set to True for mutating operations.
  """
  op = OperationDef(
    operation="Add_",
    description="In-place addition",
    std_args=[],
    variants={"torch": FrameworkVariant(api="torch.add_")},
    is_inplace=True,
  )
  assert op.is_inplace is True
