"""Test suite for the Enums Coverage module."""

from ml_switcheroo.enums import SemanticTier, LogicOp


def test_enums_member_access():
  """Verifies the behavior of enums member access."""
  assert SemanticTier.ARRAY_API == "array"
  assert SemanticTier.NEURAL == "neural"
  assert SemanticTier.NEURAL_OPS == "neural_ops"
  assert SemanticTier.EXTRAS == "extras"
  assert LogicOp.EQ == "eq"
  assert LogicOp.NEQ == "neq"
  assert LogicOp.GT == "gt"
  assert LogicOp.LT == "lt"
  assert LogicOp.GTE == "gte"
  assert LogicOp.LTE == "lte"
  assert LogicOp.IN == "in"
  assert LogicOp.NOT_IN == "not_in"
  assert LogicOp.IS_TYPE == "is_type"
