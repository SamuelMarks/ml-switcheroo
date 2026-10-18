"""Test suite for the Enums Coverage module."""

from ml_switcheroo.enums import LogicOp, SemanticTier


def test_enums_member_access() -> None:
  """Verifies the behavior of enums member access."""
  assert SemanticTier.ARRAY_API.value == "array"
  assert SemanticTier.NEURAL.value == "neural"
  assert SemanticTier.NEURAL_OPS.value == "neural_ops"
  assert SemanticTier.EXTRAS.value == "extras"
  assert LogicOp.EQ.value == "eq"
  assert LogicOp.NEQ.value == "neq"
  assert LogicOp.GT.value == "gt"
  assert LogicOp.LT.value == "lt"
  assert LogicOp.GTE.value == "gte"
  assert LogicOp.LTE.value == "lte"
  assert LogicOp.IN.value == "in"
  assert LogicOp.NOT_IN.value == "not_in"
  assert LogicOp.IS_TYPE.value == "is_type"
