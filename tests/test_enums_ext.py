"""Test suite for `SemanticTier` and `LogicOp` enumerations and behavioral evaluation."""

import libcst as cst

from ml_switcheroo.core.dsl import Rule as _Rule
from ml_switcheroo.core.rewriter.calls.dispatch import evaluate_dispatch_rules
from ml_switcheroo.enums import LogicOp, SemanticTier
from ml_switcheroo.semantics.merging import TIER_PRECEDENCE, infer_tier_from_priority


class _DummyRewriter:
  """Typed rewriter test helper."""

  source_fw: str = "torch"

  def _is_module_alias(self, _name: cst.CSTNode) -> bool:
    """Return False for module alias check in tests.

    Args:
        _name: CST node to inspect.

    Returns:
        False.
    """
    return False


def test_enums_member_access_and_behavioral_dispatch() -> None:
  """Verify `SemanticTier` precedence and `LogicOp` conditional dispatch evaluation."""
  assert SemanticTier.ARRAY_API.value == "array"
  assert SemanticTier.NEURAL.value == "neural"
  assert SemanticTier.NEURAL_OPS.value == "neural_ops"
  assert SemanticTier.EXTRAS.value == "extras"

  assert infer_tier_from_priority(10) == SemanticTier.ARRAY_API
  assert infer_tier_from_priority(20) == SemanticTier.NEURAL
  assert infer_tier_from_priority(30) == SemanticTier.EXTRAS
  assert TIER_PRECEDENCE[SemanticTier.NEURAL.value] > TIER_PRECEDENCE[SemanticTier.ARRAY_API.value]
  assert TIER_PRECEDENCE[SemanticTier.ARRAY_API.value] > TIER_PRECEDENCE[SemanticTier.EXTRAS.value]

  # Behavioral evaluation of LogicOp rules in evaluate_dispatch_rules
  rewriter = _DummyRewriter()
  rules = [
    _Rule(if_arg="mode", op=LogicOp.EQ, is_val="nearest", use_api="api.nearest"),
    _Rule(if_arg="mode", op=LogicOp.NEQ, is_val="linear", use_api="api.non_linear"),
    _Rule(if_arg="k", op=LogicOp.GT, is_val=5, use_api="api.gt5"),
    _Rule(if_arg="k", op=LogicOp.LT, is_val=2, use_api="api.lt2"),
    _Rule(if_arg="k", op=LogicOp.GTE, is_val=4, use_api="api.gte4"),
    _Rule(if_arg="k", op=LogicOp.LTE, is_val=3, use_api="api.lte3"),
    _Rule(if_arg="axis", op=LogicOp.IN, is_val=[0, 1], use_api="api.in_01"),
    _Rule(if_arg="axis", op=LogicOp.NOT_IN, is_val=[0, 1], use_api="api.not_in_01"),
    _Rule(if_arg="shape", op=LogicOp.IS_TYPE, is_val="list", use_api="api.list_shape"),
  ]

  call_eq = cst.parse_expression("fn(mode='nearest')")
  assert isinstance(call_eq, cst.Call)
  assert evaluate_dispatch_rules(rewriter, call_eq, rules[:1], {"std_args": ["mode"]}) == "api.nearest"

  call_gt = cst.parse_expression("fn(10)")
  assert isinstance(call_gt, cst.Call)
  assert evaluate_dispatch_rules(rewriter, call_gt, [rules[2]], {"std_args": ["k"]}) == "api.gt5"

  call_list = cst.parse_expression("fn([1, 2])")
  assert isinstance(call_list, cst.Call)
  assert evaluate_dispatch_rules(rewriter, call_list, [rules[8]], {"std_args": ["shape"]}) == "api.list_shape"
