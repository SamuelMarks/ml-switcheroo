"""Test suite for dispatch error."""

from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.dispatch import _check_rule_condition, _node_to_literal, evaluate_dispatch_rules
from ml_switcheroo.enums import LogicOp


def test_node_to_literal_value_error() -> None:
  """Verify node to literal value error behavior."""
  node = MagicMock(spec=cst.Integer)
  node.value = "not_an_int"

  with patch("ml_switcheroo.core.rewriter.calls.dispatch.int", side_effect=ValueError):
    assert _node_to_literal(node) is None

  node_float = MagicMock(spec=cst.Float)
  node_float.value = "not_a_float"

  with patch("ml_switcheroo.core.rewriter.calls.dispatch.float", side_effect=ValueError):
    assert _node_to_literal(node_float) is None


class DummyRule:
  """Mock rule for dispatch evaluation."""

  def __init__(
    self,
    if_arg: str,
    op: str,
    is_val: int | float | str | bool | None,
    use_api: str,
  ) -> None:
    """Initialize test helper instance.

    Args:
        if_arg: The if arg (str) for   init  .
        op: The op (str) for   init  .
        is_val: The is val (int | float | str | bool | None) for   init  .
        use_api: The use api (str) for   init  .
    """
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api


class DummyRewriter:
  """Mock rewriter."""

  def __init__(self, source_framework: str) -> None:
    """Initialize test helper instance.

    Args:
        source_framework: Source framework identifier string.
    """
    self.source_framework = source_framework

  def _is_module_alias(self, name: cst.CSTNode) -> bool:
    """Execute is module alias helper logic.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `bool` from is module alias.
    """
    return False


def parse_call(code: str) -> cst.Call:
  """Parses a single call from code.

  Returns:
      Resulting `cst.Call` from parse call.
  """
  module = cst.parse_module(code)
  expr = getattr(getattr(module.body[0], "body")[0], "value")
  return cst.ensure_type(expr, cst.Call)


def test_check_rule_condition_unknown_op() -> None:
  """Verify check rule condition unknown op behavior."""
  rule = DummyRule("x", "UNKNOWN_OP", 5, "foo")
  assert not _check_rule_condition(cst.Integer("5"), rule)


def test_evaluate_dispatch_rules_arg_not_found() -> None:
  """Verify evaluate dispatch rules arg not found behavior."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(a=1)")

  rules = [
    DummyRule("b", LogicOp.EQ, 2, "special_func"),
  ]

  details = {"variants": {"torch": {"args": {"b": "b"}}}, "std_args": ["a", "b"]}

  # Should continue and return None
  assert evaluate_dispatch_rules(rewriter, call, rules, details) is None
