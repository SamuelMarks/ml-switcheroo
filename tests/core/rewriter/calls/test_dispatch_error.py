"""Test suite for dispatch error."""

from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.dispatch import _check_rule_condition, _node_to_literal, evaluate_dispatch_rules
from ml_switcheroo.enums import LogicOp


def test_node_to_literal_value_error() -> None:
  """Verify `_node_to_literal` handles value error as expected."""
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
        if_arg: Input `if_arg` parameter value (`str`).
        op: Input `op` parameter value (`str`).
        is_val: Input `is_val` parameter value (`int | float | str | bool | None`).
        use_api: Input `use_api` parameter value (`str`).
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
    """Execute `_is_module_alias` on `DummyRewriter` for dispatch error tests.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `bool` from is module alias.
    """
    return False


def parse_call(code: str) -> cst.Call:
  """Parses a single call from code.

  Args:
      code: Python or DSL source snippet.

  Returns:
      Resulting `cst.Call` from parse call.
  """
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr_stmt = cst.ensure_type(stmt.body[0], cst.Expr)
  return cst.ensure_type(expr_stmt.value, cst.Call)


def test_check_rule_condition_unknown_op() -> None:
  """Verify `_check_rule_condition` handles unknown operation as expected."""
  rule = DummyRule("x", "UNKNOWN_OP", 5, "foo")
  assert not _check_rule_condition(cst.Integer("5"), rule)


def test_evaluate_dispatch_rules_arg_not_found() -> None:
  """Verify `parse_call` handles evaluate dispatch rules arg not found as expected."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(a=1)")

  rules = [
    DummyRule("b", LogicOp.EQ, 2, "special_func"),
  ]

  details = {"variants": {"torch": {"args": {"b": "b"}}}, "std_args": ["a", "b"]}

  # Should continue and return None
  assert evaluate_dispatch_rules(rewriter, call, rules, details) is None
