"""Module docstring."""

import libcst as cst
from unittest.mock import patch, MagicMock

from ml_switcheroo.core.rewriter.calls.dispatch import _node_to_literal
from ml_switcheroo.core.rewriter.calls.dispatch import _check_rule_condition, evaluate_dispatch_rules
from ml_switcheroo.enums import LogicOp


def test_node_to_literal_value_error():
  """Docstring."""
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

  def __init__(self, if_arg, op, is_val, use_api):
    """Docstring."""
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api


class DummyRewriter:
  """Mock rewriter."""

  def __init__(self, source_fw):
    """Docstring."""
    self.source_fw = source_fw


def parse_call(code: str) -> cst.Call:
  """Parses a single call from code."""
  module = cst.parse_module(code)
  expr = module.body[0].body[0].value
  return expr


def test_check_rule_condition_unknown_op():
  """Docstring."""
  rule = DummyRule("x", "UNKNOWN_OP", 5, "foo")
  assert not _check_rule_condition(cst.Integer("5"), rule)


def test_evaluate_dispatch_rules_arg_not_found():
  """Docstring."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(a=1)")

  rules = [
    DummyRule("b", LogicOp.EQ, 2, "special_func"),
  ]

  details = {"variants": {"torch": {"args": {"b": "b"}}}, "std_args": ["a", "b"]}

  # Should continue and return None
  assert evaluate_dispatch_rules(rewriter, call, rules, details) is None
