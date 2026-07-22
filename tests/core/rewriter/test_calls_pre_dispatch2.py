"""Test suite for the Calls Pre Dispatch2 module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.calls.dispatch import (
  evaluate_dispatch_rules,
  _extract_argument_node,
  _node_to_literal,
  _check_rule_condition,
)
from ml_switcheroo.enums import LogicOp


class MockRule:
  """Mock Rule class for testing purposes."""

  def __init__(self, if_arg=None, op=None, is_val=None, use_api=None, target_variant=None):
    """Initializes the MockRule instance."""
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api
    self.target_variant = target_variant


class MockRewriterDispatch:
  """Mock Rewriter Dispatch class for testing purposes."""

  def __init__(self):
    """Initializes the MockRewriterDispatch instance."""
    self.context = type("Ctx", (), {"semantics": None, "imports": set()})()
    self.current_rule = None
    self.source_fw = "src"


class MockTraits:
  """Mock Traits class for testing purposes."""

  def __init__(self, method="apply", implicit_roots=None):
    """Initializes the MockTraits instance."""
    self.functional_execution_method = method
    self.implicit_method_roots = implicit_roots or []


def test_evaluate_dispatch_rules():
  """Evaluates dispatch rules."""
  rewriter = MockRewriterDispatch()
  details = {"variants": {"src": {"args": {"arg1": "src_arg1"}}}, "std_args": [("arg1",), {"name": "arg2"}, "arg3"]}
  rules = [
    MockRule("missing_arg", LogicOp.EQ, 1, "api_miss"),
    MockRule("arg1", LogicOp.EQ, 1, "api1"),
    MockRule("arg2", LogicOp.EQ, 2, "api2"),
  ]
  node = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Integer("1")), cst.Arg(value=cst.Integer("2"))])
  res = evaluate_dispatch_rules(rewriter, node, rules, details)
  assert res == "api1"


def test_extract_argument_node_errors():
  """Extracts argument node errors."""
  rewriter = MockRewriterDispatch()
  node = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[cst.Arg(value=cst.Name("a"))])
  res1 = _extract_argument_node(rewriter, node, "not_found", "a", ["x", "a"])
  assert res1.value == "a"
  res2 = _extract_argument_node(rewriter, node, "not_found", "not_there", ["a"])
  assert res2 is None


def test_node_to_literal():
  """Verifies the behavior of node to literal."""
  int_node = MagicMock(spec=cst.Integer)
  int_node.value = "abc"
  assert _node_to_literal(int_node) is None
  float_node = MagicMock(spec=cst.Float)
  float_node.value = "abc"
  assert _node_to_literal(float_node) is None
  assert _node_to_literal(cst.Name("True")) is True
  assert _node_to_literal(cst.Name("False")) is False
  assert _node_to_literal(cst.Name("None")) is None
  assert _node_to_literal(cst.Name("Other")) is None


def test_check_rule_condition_is_type():
  """Checks rule condition is type."""
  rule = MockRule(None, LogicOp.IS_TYPE, "int")
  assert _check_rule_condition(cst.Integer("1"), rule) is True
  rule = MockRule(None, LogicOp.IS_TYPE, "float")
  assert _check_rule_condition(cst.Float("1.0"), rule) is True
  rule = MockRule(None, LogicOp.IS_TYPE, "str")
  assert _check_rule_condition(cst.SimpleString("'s'"), rule) is True
  rule = MockRule(None, LogicOp.IS_TYPE, "list")
  assert _check_rule_condition(cst.List([]), rule) is True
  rule = MockRule(None, LogicOp.IS_TYPE, "dict")
  assert _check_rule_condition(cst.Dict([]), rule) is True
  rule = MockRule(None, LogicOp.IS_TYPE, "bool")
  assert _check_rule_condition(cst.Name("True"), rule) is True
  assert _check_rule_condition(cst.Name("False"), rule) is True
  rule = MockRule(None, LogicOp.IS_TYPE, "unknown")
  assert _check_rule_condition(cst.Name("Other"), rule) is False


def test_check_rule_condition_ops():
  """Checks rule condition ops."""
  rule_none = MockRule(None, "UNKNOWN_OP", 5)
  assert _check_rule_condition(cst.Integer("6"), rule_none) is False
  rule_gt = MockRule(None, LogicOp.GT, 5)
  assert _check_rule_condition(cst.Integer("6"), rule_gt) is True
  assert _check_rule_condition(cst.Integer("4"), rule_gt) is False
  rule_lt = MockRule(None, LogicOp.LT, 5)
  assert _check_rule_condition(cst.Integer("4"), rule_lt) is True
  assert _check_rule_condition(cst.Integer("6"), rule_lt) is False
  rule_gte = MockRule(None, LogicOp.GTE, 5)
  assert _check_rule_condition(cst.Integer("5"), rule_gte) is True
  assert _check_rule_condition(cst.Integer("4"), rule_gte) is False
  rule_lte = MockRule(None, LogicOp.LTE, 5)
  assert _check_rule_condition(cst.Integer("5"), rule_lte) is True
  assert _check_rule_condition(cst.Integer("6"), rule_lte) is False
  rule_in = MockRule(None, LogicOp.IN, [1, 2])
  assert _check_rule_condition(cst.Integer("1"), rule_in) is True
  assert _check_rule_condition(cst.Integer("3"), rule_in) is False
  rule_not_in = MockRule(None, LogicOp.NOT_IN, [1, 2])
  assert _check_rule_condition(cst.Integer("3"), rule_not_in) is True
  assert _check_rule_condition(cst.Integer("1"), rule_not_in) is False
  rule_neq = MockRule(None, LogicOp.NEQ, 1)
  assert _check_rule_condition(cst.Integer("2"), rule_neq) is True
  assert _check_rule_condition(cst.Integer("1"), rule_neq) is False


def test_check_rule_condition_val_is_none():
  """Checks rule condition value is none."""
  rule = MockRule(None, LogicOp.EQ, 1)
  assert _check_rule_condition(cst.Name("Unknown"), rule) is False
