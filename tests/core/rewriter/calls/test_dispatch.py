"""Test suite for conditional dispatch logic."""

import libcst as cst

from ml_switcheroo.enums import LogicOp
from ml_switcheroo.core.rewriter.calls.dispatch import (
  evaluate_dispatch_rules,
  _extract_argument_node,
  _node_to_literal,
  _check_rule_condition,
)


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


def test_node_to_literal():
  """Test converting CST nodes to literals."""
  assert _node_to_literal(cst.Integer("42")) == 42
  assert _node_to_literal(cst.Float("3.14")) == 3.14
  assert _node_to_literal(cst.SimpleString("'hello'")) == "hello"
  assert _node_to_literal(cst.Name("True")) is True
  assert _node_to_literal(cst.Name("False")) is False
  assert _node_to_literal(cst.Name("None")) is None

  # Unconvertible node
  assert _node_to_literal(cst.List([])) is None


def test_check_rule_condition_is_type():
  """Test rule condition IS_TYPE."""
  rule_int = DummyRule("x", LogicOp.IS_TYPE, "int", "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_int)
  assert not _check_rule_condition(cst.Float("5.0"), rule_int)

  rule_float = DummyRule("x", LogicOp.IS_TYPE, "float", "foo")
  assert _check_rule_condition(cst.Float("3.14"), rule_float)
  assert not _check_rule_condition(cst.Integer("3"), rule_float)

  rule_str = DummyRule("x", LogicOp.IS_TYPE, "str", "foo")
  assert _check_rule_condition(cst.SimpleString('"hi"'), rule_str)

  rule_list = DummyRule("x", LogicOp.IS_TYPE, "list", "foo")
  assert _check_rule_condition(cst.List([]), rule_list)

  rule_dict = DummyRule("x", LogicOp.IS_TYPE, "dict", "foo")
  assert _check_rule_condition(cst.Dict([]), rule_dict)

  rule_bool = DummyRule("x", LogicOp.IS_TYPE, "bool", "foo")
  assert _check_rule_condition(cst.Name("True"), rule_bool)
  assert not _check_rule_condition(cst.Name("None"), rule_bool)

  rule_unknown = DummyRule("x", LogicOp.IS_TYPE, "foo", "foo")
  assert not _check_rule_condition(cst.Name("var"), rule_unknown)


def test_check_rule_condition_operators():
  """Test rule conditions using various operators."""
  # EQ
  rule_eq = DummyRule("x", LogicOp.EQ, 5, "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_eq)
  assert not _check_rule_condition(cst.Integer("6"), rule_eq)

  # NEQ
  rule_neq = DummyRule("x", LogicOp.NEQ, 5, "foo")
  assert _check_rule_condition(cst.Integer("6"), rule_neq)
  assert not _check_rule_condition(cst.Integer("5"), rule_neq)

  # GT
  rule_gt = DummyRule("x", LogicOp.GT, 5, "foo")
  assert _check_rule_condition(cst.Integer("6"), rule_gt)
  assert not _check_rule_condition(cst.Integer("5"), rule_gt)

  # LT
  rule_lt = DummyRule("x", LogicOp.LT, 5, "foo")
  assert _check_rule_condition(cst.Integer("4"), rule_lt)

  # GTE
  rule_gte = DummyRule("x", LogicOp.GTE, 5, "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_gte)

  # LTE
  rule_lte = DummyRule("x", LogicOp.LTE, 5, "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_lte)

  # IN
  rule_in = DummyRule("x", LogicOp.IN, [1, 2, 3], "foo")
  assert _check_rule_condition(cst.Integer("2"), rule_in)

  # NOT_IN
  rule_notin = DummyRule("x", LogicOp.NOT_IN, [1, 2, 3], "foo")
  assert _check_rule_condition(cst.Integer("4"), rule_notin)

  # Missing literal
  assert not _check_rule_condition(cst.Name("var"), rule_eq)


def test_extract_argument_node_keyword():
  """Test extracting arguments by keyword."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(a=1, b=2)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_positional():
  """Test extracting arguments positionally."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(1, 2)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_method():
  """Test extracting arguments for methods where first arg 'x' is skipped."""

  class MethodRewriter(DummyRewriter):
    def _is_module_alias(self, name):
      return False

  rewriter = MethodRewriter("torch")
  call = parse_call("obj.func(2)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["x", "b"])
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_not_found():
  """Test when argument is not found."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(1)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])
  assert node is None

  node2 = _extract_argument_node(rewriter, call, "c", "c", ["a", "b"])
  assert node2 is None


def test_evaluate_dispatch_rules():
  """Test evaluate_dispatch_rules."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(mode='fast')")

  rules = [DummyRule("mode", LogicOp.EQ, "fast", "fast_func"), DummyRule("mode", LogicOp.EQ, "slow", "slow_func")]

  details = {"variants": {"torch": {"args": {"mode": "mode"}}}, "std_args": ["data", {"name": "mode"}]}

  result = evaluate_dispatch_rules(rewriter, call, rules, details)
  assert result == "fast_func"

  call_slow = parse_call("func(mode='slow')")
  assert evaluate_dispatch_rules(rewriter, call_slow, rules, details) == "slow_func"

  call_none = parse_call("func(mode='unknown')")
  assert evaluate_dispatch_rules(rewriter, call_none, rules, details) is None


def test_evaluate_dispatch_rules_tuple_std_args():
  """Test with tuple format in std_args."""
  rewriter = DummyRewriter("torch")
  call = parse_call("func(1, 2)")

  rules = [
    DummyRule("y", LogicOp.EQ, 2, "special_func"),
  ]

  details = {"variants": {"torch": {"args": {"y": "y"}}}, "std_args": [["x", "int"], ["y", "int"]]}

  result = evaluate_dispatch_rules(rewriter, call, rules, details)
  assert result == "special_func"
