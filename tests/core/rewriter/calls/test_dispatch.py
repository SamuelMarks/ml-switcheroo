"""Test suite for conditional dispatch logic."""

import libcst as cst
import typing

from ml_switcheroo.enums import LogicOp
from ml_switcheroo.core.rewriter.calls.dispatch import (
  evaluate_dispatch_rules,
  _extract_argument_node,
  _node_to_literal,
  _check_rule_condition,
)


class DummyRule:
  """Mock rule for dispatch evaluation."""

  def __init__(self, if_arg: str, op: str, is_val: typing.Any, use_api: str) -> None:
    """Docstring."""
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api


class DummyRewriter:
  """Mock rewriter."""

  def __init__(self, source_fw: str) -> None:
    """Docstring."""
    self.source_fw = source_fw

  def _is_module_alias(self, name: typing.Any) -> bool:
    """Docstring."""
    return False


def parse_call(code: str) -> cst.Call:
  """Parses a single call from code."""
  module: cst.Module = cst.parse_module(code)
  expr: typing.Any = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value
  return typing.cast(cst.Call, expr)


def test_node_to_literal() -> None:
  """Test converting CST nodes to literals."""
  assert _node_to_literal(cst.Integer("42")) == 42
  assert _node_to_literal(cst.Float("3.14")) == 3.14
  assert _node_to_literal(cst.SimpleString("'hello'")) == "hello"
  assert _node_to_literal(cst.Name("True")) is True
  assert _node_to_literal(cst.Name("False")) is False
  assert _node_to_literal(cst.Name("None")) is None

  # Unconvertible node
  assert _node_to_literal(cst.List([])) is None


def test_check_rule_condition_is_type() -> None:
  """Test rule condition IS_TYPE."""
  rule_int: typing.Any = DummyRule("x", LogicOp.IS_TYPE, "int", "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_int)
  assert not _check_rule_condition(cst.Float("5.0"), rule_int)

  rule_float: typing.Any = DummyRule("x", LogicOp.IS_TYPE, "float", "foo")
  assert _check_rule_condition(cst.Float("3.14"), rule_float)
  assert not _check_rule_condition(cst.Integer("3"), rule_float)

  rule_str: typing.Any = DummyRule("x", LogicOp.IS_TYPE, "str", "foo")
  assert _check_rule_condition(cst.SimpleString('"hi"'), rule_str)

  rule_list: typing.Any = DummyRule("x", LogicOp.IS_TYPE, "list", "foo")
  assert _check_rule_condition(cst.List([]), rule_list)

  rule_dict: typing.Any = DummyRule("x", LogicOp.IS_TYPE, "dict", "foo")
  assert _check_rule_condition(cst.Dict([]), rule_dict)

  rule_bool: typing.Any = DummyRule("x", LogicOp.IS_TYPE, "bool", "foo")
  assert _check_rule_condition(cst.Name("True"), rule_bool)
  assert not _check_rule_condition(cst.Name("None"), rule_bool)

  rule_unknown: typing.Any = DummyRule("x", LogicOp.IS_TYPE, "foo", "foo")
  assert not _check_rule_condition(cst.Name("var"), rule_unknown)


def test_check_rule_condition_operators() -> None:
  """Test rule conditions using various operators."""
  # EQ
  rule_eq: typing.Any = DummyRule("x", LogicOp.EQ, 5, "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_eq)
  assert not _check_rule_condition(cst.Integer("6"), rule_eq)

  # NEQ
  rule_neq: typing.Any = DummyRule("x", LogicOp.NEQ, 5, "foo")
  assert _check_rule_condition(cst.Integer("6"), rule_neq)
  assert not _check_rule_condition(cst.Integer("5"), rule_neq)

  # GT
  rule_gt: typing.Any = DummyRule("x", LogicOp.GT, 5, "foo")
  assert _check_rule_condition(cst.Integer("6"), rule_gt)
  assert not _check_rule_condition(cst.Integer("5"), rule_gt)

  # LT
  rule_lt: typing.Any = DummyRule("x", LogicOp.LT, 5, "foo")
  assert _check_rule_condition(cst.Integer("4"), rule_lt)

  # GTE
  rule_gte: typing.Any = DummyRule("x", LogicOp.GTE, 5, "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_gte)

  # LTE
  rule_lte: typing.Any = DummyRule("x", LogicOp.LTE, 5, "foo")
  assert _check_rule_condition(cst.Integer("5"), rule_lte)

  # IN
  rule_in: typing.Any = DummyRule("x", LogicOp.IN, [1, 2, 3], "foo")
  assert _check_rule_condition(cst.Integer("2"), rule_in)

  # NOT_IN
  rule_notin: typing.Any = DummyRule("x", LogicOp.NOT_IN, [1, 2, 3], "foo")
  assert _check_rule_condition(cst.Integer("4"), rule_notin)

  # Missing literal
  assert not _check_rule_condition(cst.Name("var"), rule_eq)


def test_extract_argument_node_keyword() -> None:
  """Test extracting arguments by keyword."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(a=1, b=2)")

  node: typing.Any = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])  # type: ignore
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_positional() -> None:
  """Test extracting arguments positionally."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(1, 2)")

  node: typing.Any = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])  # type: ignore
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_method() -> None:
  """Test extracting arguments for methods where first arg 'x' is skipped."""

  class MethodRewriter(DummyRewriter):
    """Docstring."""

    def _is_module_alias(self, name: typing.Any) -> bool:
      """Docstring."""
      return False

  rewriter = MethodRewriter("torch")
  call: cst.Call = parse_call("obj.func(2)")

  node: typing.Any = _extract_argument_node(rewriter, call, "b", "b", ["x", "b"])  # type: ignore
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_not_found() -> None:
  """Test when argument is not found."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(1)")

  node: typing.Any = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])  # type: ignore
  assert node is None

  node2: typing.Any = _extract_argument_node(rewriter, call, "c", "c", ["a", "b"])  # type: ignore
  assert node2 is None


def test_evaluate_dispatch_rules() -> None:
  """Test evaluate_dispatch_rules."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(mode='fast')")

  rules: list[typing.Any] = [
    DummyRule("mode", LogicOp.EQ, "fast", "fast_func"),
    DummyRule("mode", LogicOp.EQ, "slow", "slow_func"),
  ]

  details: dict[str, typing.Any] = {
    "variants": {"torch": {"args": {"mode": "mode"}}},
    "std_args": ["data", {"name": "mode"}],
  }

  result: typing.Any = evaluate_dispatch_rules(rewriter, call, rules, details)  # type: ignore
  assert result == "fast_func"

  call_slow: cst.Call = parse_call("func(mode='slow')")
  assert evaluate_dispatch_rules(rewriter, call_slow, rules, details) == "slow_func"  # type: ignore

  call_none: cst.Call = parse_call("func(mode='unknown')")
  assert evaluate_dispatch_rules(rewriter, call_none, rules, details) is None  # type: ignore


def test_evaluate_dispatch_rules_tuple_std_args() -> None:
  """Test with tuple format in std_args."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(1, 2)")

  rules: list[typing.Any] = [
    DummyRule("y", LogicOp.EQ, 2, "special_func"),
  ]

  details: dict[str, typing.Any] = {"variants": {"torch": {"args": {"y": "y"}}}, "std_args": [["x", "int"], ["y", "int"]]}

  result: typing.Any = evaluate_dispatch_rules(rewriter, call, rules, details)  # type: ignore
  assert result == "special_func"
