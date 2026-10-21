"""Test suite for conditional dispatch logic."""

import libcst as cst

from ml_switcheroo.core.rewriter.calls.dispatch import (
  _check_rule_condition,
  _extract_argument_node,
  _node_to_literal,
  evaluate_dispatch_rules,
)
from ml_switcheroo.enums import LogicOp


class DummyRule:
  """Mock rule for dispatch evaluation."""

  def __init__(
    self,
    if_arg: str,
    op: LogicOp | str,
    is_val: int | float | str | bool | None | list[int] | tuple[int, ...] | set[int],
    use_api: str,
  ) -> None:
    """Initialize test helper instance.

    Args:
        if_arg: Test argument or fixture.
        op: Test argument or fixture.
        is_val: Test argument or fixture.
        use_api: Test argument or fixture.
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
        source_framework: Test argument or fixture.
    """
    self.source_framework = source_framework

  def _is_module_alias(self, name: cst.CSTNode) -> bool:
    """Execute is module alias helper logic.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return False


def parse_call(code: str) -> cst.Call:
  """Parses a single call from code.

  Returns:
      Computed test result or fixture value.
  """
  module: cst.Module = cst.parse_module(code)
  expr = getattr(getattr(module.body[0], "body")[0], "value")
  return cst.ensure_type(expr, cst.Call)


def test_node_to_literal() -> None:
  """Verify node to literal behavior."""
  assert _node_to_literal(cst.Integer("42")) == 42
  assert _node_to_literal(cst.Float("3.14")) == 3.14
  assert _node_to_literal(cst.SimpleString("'hello'")) == "hello"
  assert _node_to_literal(cst.Name("True")) is True
  assert _node_to_literal(cst.Name("False")) is False
  assert _node_to_literal(cst.Name("None")) is None

  # Unconvertible node
  assert _node_to_literal(cst.List([])) is None


def test_check_rule_condition_is_type() -> None:
  """Verify check rule condition is type behavior."""
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


def test_check_rule_condition_operators() -> None:
  """Verify check rule condition operators behavior."""
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


def test_extract_argument_node_keyword() -> None:
  """Verify extract argument node keyword behavior."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(a=1, b=2)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_positional() -> None:
  """Verify extract argument node positional behavior."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(1, 2)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_method() -> None:
  """Verify extract argument node method behavior."""

  class MethodRewriter(DummyRewriter):
    """Test helper class MethodRewriter."""

    def _is_module_alias(self, name: cst.CSTNode) -> bool:
      """Execute is module alias helper logic.

      Args:
          name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return False

  rewriter = MethodRewriter("torch")
  call: cst.Call = parse_call("obj.func(2)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["x", "b"])
  assert isinstance(node, cst.Integer)
  assert node.value == "2"


def test_extract_argument_node_not_found() -> None:
  """Verify extract argument node not found behavior."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(1)")

  node = _extract_argument_node(rewriter, call, "b", "b", ["a", "b"])
  assert node is None

  node2 = _extract_argument_node(rewriter, call, "c", "c", ["a", "b"])
  assert node2 is None


def test_evaluate_dispatch_rules() -> None:
  """Verify evaluate dispatch rules behavior."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(mode='fast')")

  rules = [
    DummyRule("mode", LogicOp.EQ, "fast", "fast_func"),
    DummyRule("mode", LogicOp.EQ, "slow", "slow_func"),
  ]

  details = {
    "variants": {"torch": {"args": {"mode": "mode"}}},
    "std_args": ["data", {"name": "mode"}],
  }

  result = evaluate_dispatch_rules(rewriter, call, rules, details)
  assert result == "fast_func"

  call_slow: cst.Call = parse_call("func(mode='slow')")
  assert evaluate_dispatch_rules(rewriter, call_slow, rules, details) == "slow_func"

  call_none: cst.Call = parse_call("func(mode='unknown')")
  assert evaluate_dispatch_rules(rewriter, call_none, rules, details) is None


def test_evaluate_dispatch_rules_tuple_std_args() -> None:
  """Verify evaluate dispatch rules tuple std args behavior."""
  rewriter = DummyRewriter("torch")
  call: cst.Call = parse_call("func(1, 2)")

  rules = [
    DummyRule("y", LogicOp.EQ, 2, "special_func"),
  ]

  details = {"variants": {"torch": {"args": {"y": "y"}}}, "std_args": [["x", "int"], ["y", "int"]]}

  result = evaluate_dispatch_rules(rewriter, call, rules, details)
  assert result == "special_func"
