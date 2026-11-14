"""Unit tests for conditional rule dispatch (`ml_switcheroo.core.rewriter.calls.dispatch`)."""

from unittest.mock import MagicMock

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.dsl import Rule
from ml_switcheroo.core.rewriter.calls.dispatch import (
  _check_rule_condition,
  _extract_argument_node,
  _node_to_literal,
  evaluate_dispatch_rules,
)
from ml_switcheroo.enums import LogicOp


class _DispatchRewriterStub:
  """Minimal rewriter host stub providing `source_framework` and `_is_module_alias`."""

  def __init__(self, source_framework: str = "torch", *, module_alias: bool = False) -> None:
    """Initialize source framework and alias behavior.

    Args:
        source_framework: Source framework identifier.
        module_alias: Return value for `_is_module_alias`.
    """
    self.source_framework = source_framework
    self._module_alias = module_alias

  def _is_module_alias(self, _node: cst.CSTNode) -> bool:
    """Indicate whether `_node` is a registered module alias.

    Args:
        _node: Receiver CST node.

    Returns:
        Configured `self._module_alias` boolean.
    """
    return self._module_alias


def _parse_call(code: str) -> cst.Call:
  """Parse a Python call expression into `cst.Call`.

  Args:
      code: Python call expression string.

  Returns:
      Parsed `cst.Call` node.
  """
  return cst.ensure_type(cst.parse_expression(code), cst.Call)


def test_node_to_literal_conversions_and_malformed_values() -> None:
  """Verify `_node_to_literal` converts Python CST scalars and returns `None` on malformed/compound nodes."""
  assert _node_to_literal(cst.Integer("42")) == 42
  assert _node_to_literal(cst.Float("3.14")) == 3.14
  assert _node_to_literal(cst.SimpleString("'hello'")) == "hello"
  assert _node_to_literal(cst.Name("True")) is True
  assert _node_to_literal(cst.Name("False")) is False
  assert _node_to_literal(cst.Name("None")) is None
  assert _node_to_literal(cst.Name("Other")) is None
  assert _node_to_literal(cst.List([])) is None

  bad_int = MagicMock(spec=cst.Integer)
  bad_int.value = "not_int"
  assert _node_to_literal(bad_int) is None

  bad_float = MagicMock(spec=cst.Float)
  bad_float.value = "not_float"
  assert _node_to_literal(bad_float) is None


def test_check_rule_condition_type_checks_and_comparisons() -> None:
  """Verify `_check_rule_condition` across `IS_TYPE` and all comparison `LogicOp` operators."""
  assert _check_rule_condition(cst.Integer("5"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="int", use_api="f"))
  assert not _check_rule_condition(cst.Float("5.0"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="int", use_api="f"))
  assert _check_rule_condition(cst.Float("3.14"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="float", use_api="f"))
  assert not _check_rule_condition(cst.Integer("3"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="float", use_api="f"))
  assert _check_rule_condition(cst.SimpleString('"s"'), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="str", use_api="f"))
  assert _check_rule_condition(cst.List([]), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="list", use_api="f"))
  assert _check_rule_condition(cst.Dict([]), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="dict", use_api="f"))
  assert _check_rule_condition(cst.Name("True"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="bool", use_api="f"))
  assert _check_rule_condition(cst.Name("False"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="bool", use_api="f"))
  assert not _check_rule_condition(cst.Name("None"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="bool", use_api="f"))
  assert not _check_rule_condition(cst.Name("v"), Rule(if_arg="x", op=LogicOp.IS_TYPE, is_val="other", use_api="f"))

  assert _check_rule_condition(cst.Integer("5"), Rule(if_arg="x", op=LogicOp.EQ, is_val=5, use_api="f"))
  assert not _check_rule_condition(cst.Integer("6"), Rule(if_arg="x", op=LogicOp.EQ, is_val=5, use_api="f"))
  assert _check_rule_condition(cst.Integer("6"), Rule(if_arg="x", op=LogicOp.NEQ, is_val=5, use_api="f"))
  assert not _check_rule_condition(cst.Integer("5"), Rule(if_arg="x", op=LogicOp.NEQ, is_val=5, use_api="f"))
  assert _check_rule_condition(cst.Integer("6"), Rule(if_arg="x", op=LogicOp.GT, is_val=5, use_api="f"))
  assert not _check_rule_condition(cst.Integer("4"), Rule(if_arg="x", op=LogicOp.GT, is_val=5, use_api="f"))
  assert _check_rule_condition(cst.Integer("4"), Rule(if_arg="x", op=LogicOp.LT, is_val=5, use_api="f"))
  assert not _check_rule_condition(cst.Integer("6"), Rule(if_arg="x", op=LogicOp.LT, is_val=5, use_api="f"))
  assert _check_rule_condition(cst.Integer("5"), Rule(if_arg="x", op=LogicOp.GTE, is_val=5, use_api="f"))
  assert not _check_rule_condition(cst.Integer("4"), Rule(if_arg="x", op=LogicOp.GTE, is_val=5, use_api="f"))
  assert _check_rule_condition(cst.Integer("5"), Rule(if_arg="x", op=LogicOp.LTE, is_val=5, use_api="f"))
  assert not _check_rule_condition(cst.Integer("6"), Rule(if_arg="x", op=LogicOp.LTE, is_val=5, use_api="f"))
  assert _check_rule_condition(cst.Integer("2"), Rule(if_arg="x", op=LogicOp.IN, is_val=[1, 2, 3], use_api="f"))
  assert not _check_rule_condition(cst.Integer("4"), Rule(if_arg="x", op=LogicOp.IN, is_val=[1, 2, 3], use_api="f"))
  assert _check_rule_condition(cst.Integer("4"), Rule(if_arg="x", op=LogicOp.NOT_IN, is_val=[1, 2, 3], use_api="f"))
  assert not _check_rule_condition(cst.Integer("2"), Rule(if_arg="x", op=LogicOp.NOT_IN, is_val=[1, 2, 3], use_api="f"))
  assert not _check_rule_condition(cst.Name("var"), Rule(if_arg="x", op=LogicOp.EQ, is_val=5, use_api="f"))

  class _UnknownOpRule:
    """Rule stub carrying an unrecognized `op` string."""

    if_arg = "x"
    op = "UNKNOWN_OP"
    is_val = 5
    use_api = "f"

  assert _check_rule_condition(cst.Integer("6"), _UnknownOpRule()) is False
  assert _check_rule_condition(cst.Integer("6"), "not_a_rule") is False


def test_extract_argument_node_and_evaluate_dispatch_rules() -> None:
  """Verify keyword, positional, receiver-method extraction, and full `evaluate_dispatch_rules` matching."""
  rewriter = _DispatchRewriterStub("torch", module_alias=False)

  class _PlainDispatchHost:
    """Host without `_is_module_alias`."""

  assert _extract_argument_node(_PlainDispatchHost(), _parse_call("func(a=1)"), "other", "a", ["a"]) is None

  kw_node = _extract_argument_node(rewriter, _parse_call("func(a=1, b=2)"), "b", "b", ["a", "b"])
  assert isinstance(kw_node, cst.Integer)
  assert kw_node.value == "2"

  pos_node = _extract_argument_node(rewriter, _parse_call("func(1, 2)"), "b", "b", ["a", "b"])
  assert isinstance(pos_node, cst.Integer)
  assert pos_node.value == "2"

  meth_node = _extract_argument_node(rewriter, _parse_call("obj.func(2)"), "b", "b", ["x", "b"])
  assert isinstance(meth_node, cst.Integer)
  assert meth_node.value == "2"

  assert _extract_argument_node(rewriter, _parse_call("func(1)"), "b", "b", ["a", "b"]) is None
  assert _extract_argument_node(rewriter, _parse_call("func(1)"), "c", "c", ["a", "b"]) is None

  rules: list[Rule | JsonValue] = [
    "non_rule_entry",
    Rule(if_arg="missing", op=LogicOp.EQ, is_val="x", use_api="miss_func"),
    Rule(if_arg="mode", op=LogicOp.EQ, is_val="fast", use_api="fast_func"),
    Rule(if_arg="mode", op=LogicOp.EQ, is_val="slow", use_api="slow_func"),
  ]
  details: dict[str, JsonValue] = {
    "variants": {"torch": {"args": {"mode": "src_mode"}}},
    "std_args": [["data", "Tensor"], {}, {"name": "mode"}, "extra"],
  }
  assert evaluate_dispatch_rules(rewriter, _parse_call("func(src_mode='fast')"), rules, details) == "fast_func"
  assert evaluate_dispatch_rules(rewriter, _parse_call("func(src_mode='slow')"), rules, details) == "slow_func"
  assert evaluate_dispatch_rules(rewriter, _parse_call("func(src_mode='other')"), rules, details) is None
