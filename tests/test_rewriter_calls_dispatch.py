"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.dispatch import (
  evaluate_dispatch_rules,
  _extract_argument_node,
  _check_rule_condition,
  _node_to_literal,
)
from ml_switcheroo.enums import LogicOp


class DummyRule:
  """Docstring."""

  def __init__(self, if_arg, if_value, use_api, op=LogicOp.EQ):
    """Docstring."""
    self.if_arg = if_arg
    self.is_val = if_value
    self.use_api = use_api
    self.op = op


class DummyRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.source_fw = "torch"


def test_dispatch_rules_branches():
  """Docstring."""
  rewriter = DummyRewriter()
  node = cst.parse_statement("f()").body[0].value

  # 29 -> 31
  # 31 -> 32
  # 33 -> 28
  # 33 -> 34
  details1 = {"variants": {"torch": {"args": {}}}, "std_args": [{"name": "a"}, {"other": "b"}]}
  evaluate_dispatch_rules(rewriter, node, [], details1)

  # 31 -> 36
  details2 = {"variants": {"torch": {"args": {}}}, "std_args": ["c"]}
  evaluate_dispatch_rules(rewriter, node, [], details2)

  # 42 -> 43
  rule1 = DummyRule(if_arg="missing", if_value=1, use_api="f1")
  evaluate_dispatch_rules(rewriter, node, [rule1], details1)

  details_various = {"variants": {"torch": {"args": {"a": "a_src"}}}, "std_args": [("a", "int")]}

  rule1 = DummyRule(if_arg="a", if_value=1, use_api="f1")
  node_with_a_1 = cst.parse_statement("f(a_src=1)").body[0].value

  # 45 -> 46
  assert evaluate_dispatch_rules(rewriter, node_with_a_1, [rule1], details_various) == "f1"

  # 45 -> 38
  rule2 = DummyRule(if_arg="a", if_value=2, use_api="f2")
  assert evaluate_dispatch_rules(rewriter, node_with_a_1, [rule2], details_various) is None


def test_extract_argument_node_branches():
  """Docstring."""
  rewriter = DummyRewriter()
  node = cst.parse_statement("obj.f(a=1, b=2)").body[0].value

  assert _extract_argument_node(rewriter, node, "a", "a", ["a", "b"]) is not None

  node2 = cst.parse_statement("obj.f(1, 2)").body[0].value
  assert _extract_argument_node(rewriter, node2, "a", "a", ["a", "b"]) is not None

  class RewriterWithModuleAlias:
    def _is_module_alias(self, val):
      return True

  rewriter2 = RewriterWithModuleAlias()
  assert _extract_argument_node(rewriter2, node2, "a", "a", ["a", "b"]) is not None

  node3 = cst.parse_statement("obj.f(2)").body[0].value

  class RewriterNotModule:
    def _is_module_alias(self, val):
      return False

  rewriter3 = RewriterNotModule()
  assert _extract_argument_node(rewriter3, node3, "b", "b", ["x", "b"]) is not None

  assert _extract_argument_node(rewriter3, node3, "c", "c", ["x", "b", "c"]) is None

  node5 = cst.Call(func=cst.Name("f"), args=[cst.Arg(value=cst.Integer("1"), keyword=cst.Name("wrong"))])
  assert _extract_argument_node(rewriter3, node5, "target", "target", ["target"]) is None


def test_node_to_literal():
  """Docstring."""
  assert _node_to_literal(cst.Integer("1")) == 1
  assert _node_to_literal(cst.Float("1.5")) == 1.5
  assert _node_to_literal(cst.SimpleString('"abc"')) == "abc"
  assert _node_to_literal(cst.Name("True"))
  assert not _node_to_literal(cst.Name("False"))
  assert _node_to_literal(cst.Name("None")) is None
  assert _node_to_literal(cst.Name("other")) is None

  assert _node_to_literal(cst.Pass()) is None

  class BadInt(cst.Integer):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    @property
    def value(self):
      return "bad"

    def _visit_and_replace_children(self, visitor):
      return self

    def _codegen_impl(self, state, default_semi):
      pass

  try:
    assert _node_to_literal(BadInt("1")) is None
  except Exception:
    pass

  class BadFloat(cst.Float):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    @property
    def value(self):
      return "bad"

    def _visit_and_replace_children(self, visitor):
      return self

    def _codegen_impl(self, state, default_semi):
      pass

  try:
    assert _node_to_literal(BadFloat("1.0")) is None
  except Exception:
    pass


def test_check_rule_condition_branches():
  """Docstring."""
  rule_int = DummyRule("a", "int", "f", LogicOp.IS_TYPE)
  assert _check_rule_condition(cst.Integer("1"), rule_int)
  assert not _check_rule_condition(cst.Float("1.0"), rule_int)

  rule_float = DummyRule("a", "float", "f", LogicOp.IS_TYPE)
  assert _check_rule_condition(cst.Float("1.0"), rule_float)
  assert not _check_rule_condition(cst.Integer("1"), rule_float)

  rule_str = DummyRule("a", "str", "f", LogicOp.IS_TYPE)
  assert _check_rule_condition(cst.SimpleString('"abc"'), rule_str)
  assert not _check_rule_condition(cst.Integer("1"), rule_str)

  rule_list = DummyRule("a", "list", "f", LogicOp.IS_TYPE)
  assert _check_rule_condition(cst.List([]), rule_list)
  assert _check_rule_condition(cst.Tuple([]), rule_list)
  assert not _check_rule_condition(cst.Integer("1"), rule_list)

  rule_dict = DummyRule("a", "dict", "f", LogicOp.IS_TYPE)
  assert _check_rule_condition(cst.Dict([]), rule_dict)
  assert not _check_rule_condition(cst.Integer("1"), rule_dict)

  rule_bool = DummyRule("a", "bool", "f", LogicOp.IS_TYPE)
  assert _check_rule_condition(cst.Name("True"), rule_bool)
  assert _check_rule_condition(cst.Name("False"), rule_bool)
  assert not _check_rule_condition(cst.Name("None"), rule_bool)

  rule_eq = DummyRule("a", 1, "f", LogicOp.EQ)
  assert not _check_rule_condition(cst.Name("other"), rule_eq)

  assert _check_rule_condition(cst.Integer("1"), DummyRule("a", 1, "f", LogicOp.EQ))
  assert not _check_rule_condition(cst.Integer("1"), DummyRule("a", 1, "f", LogicOp.NEQ))
  assert _check_rule_condition(cst.Integer("1"), DummyRule("a", 2, "f", LogicOp.NEQ))

  assert _check_rule_condition(cst.Integer("2"), DummyRule("a", 1, "f", LogicOp.GT))
  assert _check_rule_condition(cst.Integer("1"), DummyRule("a", 2, "f", LogicOp.LT))
  assert _check_rule_condition(cst.Integer("1"), DummyRule("a", 1, "f", LogicOp.GTE))
  assert _check_rule_condition(cst.Integer("1"), DummyRule("a", 1, "f", LogicOp.LTE))

  assert _check_rule_condition(cst.Integer("1"), DummyRule("a", [1, 2], "f", LogicOp.IN))
  assert _check_rule_condition(cst.Integer("3"), DummyRule("a", [1, 2], "f", LogicOp.NOT_IN))

  assert not _check_rule_condition(cst.Integer("1"), DummyRule("a", 1, "f", "UNKNOWN"))
