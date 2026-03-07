"""Module docstring."""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.rewriter.calls.dispatch import (
  evaluate_dispatch_rules,
  _extract_argument_node,
  _node_to_literal,
  _check_rule_condition,
)
from ml_switcheroo.enums import LogicOp

# --- pre.py tests ---


class MockTraits:
  """Class docstring."""

  def __init__(self, method="apply", implicit_roots=None):
    """Function docstring."""
    self.functional_execution_method = method
    self.implicit_method_roots = implicit_roots or []


class MockType:
  """Class docstring."""

  def __init__(self, name, framework=None):
    """Function docstring."""
    self.name = name
    if framework:
      self.framework = framework


class MockSymbolTable:
  """Class docstring."""

  def __init__(self, sym_type=None):
    """Function docstring."""
    self.sym_type = sym_type

  def get_type(self, node):
    """Function docstring."""
    return self.sym_type


class MockContext:
  """Class docstring."""

  def __init__(self, symbol_table=None):
    """Function docstring."""
    self.symbol_table = symbol_table
    self.hook_context = MagicMock()


class MockRewriterPre:
  """Class docstring."""

  def __init__(
    self,
    has_traits_prop=False,
    has_traits_meth=False,
    is_stateful_val=False,
    source_fw="src",
    target_fw="tgt",
    is_module_val=False,
    no_mapping=False,
  ):
    """Function docstring."""
    if has_traits_prop:
      self.source_traits = MockTraits()

    self._is_stateful_val = is_stateful_val
    self.source_fw = source_fw
    self.target_fw = target_fw
    self.semantics = MagicMock()
    self.context = MockContext()
    self.warnings = []
    self._is_module_val = is_module_val
    self.no_mapping = no_mapping

    if has_traits_meth:
      self._get_source_traits = lambda: MockTraits()

  def _get_mapping(self, name, silent=True):
    """Function docstring."""
    if self.no_mapping:
      return None
    if "requires_plugin" in name:
      return {"requires_plugin": "yes"}
    if "api_found" in name:
      return {"api": name}
    return None

  def _is_stateful(self, name):
    """Function docstring."""
    return self._is_stateful_val

  def _report_warning(self, msg):
    """Function docstring."""
    self.warnings.append(msg)

  def _get_source_lifecycle_lists(self):
    """Function docstring."""
    return {"strip_me"}, {"warn_me"}

  def _is_module_alias(self, node):
    """Function docstring."""
    return self._is_module_val


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_prop(mock_is_functional):
  """Function docstring."""
  rewriter = MockRewriterPre(has_traits_prop=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(
    func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")),
    args=[cst.Arg(value=cst.Name("vars")), cst.Arg(value=cst.Name("x"))],
  )

  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node.func, cst.Name)
  assert node.func.value == "layer"
  assert len(node.args) == 1
  assert node.args[0].value.value == "x"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_meth(mock_is_functional):
  """Function docstring."""
  rewriter = MockRewriterPre(has_traits_meth=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")), args=[])

  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert len(node.args) == 0


def test_handle_pre_checks_plugin_claim():
  """Function docstring."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.semantics.get_definition.return_value = None
  # Mapping with requires_plugin
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  handled, node = handle_pre_checks(rewriter, orig, updated, "requires_plugin_func")
  assert not handled
  assert node is updated


def test_handle_pre_checks_is_inplace_and_unroll():
  """Function docstring."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = (None, {"is_inplace": True})
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig

  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook") as mock_get_hook:
    mock_hook = MagicMock()
    mock_hook.return_value = cst.Name("unrolled")
    mock_get_hook.return_value = mock_hook
    handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
    assert handled
    assert isinstance(node, cst.Name)


def test_handle_pre_checks_endswith_underscore_unroll():
  """Function docstring."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Name("foo_"), args=[])
  updated = orig

  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook") as mock_get_hook:
    mock_hook = MagicMock()
    mock_hook.return_value = cst.Name("unrolled_")
    mock_get_hook.return_value = mock_hook
    handled, node = handle_pre_checks(rewriter, orig, updated, "foo_")
    assert handled
    assert isinstance(node, cst.Name)


def test_handle_pre_checks_lifecycle():
  """Function docstring."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = None

  # Strip
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"

  # Warn
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"


@patch("ml_switcheroo.core.rewriter.calls.pre.rewrite_stateful_call", return_value=cst.Name("stateful_rewritten"))
def test_handle_pre_checks_stateful(mock_rewrite):
  """Function docstring."""
  rewriter = MockRewriterPre(is_stateful_val=True, no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  rewriter.semantics.get_framework_config.return_value = {"stateful_call": {"method": "apply"}}
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)


def test_resolve_implicit_method_self():
  """Function docstring."""
  rewriter = MockRewriterPre()
  orig = cst.Call(func=cst.Attribute(value=cst.Name("self"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res is None


def test_resolve_implicit_method_module():
  """Function docstring."""
  rewriter = MockRewriterPre(is_module_val=True)
  orig = cst.Call(func=cst.Attribute(value=cst.Name("mod"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res is None


def test_resolve_implicit_method_sym_table():
  """Function docstring."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.context.symbol_table = MockSymbolTable(MockType("api_found"))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res == "api_found.meth"

  # Tensor special case
  rewriter.context.symbol_table = MockSymbolTable(MockType("Tensor", framework="fw"))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("api_found")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res == "fw.Tensor.api_found"


def test_resolve_implicit_method_legacy_fallback():
  """Function docstring."""
  rewriter = MockRewriterPre(no_mapping=False)
  # Remove context symbol table
  rewriter.context.symbol_table = None
  # Add _get_target_traits to trigger fallback
  rewriter._get_target_traits = MagicMock()
  rewriter.source_traits = MockTraits(implicit_roots=["api_found"])

  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res == "api_found.meth"

  # Fallback to source_fw config if source_traits missing
  del rewriter.source_traits
  rewriter.semantics.get_framework_config.return_value = {"traits": {"implicit_method_roots": ["api_found"]}}
  res2 = resolve_implicit_method(rewriter, orig, None)
  assert res2 == "api_found.meth"


# --- dispatch.py tests ---


class MockRule:
  """Class docstring."""

  def __init__(self, if_arg, op, is_val=None, use_api=None):
    """Function docstring."""
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api


class MockRewriterDispatch:
  """Class docstring."""

  def __init__(self, source_fw="src", is_module_val=False):
    """Function docstring."""
    self.source_fw = source_fw
    self._is_module_val = is_module_val

  def _is_module_alias(self, node):
    """Function docstring."""
    return self._is_module_val


def test_evaluate_dispatch_rules():
  """Function docstring."""
  rewriter = MockRewriterDispatch()
  # std_args raw parsing: tuple, dict, str
  details = {"variants": {"src": {"args": {"arg1": "src_arg1"}}}, "std_args": [("arg1",), {"name": "arg2"}, "arg3"]}
  rules = [
    MockRule("missing_arg", LogicOp.EQ, 1, "api_miss"),  # Hits arg_node is None -> continue
    MockRule("arg1", LogicOp.EQ, 1, "api1"),
    MockRule("arg2", LogicOp.EQ, 2, "api2"),
  ]
  node = cst.Call(func=cst.Name("foo"), args=[cst.Arg(value=cst.Integer("1")), cst.Arg(value=cst.Integer("2"))])

  res = evaluate_dispatch_rules(rewriter, node, rules, details)
  assert res == "api1"


def test_extract_argument_node_errors():
  """Function docstring."""
  rewriter = MockRewriterDispatch()
  node = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[cst.Arg(value=cst.Name("a"))])

  # Method offset std_order[0] == "x"
  res1 = _extract_argument_node(rewriter, node, "not_found", "a", ["x", "a"])
  assert res1.value == "a"

  # ValueError on index
  res2 = _extract_argument_node(rewriter, node, "not_found", "not_there", ["a"])
  assert res2 is None


def test_node_to_literal():
  """Function docstring."""
  # Int and Float parse errors returning None
  int_node = MagicMock(spec=cst.Integer)
  int_node.value = "abc"
  assert _node_to_literal(int_node) is None

  float_node = MagicMock(spec=cst.Float)
  float_node.value = "abc"
  assert _node_to_literal(float_node) is None

  # Names parsing True, False, None
  assert _node_to_literal(cst.Name("True")) is True
  assert _node_to_literal(cst.Name("False")) is False
  assert _node_to_literal(cst.Name("None")) is None
  assert _node_to_literal(cst.Name("Other")) is None


def test_check_rule_condition_is_type():
  """Function docstring."""
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
  """Function docstring."""
  # op matches none fallback
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
  """Function docstring."""
  rule = MockRule(None, LogicOp.EQ, 1)
  # _node_to_literal returns None for unparseable nodes, e.g., Name that isn't True/False/None
  assert _check_rule_condition(cst.Name("Unknown"), rule) is False
