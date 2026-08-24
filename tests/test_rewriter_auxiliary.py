"""Test module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.hooks_registry import register_hook


@pytest.fixture(autouse=True)
def clean_hooks():
  """Test element."""
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


@pytest.fixture
def context():
  """Test element."""
  semantics = MagicMock()
  conf = {"traits": {"functional_execution_method": "apply"}}
  semantics.get_framework_config.return_value = conf

  config = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"

  ctx = RewriterContext(semantics=semantics, config=config)
  ctx.alias_map = {"t": "torch", "nn": "torch.nn"}
  return ctx


def test_auxiliary_pass(context):
  """Test element."""
  module = cst.parse_module("def foo():\n  pass\n")
  p = AuxiliaryPass()
  res = p.transform(module, context)
  assert res is not None


def test_aux_get_traits_cached(context):
  """Test element."""
  transformer = AuxiliaryTransformer(context)
  traits = transformer._get_traits()
  assert traits.functional_execution_method == "apply"

  # second call hits cache
  traits2 = transformer._get_traits()
  assert traits is traits2


def test_aux_get_traits_no_conf(context):
  """Test element."""
  context.semantics.get_framework_config.return_value = None
  transformer = AuxiliaryTransformer(context)
  traits = transformer._get_traits()
  assert traits is not None


def test_aux_get_qualified_name(context):
  """Test element."""
  transformer = AuxiliaryTransformer(context)

  node = cst.parse_expression("t.Tensor")
  assert transformer._get_qualified_name(node) == "torch.Tensor"

  node = cst.parse_expression("unknown.Tensor")
  assert transformer._get_qualified_name(node) == "unknown.Tensor"

  node = cst.parse_expression("t")
  assert transformer._get_qualified_name(node) == "torch"

  node = cst.parse_statement("a = 1")
  assert transformer._get_qualified_name(node) is None


def test_aux_create_dotted_name(context):
  """Test element."""
  transformer = AuxiliaryTransformer(context)
  node = transformer._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"
  assert node.value.attr.value == "b"
  assert node.value.value.value == "a"

  node = transformer._create_dotted_name("foo")
  assert isinstance(node, cst.Name)
  assert node.value == "foo"


def test_aux_report(context):
  """Test element."""
  transformer = AuxiliaryTransformer(context)
  transformer._report_failure("error1")
  transformer._report_warning("warn1")
  assert "error1" in context.current_stmt_errors
  assert "warn1" in context.current_stmt_warnings

  stmt = cst.parse_statement("a = 1")
  # Both warnings and errors
  res = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res != stmt
  # Can't assert isinstance SimpleStatementLine due to flatten sentinel


def test_aux_report_warning_only(context):
  """Test element."""
  transformer = AuxiliaryTransformer(context)
  transformer._report_warning("warn1")
  transformer._report_warning("warn1")  # duplicate
  stmt = cst.parse_statement("a = 1")
  res = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res != stmt


def test_aux_decorator(context):
  """Test element."""
  context.semantics.get_definition.return_value = ("abs_id", {"variants": {"jax": {"api": "jax.jit"}}})

  transformer = AuxiliaryTransformer(context)
  mod = cst.parse_module("@t.jit\ndef foo(): pass")
  decorator = mod.body[0].decorators[0]

  res = transformer.leave_Decorator(decorator, decorator)
  assert isinstance(res, cst.Decorator)

  # decorator with call
  mod2 = cst.parse_module("@t.jit(1)\ndef foo(): pass")
  decorator_call = mod2.body[0].decorators[0]
  res_call = transformer.leave_Decorator(decorator_call, decorator_call)
  assert isinstance(res_call, cst.Decorator)

  # decorator removal
  context.semantics.get_definition.return_value = ("abs_id", {"variants": {"jax": None}})
  res_rem = transformer.leave_Decorator(decorator, decorator)
  assert res_rem == cst.RemoveFromParent()

  # target fw not in variants -> no cover skipped, but let's test if we can
  context.semantics.get_definition.return_value = ("abs_id", {"variants": {"other": None}})
  transformer.leave_Decorator(decorator, decorator)
  # wait, target fw not in variants returns original node.

  # None definition -> returns original node


def test_aux_for_loop(context):
  """Test element."""
  transformer = AuxiliaryTransformer(context)
  mod = cst.parse_module("for i in range(10): pass")
  for_node = mod.body[0]

  # Test without hooks
  res = transformer.leave_For(for_node, for_node)
  assert res is for_node

  # Test static unroll success
  @register_hook("transform_for_loop_static")
  def mock_static(node, ctx):
    return cst.parse_statement("i = 0")

  res2 = transformer.leave_For(for_node, for_node)
  assert res2 is not for_node

  # Test static unroll fail
  import ml_switcheroo.core.hooks_registry as hr

  hr._HOOKS.clear()

  @register_hook("transform_for_loop_static")
  def mock_static_fail(node, ctx):
    raise ValueError("static fail")

  res3 = transformer.leave_For(for_node, for_node)
  assert res3 is for_node
  assert len(context.current_stmt_warnings) == 1

  # Test normal loop transform success
  hr._HOOKS.clear()

  @register_hook("transform_for_loop")
  def mock_loop(node, ctx):
    return cst.parse_statement("pass")

  res4 = transformer.leave_For(for_node, for_node)
  assert res4 is not for_node

  # Test normal loop transform fail
  hr._HOOKS.clear()

  @register_hook("transform_for_loop")
  def mock_loop_fail(node, ctx):
    raise ValueError("loop fail")

  res5 = transformer.leave_For(for_node, for_node)
  # wraps in escape hatch failure
  assert res5 is not for_node
