"""Test module."""

from typing import Any, Generator, Union
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks_registry import register_hook
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer


@pytest.fixture(autouse=True)
def clean_hooks() -> Generator[None, None, None]:
  """Docstring."""
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


@pytest.fixture
def context() -> RewriterContext:
  """Docstring."""
  semantics: MagicMock = MagicMock()
  conf: dict = {"traits": {"functional_execution_method": "apply"}}
  semantics.get_framework_config.return_value = conf

  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"

  ctx: RewriterContext = RewriterContext(semantics=semantics, config=config)
  ctx.alias_map = {"t": "torch", "nn": "torch.nn"}
  return ctx


def test_auxiliary_pass(context: RewriterContext) -> None:
  """Docstring."""
  module: cst.Module = cst.parse_module("def foo():\n  pass\n")
  p: AuxiliaryPass = AuxiliaryPass()
  res: cst.Module = p.transform(module, context)
  assert res is not None


def test_aux_get_traits_cached(context: RewriterContext) -> None:
  """Docstring."""
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  traits: Any = transformer._get_traits()
  assert getattr(traits, "functional_execution_method", None) == "apply"

  # second call hits cache
  traits2: Any = transformer._get_traits()
  assert traits is traits2


def test_aux_get_traits_no_conf(context: RewriterContext) -> None:
  """Docstring."""
  context.semantics.get_framework_config.return_value = None
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  traits: Any = transformer._get_traits()
  assert traits is not None


def test_aux_get_qualified_name(context: RewriterContext) -> None:
  """Docstring."""
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)

  node: cst.BaseExpression = cst.parse_expression("t.Tensor")
  assert transformer._get_qualified_name(node) == "torch.Tensor"

  node2: cst.BaseExpression = cst.parse_expression("unknown.Tensor")
  assert transformer._get_qualified_name(node2) == "unknown.Tensor"

  node3: cst.BaseExpression = cst.parse_expression("t")
  assert transformer._get_qualified_name(node3) == "torch"

  node4: cst.SimpleStatementLine = getattr(cst.parse_module("a = 1"), "body")[0]
  assert transformer._get_qualified_name(node4) is None


def test_aux_create_dotted_name(context: RewriterContext) -> None:
  """Docstring."""
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  node: cst.BaseExpression = transformer._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "b"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "a"

  node2: cst.BaseExpression = transformer._create_dotted_name("foo")
  assert isinstance(node2, cst.Name)
  assert node2.value == "foo"


def test_aux_report(context: RewriterContext) -> None:
  """Docstring."""
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  transformer._report_failure("error1")
  transformer._report_warning("warn1")
  assert "error1" in context.current_stmt_errors
  assert "warn1" in context.current_stmt_warnings

  stmt: cst.SimpleStatementLine = getattr(cst.parse_module("a = 1"), "body")[0]
  # Both warnings and errors
  res: Union[cst.CSTNode, cst.FlattenSentinel, cst.RemovalSentinel] = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res != stmt
  # Can't assert isinstance SimpleStatementLine due to flatten sentinel


def test_aux_report_warning_only(context: RewriterContext) -> None:
  """Docstring."""
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  transformer._report_warning("warn1")
  transformer._report_warning("warn1")  # duplicate
  stmt: cst.SimpleStatementLine = getattr(cst.parse_module("a = 1"), "body")[0]
  res: Union[cst.CSTNode, cst.FlattenSentinel, cst.RemovalSentinel] = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res != stmt


def test_aux_decorator(context: RewriterContext) -> None:
  """Docstring."""
  context.semantics.get_definition.return_value = ("abs_id", {"variants": {"jax": {"api": "jax.jit"}}})

  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  mod: cst.Module = cst.parse_module("@t.jit\ndef foo(): pass")
  decorator: cst.Decorator = getattr(getattr(mod, "body")[0], "decorators")[0]

  res: Union[cst.Decorator, cst.RemovalSentinel] = transformer.leave_Decorator(decorator, decorator)
  assert isinstance(res, cst.Decorator)

  # decorator with call
  mod2: cst.Module = cst.parse_module("@t.jit(1)\ndef foo(): pass")
  decorator_call: cst.Decorator = getattr(getattr(mod2, "body")[0], "decorators")[0]
  res_call: Union[cst.Decorator, cst.RemovalSentinel] = transformer.leave_Decorator(decorator_call, decorator_call)
  assert isinstance(res_call, cst.Decorator)

  # decorator removal
  context.semantics.get_definition.return_value = ("abs_id", {"variants": {"jax": None}})
  res_rem: Union[cst.Decorator, cst.RemovalSentinel] = transformer.leave_Decorator(decorator, decorator)
  assert isinstance(res_rem, cst.RemovalSentinel)

  # target fw not in variants -> no cover skipped, but let's test if we can
  context.semantics.get_definition.return_value = ("abs_id", {"variants": {"other": None}})
  transformer.leave_Decorator(decorator, decorator)
  # wait, target fw not in variants returns original node.

  # None definition -> returns original node


def test_aux_for_loop(context: RewriterContext) -> None:
  """Docstring."""
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  mod: cst.Module = cst.parse_module("for i in range(10): pass")
  for_node: cst.For = getattr(mod, "body")[0]

  # Test without hooks
  res: Union[cst.BaseStatement, cst.FlattenSentinel, cst.RemovalSentinel] = transformer.leave_For(for_node, for_node)
  assert res is for_node

  # Test static unroll success
  @register_hook("transform_for_loop_static")
  def mock_static(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Docstring."""
    return cst.parse_statement("i = 0")

  res2: Union[cst.BaseStatement, cst.FlattenSentinel, cst.RemovalSentinel] = transformer.leave_For(for_node, for_node)
  assert res2 is not for_node

  # Test static unroll fail
  import ml_switcheroo.core.hooks_registry as hr

  hr._HOOKS.clear()

  @register_hook("transform_for_loop_static")
  def mock_static_fail(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Docstring."""
    raise ValueError("static fail")

  res3: Union[cst.BaseStatement, cst.FlattenSentinel, cst.RemovalSentinel] = transformer.leave_For(for_node, for_node)
  assert res3 is for_node
  assert len(context.current_stmt_warnings) == 1

  # Test normal loop transform success
  hr._HOOKS.clear()

  @register_hook("transform_for_loop")
  def mock_loop(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Docstring."""
    return cst.parse_statement("pass")

  res4: Union[cst.BaseStatement, cst.FlattenSentinel, cst.RemovalSentinel] = transformer.leave_For(for_node, for_node)
  assert res4 is not for_node

  # Test normal loop transform fail
  hr._HOOKS.clear()

  @register_hook("transform_for_loop")
  def mock_loop_fail(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Docstring."""
    raise ValueError("loop fail")

  res5: Union[cst.BaseStatement, cst.FlattenSentinel, cst.RemovalSentinel] = transformer.leave_For(for_node, for_node)
  # wraps in escape hatch failure
  assert res5 is not for_node
