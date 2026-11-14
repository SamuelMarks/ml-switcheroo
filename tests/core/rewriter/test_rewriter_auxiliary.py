"""Test suite for rewriter auxiliary."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks_registry import register_hook
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer


@pytest.fixture(autouse=True)
def clean_hooks() -> typing.Generator[None, None, None]:
  """Create a configured `typing.Generator[None, None, None]` fixture for rewriter auxiliary tests.

  Yields:
      Configured `clean_hooks` fixture value (typing.Generator[None, None, None]).
  """
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks(mark_loaded=True)
  yield
  hr.clear_hooks()


@pytest.fixture
def context() -> RewriterContext:
  """Create a configured `RewriterContext` fixture for rewriter auxiliary tests.

  Returns:
      Configured `context` fixture value (RewriterContext).
  """
  semantics: MagicMock = MagicMock()
  conf = {"traits": {"functional_execution_method": "apply"}}
  semantics.get_framework_config.return_value = conf

  config: MagicMock = MagicMock()
  config.effective_source = "torch"
  config.effective_target = "jax"

  ctx: RewriterContext = RewriterContext(semantics=semantics, config=config)
  ctx.alias_map = {"t": "torch", "nn": "torch.nn"}
  return ctx


def test_auxiliary_pass(context: RewriterContext) -> None:
  """Verify `AuxiliaryPass` handles auxiliary pass as expected.

  Args:
      context: Execution or rewriter context instance.
  """
  module: cst.Module = cst.parse_module("def foo():\n  pass\n")
  p: AuxiliaryPass = AuxiliaryPass()
  res: cst.Module = p.transform(module, context)
  assert isinstance(res, cst.Module)
  assert len(res.body) == 1
  assert res.code == "def foo():\n  pass\n"


def test_aux_get_traits_cached(context: RewriterContext) -> None:
  """Verify `AuxiliaryTransformer` handles aux get traits cached as expected.

  Args:
      context: Execution or rewriter context instance.
  """
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  traits = transformer._get_traits()
  assert traits.functional_execution_method == "apply"

  # second call hits cache
  traits2 = transformer._get_traits()
  assert traits is traits2


def test_aux_get_traits_no_conf(context: RewriterContext, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `AuxiliaryTransformer` handles aux get traits no conf as expected.

  Args:
      context: Execution or rewriter context instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(context.semantics.get_framework_config, "return_value", None, raising=False)
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  traits = transformer._get_traits()
  assert traits.functional_execution_method is None
  assert traits.init_method_name == "__init__"
  assert traits.requires_super_init is False


def test_aux_get_qualified_name(context: RewriterContext) -> None:
  """Verify `AuxiliaryTransformer` handles aux get qualified name as expected.

  Args:
      context: Execution or rewriter context instance.
  """
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)

  node: cst.BaseExpression = cst.parse_expression("t.Tensor")
  assert transformer._get_qualified_name(node) == "torch.Tensor"

  node2: cst.BaseExpression = cst.parse_expression("unknown.Tensor")
  assert transformer._get_qualified_name(node2) == "unknown.Tensor"

  node3: cst.BaseExpression = cst.parse_expression("t")
  assert transformer._get_qualified_name(node3) == "torch"

  node4: cst.BaseExpression = cst.parse_expression("1")
  assert transformer._get_qualified_name(node4) is None


def test_aux_create_dotted_name(context: RewriterContext) -> None:
  """Verify `AuxiliaryTransformer` handles aux create dotted name as expected.

  Args:
      context: Execution or rewriter context instance.
  """
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
  """Verify `AuxiliaryTransformer` emits `error1` for aux report.

  Args:
      context: Execution or rewriter context instance.
  """
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  transformer._report_failure("error1")
  transformer._report_warning("warn1")
  assert "error1" in context.current_stmt_errors
  assert "warn1" in context.current_stmt_warnings
  _raw_stmt = cst.parse_module("a = 1").body[0]
  assert isinstance(_raw_stmt, cst.SimpleStatementLine)

  stmt: cst.SimpleStatementLine = _raw_stmt
  # Both warnings and errors
  res = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res != stmt
  # Can't assert isinstance SimpleStatementLine due to flatten sentinel


def test_aux_report_warning_only(context: RewriterContext) -> None:
  """Verify `AuxiliaryTransformer` handles aux report warning only as expected.

  Args:
      context: Execution or rewriter context instance.
  """
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  transformer._report_warning("warn1")
  transformer._report_warning("warn1")  # duplicate
  _raw_stmt = cst.parse_module("a = 1").body[0]
  assert isinstance(_raw_stmt, cst.SimpleStatementLine)
  stmt: cst.SimpleStatementLine = _raw_stmt
  res = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res != stmt


def test_aux_decorator(context: RewriterContext, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `AuxiliaryTransformer` handles aux decorator as expected.

  Args:
      context: Execution or rewriter context instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(
    context.semantics.get_definition,
    "return_value",
    (
      "abs_id",
      {"variants": {"jax": {"api": "jax.jit"}}},
    ),
    raising=False,
  )

  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  mod: cst.Module = cst.parse_module("@t.jit\ndef foo(): pass")
  decorator: cst.Decorator = cst.ensure_type(mod.body[0], cst.FunctionDef).decorators[0]

  res: cst.Decorator | cst.RemovalSentinel = transformer.leave_Decorator(decorator, decorator)
  assert isinstance(res, cst.Decorator)
  assert cst.Module([]).code_for_node(res.decorator) == "jax.jit"

  # decorator with call
  mod2: cst.Module = cst.parse_module("@t.jit(1)\ndef foo(): pass")
  decorator_call: cst.Decorator = cst.ensure_type(mod2.body[0], cst.FunctionDef).decorators[0]
  res_call: cst.Decorator | cst.RemovalSentinel = transformer.leave_Decorator(decorator_call, decorator_call)
  assert isinstance(res_call, cst.Decorator)
  assert cst.Module([]).code_for_node(res_call.decorator) == "jax.jit(1)"

  # decorator removal
  monkeypatch.setattr(
    context.semantics.get_definition,
    "return_value",
    (
      "abs_id",
      {"variants": {"jax": None}},
    ),
    raising=False,
  )
  res_rem: cst.Decorator | cst.RemovalSentinel = transformer.leave_Decorator(decorator, decorator)
  assert isinstance(res_rem, cst.RemovalSentinel)
  assert res_rem == cst.RemovalSentinel.REMOVE

  # target fw not in variants -> no cover skipped, but let's test if we can
  monkeypatch.setattr(
    context.semantics.get_definition,
    "return_value",
    (
      "abs_id",
      {"variants": {"other": None}},
    ),
    raising=False,
  )
  res_unchanged: cst.Decorator | cst.RemovalSentinel = transformer.leave_Decorator(decorator, decorator)
  assert res_unchanged is decorator
  # wait, target fw not in variants returns original node.

  # None definition -> returns original node


def test_aux_for_loop(context: RewriterContext) -> None:
  """Verify `AuxiliaryTransformer` returns `1` for aux for loop.

  Args:
      context: Execution or rewriter context instance.
  """
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  mod: cst.Module = cst.parse_module("for i in range(10): pass")
  for_node: cst.For = cst.ensure_type(mod.body[0], cst.For)

  # Test without hooks
  res = transformer.leave_For(for_node, for_node)
  assert res is for_node

  # Test static unroll success
  @register_hook("transform_for_loop_static")
  def mock_static(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Execute `register_hook` to support mock static in rewriter auxiliary tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext active during transformation.

    Returns:
        Resulting `cst.CSTNode` from mock static.
    """
    return cst.parse_statement("i = 0")

  res2 = transformer.leave_For(for_node, for_node)
  assert res2 is not for_node

  # Test static unroll fail
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks(mark_loaded=True)

  @register_hook("transform_for_loop_static")
  def mock_static_fail(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Execute `register_hook` to support mock static fail in rewriter auxiliary tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext active during transformation.

    Returns:
        Resulting `cst.CSTNode` from mock static fail.
    """
    raise ValueError("static fail")

  res3 = transformer.leave_For(for_node, for_node)
  assert res3 is for_node
  assert len(context.current_stmt_warnings) == 1

  # Test normal loop transform success
  hr.clear_hooks(mark_loaded=True)

  @register_hook("transform_for_loop")
  def mock_loop(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Execute `register_hook` to support mock loop in rewriter auxiliary tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext active during transformation.

    Returns:
        Resulting `cst.CSTNode` from mock loop.
    """
    return cst.parse_statement("pass")

  res4 = transformer.leave_For(for_node, for_node)
  assert res4 is not for_node

  # Test normal loop transform fail
  hr.clear_hooks(mark_loaded=True)

  @register_hook("transform_for_loop")
  def mock_loop_fail(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Execute `register_hook` to support mock loop fail in rewriter auxiliary tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext active during transformation.

    Returns:
        Resulting `cst.CSTNode` from mock loop fail.
    """
    raise ValueError("loop fail")

  res5 = transformer.leave_For(for_node, for_node)
  # wraps in escape hatch failure
  assert res5 is not for_node


def test_aux_cst_to_string_attribute_with_non_convertible_base(context: RewriterContext) -> None:
  """Test _cst_to_string when the base of an Attribute cannot be flattened.

  Args:
      context: Execution or rewriter context instance.
  """
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  node = cst.Attribute(value=cst.Call(func=cst.Name("fn")), attr=cst.Name("member"))
  assert transformer._cst_to_string(node) is None


def test_aux_report_errors_only(context: RewriterContext) -> None:
  """Test leave_SimpleStatementLine when only errors are present in current_stmt_errors.

  Args:
      context: Execution or rewriter context instance.
  """
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  transformer._report_failure("error_only_1")
  transformer._report_failure("error_only_1")  # duplicate to test deduplication
  _raw_stmt = cst.parse_module("b = 2").body[0]
  assert isinstance(_raw_stmt, cst.SimpleStatementLine)
  stmt: cst.SimpleStatementLine = _raw_stmt
  res = transformer.leave_SimpleStatementLine(stmt, stmt)
  assert res != stmt


def test_aux_decorator_definition_not_found(context: RewriterContext, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test leave_Decorator when semantics definition is None.

  Args:
      context: Execution or rewriter context instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(context.semantics.get_definition, "return_value", None, raising=False)
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  mod: cst.Module = cst.parse_module("@t.jit\ndef foo(): pass")
  dec: cst.Decorator = cst.ensure_type(mod.body[0], cst.FunctionDef).decorators[0]
  res = transformer.leave_Decorator(dec, dec)
  assert res is dec


def test_aux_decorator_variant_without_api(context: RewriterContext, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test leave_Decorator when target variant exists but contains no target api.

  Args:
      context: Execution or rewriter context instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(
    context.semantics.get_definition,
    "return_value",
    (
      "abs_id",
      {"variants": {"jax": {}}},
    ),
    raising=False,
  )
  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  mod: cst.Module = cst.parse_module("@t.jit\ndef foo(): pass")
  dec: cst.Decorator = cst.ensure_type(mod.body[0], cst.FunctionDef).decorators[0]
  res = transformer.leave_Decorator(dec, dec)
  assert res is dec


def test_aux_for_loop_noops_and_non_statement_escape_hatch(
  context: RewriterContext, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Test leave_For when hooks return unchanged node, and when EscapeHatch returns non-statement.

  Args:
      context: Execution or rewriter context instance.
      monkeypatch: Pytest monkeypatch fixture for patching attributes and environment.
  """
  import ml_switcheroo.core.hooks_registry as hr
  from ml_switcheroo.core.escape_hatch import EscapeHatch

  transformer: AuxiliaryTransformer = AuxiliaryTransformer(context)
  mod: cst.Module = cst.parse_module("for i in range(5): pass")
  for_node: cst.For = cst.ensure_type(mod.body[0], cst.For)

  # 1. Both static hook and normal hook return updated_node (no-op)
  hr.clear_hooks(mark_loaded=True)

  @register_hook("transform_for_loop_static")
  def mock_static_noop(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Noop static loop hook.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext active during transformation.

    Returns:
        Resulting `cst.CSTNode` from mock static noop.
    """
    return node

  @register_hook("transform_for_loop")
  def mock_loop_noop(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Noop loop hook.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext active during transformation.

    Returns:
        Resulting `cst.CSTNode` from mock loop noop.
    """
    return node

  res_noop = transformer.leave_For(for_node, for_node)
  assert res_noop is for_node

  # 2. Loop hook raises, and EscapeHatch.mark_failure returns an expression (not BaseStatement/FlattenSentinel)
  hr.clear_hooks(mark_loaded=True)

  @register_hook("transform_for_loop")
  def mock_loop_err(node: cst.CSTNode, ctx: RewriterContext) -> cst.CSTNode:
    """Error loop hook.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext active during transformation.

    Returns:
        Resulting `cst.CSTNode` from mock loop err.
    """
    raise RuntimeError("test error")

  monkeypatch.setattr(EscapeHatch, "mark_failure", lambda node, reason: cst.Name("expr_not_stmt"))
  res_err = transformer.leave_For(for_node, for_node)
  assert res_err is for_node
