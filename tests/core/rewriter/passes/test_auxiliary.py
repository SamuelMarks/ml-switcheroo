"""Test suite for the Auxiliary module."""

import typing
from unittest.mock import MagicMock, patch

import libcst as _cst
import pytest
from libcst.codemod import CodemodContext

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer
from ml_switcheroo.semantics.manager import SemanticsManager

cst: typing.Any = _cst


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: typing.Any = {}
    jit_def: typing.Any = {"variants": {"jax": {"api": "jax.jit"}, "torch": {"api": "torch.jit.script"}}}
    inf_def: typing.Any = {"variants": {"jax": None, "torch": {"api": "torch.inference_mode"}}}
    self._reverse_index: typing.Any = {
      "torch.jit.script": ("Jit", jit_def),
      "torch.inference_mode": ("InfMode", inf_def),
    }
    self.framework_configs: typing.Any = {}

  def get_definition(self, name: typing.Any) -> typing.Any:
    """Mock implementation of get definition.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._reverse_index.get(name)

  def get_framework_config(self, fw: typing.Any) -> typing.Any:
    """Mock implementation of get framework configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {}


@pytest.fixture
def run_pass() -> typing.Any:
  """Execute run pass helper logic.

  Returns:
      Computed test result or fixture value.
  """
  semantics: typing.Any = MockSemantics()
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx: typing.Any = RewriterContext(semantics, config)

  def _transform(code: typing.Any) -> typing.Any:
    """Helper to  transform.

    Args:
        code: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    module: typing.Any = cst.parse_module(code)
    aux_pass: typing.Any = AuxiliaryPass()
    return aux_pass.transform(module, ctx).code

  return _transform


@pytest.fixture(autouse=True)
def clean_hooks() -> typing.Any:
  """Helper to clean hooks.

  Returns:
      Computed test result or fixture value.
  """
  pass
  yield
  pass


def test_decorator_renaming(run_pass: typing.Any) -> None:
  """Verifies the behavior of decorator renaming.

  Args:
      run_pass: Test argument or fixture.
  """
  code: typing.Any = "\n@torch.jit.script\ndef f(): pass\n"
  res: typing.Any = run_pass(code)
  assert "@jax.jit" in res
  assert "@torch" not in res


def test_decorator_removal(run_pass: typing.Any) -> None:
  """Verifies the behavior of decorator removal.

  Args:
      run_pass: Test argument or fixture.
  """
  code: typing.Any = "\n@torch.inference_mode\ndef f(): pass\n"
  res: typing.Any = run_pass(code)
  assert "@torch" not in res
  assert "def f():" in res


def test_decorator_with_args(run_pass: typing.Any) -> None:
  """Verifies the behavior of decorator with arguments.

  Args:
      run_pass: Test argument or fixture.
  """
  code: typing.Any = "\n@torch.jit.script(optimize=True)\ndef f(): pass\n"
  res: typing.Any = run_pass(code)
  assert "@jax.jit(optimize=True)" in res


def test_loop_static_unroll_hook(run_pass: typing.Any) -> None:
  """Verifies the behavior of loop static unroll hook.

  Args:
      run_pass: Test argument or fixture.
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop_static")
  def mock_hook(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Execute mock hook helper logic.

    Args:
        node: Test argument or fixture.
        ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return cst.FlattenSentinel([cst.SimpleStatementLine([cst.Expr(cst.Name("unrolled"))])])

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_hook if name == "transform_for_loop_static" else None,
  ):
    code: typing.Any = "for i in range(2): pass"
    res: typing.Any = run_pass(code)
  print("RES:", res)
  assert "unrolled" in res
  assert "for" not in res


def test_loop_safety_hook(run_pass: typing.Any) -> None:
  """Verifies the behavior of loop safety hook.

  Args:
      run_pass: Test argument or fixture.
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def mock_safety(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Execute mock safety helper logic.

    Args:
        node: Test argument or fixture.
        ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return EscapeHatch.mark_failure(node, "Unsafe Loop")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_safety if name == "transform_for_loop" else None,
  ):
    code: typing.Any = "for i in range(N): pass"
    res: typing.Any = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "Unsafe Loop" in res


def test_loop_error_bubbling(run_pass: typing.Any) -> None:
  """Verifies the behavior of loop correctly handling an error bubbling.

  Args:
      run_pass: Test argument or fixture.
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def crash_hook(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Helper to crash hook.

    Args:
        node: Test argument or fixture.
        ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    raise ValueError("Hook Crash")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: crash_hook if name == "transform_for_loop" else None,
  ):
    code: typing.Any = "for i in range(10): pass"
    res: typing.Any = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "Loop transformation failed: Hook Crash" in res


# --- Merged from test_auxiliary_extra.py ---


def setup_ctx(alias_map: typing.Any = None) -> typing.Any:
  """Execute setup ctx helper logic.

  Args:
      alias_map: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  config: typing.Any = RuntimeConfig(source_framework="torch", target_framework="torch")
  ctx: typing.Any = RewriterContext(semantics=SemanticsManager(), config=config)
  if alias_map:
    ctx.alias_map = alias_map
  return ctx


def test_auxiliary_traits_empty() -> None:
  """Verify auxiliary traits empty behavior."""
  ctx: typing.Any = setup_ctx()
  ctx.semantics.get_framework_config = MagicMock(return_value={})
  p: typing.Any = AuxiliaryTransformer(ctx)
  traits: typing.Any = p._get_traits()
  assert traits is not None
  assert p._get_traits() is traits


def test_auxiliary_get_qualified_name_alias_split() -> None:
  """Verify auxiliary get qualified name alias split behavior."""
  ctx: typing.Any = setup_ctx({"np": "numpy"})
  p: typing.Any = AuxiliaryTransformer(ctx)
  node: typing.Any = cst.Attribute(value=cst.Name("np"), attr=cst.Name("add"))
  assert p._get_qualified_name(node) == "numpy.add"


def test_auxiliary_get_qualified_name_no_string() -> None:
  """Verify auxiliary get qualified name no string behavior."""
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  node: typing.Any = cst.Integer("1")
  assert p._get_qualified_name(node) is None


def test_auxiliary_create_dotted_name() -> None:
  """Verify auxiliary create dotted name behavior."""
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  node: typing.Any = p._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_auxiliary_leave_simplestatementline_warnings() -> None:
  """Verify auxiliary leave simplestatementline warnings behavior."""
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  p.context.current_stmt_warnings = ["warn1"]
  node: typing.Any = cst.SimpleStatementLine(body=[cst.Pass()])
  res: typing.Any = p.leave_SimpleStatementLine(node, node)
  assert res is not node
  assert hasattr(res, "nodes")  # FlattenSentinel


def test_auxiliary_leave_simplestatementline_errors() -> None:
  """Verify auxiliary leave simplestatementline errors behavior."""
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  p.context.current_stmt_errors = ["err1"]
  node: typing.Any = cst.SimpleStatementLine(body=[cst.Pass()])
  res: typing.Any = p.leave_SimpleStatementLine(node, node)
  assert res is not node


def test_auxiliary_leave_decorator_remove() -> None:
  """Verify auxiliary leave decorator remove behavior."""
  ctx: typing.Any = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": None}}))
  p: typing.Any = AuxiliaryTransformer(ctx)
  dec: typing.Any = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert type(res).__name__ == "RemovalSentinel"


def test_auxiliary_leave_decorator_rename_noncall() -> None:
  """Verify auxiliary leave decorator rename noncall behavior."""
  ctx: typing.Any = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))
  p: typing.Any = AuxiliaryTransformer(ctx)
  dec: typing.Any = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)
  assert isinstance(res.decorator, cst.Name)
  assert res.decorator.value == "new_dec"


def test_auxiliary_leave_decorator_name_none_or_not_found() -> None:
  """Test leave_Decorator returns updated_node if name is None or not found."""
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  # name is None (e.g. integer or unsupported expression as decorator)
  dec_unnamed: typing.Any = cst.Decorator(decorator=cst.Integer("1"))
  assert p.leave_Decorator(dec_unnamed, dec_unnamed) is dec_unnamed

  # lookup is None
  ctx.semantics.get_definition = MagicMock(return_value=None)
  dec: typing.Any = cst.Decorator(decorator=cst.Name("unknown_dec"))
  assert p.leave_Decorator(dec, dec) is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop static hook behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook.

    Args:
        node: Test argument or fixture.
        hook_ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if hook_ctx is ctx.hook_context:
      return cst.Pass()
    return node

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_exception(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop static hook exception behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook.

    Args:
        node: Test argument or fixture.
        hook_ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    raise ValueError("static error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop hook behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook.

    Args:
        node: Test argument or fixture.
        hook_ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_exception(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop hook exception behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook.

    Args:
        node: Test argument or fixture.
        hook_ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    raise ValueError("loop error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert not isinstance(res, cst.For)


def test_auxiliary_traits_with_traits() -> None:
  """Verify auxiliary traits with traits behavior."""
  ctx: typing.Any = setup_ctx()
  ctx.semantics.get_framework_config = MagicMock(return_value={"traits": {}})
  p: typing.Any = AuxiliaryTransformer(ctx)
  traits: typing.Any = p._get_traits()
  assert traits is not None


def test_auxiliary_get_qualified_name_alias_no_split() -> None:
  """Verify auxiliary get qualified name alias no split behavior."""
  ctx: typing.Any = setup_ctx({"np": "numpy"})
  p: typing.Any = AuxiliaryTransformer(ctx)
  node: typing.Any = cst.Name("np")
  assert p._get_qualified_name(node) == "numpy"


def test_auxiliary_leave_decorator_rename_noncall2() -> None:
  """Verify auxiliary leave decorator rename noncall2 behavior."""
  ctx: typing.Any = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))
  p: typing.Any = AuxiliaryTransformer(ctx)
  # The actual decorator decorator is just a Name, not a Call
  dec: typing.Any = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_new_node(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop static hook new node behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook.

    Args:
        node: Test argument or fixture.
        hook_ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


# --- Merged from test_auxiliary_extra_hooks2.py ---


def test_auxiliary_leave_decorator_rename_call() -> None:
  """Verify auxiliary leave decorator rename call behavior."""
  ctx: typing.Any = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))
  p: typing.Any = AuxiliaryTransformer(ctx)
  dec: typing.Any = cst.Decorator(decorator=cst.Call(func=cst.Name("test")))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same2(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop static hook same2 behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same2(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop hook same2 behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


# --- Merged from test_auxiliary_extra_hooks3.py ---


def test_cst_to_string_none_base() -> None:
  """Verify cst to string none base behavior."""
  transformer: typing.Any = AuxiliaryTransformer(CodemodContext())
  attr_node: typing.Any = cst.Attribute(value=cst.Pass(), attr=cst.Name("bar"))
  assert transformer._cst_to_string(attr_node) is None


# --- Merged from test_auxiliary_extra_hooks.py ---


def test_auxiliary_visit_simplestatementline() -> None:
  """Verify auxiliary visit simplestatementline behavior."""
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  res: typing.Any = p.visit_SimpleStatementLine(cst.SimpleStatementLine(body=[cst.Pass()]))
  assert res is True
  assert p.context.current_stmt_errors == []
  assert p.context.current_stmt_warnings == []


def test_auxiliary_leave_decorator_rename_noncall3() -> None:
  """Verify auxiliary leave decorator rename noncall3 behavior."""
  ctx: typing.Any = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"no_api": "missing"}}}))
  p: typing.Any = AuxiliaryTransformer(ctx)
  dec: typing.Any = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert res is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop static hook same behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop hook same behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_none(mock_get_hook: typing.Any) -> None:
  """Verify auxiliary for loop hook none behavior.

  Args:
      mock_get_hook: Test argument or fixture.
  """
  ctx: typing.Any = setup_ctx()
  p: typing.Any = AuxiliaryTransformer(ctx)
  loop: typing.Any = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.return_value = None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop
