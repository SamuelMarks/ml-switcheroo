"""Test suite for the Auxiliary module."""

import typing
from unittest.mock import MagicMock, patch

import libcst as _cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits

cst = _cst


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` used in auxiliary tests."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    super().__init__(load_defaults=False)
    jit_def: dict[str, JsonValue] = {"variants": {"jax": {"api": "jax.jit"}, "torch": {"api": "torch.jit.script"}}}
    inf_def: dict[str, JsonValue] = {"variants": {"jax": None, "torch": {"api": "torch.inference_mode"}}}
    self._reverse_index = {
      "torch.jit.script": ("Jit", jit_def),
      "torch.inference_mode": ("InfMode", inf_def),
    }

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Mock implementation of get definition.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    return self._reverse_index.get(name)

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Mock implementation of get framework configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    return {}


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Create a configured `typing.Callable[[str], str]` fixture for auxiliary tests.

  Returns:
      Configured `run_pass` fixture value (typing.Callable[[str], str]).
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics, config)

  def _transform(code: str) -> str:
    """Execute `AuxiliaryPass` to support transform in auxiliary tests.

    Args:
        code: Python or DSL source code string to parse or transpile.

    Returns:
        Resulting `str` from transform.
    """
    module = cst.parse_module(code)
    aux_pass = AuxiliaryPass()
    return aux_pass.transform(module, ctx).code

  return _transform


@pytest.fixture(autouse=True)
def clean_hooks() -> None:
  """Create a configured `None` fixture for auxiliary tests."""


def test_decorator_renaming(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `@jax.jit` for decorator renaming.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code = "\n@torch.jit.script\ndef f(): pass\n"
  res = run_pass(code)
  assert "@jax.jit" in res
  assert "@torch" not in res


def test_decorator_removal(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `def f():` for decorator removal.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code = "\n@torch.inference_mode\ndef f(): pass\n"
  res = run_pass(code)
  assert "@torch" not in res
  assert "def f():" in res


def test_decorator_with_args(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `run_pass` emits `@jax.jit(optimize=True)` for decorator with args.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  code = "\n@torch.jit.script(optimize=True)\ndef f(): pass\n"
  res = run_pass(code)
  assert "@jax.jit(optimize=True)" in res


def test_loop_static_unroll_hook(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `register_hook` emits `unrolled` for loop static unroll hook.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop_static")
  def mock_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]:
    """Execute `register_hook` to support mock hook in auxiliary tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]` from mock hook.
    """
    stmts: list[cst.BaseStatement] = [cst.SimpleStatementLine([cst.Expr(cst.Name("unrolled"))])]
    return cst.FlattenSentinel(stmts)

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_hook if name == "transform_for_loop_static" else None,
  ):
    code = "for i in range(2): pass"
    res = run_pass(code)
  print("RES:", res)
  assert "unrolled" in res
  assert "for" not in res


def test_loop_safety_hook(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `register_hook` emits `Unsafe Loop` for loop safety hook.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def mock_safety(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]:
    """Execute `register_hook` to support mock safety in auxiliary tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode | cst.FlattenSentinel[cst.BaseStatement]` from mock safety.
    """
    return EscapeHatch.mark_failure(node, "Unsafe Loop")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_safety if name == "transform_for_loop" else None,
  ):
    code = "for i in range(N): pass"
    res = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "Unsafe Loop" in res


def test_loop_error_bubbling(run_pass: typing.Callable[[str], str]) -> None:
  """Verify `register_hook` handles loop error bubbling as expected.

  Args:
      run_pass: Input `run_pass` parameter value (`typing.Callable[[str], str]`).
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def crash_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Execute `register_hook` to support crash hook in auxiliary tests.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode` from crash hook.
    """
    raise ValueError("Hook Crash")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: crash_hook if name == "transform_for_loop" else None,
  ):
    code = "for i in range(10): pass"
    res = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "Loop transformation failed: Hook Crash" in res


def setup_ctx(alias_map: dict[str, str] | None = None) -> RewriterContext:
  """Execute `RuntimeConfig` to support setup ctx in auxiliary tests.

  Args:
      alias_map: Input `alias_map` parameter value (`dict[str, str] | None`).

  Returns:
      Resulting `RewriterContext` from setup ctx.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="torch")
  ctx = RewriterContext(semantics=SemanticsManager(), config=config)
  if alias_map:
    ctx.alias_map = alias_map
  return ctx


def test_auxiliary_traits_empty(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `setup_ctx` handles auxiliary traits empty as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  monkeypatch.setattr(ctx.semantics, "get_framework_config", MagicMock(return_value={}), raising=False)
  p = AuxiliaryTransformer(ctx)
  traits = p._get_traits()
  assert traits is not None
  assert p._get_traits() is traits


def test_auxiliary_get_qualified_name_alias_split() -> None:
  """Verify `setup_ctx` handles auxiliary get qualified name alias split as expected."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Attribute(value=cst.Name("np"), attr=cst.Name("add"))
  assert p._get_qualified_name(node) == "numpy.add"


def test_auxiliary_get_qualified_name_no_string() -> None:
  """Verify `setup_ctx` handles auxiliary get qualified name no string as expected."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node = cst.Integer("1")
  assert p._get_qualified_name(node) is None


def test_auxiliary_create_dotted_name() -> None:
  """Verify `setup_ctx` handles auxiliary create dotted name as expected."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node = p._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_auxiliary_leave_simplestatementline_warnings() -> None:
  """Verify `setup_ctx` handles auxiliary leave simplestatementline warnings as expected."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_warnings = ["warn1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res = p.leave_SimpleStatementLine(node, node)
  assert res is not node
  assert isinstance(res, cst.FlattenSentinel)


def test_auxiliary_leave_simplestatementline_errors() -> None:
  """Verify `setup_ctx` handles auxiliary leave simplestatementline errors as expected."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_errors = ["err1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res = p.leave_SimpleStatementLine(node, node)
  assert res is not node


def test_auxiliary_leave_decorator_remove(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `setup_ctx` handles auxiliary leave decorator remove as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  monkeypatch.setattr(
    ctx.semantics, "get_definition", MagicMock(return_value=("id", {"variants": {"torch": None}})), raising=False
  )
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert type(res).__name__ == "RemovalSentinel"


def test_auxiliary_leave_decorator_rename_noncall(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `setup_ctx` handles auxiliary leave decorator rename noncall as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  monkeypatch.setattr(
    ctx.semantics,
    "get_definition",
    MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}})),
    raising=False,
  )
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)
  assert isinstance(res.decorator, cst.Name)
  assert res.decorator.value == "new_dec"


def test_auxiliary_leave_decorator_name_none_or_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test leave_Decorator returns updated_node if name is None or not found.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  # name is None (e.g. integer or unsupported expression as decorator)
  dec_unnamed = cst.Decorator(decorator=cst.Integer("1"))
  assert p.leave_Decorator(dec_unnamed, dec_unnamed) is dec_unnamed

  # lookup is None
  monkeypatch.setattr(ctx.semantics, "get_definition", MagicMock(return_value=None), raising=False)
  dec = cst.Decorator(decorator=cst.Name("unknown_dec"))
  assert p.leave_Decorator(dec, dec) is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop static hook as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: cst.CSTNode, hook_ctx: HookContext) -> cst.CSTNode:
    """Mocks the hook.

    Args:
        node: CST or AST node under inspection or transformation.
        hook_ctx: Input `hook_ctx` parameter value (`HookContext`).

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    if hook_ctx is ctx.hook_context:
      return cst.Pass()
    return node

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)
  assert cst.Module([]).code_for_node(res).strip() == "pass"


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_exception(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop static hook exception as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: cst.CSTNode, hook_ctx: HookContext) -> cst.CSTNode:
    """Mocks the hook.

    Args:
        node: CST or AST node under inspection or transformation.
        hook_ctx: Input `hook_ctx` parameter value (`HookContext`).

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    raise ValueError("static error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop hook as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: cst.CSTNode, hook_ctx: HookContext) -> cst.CSTNode:
    """Mocks the hook.

    Args:
        node: CST or AST node under inspection or transformation.
        hook_ctx: Input `hook_ctx` parameter value (`HookContext`).

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)
  assert cst.Module([]).code_for_node(res).strip() == "pass"


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_exception(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop hook exception as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: cst.CSTNode, hook_ctx: HookContext) -> cst.CSTNode:
    """Mocks the hook.

    Args:
        node: CST or AST node under inspection or transformation.
        hook_ctx: Input `hook_ctx` parameter value (`HookContext`).

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    raise ValueError("loop error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert not isinstance(res, cst.For)


def test_auxiliary_traits_with_traits(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `setup_ctx` handles auxiliary traits with traits as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  monkeypatch.setattr(ctx.semantics, "get_framework_config", MagicMock(return_value={"traits": {}}), raising=False)
  p = AuxiliaryTransformer(ctx)
  traits = p._get_traits()
  assert isinstance(traits, StructuralTraits)
  assert traits.init_method_name == "__init__"


def test_auxiliary_get_qualified_name_alias_no_split() -> None:
  """Verify `setup_ctx` handles auxiliary get qualified name alias no split as expected."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Name("np")
  assert p._get_qualified_name(node) == "numpy"


def test_auxiliary_leave_decorator_rename_noncall2(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `setup_ctx` handles auxiliary leave decorator rename noncall2 as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  monkeypatch.setattr(
    ctx.semantics,
    "get_definition",
    MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}})),
    raising=False,
  )
  p = AuxiliaryTransformer(ctx)
  # The actual decorator decorator is just a Name, not a Call
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)
  assert cst.Module([]).code_for_node(res).strip() == "@new_dec"


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_new_node(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop static hook new node as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: cst.CSTNode, hook_ctx: HookContext) -> cst.CSTNode:
    """Mocks the hook.

    Args:
        node: CST or AST node under inspection or transformation.
        hook_ctx: Input `hook_ctx` parameter value (`HookContext`).

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)
  assert cst.Module([]).code_for_node(res).strip() == "pass"


def test_auxiliary_leave_decorator_rename_call(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `setup_ctx` handles auxiliary leave decorator rename call as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  monkeypatch.setattr(
    ctx.semantics,
    "get_definition",
    MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}})),
    raising=False,
  )
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Call(func=cst.Name("test")))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)
  assert cst.Module([]).code_for_node(res).strip() == "@new_dec()"


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same2(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop static hook same2 as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same2(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop hook same2 as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert res is loop


def test_cst_to_string_none_base() -> None:
  """Verify `transformer._cst_to_string` handles none base as expected."""
  transformer = AuxiliaryTransformer(setup_ctx())
  attr_node = cst.Attribute(
    value=cst.Call(func=cst.Name("foo")),
    attr=cst.Name("bar"),
  )
  assert transformer._cst_to_string(attr_node) is None


def test_auxiliary_visit_simplestatementline() -> None:
  """Verify `setup_ctx` handles auxiliary visit simplestatementline as expected."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  res = p.visit_SimpleStatementLine(cst.SimpleStatementLine(body=[cst.Pass()]))
  assert res is True
  assert p.context.current_stmt_errors == []
  assert p.context.current_stmt_warnings == []


def test_auxiliary_leave_decorator_rename_noncall3(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `setup_ctx` handles auxiliary leave decorator rename noncall3 as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = setup_ctx()
  monkeypatch.setattr(
    ctx.semantics,
    "get_definition",
    MagicMock(return_value=("id", {"variants": {"torch": {"no_api": "missing"}}})),
    raising=False,
  )
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert res is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop static hook same as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop hook same as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_none(mock_get_hook: MagicMock) -> None:
  """Verify `setup_ctx` handles auxiliary for loop hook none as expected.

  Args:
      mock_get_hook: Input `mock_get_hook` parameter value (`MagicMock`).
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.return_value = None
  res = p.leave_For(loop, loop)
  assert res is loop


def test_auxiliary_decorator_complex_callee_and_missing_target_variant() -> None:
  """Verify `AuxiliaryTransformer.leave_Decorator` preserves unresolvable and unmapped-target decorators."""
  ctx = setup_ctx()
  ctx.semantics._reverse_index["my_dec"] = ("DecOp", {"variants": {"other_fw": {"api": "other.dec"}}})
  p = AuxiliaryTransformer(ctx)

  fn_complex = cst.ensure_type(cst.parse_module("@foo()()\ndef x(): pass").body[0], cst.FunctionDef)
  dec_complex = fn_complex.decorators[0]
  assert p.leave_Decorator(dec_complex, dec_complex) is dec_complex

  fn_unmapped = cst.ensure_type(cst.parse_module("@my_dec\ndef x(): pass").body[0], cst.FunctionDef)
  dec_unmapped = fn_unmapped.decorators[0]
  assert p.leave_Decorator(dec_unmapped, dec_unmapped) is dec_unmapped

  ctx.semantics._reverse_index["bad_variants_dec"] = ("BadDec", {"variants": "not_a_dict"})
  fn_bad_var = cst.ensure_type(cst.parse_module("@bad_variants_dec\ndef x(): pass").body[0], cst.FunctionDef)
  dec_bad_var = fn_bad_var.decorators[0]
  assert p.leave_Decorator(dec_bad_var, dec_bad_var) is dec_bad_var

  from ml_switcheroo.core.hooks import HookContext

  def _failing_loop_hook(_n: cst.For, _c: HookContext) -> cst.CSTNode:
    """Raise `ValueError` inside loop transformation hook.

    Args:
        _n: Loop node.
        _c: Hook context.

    Returns:
        Never returns; raises `ValueError`.
    """
    raise ValueError("loop failure")

  loop = cst.For(target=cst.Name("i"), iter=cst.Name("r"), body=cst.IndentedBlock(body=[]))
  with (
    patch(
      "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
      side_effect=lambda name: _failing_loop_hook if name == "transform_for_loop" else None,
    ),
    patch("ml_switcheroo.core.rewriter.passes.auxiliary.EscapeHatch.mark_failure", return_value=cst.RemoveFromParent()),
  ):
    assert p.leave_For(loop, loop) is loop
