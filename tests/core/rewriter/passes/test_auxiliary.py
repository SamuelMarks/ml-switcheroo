"""Test suite for the Auxiliary module."""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest
from libcst.codemod import CodemodContext

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Docstring."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data: dict[str, typing.Any] = {}
    jit_def: dict[str, typing.Any] = {"variants": {"jax": {"api": "jax.jit"}, "torch": {"api": "torch.jit.script"}}}
    inf_def: dict[str, typing.Any] = {"variants": {"jax": None, "torch": {"api": "torch.inference_mode"}}}
    self._reverse_index: dict[str, tuple[str, dict[str, typing.Any]]] = {
      "torch.jit.script": ("Jit", jit_def),
      "torch.inference_mode": ("InfMode", inf_def),
    }
    self.framework_configs: dict[str, typing.Any] = {}

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)

  def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
    """Mock implementation of get framework configuration."""
    return {}


@pytest.fixture
def run_pass() -> typing.Callable[[str], str]:
  """Docstring."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics, config)

  def _transform(code: str) -> str:
    """Helper to  transform."""
    module = cst.parse_module(code)
    aux_pass = AuxiliaryPass()
    return typing.cast(str, aux_pass.transform(module, ctx).code)

  return _transform


@pytest.fixture(autouse=True)
def clean_hooks() -> typing.Generator[None, None, None]:
  """Helper to clean hooks."""
  pass
  yield
  pass


def test_decorator_renaming(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of decorator renaming."""
  code: str = "\n@torch.jit.script\ndef f(): pass\n"
  res: str = run_pass(code)
  assert "@jax.jit" in res
  assert "@torch" not in res


def test_decorator_removal(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of decorator removal."""
  code: str = "\n@torch.inference_mode\ndef f(): pass\n"
  res: str = run_pass(code)
  assert "@torch" not in res
  assert "def f():" in res


def test_decorator_with_args(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of decorator with arguments."""
  code: str = "\n@torch.jit.script(optimize=True)\ndef f(): pass\n"
  res: str = run_pass(code)
  assert "@jax.jit(optimize=True)" in res


def test_loop_static_unroll_hook(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of loop static unroll hook."""
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop_static")
  def mock_hook(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Docstring."""
    return cst.FlattenSentinel([cst.SimpleStatementLine([cst.Expr(cst.Name("unrolled"))])])

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_hook if name == "transform_for_loop_static" else None,
  ):
    code: str = "for i in range(2): pass"
    res: str = run_pass(code)
  print("RES:", res)
  assert "unrolled" in res
  assert "for" not in res


def test_loop_safety_hook(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of loop safety hook."""
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def mock_safety(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Docstring."""
    return EscapeHatch.mark_failure(node, "Unsafe Loop")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_safety if name == "transform_for_loop" else None,
  ):
    code: str = "for i in range(N): pass"
    res: str = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "Unsafe Loop" in res


def test_loop_error_bubbling(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of loop correctly handling an error bubbling."""
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def crash_hook(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Helper to crash hook."""
    raise ValueError("Hook Crash")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: crash_hook if name == "transform_for_loop" else None,
  ):
    code: str = "for i in range(10): pass"
    res: str = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "Loop transformation failed: Hook Crash" in res


# --- Merged from test_auxiliary_extra.py ---


def setup_ctx(alias_map: typing.Optional[dict[str, str]] = None) -> RewriterContext:
  """Docstring."""
  config = RuntimeConfig(source_framework="torch", target_framework="torch")
  ctx = RewriterContext(semantics=SemanticsManager(), config=config)
  if alias_map:
    ctx.alias_map = alias_map
  return ctx


def test_auxiliary_traits_empty() -> None:
  """Docstring."""
  ctx = setup_ctx()
  ctx.semantics.get_framework_config = MagicMock(return_value={})  # type: ignore
  p = AuxiliaryTransformer(ctx)
  traits = p._get_traits()
  assert traits is not None
  assert p._get_traits() is traits


def test_auxiliary_get_qualified_name_alias_split() -> None:
  """Docstring."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Attribute(value=cst.Name("np"), attr=cst.Name("add"))
  assert p._get_qualified_name(node) == "numpy.add"


def test_auxiliary_get_qualified_name_no_string() -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node = cst.Integer("1")
  assert p._get_qualified_name(node) is None


def test_auxiliary_create_dotted_name() -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node: typing.Any = p._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_auxiliary_leave_simplestatementline_warnings() -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_warnings = ["warn1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res: typing.Any = p.leave_SimpleStatementLine(node, node)
  assert res is not node
  assert hasattr(res, "nodes")  # FlattenSentinel


def test_auxiliary_leave_simplestatementline_errors() -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_errors = ["err1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res: typing.Any = p.leave_SimpleStatementLine(node, node)
  assert res is not node


def test_auxiliary_leave_decorator_remove() -> None:
  """Docstring."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": None}}))  # type: ignore
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert type(res).__name__ == "RemovalSentinel"


def test_auxiliary_leave_decorator_rename_noncall() -> None:
  """Docstring."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))  # type: ignore
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)
  assert isinstance(res.decorator, cst.Name)
  assert res.decorator.value == "new_dec"


def test_auxiliary_leave_decorator_name_none_or_not_found() -> None:
  """Test leave_Decorator returns updated_node if name is None or not found."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  # name is None (e.g. integer or unsupported expression as decorator)
  dec_unnamed = cst.Decorator(decorator=cst.Integer("1"))
  assert p.leave_Decorator(dec_unnamed, dec_unnamed) is dec_unnamed

  # lookup is None
  ctx.semantics.get_definition = MagicMock(return_value=None)  # type: ignore
  dec = cst.Decorator(decorator=cst.Name("unknown_dec"))
  assert p.leave_Decorator(dec, dec) is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook."""
    if hook_ctx is ctx.hook_context:
      return cst.Pass()
    return node

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_exception(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook."""
    raise ValueError("static error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook."""
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_exception(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook."""
    raise ValueError("loop error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert not isinstance(res, cst.For)


def test_auxiliary_traits_with_traits() -> None:
  """Docstring."""
  ctx = setup_ctx()
  ctx.semantics.get_framework_config = MagicMock(return_value={"traits": {}})  # type: ignore
  p = AuxiliaryTransformer(ctx)
  traits: typing.Any = p._get_traits()
  assert traits is not None


def test_auxiliary_get_qualified_name_alias_no_split() -> None:
  """Docstring."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Name("np")
  assert p._get_qualified_name(node) == "numpy"


def test_auxiliary_leave_decorator_rename_noncall2() -> None:
  """Docstring."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))  # type: ignore
  p = AuxiliaryTransformer(ctx)
  # The actual decorator decorator is just a Name, not a Call
  dec = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_new_node(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node: typing.Any, hook_ctx: typing.Any) -> typing.Any:
    """Mocks the hook."""
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


# --- Merged from test_auxiliary_extra_hooks2.py ---


def test_auxiliary_leave_decorator_rename_call() -> None:
  """Docstring."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))  # type: ignore
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Call(func=cst.Name("test")))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same2(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same2(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


# --- Merged from test_auxiliary_extra_hooks3.py ---


def test_cst_to_string_none_base() -> None:
  """Docstring."""
  transformer = AuxiliaryTransformer(CodemodContext())  # type: ignore
  attr_node = cst.Attribute(value=cst.Pass(), attr=cst.Name("bar"))  # type: ignore
  assert transformer._cst_to_string(attr_node) is None


# --- Merged from test_auxiliary_extra_hooks.py ---


def test_auxiliary_visit_simplestatementline() -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  res: typing.Any = p.visit_SimpleStatementLine(cst.SimpleStatementLine(body=[cst.Pass()]))
  assert res is True
  assert p.context.current_stmt_errors == []
  assert p.context.current_stmt_warnings == []


def test_auxiliary_leave_decorator_rename_noncall3() -> None:
  """Docstring."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"no_api": "missing"}}}))  # type: ignore
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res: typing.Any = p.leave_Decorator(dec, dec)
  assert res is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_none(mock_get_hook: MagicMock) -> None:
  """Docstring."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.return_value = None
  res: typing.Any = p.leave_For(loop, loop)
  assert res is loop
