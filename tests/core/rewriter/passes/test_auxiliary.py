"""Test suite for the Auxiliary module."""

import typing
from ml_switcheroo.core.hooks import HookContext
from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import libcst as _cst
import pytest
from libcst.codemod import CodemodContext

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass, AuxiliaryTransformer
from ml_switcheroo.semantics.manager import SemanticsManager

cst = _cst


class MockSemantics(SemanticsManager):
  """Test helper class MockSemantics."""

  def __init__(self) -> None:
    """Initializes the MockSemantics instance."""
    self.data = {}
    jit_def = {"variants": {"jax": {"api": "jax.jit"}, "torch": {"api": "torch.jit.script"}}}
    inf_def = {"variants": {"jax": None, "torch": {"api": "torch.inference_mode"}}}
    setattr(
      getattr({"v": self}, "__getitem__")("v"),
      "_reverse_index",
      getattr(
        {
          "v": {
            "torch.jit.script": getattr(
              {"v": (getattr({"v": "Jit"}, "__getitem__")("v"), getattr({"v": jit_def}, "__getitem__")("v"))},
              "__getitem__",
            )("v"),
            "torch.inference_mode": getattr(
              {"v": (getattr({"v": "InfMode"}, "__getitem__")("v"), getattr({"v": inf_def}, "__getitem__")("v"))},
              "__getitem__",
            )("v"),
          }
        },
        "__getitem__",
      )("v"),
    )
    self.framework_configs = {}

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
  """Execute run pass helper logic.

  Returns:
      Configured `run_pass` fixture value (typing.Callable[[str], str]).
  """
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics, config)

  def _transform(code: str) -> str:
    """Helper to  transform.

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
def clean_hooks() -> typing.Generator[None, None, None]:
  """Helper to clean hooks.

  Returns:
      Configured `clean_hooks` fixture value (typing.Generator[None, None, None]).
  """
  pass
  yield
  pass


def test_decorator_renaming(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of decorator renaming.

  Args:
      run_pass: The run pass (typing.Callable[[str], str]) for test decorator renaming.
  """
  code = "\n@torch.jit.script\ndef f(): pass\n"
  res = run_pass(code)
  assert "@jax.jit" in res
  assert "@torch" not in res


def test_decorator_removal(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of decorator removal.

  Args:
      run_pass: The run pass (typing.Callable[[str], str]) for test decorator removal.
  """
  code = "\n@torch.inference_mode\ndef f(): pass\n"
  res = run_pass(code)
  assert "@torch" not in res
  assert "def f():" in res


def test_decorator_with_args(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of decorator with arguments.

  Args:
      run_pass: The run pass (typing.Callable[[str], str]) for test decorator with args.
  """
  code = "\n@torch.jit.script(optimize=True)\ndef f(): pass\n"
  res = run_pass(code)
  assert "@jax.jit(optimize=True)" in res


def test_loop_static_unroll_hook(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of loop static unroll hook.

  Args:
      run_pass: The run pass (typing.Callable[[str], str]) for test loop static unroll hook.
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop_static")
  def mock_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Execute mock hook helper logic.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode` from mock hook.
    """
    _ret_v: cst.CSTNode = getattr(
      {
        "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").FlattenSentinel}, "__getitem__")("v")(
          getattr(
            {
              "v": [
                getattr(
                  {
                    "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").SimpleStatementLine}, "__getitem__")("v")(
                      getattr(
                        {
                          "v": [
                            getattr(
                              {
                                "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Expr}, "__getitem__")("v")(
                                  getattr(
                                    {
                                      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")(
                                        "v"
                                      )(getattr({"v": "unrolled"}, "__getitem__")("v"))
                                    },
                                    "__getitem__",
                                  )("v")
                                )
                              },
                              "__getitem__",
                            )("v")
                          ]
                        },
                        "__getitem__",
                      )("v")
                    )
                  },
                  "__getitem__",
                )("v")
              ]
            },
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

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
  """Verifies the behavior of loop safety hook.

  Args:
      run_pass: The run pass (typing.Callable[[str], str]) for test loop safety hook.
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def mock_safety(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Execute mock safety helper logic.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode` from mock safety.
    """
    _ret_v: cst.CSTNode = getattr(
      {
        "v": getattr({"v": getattr({"v": EscapeHatch}, "__getitem__")("v").mark_failure}, "__getitem__")("v")(
          getattr({"v": node}, "__getitem__")("v"), getattr({"v": "Unsafe Loop"}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_safety if name == "transform_for_loop" else None,
  ):
    code = "for i in range(N): pass"
    res = run_pass(code)
  assert EscapeHatch.START_MARKER in res
  assert "Unsafe Loop" in res


def test_loop_error_bubbling(run_pass: typing.Callable[[str], str]) -> None:
  """Verifies the behavior of loop correctly handling an error bubbling.

  Args:
      run_pass: The run pass (typing.Callable[[str], str]) for test loop error bubbling.
  """
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def crash_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Helper to crash hook.

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
  """Execute setup ctx helper logic.

  Args:
      alias_map: The alias map (dict[str, str] | None) for setup ctx.

  Returns:
      Resulting `RewriterContext` from setup ctx.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="torch")
  ctx = RewriterContext(semantics=SemanticsManager(), config=config)
  if alias_map:
    ctx.alias_map = alias_map
  return ctx


def test_auxiliary_traits_empty() -> None:
  """Verify auxiliary traits empty behavior."""
  ctx = setup_ctx()
  setattr(ctx.semantics, "get_framework_config", MagicMock(return_value={}))
  p = AuxiliaryTransformer(ctx)
  traits = p._get_traits()
  assert traits is not None
  assert p._get_traits() is traits


def test_auxiliary_get_qualified_name_alias_split() -> None:
  """Verify auxiliary get qualified name alias split behavior."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Attribute(value=cst.Name("np"), attr=cst.Name("add"))
  assert p._get_qualified_name(node) == "numpy.add"


def test_auxiliary_get_qualified_name_no_string() -> None:
  """Verify auxiliary get qualified name no string behavior."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node = cst.Integer("1")
  assert p._get_qualified_name(node) is None


def test_auxiliary_create_dotted_name() -> None:
  """Verify auxiliary create dotted name behavior."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node = p._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_auxiliary_leave_simplestatementline_warnings() -> None:
  """Verify auxiliary leave simplestatementline warnings behavior."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_warnings = ["warn1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res = p.leave_SimpleStatementLine(node, node)
  assert res is not node
  assert hasattr(res, "nodes")  # FlattenSentinel


def test_auxiliary_leave_simplestatementline_errors() -> None:
  """Verify auxiliary leave simplestatementline errors behavior."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_errors = ["err1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res = p.leave_SimpleStatementLine(node, node)
  assert res is not node


def test_auxiliary_leave_decorator_remove() -> None:
  """Verify auxiliary leave decorator remove behavior."""
  ctx = setup_ctx()
  setattr(ctx.semantics, "get_definition", MagicMock(return_value=("id", {"variants": {"torch": None}})))
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert type(res).__name__ == "RemovalSentinel"


def test_auxiliary_leave_decorator_rename_noncall() -> None:
  """Verify auxiliary leave decorator rename noncall behavior."""
  ctx = setup_ctx()
  setattr(ctx.semantics, "get_definition", MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}})))
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
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
  setattr(ctx.semantics, "get_definition", MagicMock(return_value=None))
  dec = cst.Decorator(decorator=cst.Name("unknown_dec"))
  assert p.leave_Decorator(dec, dec) is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop static hook behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop static hook.
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
        hook_ctx: The hook ctx (HookContext) for hook mock.

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    if hook_ctx is ctx.hook_context:
      return cst.Pass()
    return node

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_exception(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop static hook exception behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop static hook exception.
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
        hook_ctx: The hook ctx (HookContext) for hook mock.

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    raise ValueError("static error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop hook behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop hook.
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
        hook_ctx: The hook ctx (HookContext) for hook mock.

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_exception(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop hook exception behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop hook exception.
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
        hook_ctx: The hook ctx (HookContext) for hook mock.

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    raise ValueError("loop error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert not isinstance(res, cst.For)


def test_auxiliary_traits_with_traits() -> None:
  """Verify auxiliary traits with traits behavior."""
  ctx = setup_ctx()
  setattr(ctx.semantics, "get_framework_config", MagicMock(return_value={"traits": {}}))
  p = AuxiliaryTransformer(ctx)
  traits = p._get_traits()
  assert traits is not None


def test_auxiliary_get_qualified_name_alias_no_split() -> None:
  """Verify auxiliary get qualified name alias no split behavior."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Name("np")
  assert p._get_qualified_name(node) == "numpy"


def test_auxiliary_leave_decorator_rename_noncall2() -> None:
  """Verify auxiliary leave decorator rename noncall2 behavior."""
  ctx = setup_ctx()
  setattr(ctx.semantics, "get_definition", MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}})))
  p = AuxiliaryTransformer(ctx)
  # The actual decorator decorator is just a Name, not a Call
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_new_node(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop static hook new node behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop static hook new node.
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
        hook_ctx: The hook ctx (HookContext) for hook mock.

    Returns:
        Resulting `cst.CSTNode` from hook mock.
    """
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


def test_auxiliary_leave_decorator_rename_call() -> None:
  """Verify auxiliary leave decorator rename call behavior."""
  ctx = setup_ctx()
  setattr(ctx.semantics, "get_definition", MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}})))
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Call(func=cst.Name("test")))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same2(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop static hook same2 behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop static hook same2.
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same2(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop hook same2 behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop hook same2.
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert res is loop


def test_cst_to_string_none_base() -> None:
  """Verify cst to string none base behavior."""
  transformer = getattr(
    {
      "v": getattr({"v": AuxiliaryTransformer}, "__getitem__")("v")(
        getattr({"v": getattr({"v": CodemodContext}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  attr_node = getattr(
    {
      "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Attribute}, "__getitem__")("v")(
        value=getattr(
          {"v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Pass}, "__getitem__")("v")()}, "__getitem__"
        )("v"),
        attr=getattr(
          {
            "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v")(
              getattr({"v": "bar"}, "__getitem__")("v")
            )
          },
          "__getitem__",
        )("v"),
      )
    },
    "__getitem__",
  )("v")
  assert transformer._cst_to_string(attr_node) is None


def test_auxiliary_visit_simplestatementline() -> None:
  """Verify auxiliary visit simplestatementline behavior."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  res = p.visit_SimpleStatementLine(cst.SimpleStatementLine(body=[cst.Pass()]))
  assert res is True
  assert p.context.current_stmt_errors == []
  assert p.context.current_stmt_warnings == []


def test_auxiliary_leave_decorator_rename_noncall3() -> None:
  """Verify auxiliary leave decorator rename noncall3 behavior."""
  ctx = setup_ctx()
  setattr(ctx.semantics, "get_definition", MagicMock(return_value=("id", {"variants": {"torch": {"no_api": "missing"}}})))
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert res is dec


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_same(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop static hook same behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop static hook same.
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_same(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop hook same behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop hook same.
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.side_effect = lambda name: (lambda n, c: n) if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_none(mock_get_hook: MagicMock) -> None:
  """Verify auxiliary for loop hook none behavior.

  Args:
      mock_get_hook: The mock get hook (MagicMock) for test auxiliary for loop hook none.
  """
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(target=cst.Name("i"), iter=cst.Name("range"), body=cst.IndentedBlock(body=[]))
  mock_get_hook.return_value = None
  res = p.leave_For(loop, loop)
  assert res is loop
