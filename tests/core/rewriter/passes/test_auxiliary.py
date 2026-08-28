"""Test suite for the Auxiliary module."""

import pytest
import typing
import libcst as cst
from unittest.mock import patch
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

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
  """Provides a mock run pass for testing."""
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
    """Provides a mock hook for testing."""
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
    """Provides a mock safety for testing."""
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
