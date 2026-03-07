"""
Tests for the Auxiliary Pass logic.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryPass
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import _HOOKS, clear_hooks
from ml_switcheroo.core.escape_hatch import EscapeHatch


class MockSemantics(SemanticsManager):
  """Class docstring."""

  def __init__(self):
    """Function docstring."""
    self.data = {}
    # Decorator Mappings
    # torch.jit.script -> jax.jit
    # torch.inference_mode -> None (Remove)
    jit_def = {"variants": {"jax": {"api": "jax.jit"}, "torch": {"api": "torch.jit.script"}}}
    inf_def = {"variants": {"jax": None, "torch": {"api": "torch.inference_mode"}}}

    self._reverse_index = {"torch.jit.script": ("Jit", jit_def), "torch.inference_mode": ("InfMode", inf_def)}
    self.framework_configs = {}

  def get_definition(self, name):
    """Function docstring."""
    return self._reverse_index.get(name)

  def get_framework_config(self, fw):
    """Function docstring."""
    return {}


@pytest.fixture
def run_pass():
  """Function docstring."""
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics, config)

  def _transform(code):
    """Function docstring."""
    module = cst.parse_module(code)
    aux_pass = AuxiliaryPass()
    return aux_pass.transform(module, ctx).code

  return _transform


@pytest.fixture(autouse=True)
def clean_hooks():
  """Function docstring."""
  clear_hooks()
  yield
  clear_hooks()


def test_decorator_renaming(run_pass):
  """Verify renaming @torch.jit.script to @jax.jit."""
  code = """
@torch.jit.script
def f(): pass
"""
  res = run_pass(code)
  assert "@jax.jit" in res
  assert "@torch" not in res


def test_decorator_removal(run_pass):
  """Verify removing @torch.inference_mode."""
  code = """
@torch.inference_mode
def f(): pass
"""
  res = run_pass(code)
  assert "@torch" not in res
  assert "def f():" in res


def test_decorator_with_args(run_pass):
  """Verify args preserved during rename."""
  # We mock a call-style decorator mapping
  # torch.compile -> jax.jit (preserving args structurally for this test scope)
  # Note: Real plugins might strip args, but AuxiliaryTransformer preserves Call node structure if simply renamed.
  code = """
@torch.jit.script(optimize=True)
def f(): pass
"""
  res = run_pass(code)
  assert "@jax.jit(optimize=True)" in res


def test_loop_static_unroll_hook(run_pass):
  """Verify static loop unrolling logic triggers."""
  # Register mock hook
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop_static")
  def mock_hook(node, ctx):
    """Function docstring."""
    return cst.FlattenSentinel([cst.SimpleStatementLine([cst.Expr(cst.Name("unrolled"))])])

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_hook if name == "transform_for_loop_static" else None,
  ):
    code = "for i in range(2): pass"
    res = run_pass(code)

  print("RES:", res)
  assert "unrolled" in res
  assert "for" not in res


def test_loop_safety_hook(run_pass):
  """Verify safety scanner logic triggers."""
  # Register mock hook that returns EscapeHatch Sentinel
  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def mock_safety(node, ctx):
    """Function docstring."""
    return EscapeHatch.mark_failure(node, "Unsafe Loop")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: mock_safety if name == "transform_for_loop" else None,
  ):
    code = "for i in range(N): pass"
    res = run_pass(code)

  assert EscapeHatch.START_MARKER in res
  assert "Unsafe Loop" in res


def test_loop_error_bubbling(run_pass):
  """Verify exception in hook bubbles to error report."""

  from ml_switcheroo.core.hooks import register_hook

  @register_hook("transform_for_loop")
  def crash_hook(node, ctx):
    """Function docstring."""
    raise ValueError("Hook Crash")

  with patch(
    "ml_switcheroo.core.rewriter.passes.auxiliary.get_hook",
    side_effect=lambda name: crash_hook if name == "transform_for_loop" else None,
  ):
    code = "for i in range(10): pass"
    res = run_pass(code)

  # Should wrap original in escape hatch with error msg
  assert EscapeHatch.START_MARKER in res
  assert "Loop transformation failed: Hook Crash" in res
