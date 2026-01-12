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
  def __init__(self):
    self.data = {}
    # Decorator Mappings
    # torch.jit.script -> jax.jit
    # torch.inference_mode -> None (Remove)
    jit_def = {"variants": {"jax": {"api": "jax.jit"}, "torch": {"api": "torch.jit.script"}}}
    inf_def = {"variants": {"jax": None, "torch": {"api": "torch.inference_mode"}}}

    self._reverse_index = {"torch.jit.script": ("Jit", jit_def), "torch.inference_mode": ("InfMode", inf_def)}
    self.framework_configs = {}

  def get_definition(self, name):
    return self._reverse_index.get(name)

  def get_framework_config(self, fw):
    return {}


@pytest.fixture
def run_pass():
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  ctx = RewriterContext(semantics, config)

  def _transform(code):
    module = cst.parse_module(code)
    aux_pass = AuxiliaryPass()
    return aux_pass.transform(module, ctx).code

  return _transform


@pytest.fixture(autouse=True)
def clean_hooks():
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
  _HOOKS["transform_for_loop_static"] = lambda node, ctx: cst.FlattenSentinel(
    [cst.SimpleStatementLine([cst.Expr(cst.Name("unrolled"))])]
  )

  code = "for i in range(2): pass"
  res = run_pass(code)

  assert "unrolled" in res
  assert "for" not in res


def test_loop_safety_hook(run_pass):
  """Verify safety scanner logic triggers."""
  # Register mock hook that returns EscapeHatch Sentinel
  _HOOKS["transform_for_loop"] = lambda node, ctx: EscapeHatch.mark_failure(node, "Unsafe Loop")

  code = "for i in range(N): pass"
  res = run_pass(code)

  assert EscapeHatch.START_MARKER in res
  assert "Unsafe Loop" in res


def test_loop_error_bubbling(run_pass):
  """Verify exception in hook bubbles to error report."""

  def crash_hook(node, ctx):
    raise ValueError("Hook Crash")

  _HOOKS["transform_for_loop"] = crash_hook

  code = "for i in range(10): pass"
  res = run_pass(code)

  # Should wrap original in escape hatch with error msg
  assert EscapeHatch.START_MARKER in res
  assert "Loop transformation failed: Hook Crash" in res
