"""
Tests for Loop Unrolling Plugin.

Verifies:
1. Passthrough on non-JAX targets.
2. Detection of `for i in range()` in JAX mode.
3. Application of Escape Hatch warnings for JAX loops (Safety First strategy).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.loop_unroll import transform_loops
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch


def rewrite_code(rewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter_factory():
  # Register hooks manually
  hooks._HOOKS["transform_for_loop"] = transform_loops
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_definition.return_value = None  # No mappings needed for loop syntax

  def create(target):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=False)
    return PivotRewriter(mgr, cfg)

  return create


def test_torch_passthrough(rewriter_factory):
  """Verify that loops remain untouched for Torch targets."""
  rewriter = rewriter_factory("torch")
  code = """
for i in range(10):
    print(i)
"""
  result = rewrite_code(rewriter, code)

  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER not in result


def test_jax_range_detection_and_warning(rewriter_factory):
  """Verify that JAX targets get the safety warning."""
  rewriter = rewriter_factory("jax")
  code = """
for i in range(10):
    x = x + i
"""
  result = rewrite_code(rewriter, code)

  assert "for i in range(10):" in result  # Code preserved
  assert EscapeHatch.START_MARKER in result
  assert "JAX requires `jax.lax.fori_loop`" in result


def test_jax_iterator_detection(rewriter_factory):
  """Verify generic iterator loops also get flagged."""
  rewriter = rewriter_factory("jax")
  code = """
for item in my_list:
    print(item)
"""
  result = rewrite_code(rewriter, code)

  assert EscapeHatch.START_MARKER in result
  assert "requires `scan`" in result
