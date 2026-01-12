"""
Tests for Static Loop Unrolling Plugin.

Verifies:
1.  **Detection**: Identifies `for i in range(N)` where N is a static integer.
2.  **Expansion**: Unrolls the loop body N times.
3.  **Substitution**: Replaces the loop variable 'i' with the literal index.
4.  **Safety**: Preserves dynamic loops or loops extending safety limits.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Correctly import the Test shim instead of the deleted core class
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.static_unroll import unroll_static_loops


def rewrite_code(rewriter, code):
  """Parses code, runs pipeline via rewriter shim, returns new code."""
  tree = cst.parse_module(code)
  try:
    new_tree = rewriter.convert(tree)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter():
  # Register hook under the generic system trigger "transform_for_loop"
  # This overrides the default safety-warning plugin
  hooks._HOOKS["transform_for_loop"] = unroll_static_loops
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()
  # No definitions needed for control flow but ensure get_definition doesn't return Truthy mock
  mgr.get_definition.return_value = None
  # Ensure trait lookups don't crash
  mgr.get_framework_config.return_value = {}

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_unroll_simple_range(rewriter):
  """
  Input:
      for i in range(2):
          print(i)
  Output:
      print(0)
      print(1)
  """
  code = "for i in range(2):\n    print(i)"
  res = rewrite_code(rewriter, code)

  assert "for" not in res
  assert "print(0)" in res
  assert "print(1)" in res


def test_unroll_dependency_replacement(rewriter):
  """
  Input:
      x = 0
      for i in range(2):
          x = x + i
  Output:
      x = 0
      x = x + 0
      x = x + 1
  """
  code = """
x = 0
for i in range(2):
    x = x + i
"""
  res = rewrite_code(rewriter, code)

  # Indentation might vary slightly depending on LibCST codegen version/whitespace
  # We check for structural presence
  assert "x = x + 0" in res
  assert "x = x + 1" in res


def test_ignore_dynamic_range(rewriter):
  """
  Input: for i in range(N): ...
  Output: Preserved (or handled by fallback).
  """
  code = "for i in range(N):\n    pass"
  res = rewrite_code(rewriter, code)

  # Logic returns original node if check fails
  assert "for i in range(N):" in res


def test_safety_limit(rewriter):
  """
  Input: range(100)
  Output: Preserved (too large to unroll).
  """
  code = "for i in range(100):\n    pass"
  res = rewrite_code(rewriter, code)
  assert "range(100)" in res
