"""
Tests for Loop Unrolling Plugin (Decoupled Logic).

Verifies:
1. Passthrough on frameworks NOT requiring functional control flow.
2. Warning generation on frameworks requiring functional control flow (via Traits).
3. Correct messages for Range vs Generic loops.
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
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code: str) -> str:
  """Helper to execute the rewrite pass."""
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter_factory():
  """
  Sets up a PivotRewriter where we can control the Semantic Traits regarding loops.
  """
  # Register hooks manually
  hooks._HOOKS["transform_for_loop"] = transform_loops
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_definition.return_value = None  # No mappings needed for loop syntax

  # Mock Framework Configs
  def get_config(fw):
    if fw == "torch":
      return {"plugin_traits": PluginTraits(requires_functional_control_flow=False)}
    if fw == "jax":
      return {"plugin_traits": PluginTraits(requires_functional_control_flow=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config

  def create(target):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=False)
    return PivotRewriter(mgr, cfg)

  return create


def test_imperative_passthrough(rewriter_factory):
  """
  Verify that loops remain untouched for frameworks that support imperative flow (like Torch).
  Control: requires_functional_control_flow = False.
  """
  rewriter = rewriter_factory("torch")
  code = """ 
for i in range(10): 
    print(i) 
"""
  result = rewrite_code(rewriter, code)

  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER not in result


def test_functional_range_warning(rewriter_factory):
  """
  Verify that frameworks requiring functional flow (like JAX) get a safety warning for range().
  Control: requires_functional_control_flow = True.
  """
  rewriter = rewriter_factory("jax")
  code = """ 
for i in range(10): 
    x = x + i
"""
  result = rewrite_code(rewriter, code)

  assert "for i in range(10):" in result  # Code preserved
  assert EscapeHatch.START_MARKER in result
  assert "JAX requires explicit functional loops" in result


def test_functional_iterator_warning(rewriter_factory):
  """
  Verify generic iterator loops also get flagged with specific scan message.
  Control: requires_functional_control_flow = True.
  """
  rewriter = rewriter_factory("jax")
  code = """ 
for item in my_list: 
    print(item) 
"""
  result = rewrite_code(rewriter, code)

  assert EscapeHatch.START_MARKER in result
  assert "requires structural rewrite (e.g. `scan`)" in result
