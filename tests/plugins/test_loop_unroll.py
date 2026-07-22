"""Test suite for the Loop Unroll module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.loop_unroll import transform_loops
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter_factory():
  """Provides a mock rewriter factory for testing."""
  hooks._HOOKS.clear()
  hooks._HOOK_METADATA.clear()
  hooks._HOOKS["transform_for_loop"] = transform_loops
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock(spec=SemanticsManager)
  mgr.get_definition.return_value = None

  def get_config(fw):
    """Gets configuration."""
    if fw == "torch":
      return {"plugin_traits": PluginTraits(requires_functional_control_flow=False)}
    if fw == "jax":
      return {"plugin_traits": PluginTraits(requires_functional_control_flow=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config

  def create(target):
    """Creates ."""
    cfg = RuntimeConfig(source_framework="torch", target_framework=target, strict_mode=False)
    return PivotRewriter(mgr, cfg)

  return create


def test_imperative_passthrough(rewriter_factory):
  """Verifies the behavior of imperative passthrough."""
  rewriter = rewriter_factory("torch")
  code = "\nfor i in range(10):\n    print(i)\n"
  result = rewrite_code(rewriter, code)
  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER not in result


def test_functional_range_warning(rewriter_factory):
  """Verifies the behavior of functional range warning."""
  rewriter = rewriter_factory("jax")
  code = "\nfor i in range(10):\n    x = x + i\n"
  result = rewrite_code(rewriter, code)
  assert "for i in range(10):" in result
  assert EscapeHatch.START_MARKER in result
  assert "JAX requires explicit functional loops" in result


def test_functional_iterator_warning(rewriter_factory):
  """Verifies the behavior of functional iterator warning."""
  rewriter = rewriter_factory("jax")
  code = "\nfor item in my_list:\n    print(item)\n"
  result = rewrite_code(rewriter, code)
  assert EscapeHatch.START_MARKER in result
  assert "requires structural rewrite (e.g. `scan`)" in result
