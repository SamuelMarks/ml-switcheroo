"""
Tests for Scatter/Gather Syntax Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.scatter import transform_scatter


def rewrite_code(rewriter, code):
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["scatter_indexer"] = transform_scatter
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define scatter variants
  scatter_def = {
    "variants": {
      "torch": {"api": "torch.Tensor.scatter_"},
      "jax": {"api": "at_set", "requires_plugin": "scatter_indexer"},
    }
  }

  # Mock Lookups
  def get_def(name):
    if "scatter" in name:
      return ("Scatter", scatter_def)
    return None

  mgr.get_definition.side_effect = get_def

  # Wiring Logic: Only JAX requests the plugin
  mgr.resolve_variant.side_effect = lambda aid, fw: scatter_def["variants"]["jax"] if fw == "jax" else None

  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Scatter": scatter_def}
  mgr.get_framework_config.return_value = {}

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_scatter_simple_rewrite(rewriter):
  """
  Input: x.scatter_(1, idx, src)
  Output: x.at[idx].set(src)
  """
  code = "res = x.scatter_(1, idx, src)"
  res = rewrite_code(rewriter, code)

  assert "x.at[idx]" in res
  assert ".set(src)" in res
  # Ensure dim (1) is stripped (current limitation behavior)
  assert ", 1," not in res and "(1," not in res


def test_scatter_add_rewrite(rewriter):
  """
  Input: x.scatter_add_(0, idx, val)
  Output: x.at[idx].add(val)
  """
  # Need to ensure scatter_add matches definition mock
  code = "res = x.scatter_add_(0, idx, val)"
  res = rewrite_code(rewriter, code)

  assert "x.at[idx]" in res
  assert ".add(val)" in res


def test_scatter_keywords(rewriter):
  """
  Input: x.scatter_(dim=0, src=updates, index=indices)
  Output: x.at[indices].set(updates)
  """
  code = "x.scatter_(dim=0, src=updates, index=indices)"
  res = rewrite_code(rewriter, code)

  assert "x.at[indices]" in res
  assert ".set(updates)" in res


def test_ignore_tf_target(rewriter):
  """
  Verify that if the target framework is NOT JAX (and thus not wired to the plugin),
  the conversion is skipped.
  Implementation relies on the SemanticsManager returning None for 'tensorflow'.
  """
  # Update Context Config
  rewriter.context.config.target_framework = "tensorflow"
  rewriter.context.hook_context.target_fw = "tensorflow"

  code = "x.scatter_(1, i, v)"
  res = rewrite_code(rewriter, code)

  # Should not produce .at[] syntax because resolves to None
  assert ".at[" not in res
