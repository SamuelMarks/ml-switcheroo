"""
Tests for Flatten Range Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["flatten_range"] = transform_flatten
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  flatten_def = {
    "variants": {
      "torch": {"api": "torch.flatten"},
      "jax": {"api": "jnp.reshape", "requires_plugin": "flatten_range"},
    }
  }

  # Internal definitions for fallback lookup
  range_def = {"variants": {"jax": {"api": "jnp.reshape"}}}
  full_def = {"variants": {"jax": {"api": "jnp.ravel"}}}

  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  # Mock lookup_api context helper
  def resolve_variant(aid, fw):
    # Plugin looks up these IDs
    if fw == "jax":
      if aid == "flatten_range":
        return {"api": "jnp.reshape"}
      if aid == "flatten_full":
        return {"api": "jnp.ravel"}
      if aid == "Flatten":
        return flatten_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.is_verified.return_value = True

  def get_config(fw):
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter):
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in res.replace(" ", "")


def test_flatten_passthrough_missing_def(rewriter):
  # Switch target to one without definitions (e.g. numpy)
  # Update configuration on shared context
  rewriter.context.config.target_framework = "numpy"
  # Update hook context (since it persists copy)
  rewriter.context.hook_context.target_fw = "numpy"

  # Feature flag enabled, but lookups will fail because resolve_variant mock checks 'jax'
  rewriter.semantics.get_framework_config.side_effect = lambda f: {
    "plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)
  }

  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten" in res
