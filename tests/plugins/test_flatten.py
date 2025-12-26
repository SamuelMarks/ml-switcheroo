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

# Fix: Import for traits support
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["flatten_range"] = transform_flatten
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  flatten_def = {
    "variants": {"torch": {"api": "torch.flatten"}, "jax": {"api": "jnp.reshape", "requires_plugin": "flatten_range"}}
  }

  # We mock lookup for "Flatten" abstract op
  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  # We mock lookup_api context helper to return jnp.reshape for FlattenRange strategy
  def resolve_variant(aid, fw):
    # Plugin explicitly requests API for "flatten_range" or "flatten_full"
    # So we must handle those IDs
    if aid == "flatten_range":
      return {"api": "jnp.reshape"}
    if aid == "flatten_full":
      return {"api": "jnp.ravel"}
    if aid == "Flatten" and fw == "jax":
      return flatten_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.is_verified.return_value = True

  # FIX: Ensure get_framework_config returns compatible traits
  def get_config(fw):
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config

  # Setup context lookup for plugin internals via get_definition_by_id indirect mock
  # because resolve_variant uses it if not overridden. But here we override resolve_variant.

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter):
  """
  Input: y = torch.flatten(x, 1)
  Output: y = jnp.reshape(x, (x.shape[0], -1))
  """
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)

  assert "jnp.reshape" in res
  clean = res.replace(" ", "")
  assert "(x.shape[0],-1)" in clean


def test_flatten_full(rewriter):
  """
  Input: y = torch.flatten(x)
  Output: y = jnp.ravel(x)
  """
  code = "y = torch.flatten(x)"
  res = rewrite_code(rewriter, code)
  assert "jnp.ravel(x)" in res


def test_flatten_keyword_arg(rewriter):
  """
  Input: y = torch.flatten(x, start_dim=1)
  Output: reshape logic
  """
  code = "y = torch.flatten(x, start_dim=1)"
  res = rewrite_code(rewriter, code)

  clean = res.replace(" ", "")
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in clean
  assert "start_dim" not in clean
