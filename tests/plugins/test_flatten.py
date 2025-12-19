"""
Tests for Flatten Range Plugin.

Verifies:
1. `flatten(x)` -> `ravel(x)`.
2. `flatten(x, 1)` -> `reshape(x, (x.shape[0], -1))`.
3. `flatten(x, start_dim=1)` -> `reshape(x, (x.shape[0], -1))`.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.flatten import transform_flatten


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
    if aid == "Flatten" and fw == "jax":
      return flatten_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.is_verified.return_value = True

  # Setup context lookup for plugin internals
  # The plugin asks context what API to use.
  # We mock the context's lookup_api method implicitly by ensuring Manager returns data
  # But usually context calls `mgr.get_known_apis`.
  mgr.get_known_apis.return_value = {
    "flatten_range": {"variants": {"jax": {"api": "jnp.reshape"}}},
    "flatten_full": {"variants": {"jax": {"api": "jnp.ravel"}}},
  }

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
  assert "start_dim" not in clean  # Should strip kwargs


def test_complex_input_duplication(rewriter):
  """
  Input: y = torch.flatten(self.conv(x), 1)
  Output: y = jnp.reshape(self.conv(x), (self.conv(x).shape[0], -1))
  Warning: This duplicates execution in generated code, but is structurally correct translation.
  """
  code = "y = torch.flatten(self.conv(x), 1)"
  res = rewrite_code(rewriter, code)

  assert res.count("self.conv(x)") == 2
