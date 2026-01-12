"""
Tests for Gradient Clipping Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.clipping import transform_grad_clipping
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  """Executes the rewriter pipeline."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["grad_clipper"] = transform_grad_clipping
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define Clipping
  clip_def = {
    "variants": {
      "torch": {"api": "torch.nn.utils.clip_grad_norm_"},
      "jax": {
        "api": "optax.clip_by_global_norm",
        "requires_plugin": "grad_clipper",
      },
    }
  }

  mgr.get_definition.side_effect = lambda n: ("ClipGrads", clip_def) if "clip_grad" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: clip_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"ClipGrads": clip_def}
  mgr.is_verified.return_value = True

  # FIX: Enable functional state trait
  mgr.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_functional_state=True)}

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_clip_transform(rewriter):
  """
  Input: torch.nn.utils.clip_grad_norm_(grads, 1.0)
  Output: optax.clip_by_global_norm(1.0).update(grads, None)[0]
  """
  code = "torch.nn.utils.clip_grad_norm_(grads, 1.0)"
  res = rewrite_code(rewriter, code)

  assert "optax.clip_by_global_norm(1.0)" in res
  assert ".update(grads, None)" in res
  assert ")[0]" in res.strip()


def test_clip_with_variable_args(rewriter):
  """
  Input: clip_grad_norm_(g, max_norm)
  Output: optax... (max_norm) ... (g, None)[0]
  """
  code = "clip_grad_norm_(g, max_val)"
  res = rewrite_code(rewriter, code)

  assert "clip_by_global_norm(max_val)" in res
  assert "update(g," in res


def test_ignores_if_traits_missing(rewriter):
  # Disable traits
  rewriter.semantics.get_framework_config.return_value = {"plugin_traits": PluginTraits(requires_functional_state=False)}

  code = "torch.nn.utils.clip_grad_norm_(g, 1.0)"
  res = rewrite_code(rewriter, code)

  # Should NOT transform to optax
  assert "optax" not in res
