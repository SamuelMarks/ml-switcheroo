"""
Tests for Padding Normalization Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.padding import transform_padding
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["padding_converter"] = transform_padding
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  pad_def = {
    "variants": {
      "torch": {"api": "torch.nn.functional.pad"},
      "jax": {"api": "jnp.pad", "requires_plugin": "padding_converter"},
    }
  }

  mgr.get_definition.side_effect = lambda n: ("Pad", pad_def) if "pad" in n else None

  def resolve(aid, fw):
    if aid == "Pad" and fw == "jax":
      return pad_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"Pad": pad_def}
  mgr.is_verified.return_value = True

  # Enable Traits
  def get_config(fw):
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_padding_2d_nchw(rewriter):
  code = "y = F.pad(x, (1, 2, 3, 4))"
  res = rewrite_code(rewriter, code)
  assert "jnp.pad" in res
  assert "((0,0),(0,0),(3,4),(1,2))" in res.replace(" ", "")


def test_padding_passthrough_missing(rewriter):
  rewriter.ctx.target_fw = "unknown"
  code = "y = F.pad(x, (1, 2, 3, 4))"
  res = rewrite_code(rewriter, code)
  assert "F.pad" in res
