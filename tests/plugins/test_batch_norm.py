"""
Tests for BatchNorm Functionalization Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.batch_norm import transform_batch_norm
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["batch_norm_unwrap"] = transform_batch_norm
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  bn_def = {
    "variants": {
      "torch": {"api": "torch.nn.BatchNorm2d"},
      "jax": {
        "api": "flax.nnx.BatchNorm",
        "requires_plugin": "batch_norm_unwrap",
      },
    }
  }

  def get_def(name):
    return ("BatchNorm", bn_def) if "BatchNorm" in name or "bn" in name else None

  # Implement trait retrieval logic
  def get_fw_config(fw):
    if fw == "jax":
      return {"plugin_traits": PluginTraits(requires_functional_state=True)}
    return {}

  mgr.get_definition.side_effect = get_def
  mgr.get_framework_config.side_effect = get_fw_config

  def resolve(aid, fw):
    return bn_def["variants"]["jax"] if fw == "jax" and aid == "BatchNorm" else None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"BatchNorm": bn_def}
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_bn_injection_and_unwrap(rewriter):
  """
  Scenario: Standard forward pass call.
  Input: y = self.bn(x)
  Output: y = self.bn(x, use_running_average=not training, mutable=['batch_stats'])[0]
  """
  code = "y = self.bn(x)"
  res = rewrite_code(rewriter, code)
  clean = res.replace(" ", "")
  # Check for presence of injected args
  assert "use_running_average=nottraining" in clean
  assert "mutable=['batch_stats']" in clean
  # Check that it ends with subscript [0]
  assert res.strip().endswith(")[0]")


def test_bn_nested_expression(rewriter):
  """
  Scenario: Nested inside activation.
  Input: y = F.relu(self.bn(x))
  """
  code = "y = F.relu(self.bn(x))"
  res = rewrite_code(rewriter, code)
  assert "self.bn(x" in res
  assert "[0])" in res


def test_bn_preserve_existing_args(rewriter):
  code = "y = self.bn(x, other=1)"
  res = rewrite_code(rewriter, code)

  # Order isn't guaranteed by dict, but logic appends to end
  assert "other=1" in res
  assert "use_running_average" in res
