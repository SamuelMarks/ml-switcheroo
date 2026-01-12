"""
Tests for Checkpoint Keys Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.checkpoint_keys import transform_checkpoint_keys, KEY_MAPPER_SOURCE


def rewrite_code(rewriter, code):
  """Executes the rewriter pipeline."""
  tree = cst.parse_module(code)
  # Use convert() via shim
  return rewriter.convert(tree).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["checkpoint_mapper"] = transform_checkpoint_keys
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define LoadStateDict
  load_def = {
    "variants": {
      "torch": {"api": "torch.nn.Module.load_state_dict"},
      "jax": {
        "api": "CustomKeyMapper",  # Nominal
        "requires_plugin": "checkpoint_mapper",
      },
    }
  }

  mgr.get_definition.side_effect = lambda n: (("LoadState", load_def) if "load_state_dict" in n else None)
  mgr.resolve_variant.side_effect = lambda aid, fw: load_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"LoadState": load_def}
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_load_state_dict_rewrite(rewriter):
  """
  Input: model.load_state_dict(sd)
  Output: KeyMapper.from_torch(sd)
  """
  code = "model.load_state_dict(sd)"
  res = rewrite_code(rewriter, code)

  assert "KeyMapper.from_torch" in res
  assert "(sd)" in res
  assert "strict" not in res  # extra args stripped by simple plugin logic selection


def test_load_state_dict_kwargs(rewriter):
  """
  Input: x.load_state_dict(state_dict=y, strict=False)
  Output: KeyMapper.from_torch(y)
  """
  code = "x.load_state_dict(state_dict=y, strict=False)"
  res = rewrite_code(rewriter, code)

  assert "KeyMapper.from_torch(y)" in res.replace(" ", "")


def test_mapper_source_availability():
  """
  Ensure the plugin exposes the source code for the runtime utility.
  """
  assert "class KeyMapper" in KEY_MAPPER_SOURCE
  assert "map_name" in KEY_MAPPER_SOURCE
  assert "map_value" in KEY_MAPPER_SOURCE
  # Check simple heuristic presence
  assert "transpose" in KEY_MAPPER_SOURCE
  assert "replace" in KEY_MAPPER_SOURCE
