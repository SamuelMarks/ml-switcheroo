"""
Tests for Type Casting Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.casting import transform_casting


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["type_methods"] = transform_casting
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Universal definition for all cast methods
  cast_def = {
    "variants": {
      "torch": {"api": "torch.Tensor.float"},  # Placeholder
      "jax": {"api": "astype", "requires_plugin": "type_methods"},
    }
  }

  # Mock lookup
  def get_def(name):
    # Handle cases like "float", "x.float", "torch.Tensor.float"
    leaf_name = name.split(".")[-1]

    # Match any known cast name usually found in discovery
    if leaf_name in ["float", "long", "int", "half", "double", "bool"]:
      return ("Cast", cast_def)
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = lambda aid, fw: cast_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"Cast": cast_def}  # Generic bucket
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_float_cast(rewriter):
  """
  Input: x.float()
  Output: x.astype(jax.numpy.float32)
  """
  code = "y = x.float()"
  res = rewrite_code(rewriter, code)

  assert ".astype" in res
  assert "jax.numpy.float32" in res
  assert ".float(" not in res


def test_long_cast(rewriter):
  """
  Input: x.long()
  Output: x.astype(jax.numpy.int64)
  """
  code = "idx = mask.long()"
  res = rewrite_code(rewriter, code)

  assert "jax.numpy.int64" in res


def test_bool_cast(rewriter):
  """
  Input: x.bool()
  Output: x.astype(jax.numpy.bool_)
  """
  code = "mask = x.bool()"
  res = rewrite_code(rewriter, code)

  assert "jax.numpy.bool_" in res


def test_ignores_non_casts(rewriter):
  """
  Input: x.other_method()
  Output: No change (plugin returns node, base rewriter handles rename if mapped)
  """
  code = "y = x.item()"  # item is not in TYPE_MAP
  res = rewrite_code(rewriter, code)

  assert "astype" not in res
