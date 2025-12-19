"""
Tests for View Semantics Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.reshape import transform_view_semantics


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["view_semantics"] = transform_view_semantics
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  view_def = {
    "variants": {
      "torch": {"api": "torch.Tensor.view"},
      "jax": {"api": "jnp.reshape", "requires_plugin": "view_semantics"},
    }
  }

  mgr.get_definition.side_effect = lambda n: ("View", view_def) if "view" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: view_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"View": view_def}
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  # Default strict_mode = False
  cfg.strict_mode = False

  return PivotRewriter(mgr, cfg)


def test_view_basic_mapping(rewriter):
  """
  Input: x.view(a, b)
  Output: jax.numpy.reshape(x, (a, b))
  """
  code = "y = x.view(a, b)"
  res = rewrite_code(rewriter, code)

  assert "jax.numpy.reshape(x" in res
  assert "(a, b)" in res


def test_strict_mode_injection(rewriter):
  """
  Input: x.view(a, b) with strict_mode=True
  Output: jax.numpy.reshape(x, (a, b)).block_until_ready()
  """
  rewriter.ctx._runtime_config.strict_mode = True

  code = "y = x.view(a, b)"
  res = rewrite_code(rewriter, code)

  assert "jax.numpy.reshape" in res
  assert ".block_until_ready()" in res


def test_single_dim_tuple_packing(rewriter):
  """
  Input: x.view(10)
  Output: jax.numpy.reshape(x, (10,))
  """
  code = "y = x.view(10)"
  res = rewrite_code(rewriter, code)

  # Check for tuple construction
  assert "jax.numpy.reshape" in res
  clean = res.replace(" ", "")
  assert "(10,)" in clean


def test_non_strict_by_default(rewriter):
  rewriter.ctx._runtime_config.strict_mode = False
  code = "y = x.view(a)"
  res = rewrite_code(rewriter, code)
  assert "block_until_ready" not in res
