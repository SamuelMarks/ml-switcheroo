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
def rewriter_factory():
  hooks._HOOKS["view_semantics"] = transform_view_semantics
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  def_map = {
    "variants": {
      "torch": {"api": "torch.Tensor.view"},
      "jax": {"api": "jnp.reshape", "requires_plugin": "view_semantics"},
    }
  }

  mgr.get_known_apis.return_value = {"View": def_map, "Reshape": def_map}

  def resolve(aid, fw):
    if aid in ["Reshape", "View"] and fw == "jax":
      return def_map["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition.side_effect = lambda n: ("View", def_map) if "view" in n else None

  # Empty Config Default
  mgr.get_framework_config.return_value = {}

  def create(target):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_view_basic_mapping_jax(rewriter_factory):
  rw = rewriter_factory("jax")
  code = "y = x.view(a, b)"
  res = rewrite_code(rw, code)
  assert "jnp.reshape(x" in res
  assert "(a, b)" in res


def test_view_passthrough_missing(rewriter_factory):
  rw = rewriter_factory("numpy")
  code = "y = x.view(a, b)"
  res = rewrite_code(rw, code)
  assert "x.view(a, b)" in res
