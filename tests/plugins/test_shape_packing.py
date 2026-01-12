"""
Tests for Shape Packing.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim from conftest
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.shape_packing import transform_shape_packing
from ml_switcheroo.frameworks.base import register_framework


def rewrite_code(rewriter, code: str) -> str:
  """Executes the rewriter pipeline."""
  tree = cst.parse_module(code)
  # Use .convert() for pipeline execution
  return rewriter.convert(tree).code


@register_framework("custom_fw")
class CustomAdapter:
  @property
  def harness_imports(self):
    return []

  def get_harness_init_code(self):
    return ""

  def get_to_numpy_code(self) -> str:
    """Implement required protocol method."""
    return "return str(obj)"

  @property
  def declared_magic_args(self):
    return []


@pytest.fixture
def rewriter_factory():
  hooks._HOOKS["pack_shape_args"] = transform_shape_packing
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  def_map = {
    "variants": {
      "torch": {"api": "torch.view"},
      "jax": {"api": "jnp.reshape", "requires_plugin": "pack_shape_args"},
      "custom_fw": {
        "api": "custom.ops.reshape",
        "requires_plugin": "pack_shape_args",
      },
    }
  }

  mgr.get_known_apis.return_value = {"Reshape": def_map}

  def resolve(aid, fw):
    if aid == "Reshape":
      return def_map["variants"].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition.side_effect = lambda n: ("Reshape", def_map) if "view" in n else None
  mgr.get_framework_config.return_value = {}

  def create(target):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_packing_jax(rewriter_factory):
  rw = rewriter_factory("jax")
  code = "y = x.view(1, 2)"
  res = rewrite_code(rw, code)
  assert "jnp.reshape(x" in res
  assert "(1, 2)" in res


def test_packing_custom_fw(rewriter_factory):
  rw = rewriter_factory("custom_fw")
  code = "y = x.view(1, 2)"
  res = rewrite_code(rw, code)
  assert "custom.ops.reshape(x" in res
  assert "(1, 2)" in res


def test_packing_missing_passthrough(rewriter_factory):
  rw = rewriter_factory("numpy")
  code = "y = x.view(1, 2)"
  res = rewrite_code(rw, code)
  assert "x.view(1, 2)" in res
