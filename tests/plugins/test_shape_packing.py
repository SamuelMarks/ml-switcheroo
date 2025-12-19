"""
Tests for View/Reshape Packer Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.shape_packing import transform_shape_packing


def rewrite_code(rewriter, code):
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter():
  hooks._HOOKS["pack_shape_args"] = transform_shape_packing
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define Reshape and View
  # Both map to the same target logic
  def_map = {
    "variants": {
      "torch": {"api": "torch.Tensor.view"},
      "jax": {"api": "jnp.reshape", "requires_plugin": "pack_shape_args"},
    }
  }

  # Context lookup simulation
  mgr.get_known_apis.return_value = {"Reshape": def_map, "View": def_map}

  def resolve(aid, fw):
    return def_map["variants"]["jax"]

  # We patch get_definition to catch "view" method call on tensor
  def get_def(name):
    if "view" in name or "reshape" in name:
      return ("Reshape", def_map)
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_method_varargs_packing(rewriter):
  """
  Input: x.view(1, 2, -1)
  Output: jnp.reshape(x, (1, 2, -1))
  """
  code = "y = x.view(1, 2, -1)"
  res = rewrite_code(rewriter, code)

  assert "jnp.reshape(x" in res
  assert "(1, 2, -1)" in res


def test_method_single_var_passthrough(rewriter):
  """
  Input: x.view(shape)
  Output: jnp.reshape(x, shape)
  """
  code = "y = x.view(new_shape)"
  res = rewrite_code(rewriter, code)

  assert "jnp.reshape(x, new_shape)" in res


def test_method_single_int_tuple(rewriter):
  """
  Input: x.view(10)  -- 1D view
  Output: jnp.reshape(x, (10,)) -- Safer explicit tuple
  """
  code = "y = x.view(10)"
  res = rewrite_code(rewriter, code)

  clean = res.replace(" ", "")
  assert "(10,)" in clean


def test_function_style_reshape(rewriter):
  """
  Input: torch.reshape(x, (a, b))
  Output: jnp.reshape(x, (a, b))
  """
  code = "y = torch.reshape(x, (a, b))"
  res = rewrite_code(rewriter, code)

  assert "jnp.reshape(x, (a, b))" in res


def test_function_style_unpacking(rewriter):
  """
  Input: torch.reshape(x, a, b) -- if supported by source
  Output: jnp.reshape(x, (a, b))
  """
  # Note: torch.reshape usually requires tuple for shape, but if user made wrapper...
  # Plugin supports arbitrary callable structure
  code = "y = torch.reshape(x, a, b)"
  res = rewrite_code(rewriter, code)
  assert "jnp.reshape(x, (a, b))" in res
