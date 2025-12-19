"""
Tests for Gather Semantics Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.gather import transform_gather


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["gather_adapter"] = transform_gather
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  gather_def = {
    "variants": {
      "torch": {"api": "torch.gather"},
      "jax": {"api": "jnp.take_along_axis", "requires_plugin": "gather_adapter"},
    }
  }

  mgr.get_definition.side_effect = lambda n: ("Gather", gather_def) if "gather" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: gather_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"Gather": gather_def}
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_gather_method_reorder(rewriter):
  """
  Input: x.gather(1, indices)  -- (dim, index)
  Output: jnp.take_along_axis(x, indices, 1) -- (arr, indices, axis)
  """
  code = "y = x.gather(1, indices)"
  res = rewrite_code(rewriter, code)

  assert "jnp.take_along_axis" in res
  # Check order: x, indices, 1
  # We stripped keywords so simple string check works
  clean = res.replace(" ", "")
  # Robust check against trailing commas
  assert "(x,indices,1)" in clean or "(x,indices,1,)" in clean


def test_gather_function_reorder(rewriter):
  """
  Input: torch.gather(x, 1, idx)
  Output: jnp.take_along_axis(x, idx, 1)
  """
  code = "y = torch.gather(x, 1, idx)"
  res = rewrite_code(rewriter, code)

  clean = res.replace(" ", "")
  assert "(x,idx,1)" in clean or "(x,idx,1,)" in clean


def test_gather_keywords(rewriter):
  """
  Input: x.gather(dim=1, index=idx)
  Output: jnp.take_along_axis(x, idx, 1)
  """
  code = "y = x.gather(dim=1, index=idx)"
  res = rewrite_code(rewriter, code)

  clean = res.replace(" ", "")
  assert "(x,idx,1)" in clean or "(x,idx,1,)" in clean


def test_gather_ignores_extras(rewriter):
  """
  Input: torch.gather(x, 1, idx, out=y, sparse_grad=True)
  Output: jnp.take_along_axis(x, idx, 1)
  """
  code = "y = torch.gather(x, 1, idx, out=z)"
  res = rewrite_code(rewriter, code)

  assert "out=" not in res
  assert "z" not in res  # Arg removed entirely by plugin logic that only picks supported args
