"""
Integration Tests for TopK Semantics.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.topk import transform_topk


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["topk_adapter"] = transform_topk
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  topk_def = {
    "variants": {"torch": {"api": "torch.topk"}, "jax": {"api": "jax.lax.top_k", "requires_plugin": "topk_adapter"}}
  }
  mgr.get_definition.side_effect = lambda n: ("TopK", topk_def) if "topk" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: topk_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"TopK": topk_def}
  mgr.is_verified.return_value = True
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_topk_rewrapping(rewriter):
  code = "res = torch.topk(x, 5)"
  res = rewrite_code(rewriter, code)
  assert "collections.namedtuple" in res
  assert '"TopK"' in res
  assert "jax.lax.top_k(x, 5)" in res
  assert "(*" in res


def test_topk_strip_unsupported(rewriter):
  code = "res = torch.topk(x, 5, sorted=True)"
  res = rewrite_code(rewriter, code)
  assert "sorted" not in res
  # Assert presence of inner call
  assert "jax.lax.top_k(x, 5, )" in res or "jax.lax.top_k(x, 5)" in res


def test_topk_functional_call(rewriter):
  code = "res = torch.topk(t, k)"
  res = rewrite_code(rewriter, code)
  assert "jax.lax.top_k(t, k)" in res
