"""
Tests for TopK Semantics Plugin (Decoupled).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.topk import transform_topk


def rewrite_code(rewriter, code):
  tree = cst.parse_module(code)
  # Use pipeline conversion
  new_tree = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter_factory():
  hooks._HOOKS["topk_adapter"] = transform_topk
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  # Define TopK with multi-backend support
  topk_def = {
    "variants": {
      "torch": {"api": "torch.topk"},
      "jax": {
        "api": "jax.lax.top_k",
        "requires_plugin": "topk_adapter",
      },
      "tensorflow": {
        "api": "tf.math.top_k",
        "requires_plugin": "topk_adapter",
      },
    }
  }

  mgr.get_definition.side_effect = lambda n: (("TopK", topk_def) if "topk" in n else None)

  # Dynamic resolver
  def resolve(aid, fw):
    if aid == "TopK" and fw in topk_def["variants"]:
      return topk_def["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = {"TopK": topk_def}
  mgr.is_verified.return_value = True

  def create(target):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_topk_rewrapping_jax(rewriter_factory):
  """Verify JAX mapping."""
  rewriter = rewriter_factory("jax")
  code = "res = torch.topk(x, 5)"
  res = rewrite_code(rewriter, code)

  assert "jax.lax.top_k" in res
  assert "collections.namedtuple" in res


def test_topk_passthrough_missing_target(rewriter_factory):
  """Verify missing target returns original."""
  rewriter = rewriter_factory("numpy")
  rewriter.context.hook_context.target_fw = "numpy"
  code = "res = torch.topk(x, 5)"
  res = rewrite_code(rewriter, code)

  assert "torch.topk" in res
  assert "collections.namedtuple" not in res
