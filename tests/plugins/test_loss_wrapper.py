"""
Tests for Loss Reduction Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["loss_reduction"] = transform_loss_reduction
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define CrossEntropy
  ce_def = {
    "variants": {
      "torch": {"api": "torch.nn.functional.cross_entropy"},
      "jax": {
        "api": "optax.softmax_cross_entropy_with_integer_labels",
        "requires_plugin": "loss_reduction",
      },
    }
  }

  def get_def(name):
    return ("CrossEntropy", ce_def) if "cross_entropy" in name else None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = lambda aid, fw: ce_def["variants"]["jax"]
  mgr.get_known_apis.return_value = {"CrossEntropy": ce_def}
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  rw = PivotRewriter(mgr, cfg)

  # Hack: Inject current_op_id into context for the hook to find the name
  # The hook logic relies on rewriter context state
  rw.ctx.current_op_id = "CrossEntropy"

  return rw


def test_default_reduction_is_mean(rewriter):
  """
  Input: F.cross_entropy(logits, target)
  Output: jax.numpy.mean(optax.softmax_cross_entropy...(logits, target))
  """
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)

  # Updated: Plugin uses full path 'jax.numpy.mean'
  assert "jax.numpy.mean" in res
  assert "optax.softmax_cross_entropy" in res
  assert "reduction" not in res


def test_explicit_sum_reduction(rewriter):
  """
  Input: F.cross_entropy(..., reduction='sum')
  Output: jax.numpy.sum(...)
  """
  code = "loss = F.cross_entropy(pred, y, reduction='sum')"
  res = rewrite_code(rewriter, code)

  assert "jax.numpy.sum" in res
  assert "reduction" not in res


def test_reduction_none(rewriter):
  """
  Input: F.cross_entropy(..., reduction='none')
  Output: optax.softmax_cross_(...) (No wrapper)
  """
  code = "loss = F.cross_entropy(x, y, reduction='none')"
  res = rewrite_code(rewriter, code)

  assert "jax.numpy.mean" not in res
  assert "jax.numpy.sum" not in res
  assert "optax.softmax" in res


def test_strip_reduction_arg(rewriter):
  """
  Ensure the reduction argument is actually removed from the inner call.
  """
  code = "loss = F.cross_entropy(x, y, reduction='mean')"
  res = rewrite_code(rewriter, code)

  # Inner call shouldn't have reduction
  # Valid split key matching plugin output
  inner = res.split("jax.numpy.mean(")[1]
  assert "reduction" not in inner
