"""Test suite for the Loss Wrapper module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter_factory():
  """Provides a mock rewriter factory for testing."""
  hooks._HOOKS["loss_reduction"] = transform_loss_reduction
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  ce_def = {
    "variants": {
      "torch": {"api": "torch.nn.functional.cross_entropy"},
      "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      "tensorflow": {"api": "tf.nn.sparse_softmax_cross_entropy_with_logits", "requires_plugin": "loss_reduction"},
    }
  }
  mean_def = {"variants": {"jax": {"api": "jnp.mean"}, "tensorflow": {"api": "tf.reduce_mean"}}}
  sum_def = {"variants": {"jax": {"api": "jnp.sum"}, "tensorflow": {"api": "tf.reduce_sum"}}}
  all_defs = {"CrossEntropyLoss": ce_def, "Mean": mean_def, "Sum": sum_def}

  def get_def(name):
    """Gets def."""
    return ("CrossEntropyLoss", ce_def) if "cross_entropy" in name else None

  def resolve_variant(aid, fw):
    """Resolves variant."""
    if aid in all_defs and fw in all_defs[aid]["variants"]:
      return all_defs[aid]["variants"][fw]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve_variant
  mgr.get_known_apis.return_value = all_defs
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  def create(target_fw):
    """Creates ."""
    cfg = RuntimeConfig(source_framework="torch", target_framework=target_fw)
    rw = PivotRewriter(mgr, cfg)
    rw.ctx.current_op_id = "CrossEntropyLoss"
    return rw

  return create


def test_jax_mean_reduction(rewriter_factory):
  """Verifies the behavior of JAX mean reduction."""
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)
  assert "jnp.mean" in res
  assert "optax.softmax_cross_entropy" in res
  assert "reduction" not in res


def test_tensorflow_mean_reduction(rewriter_factory):
  """Verifies the behavior of TensorFlow mean reduction."""
  rewriter = rewriter_factory("tensorflow")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)
  assert "tf.reduce_mean" in res
  assert "tf.nn.sparse_softmax_cross_entropy" in res


def test_explicit_sum_reduction(rewriter_factory):
  """Verifies the behavior of explicit sum reduction."""
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(pred, y, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "jnp.sum" in res
  assert "reduction" not in res


def test_reduction_none(rewriter_factory):
  """Verifies the behavior of reduction none."""
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(x, y, reduction='none')"
  res = rewrite_code(rewriter, code)
  assert "jnp.mean" not in res
  assert "jnp.sum" not in res
  assert "optax.softmax" in res
