"""
Tests for Loss Reduction Plugin via Decoupled Logic.

Verifies:
1.  Dynamic Lookup of "Mean" and "Sum" APIs.
2.  Handling of 'mean', 'sum', and 'none' reduction modes.
3.  Stripping of the 'reduction' keyword argument.
4.  Correct renaming of the inner loss function API.
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
def rewriter_factory():
  """
  Factory to create rewriter with configurable target backend mocks.
  """
  hooks._HOOKS["loss_reduction"] = transform_loss_reduction
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # Define CrossEntropy
  # Note: We include mappings for multiple frameworks to prove decoupling
  ce_def = {
    "variants": {
      "torch": {"api": "torch.nn.functional.cross_entropy"},
      "jax": {
        "api": "optax.softmax_cross_entropy_with_integer_labels",
        "requires_plugin": "loss_reduction",
      },
      "tensorflow": {"api": "tf.nn.sparse_softmax_cross_entropy_with_logits", "requires_plugin": "loss_reduction"},
    }
  }

  # Define Mean/Sum mappings
  mean_def = {
    "variants": {
      "jax": {"api": "jnp.mean"},
      "tensorflow": {"api": "tf.reduce_mean"},
    }
  }
  sum_def = {
    "variants": {
      "jax": {"api": "jnp.sum"},
      "tensorflow": {"api": "tf.reduce_sum"},
    }
  }

  # Mock Dictionary Access
  all_defs = {"CrossEntropyLoss": ce_def, "Mean": mean_def, "Sum": sum_def}

  def get_def(name):
    return ("CrossEntropyLoss", ce_def) if "cross_entropy" in name else None

  # Mock Resolve Logic
  def resolve_variant(aid, fw):
    if aid in all_defs and fw in all_defs[aid]["variants"]:
      return all_defs[aid]["variants"][fw]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve_variant
  mgr.get_known_apis.return_value = all_defs
  mgr.is_verified.return_value = True

  def create(target_fw):
    cfg = RuntimeConfig(source_framework="torch", target_framework=target_fw)
    rw = PivotRewriter(mgr, cfg)
    # Inject Op ID for context lookup
    rw.ctx.current_op_id = "CrossEntropyLoss"
    return rw

  return create


def test_jax_mean_reduction(rewriter_factory):
  """
  Scenario: Target JAX. Default reduction (mean).
  Input: F.cross_entropy(logits, target)
  Output: jnp.mean(optax.softmax_cross_entropy...(logits, target))
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)

  # Verify Dynamic Lookup: "Mean" -> "jnp.mean"
  assert "jnp.mean" in res
  assert "optax.softmax_cross_entropy" in res
  assert "reduction" not in res


def test_tensorflow_mean_reduction(rewriter_factory):
  """
  Scenario: Target TensorFlow. Default reduction.
  Input: F.cross_entropy(logits, target)
  Output: tf.reduce_mean(tf.nn.sparse_softmax...(logits, target))

  Proves removal of hardcoded jax strings.
  """
  rewriter = rewriter_factory("tensorflow")
  code = "loss = F.cross_entropy(logits, target)"
  res = rewrite_code(rewriter, code)

  assert "tf.reduce_mean" in res
  assert "tf.nn.sparse_softmax_cross_entropy" in res


def test_explicit_sum_reduction(rewriter_factory):
  """
  Scenario: Target JAX. Explicit sum reduction.
  Input: F.cross_entropy(..., reduction='sum')
  Output: jnp.sum(...)
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(pred, y, reduction='sum')"
  res = rewrite_code(rewriter, code)

  assert "jnp.sum" in res
  assert "reduction" not in res


def test_reduction_none(rewriter_factory):
  """
  Scenario: Target JAX. reduction='none'.
  Output: optax.softmax_cross_(...) (No wrapper)
  """
  rewriter = rewriter_factory("jax")
  code = "loss = F.cross_entropy(x, y, reduction='none')"
  res = rewrite_code(rewriter, code)

  assert "jnp.mean" not in res
  assert "jnp.sum" not in res
  assert "optax.softmax" in res
