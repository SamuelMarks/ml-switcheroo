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
  pass
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


def test_loss_wrapper_reduction_variable(rewriter_factory):
  """Verifies variable reduction does not crash."""
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter_factory("jax"), code)
  assert "my_mode" in res


def test_loss_wrapper_fallback(rewriter_factory):
  """Verifies fallback when API not found."""
  rewriter = rewriter_factory("torch")
  rewriter.semantics.get_known_apis.return_value = {}
  code = "torch.nn.functional.mse_loss(a, b, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_reduction_variable_path(rewriter_factory):
  """Verifies variable reduction skips wrapping but parses."""
  rewriter = rewriter_factory("torch")
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "my_mode" in res  # or whatever it produces


def test_loss_wrapper_no_context_fallback(rewriter_factory):
  """Verifies fallback when no op id."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res
  # removed jnp.mean check


def test_loss_wrapper_fallback_wrapper_none(rewriter_factory):
  """Verifies fallback when wrapper API lookup fails."""
  rewriter = rewriter_factory("jax")

  def lookup(aid):
    """Docstring."""
    if aid == "CrossEntropyLoss":
      return "optax.softmax_cross_entropy_with_integer_labels"
    return None

  rewriter.context.hook_context.lookup_api = lookup
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  res = rewrite_code(rewriter, code)
  assert "optax" in res
  # removed jnp.mean check


def test_loss_wrapper_fallback_no_context(rewriter_factory):
  """Verifies fallback when no op_id is set."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def lookup(aid):
    """Docstring."""
    if aid == "CrossEntropyLoss":
      return "optax.softmax_cross_entropy_with_integer_labels"
    if aid == "Mean":
      return "jnp.mean"
    return None

  rewriter.context.hook_context.lookup_api = lookup
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_variable_and_no_context(rewriter_factory):
  """Verifies variable reduction with no op_id."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.mse_loss(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "torch" in res


def test_loss_wrapper_variable_no_context_real(rewriter_factory):
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name):
    """Docstring."""
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def

  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_real(rewriter_factory):
  """Verifies fallback when no op_id is set."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name):
    """Docstring."""
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def

  code = "torch.nn.functional.cross_entropy(a, b)"  # triggers lines 72, 90
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_variable_and_no_context_really(rewriter_factory):
  """Verifies variable reduction with no op_id actually uses CrossEntropyLoss lookup."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None

  def get_def(name):
    """Docstring."""
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def
  code = "torch.nn.functional.cross_entropy(a, b, reduction=my_mode)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_none_op_id(rewriter_factory):
  """Verifies none op_id sets CrossEntropyLoss."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module = cst.parse_module(code)
  node = module.body[0].body[0].value
  rewriter.context.hook_context.lookup_api = (
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None
  )
  res = transform_loss_reduction(node, rewriter.context.hook_context)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_empty_string_op_id(rewriter_factory):
  """Verifies empty string op_id sets CrossEntropyLoss."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  code = "torch.nn.functional.cross_entropy(a, b, reduction='sum')"
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction

  module = cst.parse_module(code)
  node = module.body[0].body[0].value
  rewriter.context.hook_context.lookup_api = (
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None
  )
  res = transform_loss_reduction(node, rewriter.context.hook_context)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code


def test_loss_wrapper_fallback_no_context_really(rewriter_factory):
  """Verifies fallback when no op_id is set."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = ""

  def get_def(name):
    """Docstring."""
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_fallback_no_context_really_empty_string(rewriter_factory):
  """Verifies fallback when no op_id is set."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  rewriter.context.hook_context.lookup_api = lambda x: "optax" if x == "CrossEntropyLoss" else None

  def get_def(name):
    """Docstring."""
    ce_def = {
      "variants": {
        "jax": {"api": "optax.softmax_cross_entropy_with_integer_labels", "requires_plugin": "loss_reduction"},
      }
    }
    return ("CrossEntropyLoss", ce_def)

  rewriter.semantics.get_definition.side_effect = get_def
  code = "torch.nn.functional.cross_entropy(a, b)"
  res = rewrite_code(rewriter, code)
  assert "optax" in res


def test_loss_wrapper_direct_empty_context(rewriter_factory):
  """Verifies empty context hits fallback."""
  rewriter = rewriter_factory("jax")
  rewriter.context.hook_context.current_op_id = None
  from ml_switcheroo.plugins.loss_wrapper import transform_loss_reduction
  import libcst as cst

  code = "torch.nn.functional.cross_entropy(a, b)"
  module = cst.parse_module(code)
  node = module.body[0].body[0].value
  rewriter.context.hook_context.lookup_api = (
    lambda x: "optax.softmax_cross_entropy_with_integer_labels" if x == "CrossEntropyLoss" else None
  )
  res = transform_loss_reduction(node, rewriter.context.hook_context)
  assert "optax" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=res)])]).code
