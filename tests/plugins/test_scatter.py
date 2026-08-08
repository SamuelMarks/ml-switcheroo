"""Test suite for the Scatter module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.scatter import transform_scatter


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["scatter_indexer"] = transform_scatter
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  scatter_def = {
    "variants": {
      "torch": {"api": "torch.Tensor.scatter_"},
      "jax": {"api": "at_set", "requires_plugin": "scatter_indexer"},
    }
  }

  def get_def(name):
    """Gets def."""
    if "scatter" in name:
      return ("Scatter", scatter_def)
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = lambda aid, fw: scatter_def["variants"]["jax"] if fw == "jax" else None
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Scatter": scatter_def}
  mgr.get_framework_config.return_value = {}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_scatter_simple_rewrite(rewriter):
  """Verifies the behavior of scatter simple rewrite."""
  code = "res = x.scatter_(1, idx, src)"
  res = rewrite_code(rewriter, code)
  assert "x.at[idx]" in res
  assert ".set(src)" in res
  assert ", 1," not in res and "(1," not in res


def test_scatter_add_rewrite(rewriter):
  """Verifies the behavior of scatter add rewrite."""
  code = "res = x.scatter_add_(0, idx, val)"
  res = rewrite_code(rewriter, code)
  assert "x.at[idx]" in res
  assert ".add(val)" in res


def test_scatter_keywords(rewriter):
  """Verifies the behavior of scatter keywords."""
  code = "x.scatter_(dim=0, src=updates, index=indices)"
  res = rewrite_code(rewriter, code)
  assert "x.at[indices]" in res
  assert ".set(updates)" in res


def test_ignore_tf_target(rewriter):
  """Verifies the behavior of ignore tf target."""
  rewriter.context.config.target_framework = "tensorflow"
  rewriter.context.hook_context.target_fw = "tensorflow"
  code = "x.scatter_(1, i, v)"
  res = rewrite_code(rewriter, code)
  assert ".at[" not in res


def test_missing_attribute_func():
  """Verifies behavior when node.func is not an Attribute."""
  node = cst.Call(
    func=cst.Name("scatter"), args=[cst.Arg(cst.Integer("1")), cst.Arg(cst.Integer("2")), cst.Arg(cst.Integer("3"))]
  )
  ctx = MagicMock()
  ctx.target_fw = "jax"
  res = transform_scatter(node, ctx)
  assert res is node


def test_missing_args():
  """Verifies behavior when there are fewer than 3 arguments."""
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")),
    args=[cst.Arg(cst.Integer("1")), cst.Arg(cst.Name("idx"))],
  )
  ctx = MagicMock()
  ctx.target_fw = "jax"
  res = transform_scatter(node, ctx)
  assert res is node
