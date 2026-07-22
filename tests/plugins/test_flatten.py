"""Test suite for the Flatten module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["flatten_range"] = transform_flatten
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  flatten_def = {
    "variants": {"torch": {"api": "torch.flatten"}, "jax": {"api": "jnp.reshape", "requires_plugin": "flatten_range"}}
  }
  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  def resolve_variant(aid, fw):
    """Resolves variant."""
    if fw == "jax":
      if aid == "flatten_range":
        return {"api": "jnp.reshape"}
      if aid == "flatten_full":
        return {"api": "jnp.ravel"}
      if aid == "Flatten":
        return flatten_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.is_verified.return_value = True

  def get_config(fw):
    """Gets configuration."""
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter):
  """Verifies the behavior of flatten batch preserve."""
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in res.replace(" ", "")


def test_flatten_passthrough_missing_def(rewriter):
  """Verifies the behavior of flatten passthrough missing def."""
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  rewriter.semantics.get_framework_config.side_effect = lambda f: {
    "plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)
  }
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten" in res


def test_flatten_empty_args(rewriter):
  """Verifies the behavior of flatten empty arguments."""
  code = "y = torch.flatten()"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten()" in res


def test_flatten_positional_args_jax_collapse(rewriter):
  """Verifies the behavior of flatten positional arguments JAX collapse."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code = "y = torch.flatten(x, 1, 2)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_kwargs_jax_collapse(rewriter):
  """Verifies the behavior of flatten keyword arguments JAX collapse."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code = "y = torch.flatten(x, start_dim=1, end_dim=-1)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jax.lax.collapse(x, 1, x.ndim)" in res


def test_flatten_full_ravel(rewriter):
  """Verifies the behavior of flatten full ravel."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jnp.ravel")
  code = "y = torch.flatten(x, 0, -1)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.ravel(x" in res


def test_flatten_value_errors(rewriter):
  """Verifies the behavior of flatten value errors."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code = "y = torch.flatten(x, 0x1A, 0x1B)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jax.lax.collapse(x, 0, x.ndim)" in res


def test_flatten_end_dim_kwarg(rewriter):
  """Verifies the behavior of flatten end dim keyword argument."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code = "y = torch.flatten(x, start_dim=1, end_dim=2)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_fallback_lookups():
  """Verifies the behavior of flatten fallback lookups."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid):
    """Provides a mock lookup for testing."""
    if aid == "flatten_full":
      return "jnp.ravel"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  node = cst.parse_module("torch.flatten(x, 0, -1)").body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.ravel(x" in res


def test_flatten_fallback_lookups_range():
  """Verifies the behavior of flatten fallback lookups range."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid):
    """Provides a mock lookup for testing."""
    if aid == "flatten_range":
      return "jnp.reshape"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  node = cst.parse_module("torch.flatten(x, 1)").body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.reshape" in res


def test_flatten_comma_injection(rewriter):
  """Verifies the behavior of flatten comma injection."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jnp.reshape")
  code = "y = torch.flatten(x, start_dim=1)"
  node = cst.parse_module(code).body[0].body[0].value
  arg_x = node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  node = node.with_changes(args=[arg_x, node.args[1]])
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.reshape(x,(x.shape[0],-1))" in res.replace(" ", "")


def test_flatten_return_node_end():
  """Verifies the behavior of flatten return node end."""
  ctx = MagicMock()
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None
  node = cst.parse_module("torch.flatten(x, 2, 3)").body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "torch.flatten(x, 2, 3)" in res


def test_flatten_with_comma(rewriter):
  """Verifies the behavior of flatten with comma."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jnp.reshape")
  code = "y = torch.flatten(x, 1)"
  node = cst.parse_module(code).body[0].body[0].value
  node = node.with_changes(args=[node.args[0].with_changes(comma=cst.Comma()), node.args[1]])
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.reshape" in res
