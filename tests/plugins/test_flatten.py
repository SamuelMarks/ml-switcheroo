"""
Tests for Flatten Range Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.semantics.schema import PluginTraits


def rewrite_code(rewriter, code):
  """Executes pipeline."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Function docstring."""
  hooks._HOOKS["flatten_range"] = transform_flatten
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  flatten_def = {
    "variants": {
      "torch": {"api": "torch.flatten"},
      "jax": {"api": "jnp.reshape", "requires_plugin": "flatten_range"},
    }
  }

  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  # Mock lookup_api context helper
  def resolve_variant(aid, fw):
    """Function docstring."""
    # Plugin looks up these IDs
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
    """Function docstring."""
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter):
  """Function docstring."""
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in res.replace(" ", "")


def test_flatten_passthrough_missing_def(rewriter):
  """Function docstring."""
  # Switch target to one without definitions (e.g. numpy)
  # Update configuration on shared context
  rewriter.context.config.target_framework = "numpy"
  # Update hook context (since it persists copy)
  rewriter.context.hook_context.target_fw = "numpy"

  # Feature flag enabled, but lookups will fail because resolve_variant mock checks 'jax'
  rewriter.semantics.get_framework_config.side_effect = lambda f: {
    "plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)
  }

  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten" in res


def test_flatten_empty_args(rewriter):
  """Test flatten with empty args."""
  code = "y = torch.flatten()"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten()" in res


def test_flatten_positional_args_jax_collapse(rewriter):
  """Test flatten with positional args mapping to JAX collapse."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code = "y = torch.flatten(x, 1, 2)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_kwargs_jax_collapse(rewriter):
  """Test flatten with kwargs mapping to JAX collapse."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code = "y = torch.flatten(x, start_dim=1, end_dim=-1)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jax.lax.collapse(x, 1, x.ndim)" in res


def test_flatten_full_ravel(rewriter):
  """Test flatten mapping to ravel."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jnp.ravel")
  code = "y = torch.flatten(x, 0, -1)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.ravel(x" in res


def test_flatten_value_errors(rewriter):
  """Test value errors when parsing dims."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  # 0x1A will parse as cst.Integer but int('0x1A') raises ValueError (base 10)
  code = "y = torch.flatten(x, 0x1A, 0x1B)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  # Defaults 0, -1 used
  assert "jax.lax.collapse(x, 0, x.ndim)" in res


def test_flatten_end_dim_kwarg(rewriter):
  """Test end_dim kwarg parsing."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code = "y = torch.flatten(x, start_dim=1, end_dim=2)"
  node = cst.parse_module(code).body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_fallback_lookups():
  """Test that fallback target API lookups work."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid):
    if aid == "flatten_full":
      return "jnp.ravel"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  node = cst.parse_module("torch.flatten(x, 0, -1)").body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.ravel(x" in res


def test_flatten_fallback_lookups_range():
  """Test that fallback target API lookups work for flatten_range."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid):
    if aid == "flatten_range":
      return "jnp.reshape"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  node = cst.parse_module("torch.flatten(x, 1)").body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.reshape" in res


def test_flatten_comma_injection(rewriter):
  """Test comma injection when start_dim=1 but input_arg has no comma."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jnp.reshape")
  # Construct a call manually where start_dim is from kwarg and x has no comma
  # e.g., torch.flatten(x, start_dim=1)
  code = "y = torch.flatten(x, start_dim=1)"
  node = cst.parse_module(code).body[0].body[0].value
  # Strip comma from x manually to trigger line 130
  arg_x = node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  node = node.with_changes(args=[arg_x, node.args[1]])
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.reshape(x,(x.shape[0],-1))" in res.replace(" ", "")


def test_flatten_return_node_end():
  """Test that node is returned if no matching strategies are found."""
  ctx = MagicMock()
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None
  node = cst.parse_module("torch.flatten(x, 2, 3)").body[0].body[0].value
  res_node = transform_flatten(node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "torch.flatten(x, 2, 3)" in res


def test_flatten_with_comma(rewriter):
  """Test flatten where input_arg has a comma already."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jnp.reshape")
  code = "y = torch.flatten(x, 1)"
  node = cst.parse_module(code).body[0].body[0].value
  node = node.with_changes(args=[node.args[0].with_changes(comma=cst.Comma()), node.args[1]])
  res_node = transform_flatten(node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code
  assert "jnp.reshape" in res
