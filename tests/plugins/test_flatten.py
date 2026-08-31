"""Test suite for the Flatten module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.semantics.schema import PluginTraits
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Docstring."""
  hooks._HOOKS["flatten_range"] = transform_flatten
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  flatten_def: dict[str, typing.Any] = {
    "variants": {"torch": {"api": "torch.flatten"}, "jax": {"api": "jnp.reshape", "requires_plugin": "flatten_range"}}
  }
  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  def resolve_variant(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves variant."""
    if fw == "jax":
      if aid == "flatten_range":
        return {"api": "jnp.reshape"}
      if aid == "flatten_full":
        return {"api": "jnp.ravel"}
      if aid == "Flatten":
        return typing.cast(dict[str, typing.Any], flatten_def["variants"]["jax"])
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.is_verified.return_value = True

  def get_config(fw: str) -> dict[str, typing.Any]:
    """Gets configuration."""
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten batch preserve."""
  code: str = "y = torch.flatten(x, 1)"
  res: str = rewrite_code(rewriter, code)
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in res.replace(" ", "")


def test_flatten_passthrough_missing_def(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten passthrough missing def."""
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  rewriter.semantics.get_framework_config.side_effect = lambda f: {
    "plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)
  }
  code: str = "y = torch.flatten(x, 1)"
  res: str = rewrite_code(rewriter, code)
  assert "torch.flatten" in res


def test_flatten_empty_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten empty arguments."""
  code: str = "y = torch.flatten()"
  res: str = rewrite_code(rewriter, code)
  assert "torch.flatten()" in res


def test_flatten_positional_args_jax_collapse(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten positional arguments JAX collapse."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code: str = "y = torch.flatten(x, 1, 2)"
  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Assign, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_kwargs_jax_collapse(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten keyword arguments JAX collapse."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code: str = "y = torch.flatten(x, start_dim=1, end_dim=-1)"
  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Assign, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 1, x.ndim)" in res


def test_flatten_full_ravel(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten full ravel."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jnp.ravel")
  code: str = "y = torch.flatten(x, 0, -1)"
  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Assign, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_value_errors(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten value errors."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code: str = "y = torch.flatten(x, 0x1A, 0x1B)"
  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Assign, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, ctx)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 0, x.ndim)" in res


def test_flatten_end_dim_kwarg(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten end dim keyword argument."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code: str = "y = torch.flatten(x, start_dim=1, end_dim=2)"
  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Assign, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, ctx)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_fallback_lookups() -> None:
  """Verifies the behavior of flatten fallback lookups."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: str) -> typing.Optional[str]:
    """Docstring."""
    if aid == "flatten_full":
      return "jnp.ravel"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  module = cst.parse_module("torch.flatten(x, 0, -1)")
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Expr, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, ctx)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_fallback_lookups_range() -> None:
  """Verifies the behavior of flatten fallback lookups range."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: str) -> typing.Optional[str]:
    """Docstring."""
    if aid == "flatten_range":
      return "jnp.reshape"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  module = cst.parse_module("torch.flatten(x, 1)")
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Expr, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, ctx)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.reshape" in res


def test_flatten_comma_injection(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten comma injection."""
  ctx = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jnp.reshape")
  code: str = "y = torch.flatten(x, start_dim=1)"
  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Call, typing.cast(cst.Assign, node).value)
  arg_x = call_node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  call_node = call_node.with_changes(args=[arg_x, call_node.args[1]])
  res_node: cst.CSTNode = transform_flatten(call_node, ctx)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.reshape(x,(x.shape[0],-1))" in res.replace(" ", "")


def test_flatten_return_node_end() -> None:
  """Verifies the behavior of flatten return node end."""
  ctx = MagicMock()
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None
  module = cst.parse_module("torch.flatten(x, 2, 3)")
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Expr, node).value
  res_node: cst.CSTNode = transform_flatten(call_node, ctx)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "torch.flatten(x, 2, 3)" in res


def test_flatten_with_comma(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten with comma."""
  rewriter.context.hook_context.lookup_api = MagicMock(return_value="jnp.reshape")
  code: str = "y = torch.flatten(x, 1)"
  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Call, typing.cast(cst.Assign, node).value)
  call_node = call_node.with_changes(args=[call_node.args[0].with_changes(comma=cst.Comma()), call_node.args[1]])
  res_node: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  res: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.reshape" in res


def test_flatten_kwarg_start_dim_negative(rewriter: PivotRewriter) -> None:
  """Verifies start_dim negative parsing."""
  rewriter.context.hook_context.target_fw = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: str = "torch.flatten(x, start_dim=-2)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Expr, node).value
  rewriter.context.hook_context.lookup_api = lambda x: "jnp.reshape"
  res: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


def test_flatten_kwarg_end_dim_negative(rewriter: PivotRewriter) -> None:
  """Verifies end_dim negative parsing."""
  rewriter.context.hook_context.target_fw = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: str = "torch.flatten(x, end_dim=-2)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Expr, node).value
  rewriter.context.hook_context.lookup_api = lambda x: "jnp.reshape"
  res: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


def test_flatten_ravel_exact(rewriter: PivotRewriter) -> None:
  """Verifies ravel transformation."""
  rewriter.context.hook_context.target_fw = "numpy"
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: str = "torch.flatten(x)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Expr, node).value
  rewriter.context.hook_context.lookup_api = lambda x: "numpy.ravel"
  res: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "ravel"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


def test_flatten_callable_class(rewriter: PivotRewriter) -> None:
  """Verifies callable class instantiation."""
  rewriter.context.hook_context.target_fw = "tensorflow"
  rewriter.context.config.target_framework = "tensorflow"
  rewriter.context.hook_context._current_variant = MagicMock(op_type=MagicMock(value="class"))

  def resolve_variant(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Docstring."""
    if aid == "Flatten":
      return {"api": "tf.keras.layers.Flatten"}
    return None

  rewriter.semantics.resolve_variant.side_effect = resolve_variant
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: str = "torch.flatten(x)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node = typing.cast(cst.Expr, node).value
  rewriter.context.hook_context.lookup_api = lambda x: "tf.keras.layers.Flatten"
  res: cst.CSTNode = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "tf.keras.layers.Flatten()(x)"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


# --- Merged from test_flatten_missing.py ---


def test_flatten_unhandled_fw():
  """Verifies the behavior of flatten unhandled framework."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "unknown"
  ctx.current_op_id = "Flatten"
  node = cst.Call(func=cst.Name("flatten"))
  res = transform_flatten(node, ctx)
  assert res == node
