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


def rewrite_code(rewriter: typing.Any, code: typing.Any) -> typing.Any:
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> typing.Any:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  hooks._HOOKS["flatten_range"] = transform_flatten
  hooks._PLUGINS_LOADED = True
  mgr: typing.Any = MagicMock()
  flatten_def: typing.Any = {
    "variants": {"torch": {"api": "torch.flatten"}, "jax": {"api": "jnp.reshape", "requires_plugin": "flatten_range"}}
  }
  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  def resolve_variant(aid: typing.Any, fw: typing.Any) -> typing.Any:
    """Resolves variant.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
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

  def get_config(fw: typing.Any) -> typing.Any:
    """Gets configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg: typing.Any = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten batch preserve.

  Args:
      rewriter: Test argument or fixture.
  """
  code: typing.Any = "y = torch.flatten(x, 1)"
  res: typing.Any = rewrite_code(rewriter, code)
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in res.replace(" ", "")


def test_flatten_passthrough_missing_def(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten passthrough missing def.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  rewriter.semantics.get_framework_config.side_effect = lambda f: {
    "plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)
  }
  code: typing.Any = "y = torch.flatten(x, 1)"
  res: typing.Any = rewrite_code(rewriter, code)
  assert "torch.flatten" in res


def test_flatten_empty_args(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten empty arguments.

  Args:
      rewriter: Test argument or fixture.
  """
  code: typing.Any = "y = torch.flatten()"
  res: typing.Any = rewrite_code(rewriter, code)
  assert "torch.flatten()" in res


def test_flatten_positional_args_jax_collapse(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten positional arguments JAX collapse.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jax.lax.collapse"))
  code: typing.Any = "y = torch.flatten(x, 1, 2)"
  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Assign, node).value
  res_node: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_kwargs_jax_collapse(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten keyword arguments JAX collapse.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jax.lax.collapse"))
  code: typing.Any = "y = torch.flatten(x, start_dim=1, end_dim=-1)"
  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Assign, node).value
  res_node: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 1, x.ndim)" in res


def test_flatten_full_ravel(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten full ravel.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jnp.ravel"))
  code: typing.Any = "y = torch.flatten(x, 0, -1)"
  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Assign, node).value
  res_node: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_value_errors(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten value errors.

  Args:
      rewriter: Test argument or fixture.
  """
  ctx: typing.Any = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code: typing.Any = "y = torch.flatten(x, 0x1A, 0x1B)"
  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Assign, node).value
  res_node: typing.Any = transform_flatten(call_node, ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 0, x.ndim)" in res


def test_flatten_end_dim_kwarg(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten end dim keyword argument.

  Args:
      rewriter: Test argument or fixture.
  """
  ctx: typing.Any = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jax.lax.collapse")
  code: typing.Any = "y = torch.flatten(x, start_dim=1, end_dim=2)"
  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Assign, node).value
  res_node: typing.Any = transform_flatten(call_node, ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_fallback_lookups() -> None:
  """Verifies the behavior of flatten fallback lookups."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: typing.Any) -> typing.Any:
    """Execute mock lookup helper logic.

    Args:
        aid: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "flatten_full":
      return "jnp.ravel"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  module: typing.Any = cst.parse_module("torch.flatten(x, 0, -1)")
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Expr, node).value
  res_node: typing.Any = transform_flatten(call_node, ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_fallback_lookups_range() -> None:
  """Verifies the behavior of flatten fallback lookups range."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: typing.Any) -> typing.Any:
    """Execute mock lookup helper logic.

    Args:
        aid: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "flatten_range":
      return "jnp.reshape"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  module: typing.Any = cst.parse_module("torch.flatten(x, 1)")
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Expr, node).value
  res_node: typing.Any = transform_flatten(call_node, ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.reshape" in res


def test_flatten_comma_injection(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten comma injection.

  Args:
      rewriter: Test argument or fixture.
  """
  ctx: typing.Any = rewriter.context.hook_context
  ctx.lookup_api = MagicMock(return_value="jnp.reshape")
  code: typing.Any = "y = torch.flatten(x, start_dim=1)"
  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Call, typing.cast(cst.Assign, node).value)
  arg_x: typing.Any = call_node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  call_node = call_node.with_changes(args=[arg_x, call_node.args[1]])
  res_node: typing.Any = transform_flatten(call_node, ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.reshape(x,(x.shape[0],-1))" in res.replace(" ", "")


def test_flatten_return_node_end() -> None:
  """Verifies the behavior of flatten return node end."""
  ctx: typing.Any = MagicMock()
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None
  module: typing.Any = cst.parse_module("torch.flatten(x, 2, 3)")
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Expr, node).value
  res_node: typing.Any = transform_flatten(call_node, ctx)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "torch.flatten(x, 2, 3)" in res


def test_flatten_with_comma(rewriter: typing.Any) -> None:
  """Verifies the behavior of flatten with comma.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jnp.reshape"))
  code: typing.Any = "y = torch.flatten(x, 1)"
  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Call, typing.cast(cst.Assign, node).value)
  call_node = call_node.with_changes(args=[call_node.args[0].with_changes(comma=cst.Comma()), call_node.args[1]])
  res_node: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  res: typing.Any = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res_node))])]).code
  assert "jnp.reshape" in res


def test_flatten_kwarg_start_dim_negative(rewriter: typing.Any) -> None:
  """Verifies start_dim negative parsing.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_fw = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: typing.Any = "torch.flatten(x, start_dim=-2)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Expr, node).value
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "jnp.reshape")
  res: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


def test_flatten_kwarg_end_dim_negative(rewriter: typing.Any) -> None:
  """Verifies end_dim negative parsing.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_fw = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: typing.Any = "torch.flatten(x, end_dim=-2)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Expr, node).value
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "jnp.reshape")
  res: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


def test_flatten_ravel_exact(rewriter: typing.Any) -> None:
  """Verifies ravel transformation.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_fw = "numpy"
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: typing.Any = "torch.flatten(x)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Expr, node).value
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "numpy.ravel")
  res: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "ravel"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


def test_flatten_callable_class(rewriter: typing.Any) -> None:
  """Verifies callable class instantiation.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_fw = "tensorflow"
  rewriter.context.config.target_framework = "tensorflow"
  rewriter.context.hook_context._current_variant = MagicMock(op_type=MagicMock(value="class"))

  def resolve_variant(aid: typing.Any, fw: typing.Any) -> typing.Any:
    """Execute resolve variant helper logic.

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "Flatten":
      return {"api": "tf.keras.layers.Flatten"}
    return None

  rewriter.semantics.resolve_variant.side_effect = resolve_variant
  rewriter.context.hook_context.current_op_id = "Flatten"
  code: typing.Any = "torch.flatten(x)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module: typing.Any = cst.parse_module(code)
  node: typing.Any = typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]
  call_node: typing.Any = typing.cast(cst.Expr, node).value
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "tf.keras.layers.Flatten")
  res: typing.Any = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "tf.keras.layers.Flatten()(x)"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=typing.cast(cst.BaseExpression, res))])]).code
  )


# --- Merged from test_flatten_missing.py ---


def test_flatten_unhandled_fw() -> None:
  """Verifies the behavior of flatten unhandled framework."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.target_fw = "unknown"
  ctx.current_op_id = "Flatten"
  node: typing.Any = cst.Call(func=cst.Name("flatten"))
  res: typing.Any = transform_flatten(node, ctx)
  assert res == node
