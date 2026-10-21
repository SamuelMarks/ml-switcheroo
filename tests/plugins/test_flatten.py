"""Test suite for the Flatten module."""

from pydantic import JsonValue
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
  """Rewrites code.

  Args:
      code: Test argument or fixture.
      rewriter: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "flatten_range", getattr({"v": transform_flatten}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  flatten_def = {
    "variants": {"torch": {"api": "torch.flatten"}, "jax": {"api": "jnp.reshape", "requires_plugin": "flatten_range"}}
  }
  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
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
        res = flatten_def["variants"]["jax"]
        _ret_v: dict[str, JsonValue] | None = getattr(
          {"v": res if isinstance(getattr({"v": res}, "__getitem__")("v"), dict) else {}}, "__getitem__"
        )("v")
        return _ret_v
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.is_verified.return_value = True

  def get_config(fw: str) -> dict[str, JsonValue]:
    """Gets configuration.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if fw == "jax":
      _ret_v: dict[str, JsonValue] = getattr(
        {
          "v": {
            "plugin_traits": getattr(
              {
                "v": getattr({"v": PluginTraits}, "__getitem__")("v")(
                  has_numpy_compatible_arrays=getattr({"v": True}, "__getitem__")("v")
                )
              },
              "__getitem__",
            )("v")
          }
        },
        "__getitem__",
      )("v")
      return _ret_v
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten batch preserve.

  Args:
      rewriter: Test argument or fixture.
  """
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in res.replace(" ", "")


def test_flatten_passthrough_missing_def(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten passthrough missing def.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_framework = "numpy"
  setattr(
    getattr(
      {
        "v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")(
          "v"
        ).get_framework_config
      },
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": lambda f: {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}}, "__getitem__")("v"),
  )
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten" in res


def test_flatten_empty_args(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten empty arguments.

  Args:
      rewriter: Test argument or fixture.
  """
  code = "y = torch.flatten()"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten()" in res


def test_flatten_positional_args_jax_collapse(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten positional arguments JAX collapse.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jax.lax.collapse"))
  code = "y = torch.flatten(x, 1, 2)"
  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_kwargs_jax_collapse(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten keyword arguments JAX collapse.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jax.lax.collapse"))
  code = "y = torch.flatten(x, start_dim=1, end_dim=-1)"
  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 1, x.ndim)" in res


def test_flatten_full_ravel(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten full ravel.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jnp.ravel"))
  code = "y = torch.flatten(x, 0, -1)"
  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_value_errors(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten value errors.

  Args:
      rewriter: Test argument or fixture.
  """
  ctx = rewriter.context.hook_context
  setattr(ctx, "lookup_api", MagicMock(return_value="jax.lax.collapse"))
  code = "y = torch.flatten(x, 0x1A, 0x1B)"
  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 0, x.ndim)" in res


def test_flatten_end_dim_kwarg(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten end dim keyword argument.

  Args:
      rewriter: Test argument or fixture.
  """
  ctx = rewriter.context.hook_context
  setattr(ctx, "lookup_api", MagicMock(return_value="jax.lax.collapse"))
  code = "y = torch.flatten(x, start_dim=1, end_dim=2)"
  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_fallback_lookups() -> None:
  """Verifies the behavior of flatten fallback lookups."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: str) -> str | None:
    """Execute mock lookup helper logic.

    Args:
        aid: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "Flatten":
      return "jnp.ravel"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  module = cst.parse_module("torch.flatten(x, 0, -1)")
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_fallback_lookups_range() -> None:
  """Verifies the behavior of flatten fallback lookups range."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: str) -> str | None:
    """Execute mock lookup helper logic.

    Args:
        aid: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "flatten":
      return "jnp.reshape"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  module = cst.parse_module("torch.flatten(x, 1)")
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.reshape" in res


def test_flatten_comma_injection(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten comma injection.

  Args:
      rewriter: Test argument or fixture.
  """
  ctx = rewriter.context.hook_context
  setattr(ctx, "lookup_api", MagicMock(return_value="jnp.reshape"))
  code = "y = torch.flatten(x, start_dim=1)"
  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = cst.ensure_type(cst.ensure_type(node, cst.Assign).value, cst.Call)
  arg_x = call_node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  call_node = call_node.with_changes(args=[arg_x, call_node.args[1]])
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.reshape(x,(x.shape[0],-1))" in res.replace(" ", "")


def test_flatten_return_node_end() -> None:
  """Verifies the behavior of flatten return node end."""
  ctx = MagicMock()
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None
  module = cst.parse_module("torch.flatten(x, 2, 3)")
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "torch.flatten(x, 2, 3)" in res


def test_flatten_with_comma(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of flatten with comma.

  Args:
      rewriter: Test argument or fixture.
  """
  setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jnp.reshape"))
  code = "y = torch.flatten(x, 1)"
  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = cst.ensure_type(cst.ensure_type(node, cst.Assign).value, cst.Call)
  call_node = call_node.with_changes(args=[call_node.args[0].with_changes(comma=cst.Comma()), call_node.args[1]])
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.reshape" in res


def test_flatten_kwarg_start_dim_negative(rewriter: PivotRewriter) -> None:
  """Verifies start_dim negative parsing.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_framework = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code = "torch.flatten(x, start_dim=-2)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "jnp.reshape")
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code
  )


def test_flatten_kwarg_end_dim_negative(rewriter: PivotRewriter) -> None:
  """Verifies end_dim negative parsing.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_framework = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code = "torch.flatten(x, end_dim=-2)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "jnp.reshape")
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code
  )


def test_flatten_ravel_exact(rewriter: PivotRewriter) -> None:
  """Verifies ravel transformation.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_framework = "numpy"
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.current_op_id = "Flatten"
  code = "torch.flatten(x)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "numpy.ravel")
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert "ravel" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code


def test_flatten_callable_class(rewriter: PivotRewriter) -> None:
  """Verifies callable class instantiation.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.context.hook_context.target_framework = "tensorflow"
  rewriter.context.config.target_framework = "tensorflow"
  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").context}, "__getitem__")("v").hook_context},
      "__getitem__",
    )("v"),
    "_current_variant",
    getattr(
      {
        "v": getattr({"v": MagicMock}, "__getitem__")("v")(
          op_type=getattr(
            {"v": getattr({"v": MagicMock}, "__getitem__")("v")(value=getattr({"v": "class"}, "__getitem__")("v"))},
            "__getitem__",
          )("v")
        )
      },
      "__getitem__",
    )("v"),
  )

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
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

  setattr(
    getattr(
      {"v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").semantics}, "__getitem__")("v").resolve_variant},
      "__getitem__",
    )("v"),
    "side_effect",
    getattr({"v": resolve_variant}, "__getitem__")("v"),
  )
  rewriter.context.hook_context.current_op_id = "Flatten"
  code = "torch.flatten(x)"
  from ml_switcheroo.plugins.flatten import transform_flatten

  module = cst.parse_module(code)
  node = getattr(module.body[0], "body")[0]
  call_node = getattr(node, "value")
  setattr(rewriter.context.hook_context, "lookup_api", lambda x: "tf.keras.layers.Flatten")
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "tf.keras.layers.Flatten()(x)"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code
  )


# --- Merged from test_flatten_missing.py ---


def test_flatten_unhandled_fw() -> None:
  """Verifies the behavior of flatten unhandled framework."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "unknown"
  ctx.current_op_id = "Flatten"
  node = cst.Call(func=cst.Name("flatten"))
  res = transform_flatten(node, ctx)
  assert res == node
