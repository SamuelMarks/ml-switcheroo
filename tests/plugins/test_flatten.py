"""Test suite for the Flatten module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.semantics.schema import PluginTraits
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


def _extract_first_call(code: str) -> cst.Call:
  """Parse a single-statement snippet and return its Call node.

  Args:
      code: Python snippet containing an assignment or expression statement.

  Returns:
      Extracted `cst.Call` node.
  """
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  small = stmt.body[0]
  assert isinstance(small, (cst.Assign, cst.Expr))
  return cst.ensure_type(small.value, cst.Call)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for flatten tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.register_hook("flatten_range")(transform_flatten)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  jax_variant: dict[str, JsonValue] = {"api": "jnp.reshape", "requires_plugin": "flatten_range"}
  flatten_def: dict[str, JsonValue] = {"variants": {"torch": {"api": "torch.flatten"}, "jax": jax_variant}}
  mgr.get_definition.side_effect = lambda n: ("Flatten", flatten_def) if "flatten" in n else None

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if fw == "jax":
      if aid == "flatten_range":
        return {"api": "jnp.reshape"}
      if aid == "flatten_full":
        return {"api": "jnp.ravel"}
      if aid == "Flatten":
        return jax_variant
    return None

  mgr.resolve_variant.side_effect = resolve_variant
  mgr.is_verified.return_value = True

  def get_config(fw: str) -> dict[str, PluginTraits]:
    """Gets configuration.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, PluginTraits]` from get config.
    """
    if fw == "jax":
      return {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}
    return {}

  mgr.get_framework_config.side_effect = get_config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_flatten_batch_preserve(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `jnp.reshape` for flatten batch preserve.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "jnp.reshape" in res
  assert "(x.shape[0],-1)" in res.replace(" ", "")


def test_flatten_passthrough_missing_def(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `rewrite_code` emits `torch.flatten` for flatten passthrough missing def.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_framework = "numpy"
  monkeypatch.setattr(
    rewriter.semantics.get_framework_config,
    "side_effect",
    (lambda f: {"plugin_traits": PluginTraits(has_numpy_compatible_arrays=True)}),
    raising=False,
  )
  code = "y = torch.flatten(x, 1)"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten" in res


def test_flatten_empty_args(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `torch.flatten()` for flatten empty args.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code = "y = torch.flatten()"
  res = rewrite_code(rewriter, code)
  assert "torch.flatten()" in res


def test_flatten_positional_args_jax_collapse(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `transform_flatten` handles flatten positional args JAX collapse as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(
    rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jax.lax.collapse"), raising=False
  )
  call_node = _extract_first_call("y = torch.flatten(x, 1, 2)")
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_kwargs_jax_collapse(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `transform_flatten` handles flatten kwargs JAX collapse as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(
    rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jax.lax.collapse"), raising=False
  )
  call_node = _extract_first_call("y = torch.flatten(x, start_dim=1, end_dim=-1)")
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 1, x.ndim)" in res


def test_flatten_full_ravel(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `transform_flatten` emits `jnp.ravel(x` for flatten full ravel.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jnp.ravel"), raising=False)
  call_node = _extract_first_call("y = torch.flatten(x, 0, -1)")
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_value_errors(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `transform_flatten` handles flatten value errors as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = rewriter.context.hook_context
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="jax.lax.collapse"), raising=False)
  call_node = _extract_first_call("y = torch.flatten(x, 0x1A, 0x1B)")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 0, x.ndim)" in res


def test_flatten_end_dim_kwarg(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `transform_flatten` emits `jax.lax.collapse(x, 1, 3)` for flatten end dim kwarg.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = rewriter.context.hook_context
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="jax.lax.collapse"), raising=False)
  call_node = _extract_first_call("y = torch.flatten(x, start_dim=1, end_dim=2)")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jax.lax.collapse(x, 1, 3)" in res


def test_flatten_fallback_lookups() -> None:
  """Verify `transform_flatten` emits `jnp.ravel(x` for flatten fallback lookups."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: str) -> str | None:
    """Execute `mock_lookup` helper in flatten tests.

    Args:
        aid: Abstract operation identifier string.

    Returns:
        Resulting `str | None` from mock lookup.
    """
    if aid == "Flatten":
      return "jnp.ravel"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  call_node = _extract_first_call("torch.flatten(x, 0, -1)")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.ravel(x" in res


def test_flatten_fallback_lookups_range() -> None:
  """Verify `transform_flatten` emits `jnp.reshape` for flatten fallback lookups range."""
  ctx = MagicMock()
  ctx.current_op_id = None

  def mock_lookup(aid: str) -> str | None:
    """Execute `mock_lookup` helper in flatten tests.

    Args:
        aid: Abstract operation identifier string.

    Returns:
        Resulting `str | None` from mock lookup.
    """
    if aid == "flatten":
      return "jnp.reshape"
    return None

  ctx.lookup_api.side_effect = mock_lookup
  call_node = _extract_first_call("torch.flatten(x, 1)")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.reshape" in res


def test_flatten_comma_injection(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles flatten comma injection as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  ctx = rewriter.context.hook_context
  monkeypatch.setattr(ctx, "lookup_api", MagicMock(return_value="jnp.reshape"), raising=False)
  call_node = _extract_first_call("y = torch.flatten(x, start_dim=1)")
  arg_x = call_node.args[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)
  call_node = call_node.with_changes(args=[arg_x, call_node.args[1]])
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.reshape(x,(x.shape[0],-1))" in res.replace(" ", "")


def test_flatten_return_node_end() -> None:
  """Verify `transform_flatten` emits `torch.flatten(x, 2, 3)` for flatten return node end."""
  ctx = MagicMock()
  ctx.current_op_id = None
  ctx.lookup_api.return_value = None
  call_node = _extract_first_call("torch.flatten(x, 2, 3)")
  res_node = transform_flatten(call_node, ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "torch.flatten(x, 2, 3)" in res


def test_flatten_with_comma(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` emits `jnp.reshape` for flatten with comma.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", MagicMock(return_value="jnp.reshape"), raising=False)
  call_node = _extract_first_call("y = torch.flatten(x, 1)")
  call_node = call_node.with_changes(args=[call_node.args[0].with_changes(comma=cst.Comma()), call_node.args[1]])
  res_node = transform_flatten(call_node, rewriter.context.hook_context)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res_node, cst.Call))])]).code
  assert "jnp.reshape" in res


def test_flatten_kwarg_start_dim_negative(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies start_dim negative parsing.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter.context.hook_context.target_framework = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  call_node = _extract_first_call("torch.flatten(x, start_dim=-2)")
  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", lambda x: "jnp.reshape", raising=False)
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code
  )


def test_flatten_kwarg_end_dim_negative(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies end_dim negative parsing.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter.context.hook_context.target_framework = "jax"
  rewriter.context.config.target_framework = "jax"
  rewriter.context.hook_context.current_op_id = "Flatten"
  call_node = _extract_first_call("torch.flatten(x, end_dim=-2)")
  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", lambda x: "jnp.reshape", raising=False)
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "flatten" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code
  )


def test_flatten_ravel_exact(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies ravel transformation.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter.context.hook_context.target_framework = "numpy"
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.current_op_id = "Flatten"
  call_node = _extract_first_call("torch.flatten(x)")
  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", lambda x: "numpy.ravel", raising=False)
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert "ravel" in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code


def test_flatten_callable_class(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies callable class instantiation.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter.context.hook_context.target_framework = "tensorflow"
  rewriter.context.config.target_framework = "tensorflow"
  monkeypatch.setattr(
    rewriter.context.hook_context,
    "_current_variant",
    MagicMock(op_type=MagicMock(value="class")),
    raising=False,
  )

  def resolve_variant(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `resolve_variant` helper in flatten tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    if aid == "Flatten":
      return {"api": "tf.keras.layers.Flatten"}
    return None

  monkeypatch.setattr(
    rewriter.semantics.resolve_variant,
    "side_effect",
    resolve_variant,
    raising=False,
  )
  rewriter.context.hook_context.current_op_id = "Flatten"
  call_node = _extract_first_call("torch.flatten(x)")
  monkeypatch.setattr(rewriter.context.hook_context, "lookup_api", lambda x: "tf.keras.layers.Flatten", raising=False)
  res = transform_flatten(call_node, rewriter.context.hook_context)
  assert (
    "tf.keras.layers.Flatten()(x)"
    in cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.ensure_type(res, cst.Call))])]).code
  )


def test_flatten_unhandled_fw() -> None:
  """Verify `Call` handles flatten unhandled fw as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "unknown"
  ctx.current_op_id = "Flatten"
  node = cst.Call(func=cst.Name("flatten"))
  res = transform_flatten(node, ctx)
  assert res == node


def test_flatten_branches_ext() -> None:
  """Verify flatten plugin edge-case branches across JAX, NumPy, TensorFlow, and PyTorch."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "jax"
  ctx.current_op_id = None
  ctx.current_variant = None
  ctx.lookup_api.return_value = None
  node0 = cst.ensure_type(cst.parse_expression("flatten()"), cst.Call)
  assert transform_flatten(node0, ctx) == node0

  ctx.target_framework = "numpy"
  node1 = cst.ensure_type(cst.parse_expression("flatten(x)"), cst.Call)
  assert transform_flatten(node1, ctx) == node1

  ctx.target_framework = "tensorflow"
  ctx.lookup_api.return_value = "tf.reshape"
  node_tf = cst.ensure_type(cst.parse_expression("flatten(x, start_dim=1, end_dim=-1)"), cst.Call)
  res_tf = transform_flatten(node_tf, ctx)
  assert cst.Module(body=[]).code_for_node(res_tf) == "tf.reshape(x, (x.shape[0], -1))"

  ctx.target_framework = "torch"
  ctx.lookup_api.return_value = "torch.flatten"
  node_torch = cst.ensure_type(cst.parse_expression("flatten(x, start_dim=0, end_dim=2)"), cst.Call)
  res_torch = transform_flatten(node_torch, ctx)
  assert cst.Module(body=[]).code_for_node(res_torch) == "flatten(x, start_dim=0, end_dim=2)"
