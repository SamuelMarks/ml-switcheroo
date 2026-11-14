"""Tests for the new flatten plugin transformations."""

import typing
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant, OpType
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.semantics.manager import SemanticsManager


def _parse_call(code: str) -> cst.Call:
  """Parse a single-statement call expression into a LibCST Call node.

  Args:
      code: Python call expression source string.

  Returns:
      Parsed `cst.Call` AST node.
  """
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  return cst.ensure_type(expr.value, cst.Call)


def get_context(
  api_name: str | None,
  op_type: str = "function",
  variant: FrameworkVariant | None = None,
  lookup_fn: typing.Callable[[str], str | None] | None = None,
) -> HookContext:
  """Construct a typed HookContext test fixture for flatten plugin tests.

  Args:
      api_name: Input `api_name` parameter value (`str | None`).
      op_type: Input `op_type` parameter value (`str`).
      variant: Input `variant` parameter value (`FrameworkVariant | None`).
      lookup_fn: Optional custom lookup_api callable.

  Returns:
      Resulting `HookContext` from get context.
  """

  class MockConfig(RuntimeConfig):
    """Minimal `RuntimeConfig` test double used in plugin flatten new tests."""

    source_framework: str = "torch"
    target_framework: str = "numpy"

  active_variant: FrameworkVariant = (
    variant if variant is not None else FrameworkVariant(api=api_name, op_type=OpType(op_type))
  )

  class _ScopedHookContext(HookContext):
    """Scoped HookContext subclass overriding current_variant and lookup_api without class-level leaks."""

    @property
    def current_variant(self) -> FrameworkVariant | None:
      """Return scoped test variant.

      Returns:
          Active test variant.
      """
      return active_variant

    def lookup_api(self, op_name: str) -> str | None:
      """Return configured test API string.

      Args:
          op_name: Queried operation name.

      Returns:
          Configured API name or None.
      """
      if lookup_fn is not None:
        return lookup_fn(op_name)
      return api_name

  config = MockConfig(source_framework="torch", target_framework="numpy")
  sem = MagicMock(spec=SemanticsManager)
  ctx = _ScopedHookContext(semantics=sem, config=config)
  ctx.current_op_id = "flatten"
  return ctx


def test_flatten_empty_args() -> None:
  """Verify `ensure_type` handles flatten empty args as expected."""
  call = _parse_call("flatten()")
  ctx = get_context("numpy.ravel")
  res = transform_flatten(call, ctx)
  assert res is call


def test_flatten_jax_collapse() -> None:
  """Verify `ensure_type` handles flatten JAX collapse as expected."""
  call = _parse_call("flatten(x, 1)")
  ctx = get_context("jax.lax.collapse")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, x.ndim)"


def test_flatten_jax_collapse_end_dim() -> None:
  """Verify `ensure_type` handles flatten JAX collapse end dim as expected."""
  call = _parse_call("flatten(x, 1, 2)")
  ctx = get_context("jax.lax.collapse")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, 3)"


def test_flatten_numpy_ravel() -> None:
  """Verify `ensure_type` handles flatten numpy ravel as expected."""
  call = _parse_call("flatten(x)")
  ctx = get_context("numpy.ravel")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.ravel(x)"


def test_flatten_numpy_reshape_batch() -> None:
  """Verify `ensure_type` handles flatten numpy reshape batch as expected."""
  call = _parse_call("flatten(x, 1)")
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.reshape(x, (x.shape[0], -1))"


def test_flatten_numpy_reshape_no_comma() -> None:
  """Verify `Call` handles flatten numpy reshape no comma as expected."""
  # Construct manually to trigger MaybeSentinel.DEFAULT logic
  call = cst.Call(func=cst.Name("flatten"), args=[cst.Arg(value=cst.Name("x"))])
  # If we pass start_dim=1 as keyword but input has no comma...
  call = call.with_changes(
    args=[
      cst.Arg(value=cst.Name("x"), comma=cst.MaybeSentinel.DEFAULT),
      cst.Arg(value=cst.Integer("1"), keyword=cst.Name("start_dim")),
    ]
  )
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert isinstance(new_call, cst.Call)
  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.reshape(x, (x.shape[0], -1))"


def test_flatten_mlx() -> None:
  """Verify `ensure_type` handles flatten MLX as expected."""
  call = _parse_call("flatten(x, 1)")
  ctx = get_context("mlx.core.flatten")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "mlx.core.flatten(x, 1, -1)"


def test_flatten_keras_layer() -> None:
  """Verify `ensure_type` handles flatten keras layer as expected."""
  call = _parse_call("flatten(x)")
  ctx = get_context("keras.layers.Flatten")
  # To force the `target_variant.op_type` condition false but fall through by string match
  new_call = transform_flatten(call, ctx)
  assert cst.Module(body=[]).code_for_node(new_call) == "keras.layers.Flatten()(x)"


def test_flatten_value_error_positional() -> None:
  """Verify `ensure_type` handles flatten value error positional as expected."""
  call = _parse_call("flatten(x, 'a', 'b')")
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert isinstance(new_call, cst.Call)
  assert new_call is call
  assert cst.Module(body=[]).code_for_node(new_call) == "flatten(x, 'a', 'b')"


def test_flatten_negative_end_dim_kwargs() -> None:
  """Verify `ensure_type` handles flatten negative end dim kwargs as expected."""
  call = _parse_call("flatten(x, start_dim=-1, end_dim=-2)")
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert isinstance(new_call, cst.Call)
  assert new_call is call
  assert cst.Module(body=[]).code_for_node(new_call) == "flatten(x, start_dim=-1, end_dim=-2)"


def test_flatten_keyword_args_positive() -> None:
  """Verify `ensure_type` handles flatten keyword args positive as expected."""
  call = _parse_call("flatten(x, start_dim=1, end_dim=2)")
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert isinstance(new_call, cst.Call)
  assert new_call is call
  assert cst.Module(body=[]).code_for_node(new_call) == "flatten(x, start_dim=1, end_dim=2)"


def test_flatten_value_error_hex() -> None:
  """Verify `ensure_type` handles flatten value error hex as expected."""
  call = _parse_call("flatten(x, 0x1, 0x2)")
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert isinstance(new_call, cst.Call)
  assert new_call is call
  assert cst.Module(body=[]).code_for_node(new_call) == "flatten(x, 0x1, 0x2)"


def test_flatten_fallback_lookups() -> None:
  """Verify `ensure_type` handles flatten fallback lookups as expected."""
  call = _parse_call("flatten(x)")

  def side_effect(k: str) -> str | None:
    """Return fallback API only for capitalized Flatten operation key.

    Args:
        k: Queried operation identifier.

    Returns:
        Matched API string or None.
    """
    if k == "Flatten":
      return "fallback.flatten"
    return None

  ctx = get_context("fallback.flatten", lookup_fn=side_effect)
  ctx.current_op_id = None

  res = transform_flatten(call, ctx)
  assert res is not call


def test_flatten_fallback_none() -> None:
  """Verify `ensure_type` handles flatten fallback none as expected."""
  call = _parse_call("flatten(x)")
  ctx = get_context("fallback.flatten", lookup_fn=lambda k: None)
  ctx.current_op_id = None

  res = transform_flatten(call, ctx)
  assert res is call


def test_flatten_negative_end_dim_kwargs_variables() -> None:
  """Verify `ensure_type` handles flatten negative end dim kwargs variables as expected."""
  call = _parse_call("flatten(x, start_dim=my_start, end_dim=my_end)")
  ctx = get_context("numpy.reshape")
  _res = transform_flatten(call, ctx)
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[]).code_for_node(_res) == "flatten(x, start_dim=my_start, end_dim=my_end)"


def test_flatten_no_variant() -> None:
  """Verify `ensure_type` handles flatten no variant as expected."""
  call = _parse_call("flatten(x)")
  ctx = HookContext(semantics=None, config=RuntimeConfig(source_framework="torch", target_framework="numpy"))
  ctx.current_op_id = "flatten"
  _res = transform_flatten(call, ctx)
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[]).code_for_node(_res) == "flatten(x)"
