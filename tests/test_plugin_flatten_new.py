"""Tests for the new flatten plugin transformations."""

import typing

from typing import Any

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten


def get_context(api_name: typing.Any, op_type: typing.Any = "function", variant: Any = None) -> typing.Any:
  """Execute get context helper logic.

  Args:
      api_name: Test argument or fixture.
      op_type: Test argument or fixture.
      variant: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  class MockSemantics:
    """Test helper class MockSemantics."""

    def resolve_variant(self, op_name: typing.Any, target_fw: typing.Any) -> typing.Any:
      """Execute resolve variant helper logic.

      Args:
          op_name: Test argument or fixture.
          target_fw: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """

      class MockVal:
        """Test helper class MockVal."""

        def __init__(self, v: typing.Any) -> None:
          """Initialize test helper instance.

          Args:
              v: Test argument or fixture.
          """
          self.value = v

      return {"api": api_name, "op_type": MockVal(op_type)}

    def get_operation(self, op_name: typing.Any) -> typing.Any:
      """Execute get operation helper logic.

      Args:
          op_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """

      class MockOp:
        """Test helper class MockOp."""

        def __init__(self, is_loss: typing.Any = False) -> None:
          """Initialize test helper instance.

          Args:
              is_loss: Test argument or fixture.
          """
          self.is_loss = is_loss
          self.sharding_supported = False

      return MockOp()

  class MockConfig(RuntimeConfig):
    """Test helper class MockConfig."""

    source_framework: typing.Any = "torch"
    target_framework: typing.Any = "numpy"

  config: typing.Any = MockConfig(source_framework="torch", target_framework="numpy")
  ctx: typing.Any = HookContext(semantics=MockSemantics(), config=config)
  ctx.current_op_id = "flatten"
  if variant:
    from unittest.mock import PropertyMock

    type(ctx).current_variant = PropertyMock(return_value=variant)
  else:

    class DummyVariant:
      """Test helper class DummyVariant."""

      pass

    dv: typing.Any = DummyVariant()

    class MockVal:
      """Test helper class MockVal."""

      def __init__(self, v: typing.Any) -> None:
        """Initialize test helper instance.

        Args:
            v: Test argument or fixture.
        """
        self.value = v

    dv.op_type = MockVal(op_type)
    from unittest.mock import PropertyMock

    type(ctx).current_variant = PropertyMock(return_value=dv)

  # Provide a real mock for lookup_api
  from unittest.mock import MagicMock

  ctx.lookup_api = MagicMock()
  ctx.lookup_api.return_value = api_name

  return ctx


def test_flatten_empty_args() -> None:
  """Verify flatten empty args behavior."""
  code: typing.Any = "flatten()"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("numpy.ravel")
  res: typing.Any = transform_flatten(call, ctx)
  assert res is call


def test_flatten_jax_collapse() -> None:
  """Verify flatten jax collapse behavior."""
  code: typing.Any = "flatten(x, 1)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value

  ctx: typing.Any = get_context("jax.lax.collapse")
  new_call: typing.Any = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, x.ndim)"


def test_flatten_jax_collapse_end_dim() -> None:
  """Verify flatten jax collapse end dim behavior."""
  code: typing.Any = "flatten(x, 1, 2)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value

  ctx: typing.Any = get_context("jax.lax.collapse")
  new_call: typing.Any = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, 3)"


def test_flatten_numpy_ravel() -> None:
  """Verify flatten numpy ravel behavior."""
  code: typing.Any = "flatten(x)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value

  ctx: typing.Any = get_context("numpy.ravel")
  new_call: typing.Any = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.ravel(x)"


def test_flatten_numpy_reshape_batch() -> None:
  """Verify flatten numpy reshape batch behavior."""
  code: typing.Any = "flatten(x, 1)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value

  ctx: typing.Any = get_context("numpy.reshape")
  new_call: typing.Any = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.reshape(x, (x.shape[0], -1))"


def test_flatten_numpy_reshape_no_comma() -> None:
  """Verify flatten numpy reshape no comma behavior."""
  # Construct manually to trigger MaybeSentinel.DEFAULT logic
  call: typing.Any = cst.Call(func=cst.Name("flatten"), args=[cst.Arg(value=cst.Name("x"))])
  # If we pass start_dim=1 as keyword but input has no comma...
  call = call.with_changes(
    args=[
      cst.Arg(value=cst.Name("x"), comma=cst.MaybeSentinel.DEFAULT),
      cst.Arg(value=cst.Integer("1"), keyword=cst.Name("start_dim")),
    ]
  )
  ctx: typing.Any = get_context("numpy.reshape")
  new_call: typing.Any = transform_flatten(call, ctx)
  assert isinstance(new_call, cst.Call)


def test_flatten_mlx() -> None:
  """Verify flatten mlx behavior."""
  code: typing.Any = "flatten(x, 1)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value

  ctx: typing.Any = get_context("mlx.core.flatten")
  new_call: typing.Any = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "mlx.core.flatten(x, 1, -1)"


def test_flatten_keras_layer() -> None:
  """Verify flatten keras layer behavior."""
  code: typing.Any = "flatten(x)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value

  ctx: typing.Any = get_context("keras.layers.Flatten")
  # To force the `target_variant.op_type` condition false but fall through by string match
  new_call: typing.Any = transform_flatten(call, ctx)
  assert cst.Module(body=[]).code_for_node(new_call) == "keras.layers.Flatten()(x)"


def test_flatten_value_error_positional() -> None:
  """Verify flatten value error positional behavior."""
  code: typing.Any = "flatten(x, 'a', 'b')"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("numpy.reshape")
  new_call: typing.Any = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_negative_end_dim_kwargs() -> None:
  """Verify flatten negative end dim kwargs behavior."""
  code: typing.Any = "flatten(x, start_dim=-1, end_dim=-2)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("numpy.reshape")
  new_call: typing.Any = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_keyword_args_positive() -> None:
  """Verify flatten keyword args positive behavior."""
  code: typing.Any = "flatten(x, start_dim=1, end_dim=2)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("numpy.reshape")
  new_call: typing.Any = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_value_error_hex() -> None:
  """Verify flatten value error hex behavior."""
  code: typing.Any = "flatten(x, 0x1, 0x2)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("numpy.reshape")
  new_call: typing.Any = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_fallback_lookups() -> None:
  """Verify flatten fallback lookups behavior."""
  code: typing.Any = "flatten(x)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("fallback.flatten")

  # 118->121 false branch
  ctx.current_op_id = None

  def side_effect(k: typing.Any) -> typing.Any:
    """Execute side effect helper logic.

    Args:
        k: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if k == "flatten":
      return None
    if k == "Flatten":
      return None
    if k == "flatten_range":
      return None
    if k == "flatten_full":
      return "fallback.flatten"
    return None

  ctx.lookup_api.side_effect = side_effect

  res: typing.Any = transform_flatten(call, ctx)
  assert res is not call


def test_flatten_fallback_none() -> None:
  """Verify flatten fallback none behavior."""
  code: typing.Any = "flatten(x)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("fallback.flatten")

  ctx.current_op_id = None
  ctx.lookup_api.side_effect = lambda k: None

  res: typing.Any = transform_flatten(call, ctx)
  assert res is call


def test_flatten_negative_end_dim_kwargs_variables() -> None:
  """Verify flatten negative end dim kwargs variables behavior."""
  code: typing.Any = "flatten(x, start_dim=my_start, end_dim=my_end)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("numpy.reshape")
  _res: typing.Any = transform_flatten(call, ctx)
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[]).code_for_node(_res) == "flatten(x, start_dim=my_start, end_dim=my_end)"


def test_flatten_no_variant() -> None:
  """Verify flatten no variant behavior."""
  code: typing.Any = "flatten(x)"
  module: typing.Any = cst.parse_module(code)
  call: typing.Any = module.body[0].body[0].value
  ctx: typing.Any = get_context("numpy.reshape", variant=None)
  # override variant mock
  from unittest.mock import PropertyMock

  type(ctx).current_variant = PropertyMock(return_value=None)
  _res: typing.Any = transform_flatten(call, ctx)
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[]).code_for_node(_res) == "flatten(x)"
