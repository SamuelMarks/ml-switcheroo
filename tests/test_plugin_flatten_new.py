"""Tests for the new flatten plugin transformations."""

from pydantic import JsonValue

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten


def get_context(
  api_name: str | None,
  op_type: str = "function",
  variant: FrameworkVariant | None = None,
) -> HookContext:
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

    def resolve_variant(self, op_name: str, target_framework: str) -> dict[str, JsonValue] | None:
      """Execute resolve variant helper logic.

      Args:
          op_name: Test argument or fixture.
          target_framework: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """

      class MockVal:
        """Test helper class MockVal."""

        def __init__(self, v: str) -> None:
          """Initialize test helper instance.

          Args:
              v: Test argument or fixture.
          """
          self.value = v

      _ret_v: dict[str, JsonValue] | None = getattr(
        {
          "v": {
            "api": getattr({"v": api_name}, "__getitem__")("v"),
            "op_type": getattr(
              {"v": getattr({"v": MockVal}, "__getitem__")("v")(getattr({"v": op_type}, "__getitem__")("v"))},
              "__getitem__",
            )("v"),
          }
        },
        "__getitem__",
      )("v")
      return _ret_v

    def get_operation(self, op_name: str) -> "MockOp":
      """Execute get operation helper logic.

      Args:
          op_name: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return MockOp()

  class MockOp:
    """Test helper class MockOp."""

    def __init__(self, is_loss: bool = False) -> None:
      """Initialize test helper instance.

      Args:
          is_loss: Test argument or fixture.
      """
      self.is_loss = is_loss
      self.sharding_supported = False

  class MockConfig(RuntimeConfig):
    """Test helper class MockConfig."""

    source_framework: str = "torch"
    target_framework: str = "numpy"

  config = MockConfig(source_framework="torch", target_framework="numpy")
  ctx = getattr(
    {
      "v": getattr({"v": HookContext}, "__getitem__")("v")(
        semantics=getattr({"v": getattr({"v": MockSemantics}, "__getitem__")("v")()}, "__getitem__")("v"),
        config=getattr({"v": config}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  ctx.current_op_id = "flatten"
  if variant:
    from unittest.mock import PropertyMock

    setattr(
      getattr({"v": getattr({"v": type}, "__getitem__")("v")(getattr({"v": ctx}, "__getitem__")("v"))}, "__getitem__")(
        "v"
      ),
      "current_variant",
      getattr(
        {"v": getattr({"v": PropertyMock}, "__getitem__")("v")(return_value=getattr({"v": variant}, "__getitem__")("v"))},
        "__getitem__",
      )("v"),
    )
  else:

    class DummyVariant:
      """Test helper class DummyVariant."""

      pass

    dv = DummyVariant()

    class MockVal:
      """Test helper class MockVal."""

      def __init__(self, v: str) -> None:
        """Initialize test helper instance.

        Args:
            v: Test argument or fixture.
        """
        self.value = v

    setattr(
      getattr({"v": dv}, "__getitem__")("v"),
      "op_type",
      getattr(
        {"v": getattr({"v": MockVal}, "__getitem__")("v")(getattr({"v": op_type}, "__getitem__")("v"))}, "__getitem__"
      )("v"),
    )
    from unittest.mock import PropertyMock

    setattr(type(ctx), "current_variant", PropertyMock(return_value=dv))

  # Provide a real mock for lookup_api
  from unittest.mock import MagicMock

  setattr(ctx, "lookup_api", MagicMock())
  setattr(
    getattr({"v": getattr({"v": ctx}, "__getitem__")("v").lookup_api}, "__getitem__")("v"),
    "return_value",
    getattr({"v": api_name}, "__getitem__")("v"),
  )

  _ret_ctx: HookContext = ctx
  return _ret_ctx


def test_flatten_empty_args() -> None:
  """Verify flatten empty args behavior."""
  code = "flatten()"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.ravel")
  res = transform_flatten(call, ctx)
  assert res is call


def test_flatten_jax_collapse() -> None:
  """Verify flatten jax collapse behavior."""
  code = "flatten(x, 1)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("jax.lax.collapse")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, x.ndim)"


def test_flatten_jax_collapse_end_dim() -> None:
  """Verify flatten jax collapse end dim behavior."""
  code = "flatten(x, 1, 2)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("jax.lax.collapse")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, 3)"


def test_flatten_numpy_ravel() -> None:
  """Verify flatten numpy ravel behavior."""
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.ravel")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.ravel(x)"


def test_flatten_numpy_reshape_batch() -> None:
  """Verify flatten numpy reshape batch behavior."""
  code = "flatten(x, 1)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.reshape(x, (x.shape[0], -1))"


def test_flatten_numpy_reshape_no_comma() -> None:
  """Verify flatten numpy reshape no comma behavior."""
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


def test_flatten_mlx() -> None:
  """Verify flatten mlx behavior."""
  code = "flatten(x, 1)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("mlx.core.flatten")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "mlx.core.flatten(x, 1, -1)"


def test_flatten_keras_layer() -> None:
  """Verify flatten keras layer behavior."""
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("keras.layers.Flatten")
  # To force the `target_variant.op_type` condition false but fall through by string match
  new_call = transform_flatten(call, ctx)
  assert cst.Module(body=[]).code_for_node(new_call) == "keras.layers.Flatten()(x)"


def test_flatten_value_error_positional() -> None:
  """Verify flatten value error positional behavior."""
  code = "flatten(x, 'a', 'b')"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_negative_end_dim_kwargs() -> None:
  """Verify flatten negative end dim kwargs behavior."""
  code = "flatten(x, start_dim=-1, end_dim=-2)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_keyword_args_positive() -> None:
  """Verify flatten keyword args positive behavior."""
  code = "flatten(x, start_dim=1, end_dim=2)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_value_error_hex() -> None:
  """Verify flatten value error hex behavior."""
  code = "flatten(x, 0x1, 0x2)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_fallback_lookups() -> None:
  """Verify flatten fallback lookups behavior."""
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("fallback.flatten")

  # 118->121 false branch
  ctx.current_op_id = None

  def side_effect(k: str) -> str | None:
    """Execute side effect helper logic.

    Args:
        k: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if k == "flatten":
      return None
    if k == "Flatten":
      return "fallback.flatten"
    return None

  setattr(
    getattr({"v": getattr({"v": ctx}, "__getitem__")("v").lookup_api}, "__getitem__")("v"),
    "side_effect",
    getattr({"v": side_effect}, "__getitem__")("v"),
  )

  res = transform_flatten(call, ctx)
  assert res is not call


def test_flatten_fallback_none() -> None:
  """Verify flatten fallback none behavior."""
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("fallback.flatten")

  ctx.current_op_id = None
  setattr(
    getattr({"v": getattr({"v": ctx}, "__getitem__")("v").lookup_api}, "__getitem__")("v"),
    "side_effect",
    getattr({"v": lambda k: None}, "__getitem__")("v"),
  )

  res = transform_flatten(call, ctx)
  assert res is call


def test_flatten_negative_end_dim_kwargs_variables() -> None:
  """Verify flatten negative end dim kwargs variables behavior."""
  code = "flatten(x, start_dim=my_start, end_dim=my_end)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.reshape")
  _res = transform_flatten(call, ctx)
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[]).code_for_node(_res) == "flatten(x, start_dim=my_start, end_dim=my_end)"


def test_flatten_no_variant() -> None:
  """Verify flatten no variant behavior."""
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  ctx = get_context("numpy.reshape", variant=None)
  # override variant mock
  from unittest.mock import PropertyMock

  setattr(type(ctx), "current_variant", PropertyMock(return_value=None))
  _res = transform_flatten(call, ctx)
  assert isinstance(_res, cst.Call)
  assert cst.Module(body=[]).code_for_node(_res) == "flatten(x)"
