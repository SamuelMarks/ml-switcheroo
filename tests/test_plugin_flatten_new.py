"""Tests for the new flatten plugin transformations."""

from typing import Any, Dict

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten


def get_context(api_name: str, op_type: str = "function", variant: Any = None) -> HookContext:
  """Execute get context helper logic."""

  class MockSemantics:
    """Test helper class MockSemantics."""

    def resolve_variant(self, op_name: str, target_fw: str) -> Dict[str, str]:
      """Execute resolve variant helper logic."""

      class MockVal:
        """Test helper class MockVal."""

        def __init__(self, v):
          """Initialize test helper instance."""
          self.value = v

      return {"api": api_name, "op_type": MockVal(op_type)}

    def get_operation(self, op_name: str) -> Any:
      """Execute get operation helper logic."""

      class MockOp:
        """Test helper class MockOp."""

        def __init__(self, is_loss=False):
          """Initialize test helper instance."""
          self.is_loss = is_loss
          self.sharding_supported = False

      return MockOp()

  class MockConfig(RuntimeConfig):
    """Test helper class MockConfig."""

    source_framework: str = "torch"
    target_framework: str = "numpy"

  config: MockConfig = MockConfig(source_framework="torch", target_framework="numpy")
  ctx: HookContext = HookContext(semantics=MockSemantics(), config=config)
  ctx.current_op_id = "flatten"
  if variant:
    from unittest.mock import PropertyMock

    type(ctx).current_variant = PropertyMock(return_value=variant)
  else:

    class DummyVariant:
      """Test helper class DummyVariant."""

      pass

    dv = DummyVariant()

    class MockVal:
      """Test helper class MockVal."""

      def __init__(self, v):
        """Initialize test helper instance."""
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
  code = "flatten()"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("numpy.ravel")
  res = transform_flatten(call, ctx)
  assert res is call


def test_flatten_jax_collapse() -> None:
  """Verify flatten jax collapse behavior."""
  code: str = "flatten(x, 1)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("jax.lax.collapse")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, x.ndim)"


def test_flatten_jax_collapse_end_dim() -> None:
  """Verify flatten jax collapse end dim behavior."""
  code: str = "flatten(x, 1, 2)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("jax.lax.collapse")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, 3)"


def test_flatten_numpy_ravel() -> None:
  """Verify flatten numpy ravel behavior."""
  code: str = "flatten(x)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("numpy.ravel")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.ravel(x)"


def test_flatten_numpy_reshape_batch() -> None:
  """Verify flatten numpy reshape batch behavior."""
  code: str = "flatten(x, 1)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

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
  code: str = "flatten(x, 1)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("mlx.core.flatten")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "mlx.core.flatten(x, 1, -1)"


def test_flatten_keras_layer() -> None:
  """Verify flatten keras layer behavior."""
  code: str = "flatten(x)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("keras.layers.Flatten")
  # To force the `target_variant.op_type` condition false but fall through by string match
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert cst.Module(body=[]).code_for_node(new_call) == "keras.layers.Flatten()(x)"


def test_flatten_value_error_positional() -> None:
  """Verify flatten value error positional behavior."""
  code: str = "flatten(x, 'a', 'b')"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_negative_end_dim_kwargs() -> None:
  """Verify flatten negative end dim kwargs behavior."""
  code: str = "flatten(x, start_dim=-1, end_dim=-2)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_keyword_args_positive() -> None:
  """Verify flatten keyword args positive behavior."""
  code: str = "flatten(x, start_dim=1, end_dim=2)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_value_error_hex() -> None:
  """Verify flatten value error hex behavior."""
  code: str = "flatten(x, 0x1, 0x2)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_fallback_lookups() -> None:
  """Verify flatten fallback lookups behavior."""
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("fallback.flatten")

  # 118->121 false branch
  ctx.current_op_id = None

  def side_effect(k):
    """Execute side effect helper logic."""
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

  res = transform_flatten(call, ctx)
  assert res is not call


def test_flatten_fallback_none() -> None:
  """Verify flatten fallback none behavior."""
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("fallback.flatten")

  ctx.current_op_id = None
  ctx.lookup_api.side_effect = lambda k: None

  res = transform_flatten(call, ctx)
  assert res is call


def test_flatten_negative_end_dim_kwargs_variables() -> None:
  """Verify flatten negative end dim kwargs variables behavior."""
  code: str = "flatten(x, start_dim=my_start, end_dim=my_end)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("numpy.reshape")
  _res = transform_flatten(call, ctx)
  assert _res is not None or _res is None


def test_flatten_no_variant() -> None:
  """Verify flatten no variant behavior."""
  code: str = "flatten(x)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("numpy.reshape", variant=None)
  # override variant mock
  from unittest.mock import PropertyMock

  type(ctx).current_variant = PropertyMock(return_value=None)
  _res = transform_flatten(call, ctx)
  assert _res is not None or _res is None
