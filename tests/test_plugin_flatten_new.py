"""Tests for the new flatten plugin transformations."""

import libcst as cst
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten
from ml_switcheroo.config import RuntimeConfig


def get_context(api_name: str, op_type: str = "function") -> HookContext:
  """Get a mock hook context for testing."""

  class MockSemantics:
    def resolve_variant(self, op_name, target_fw):
      return {"api": api_name, "op_type": op_type}

  class MockConfig(RuntimeConfig):
    source_framework: str = "torch"
    target_framework: str = "numpy"

  config = MockConfig(source_framework="torch", target_framework="numpy")
  ctx = HookContext(semantics=MockSemantics(), config=config)
  ctx.current_op_id = "flatten"
  return ctx


def test_flatten_jax_collapse():
  """Test flatten to jax collapse transformation."""
  # torch.flatten(x, 1) -> jax.lax.collapse(x, 1, x.ndim)
  code = "flatten(x, 1)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value

  ctx = get_context("jax.lax.collapse")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, x.ndim)"


def test_flatten_numpy_ravel():
  """Test flatten to numpy ravel transformation."""
  # torch.flatten(x) -> numpy.ravel(x)
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value

  ctx = get_context("numpy.ravel")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.ravel(x)"


def test_flatten_numpy_reshape_batch():
  """Test flatten to numpy reshape transformation."""
  # torch.flatten(x, 1) -> numpy.reshape(x, (x.shape[0], -1))
  code = "flatten(x, 1)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value

  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.reshape(x, (x.shape[0], -1))"


def test_flatten_mlx():
  """Test flatten to mlx transformation."""
  # torch.flatten(x, 1) -> mlx.core.flatten(x, 1, -1)
  code = "flatten(x, 1)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value

  ctx = get_context("mlx.core.flatten")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "mlx.core.flatten(x, 1, -1)"


def test_flatten_keras_layer():
  """Test flatten to keras layer transformation."""
  # torch.flatten(x) -> keras.layers.Flatten()(x)
  code = "flatten(x)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value

  ctx = get_context("keras.layers.Flatten", op_type="class")
  new_call = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "keras.layers.Flatten()(x)"


def test_flatten_value_error_positional():
  """Test value error in positional argument processing."""
  code = "flatten(x, 'a', 'b')"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_negative_end_dim_kwargs():
  """Test negative end_dim in kwargs."""
  code = "flatten(x, start_dim=-1, end_dim=-2)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_keyword_args_positive():
  """Test positive keyword args."""
  code = "flatten(x, start_dim=1, end_dim=2)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_value_error_hex():
  """Test value error with hex."""
  code = "flatten(x, 0x1, 0x2)"
  module = cst.parse_module(code)
  call = module.body[0].body[0].value
  ctx = get_context("numpy.reshape")
  new_call = transform_flatten(call, ctx)
  assert new_call is not None
