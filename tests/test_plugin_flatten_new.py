"""Tests for the new flatten plugin transformations."""

from typing import Dict

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.flatten import transform_flatten


def get_context(api_name: str, op_type: str = "function") -> HookContext:
  """Docstring."""

  class MockSemantics:
    def resolve_variant(self, op_name: str, target_fw: str) -> Dict[str, str]:
      return {"api": api_name, "op_type": op_type}

  class MockConfig(RuntimeConfig):
    source_framework: str = "torch"
    target_framework: str = "numpy"

  config: MockConfig = MockConfig(source_framework="torch", target_framework="numpy")
  ctx: HookContext = HookContext(semantics=MockSemantics(), config=config)
  ctx.current_op_id = "flatten"
  return ctx


def test_flatten_jax_collapse() -> None:
  """Docstring."""
  # torch.flatten(x, 1) -> jax.lax.collapse(x, 1, x.ndim)
  code: str = "flatten(x, 1)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("jax.lax.collapse")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "jax.lax.collapse(x, 1, x.ndim)"


def test_flatten_numpy_ravel() -> None:
  """Docstring."""
  # torch.flatten(x) -> numpy.ravel(x)
  code: str = "flatten(x)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("numpy.ravel")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.ravel(x)"


def test_flatten_numpy_reshape_batch() -> None:
  """Docstring."""
  # torch.flatten(x, 1) -> numpy.reshape(x, (x.shape[0], -1))
  code: str = "flatten(x, 1)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "numpy.reshape(x, (x.shape[0], -1))"


def test_flatten_mlx() -> None:
  """Docstring."""
  # torch.flatten(x, 1) -> mlx.core.flatten(x, 1, -1)
  code: str = "flatten(x, 1)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("mlx.core.flatten")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "mlx.core.flatten(x, 1, -1)"


def test_flatten_keras_layer() -> None:
  """Docstring."""
  # torch.flatten(x) -> keras.layers.Flatten()(x)
  code: str = "flatten(x)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value

  ctx: HookContext = get_context("keras.layers.Flatten", op_type="class")
  new_call: cst.CSTNode = transform_flatten(call, ctx)

  assert cst.Module(body=[]).code_for_node(new_call) == "keras.layers.Flatten()(x)"


def test_flatten_value_error_positional() -> None:
  """Docstring."""
  code: str = "flatten(x, 'a', 'b')"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_negative_end_dim_kwargs() -> None:
  """Docstring."""
  code: str = "flatten(x, start_dim=-1, end_dim=-2)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_keyword_args_positive() -> None:
  """Docstring."""
  code: str = "flatten(x, start_dim=1, end_dim=2)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None


def test_flatten_value_error_hex() -> None:
  """Docstring."""
  code: str = "flatten(x, 0x1, 0x2)"
  module: cst.Module = cst.parse_module(code)
  call: cst.BaseExpression = module.body[0].body[0].value
  ctx: HookContext = get_context("numpy.reshape")
  new_call: cst.CSTNode = transform_flatten(call, ctx)
  assert new_call is not None
