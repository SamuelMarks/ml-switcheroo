"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.optimizer_step import (
  transform_optimizer_init,
  transform_optimizer_step,
  strip_zero_grad,
  _get_func_name,
)


def test_transform_optimizer_init():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  code = "optax.adam(params, lr=0.01)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_optimizer_init(call_node, ctx)
  assert len(transformed.args) == 1
  assert transformed.args[0].keyword.value == "lr"


def test_transform_optimizer_init_no_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  code = "optax.adam()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_optimizer_init(call_node, ctx)
  assert len(transformed.args) == 0


def test_transform_optimizer_step():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  code = "optimizer.step()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_optimizer_step(call_node, ctx)
  assert isinstance(transformed, cst.Call)


def test_strip_zero_grad():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  code = "optimizer.zero_grad()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = strip_zero_grad(call_node, ctx)
  assert isinstance(transformed.func, cst.Name)
  assert transformed.func.value == "None"
  assert len(transformed.args) == 0


def test_get_func_name():
  """Docstring."""
  node = cst.Call(func=cst.Name("my_func"), args=[])
  assert _get_func_name(node) == "step"
