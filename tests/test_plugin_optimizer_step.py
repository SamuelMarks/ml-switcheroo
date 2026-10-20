"""Test suite for plugin optimizer step."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.optimizer_step import (
  _get_func_name,
  strip_zero_grad,
  transform_optimizer_init,
  transform_optimizer_step,
)


def test_transform_optimizer_init() -> None:
  """Verify transform optimizer init behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "optax.adam(params, lr=0.01)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)

  transformed: cst.CSTNode = transform_optimizer_init(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 1
  assert getattr(transformed.args[0].keyword, "value", None) == "lr"


def test_transform_optimizer_init_no_args() -> None:
  """Verify transform optimizer init no args behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "optax.adam()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)

  transformed: cst.CSTNode = transform_optimizer_init(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 0


def test_transform_optimizer_step() -> None:
  """Verify transform optimizer step behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "optimizer.step()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)

  transformed: cst.CSTNode = transform_optimizer_step(call_node, ctx)  # type: ignore[assignment]
  assert isinstance(transformed, cst.Call)


def test_strip_zero_grad() -> None:
  """Verify strip zero grad behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "optimizer.zero_grad()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.Call = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)

  transformed: cst.CSTNode = strip_zero_grad(call_node, ctx)  # type: ignore[assignment]
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Name)
  assert transformed.func.value == "None"
  assert len(transformed.args) == 0


def test_get_func_name() -> None:
  """Verify get func name behavior."""
  node: cst.Call = cst.Call(func=cst.Name("my_func"), args=[])
  assert _get_func_name(node) == "step"
