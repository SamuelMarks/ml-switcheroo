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


def test_transform_optimizer_init() -> None:
  """Verify `ensure_type` returns `1` for transform optimizer init."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  call_node: cst.Call = _parse_call("optax.adam(params, lr=0.01)")

  transformed: cst.CSTNode = transform_optimizer_init(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 1
  assert isinstance(transformed.args[0].keyword, cst.Name)
  assert transformed.args[0].keyword.value == "lr"


def test_transform_optimizer_init_no_args() -> None:
  """Verify `ensure_type` returns `0` for transform optimizer init no args."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  call_node: cst.Call = _parse_call("optax.adam()")

  transformed: cst.CSTNode = transform_optimizer_init(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 0


def test_transform_optimizer_step() -> None:
  """Verify `ensure_type` handles transform optimizer step as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  call_node: cst.Call = _parse_call("optimizer.step()")

  transformed = transform_optimizer_step(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert transformed is call_node
  assert len(transformed.args) == 0
  assert cst.Module([]).code_for_node(transformed) == "optimizer.step()"


def test_strip_zero_grad() -> None:
  """Verify `ensure_type` returns `0` for strip zero grad."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  call_node: cst.Call = _parse_call("optimizer.zero_grad()")

  transformed = strip_zero_grad(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Name)
  assert transformed.func.value == "None"
  assert len(transformed.args) == 0


def test_get_func_name() -> None:
  """Verify `Call` handles get func name as expected."""
  node: cst.Call = cst.Call(func=cst.Name("my_func"), args=[])
  assert _get_func_name(node) == "step"
