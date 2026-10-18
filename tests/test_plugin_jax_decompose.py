"""Test suite for plugin jax decompose."""

import typing

from unittest.mock import MagicMock

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax

cst: typing.Any = _cst


def test_decompose_via_jax() -> None:
  """Verify decompose via jax behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Hardswish"), args=[cst.Arg(value=cst.Name("x"))])
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = "Hardswish"

  result: typing.Any = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "hardswish"
  func_val: typing.Any = getattr(result.func, "value", None)
  assert func_val is not None
  assert getattr(func_val, "attr", None) is not None
  assert getattr(func_val.attr, "value", None) == "numpy"
  assert getattr(getattr(func_val, "value", None), "value", None) == "jax"
  assert len(result.args) == 1


def test_decompose_via_jax_no_op_id() -> None:
  """Verify decompose via jax no op id behavior."""
  node: typing.Any = cst.Call(func=cst.Name("Unknown"), args=[cst.Arg(value=cst.Name("x"))])
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.current_op_id = None

  result: typing.Any = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "unknownop"
