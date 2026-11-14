"""Test suite for plugin jax decompose."""

from unittest.mock import MagicMock

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax

cst = _cst


def test_decompose_via_jax() -> None:
  """Verify `Call` returns `1` for decompose via JAX."""
  node = cst.Call(func=cst.Name("Hardswish"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Hardswish"

  result = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "hardswish"
  func_val = result.func.value
  assert isinstance(func_val, cst.Attribute)
  assert func_val.attr.value == "numpy"
  assert isinstance(func_val.value, cst.Name)
  assert func_val.value.value == "jax"
  assert len(result.args) == 1


def test_decompose_via_jax_no_op_id() -> None:
  """Verify `Call` handles decompose via JAX no operation id as expected."""
  node = cst.Call(func=cst.Name("Unknown"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None

  result = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "unknownop"
