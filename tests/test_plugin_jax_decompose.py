"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax
from ml_switcheroo.core.hooks import HookContext


def test_decompose_via_jax():
  """Docstring."""
  node = cst.Call(func=cst.Name("Hardswish"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Hardswish"

  result = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "hardswish"
  assert result.func.value.attr.value == "numpy"
  assert result.func.value.value.value == "jax"
  assert len(result.args) == 1


def test_decompose_via_jax_no_op_id():
  """Docstring."""
  node = cst.Call(func=cst.Name("Unknown"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None

  result = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "unknownop"
