"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax
from ml_switcheroo.core.hooks import HookContext


def test_decompose_via_jax() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Hardswish"), args=[cst.Arg(value=cst.Name("x"))])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "Hardswish"

  result: cst.CSTNode = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "hardswish"
  assert getattr(getattr(result.func, "value", None), "attr", None) is not None
  assert getattr(getattr(result.func, "value", None).attr, "value", None) == "numpy"
  assert getattr(getattr(getattr(result.func, "value", None), "value", None), "value", None) == "jax"
  assert len(result.args) == 1


def test_decompose_via_jax_no_op_id() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("Unknown"), args=[cst.Arg(value=cst.Name("x"))])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = None

  result: cst.CSTNode = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "unknownop"
