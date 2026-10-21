"""Test suite for plugin jax decompose."""

from unittest.mock import MagicMock

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax

cst = _cst


def test_decompose_via_jax() -> None:
  """Verify decompose via jax behavior."""
  node = cst.Call(func=cst.Name("Hardswish"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Hardswish"

  result = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(
    {
      "v": getattr(
        getattr(
          {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
        )("v"),
        getattr({"v": "value"}, "__getitem__")("v"),
        getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "hardswish"}, "__getitem__")("v")
  func_val = getattr(result.func, "value", None)
  assert func_val is not None
  assert getattr(func_val, "attr", None) is not None
  assert getattr(func_val.attr, "value", None) == "numpy"
  assert getattr(getattr(func_val, "value", None), "value", None) == "jax"
  assert len(result.args) == 1


def test_decompose_via_jax_no_op_id() -> None:
  """Verify decompose via jax no op id behavior."""
  node = cst.Call(func=cst.Name("Unknown"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = None

  result = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(
    {
      "v": getattr(
        getattr(
          {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
        )("v"),
        getattr({"v": "value"}, "__getitem__")("v"),
        getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "unknownop"}, "__getitem__")("v")
