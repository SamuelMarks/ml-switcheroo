"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.clipping import transform_grad_clipping
from ml_switcheroo.core.hooks import HookContext


def test_transform_grad_clipping_not_required() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("clip"), args=[])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = False
  result: cst.CSTNode = transform_grad_clipping(node, ctx)
  assert result is node


def test_transform_grad_clipping_missing_args() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("clip"), args=[cst.Arg(value=cst.Name("grads"))])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result: cst.CSTNode = transform_grad_clipping(node, ctx)
  assert result is node


def test_transform_grad_clipping() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Name("clip_grad_norm_"),
    args=[
      cst.Arg(value=cst.Name("grads")),
      cst.Arg(value=cst.Float("1.0")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result: cst.CSTNode = transform_grad_clipping(node, ctx)
  assert isinstance(result, cst.Subscript)
  assert isinstance(result.slice[0].slice, cst.Index)
  assert isinstance(result.slice[0].slice.value, cst.Integer)
  assert result.slice[0].slice.value.value == "0"
  update_call: cst.BaseExpression = result.value
  assert isinstance(update_call, cst.Call)
  assert isinstance(update_call.func, cst.Attribute)
  assert update_call.func.attr.value == "update"
  assert isinstance(update_call.func.value, cst.Call)
  assert isinstance(update_call.func.value.func, cst.Attribute)
  assert update_call.func.value.func.attr.value == "clip_by_global_norm"
  assert len(update_call.args) == 2
  assert isinstance(update_call.args[0].value, cst.Name)
  assert update_call.args[0].value.value == "grads"
  assert isinstance(update_call.args[1].value, cst.Name)
  assert update_call.args[1].value.value == "None"
