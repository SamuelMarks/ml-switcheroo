"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.clipping import transform_grad_clipping
from ml_switcheroo.core.hooks import HookContext


def test_transform_grad_clipping_not_required():
  """Docstring."""
  node = cst.Call(func=cst.Name("clip"), args=[])
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = False
  result = transform_grad_clipping(node, ctx)
  assert result is node


def test_transform_grad_clipping_missing_args():
  """Docstring."""
  node = cst.Call(func=cst.Name("clip"), args=[cst.Arg(value=cst.Name("grads"))])
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result = transform_grad_clipping(node, ctx)
  assert result is node


def test_transform_grad_clipping():
  """Docstring."""
  node = cst.Call(
    func=cst.Name("clip_grad_norm_"),
    args=[
      cst.Arg(value=cst.Name("grads")),
      cst.Arg(value=cst.Float("1.0")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result = transform_grad_clipping(node, ctx)
  assert isinstance(result, cst.Subscript)
  assert result.slice[0].slice.value.value == "0"
  update_call = result.value
  assert isinstance(update_call, cst.Call)
  assert update_call.func.attr.value == "update"
  assert update_call.func.value.func.attr.value == "clip_by_global_norm"
  assert len(update_call.args) == 2
  assert update_call.args[0].value.value == "grads"
  assert update_call.args[1].value.value == "None"
