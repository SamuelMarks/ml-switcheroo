"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.batch_norm import transform_batch_norm
from ml_switcheroo.core.hooks import HookContext


def test_transform_batch_norm_not_required():
  """Docstring."""
  node = cst.Call(func=cst.Name("bn1"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = False
  result = transform_batch_norm(node, ctx)
  assert result is node


def test_transform_batch_norm_injects_args():
  """Docstring."""
  node = cst.Call(func=cst.Name("bn1"), args=[cst.Arg(value=cst.Name("x"))])
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result = transform_batch_norm(node, ctx)
  assert isinstance(result, cst.Subscript)
  assert result.slice[0].slice.value.value == "0"
  call_node = result.value
  assert isinstance(call_node, cst.Call)
  assert len(call_node.args) == 3
  assert call_node.args[1].keyword.value == "use_running_average"
  assert isinstance(call_node.args[1].value, cst.UnaryOperation)
  assert call_node.args[2].keyword.value == "mutable"
  assert isinstance(call_node.args[2].value, cst.List)


def test_transform_batch_norm_args_exist():
  """Docstring."""
  args = [
    cst.Arg(value=cst.Name("x")),
    cst.Arg(keyword=cst.Name("use_running_average"), value=cst.Name("True")),
    cst.Arg(keyword=cst.Name("mutable"), value=cst.Name("False")),
  ]
  node = cst.Call(func=cst.Name("bn1"), args=args)
  ctx = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result = transform_batch_norm(node, ctx)
  assert isinstance(result, cst.Subscript)
  call_node = result.value
  assert len(call_node.args) == 3
