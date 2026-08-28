"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.batch_norm import transform_batch_norm
from ml_switcheroo.core.hooks import HookContext
from typing import List


def test_transform_batch_norm_not_required() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("bn1"), args=[cst.Arg(value=cst.Name("x"))])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = False
  result: cst.CSTNode = transform_batch_norm(node, ctx)
  assert result is node


def test_transform_batch_norm_injects_args() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("bn1"), args=[cst.Arg(value=cst.Name("x"))])
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result: cst.CSTNode = transform_batch_norm(node, ctx)
  assert isinstance(result, cst.Subscript)
  assert isinstance(result.slice[0].slice, cst.Index)
  assert isinstance(result.slice[0].slice.value, cst.Integer)
  assert result.slice[0].slice.value.value == "0"
  call_node: cst.BaseExpression = result.value
  assert isinstance(call_node, cst.Call)
  assert len(call_node.args) == 3
  assert getattr(call_node.args[1].keyword, "value", None) == "use_running_average"
  assert isinstance(call_node.args[1].value, cst.UnaryOperation)
  assert getattr(call_node.args[2].keyword, "value", None) == "mutable"
  assert isinstance(call_node.args[2].value, cst.List)


def test_transform_batch_norm_args_exist() -> None:
  """Docstring."""
  args: List[cst.Arg] = [
    cst.Arg(value=cst.Name("x")),
    cst.Arg(keyword=cst.Name("use_running_average"), value=cst.Name("True")),
    cst.Arg(keyword=cst.Name("mutable"), value=cst.Name("False")),
  ]
  node: cst.Call = cst.Call(func=cst.Name("bn1"), args=args)
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.plugin_traits.requires_functional_state = True
  result: cst.CSTNode = transform_batch_norm(node, ctx)
  assert isinstance(result, cst.Subscript)
  call_node: cst.BaseExpression = result.value
  assert isinstance(call_node, cst.Call)
  assert len(call_node.args) == 3
