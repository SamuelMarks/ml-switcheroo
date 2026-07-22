"""Test suite for the Optimizer Step module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.optimizer_step import (
  _create_dotted_name,
  transform_optimizer_init,
  transform_optimizer_step,
  strip_zero_grad,
  _get_func_name,
)


def test_create_dotted_name():
  """Creates dotted name."""
  node = _create_dotted_name("Adam")
  assert isinstance(node, cst.Name)
  assert node.value == "Adam"
  node = _create_dotted_name("torch.optim.Adam")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "Adam"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "optim"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "torch"


def test_transform_optimizer_init():
  """Transforms optimizer initialization."""
  ctx = MagicMock(spec=HookContext)
  node = cst.Call(
    func=cst.Name("Adam"),
    args=[
      cst.Arg(value=cst.Name("params")),
      cst.Arg(value=cst.Float("0.01"), keyword=cst.Name("lr"), equal=cst.AssignEqual()),
    ],
  )
  new_node = transform_optimizer_init(node, ctx)
  assert len(new_node.args) == 1
  assert new_node.args[0].keyword is not None
  assert new_node.args[0].keyword.value == "lr"
  node_empty = cst.Call(func=cst.Name("Adam"), args=[])
  new_node_empty = transform_optimizer_init(node_empty, ctx)
  assert len(new_node_empty.args) == 0
  node_kwargs = cst.Call(
    func=cst.Name("Adam"), args=[cst.Arg(value=cst.Float("0.01"), keyword=cst.Name("lr"), equal=cst.AssignEqual())]
  )
  new_node_kwargs = transform_optimizer_init(node_kwargs, ctx)
  assert len(new_node_kwargs.args) == 1
  assert new_node_kwargs.args[0].keyword.value == "lr"


def test_get_func_name():
  """Gets function name."""
  node_attr = cst.Call(func=cst.Attribute(value=cst.Name("opt"), attr=cst.Name("step")), args=[])
  assert _get_func_name(node_attr) == "step"
  node_name = cst.Call(func=cst.Name("step_func"), args=[])
  assert _get_func_name(node_name) == "step"


def test_transform_optimizer_step():
  """Transforms optimizer step."""
  ctx = MagicMock(spec=HookContext)
  node = cst.Call(func=cst.Attribute(value=cst.Name("opt"), attr=cst.Name("step")), args=[])
  result = transform_optimizer_step(node, ctx)
  assert result is node


def test_transform_optimizer_step_no_leading_lines():
  """Transforms optimizer step no leading lines."""
  ctx = MagicMock(spec=HookContext)
  node = cst.Call(func=cst.Name("step"), args=[])
  result = transform_optimizer_step(node, ctx)
  assert result is node


def test_strip_zero_grad():
  """Verifies the behavior of strip zero grad."""
  ctx = MagicMock(spec=HookContext)
  node = cst.Call(func=cst.Attribute(value=cst.Name("opt"), attr=cst.Name("zero_grad")), args=[])
  new_node = strip_zero_grad(node, ctx)
  assert isinstance(new_node.func, cst.Name)
  assert new_node.func.value == "None"
  assert len(new_node.args) == 0
