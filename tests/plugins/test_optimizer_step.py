"""Test suite for the Optimizer Step module."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.optimizer_step import (
  _create_dotted_name,
  _get_func_name,
  strip_zero_grad,
  transform_optimizer_init,
  transform_optimizer_step,
)


def test_create_dotted_name() -> None:
  """Creates dotted name."""
  node = _create_dotted_name("Adam")
  assert isinstance(node, cst.Name)
  assert node.value == "Adam"
  node2 = _create_dotted_name("torch.optim.Adam")
  assert isinstance(node2, cst.Attribute)
  assert node2.attr.value == "Adam"
  assert isinstance(node2.value, cst.Attribute)
  assert node2.value.attr.value == "optim"
  assert isinstance(node2.value.value, cst.Name)
  assert node2.value.value.value == "torch"


def test_transform_optimizer_init() -> None:
  """Transforms optimizer initialization."""
  ctx: HookContext = MagicMock(spec=HookContext)
  node: cst.Call = cst.Call(
    func=cst.Name("Adam"),
    args=[
      cst.Arg(value=cst.Name("params")),
      cst.Arg(value=cst.Float("0.01"), keyword=cst.Name("lr"), equal=cst.AssignEqual()),
    ],
  )
  new_node: cst.Call = transform_optimizer_init(node, ctx)
  assert len(new_node.args) == 1
  assert new_node.args[0].keyword is not None
  assert new_node.args[0].keyword.value == "lr"
  node_empty: cst.Call = cst.Call(func=cst.Name("Adam"), args=[])
  new_node_empty: cst.Call = transform_optimizer_init(node_empty, ctx)
  assert len(new_node_empty.args) == 0
  node_kwargs: cst.Call = cst.Call(
    func=cst.Name("Adam"), args=[cst.Arg(value=cst.Float("0.01"), keyword=cst.Name("lr"), equal=cst.AssignEqual())]
  )
  new_node_kwargs: cst.Call = transform_optimizer_init(node_kwargs, ctx)
  assert len(new_node_kwargs.args) == 1
  assert new_node_kwargs.args[0].keyword is not None
  assert new_node_kwargs.args[0].keyword.value == "lr"


def test_get_func_name() -> None:
  """Gets function name."""
  node_attr: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("opt"), attr=cst.Name("step")), args=[])
  assert _get_func_name(node_attr) == "step"
  node_name: cst.Call = cst.Call(func=cst.Name("step"), args=[])
  assert _get_func_name(node_name) == "step"


def test_transform_optimizer_step() -> None:
  """Transforms optimizer step."""
  ctx: HookContext = MagicMock(spec=HookContext)
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("opt"), attr=cst.Name("step")), args=[])
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": transform_optimizer_step}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert result is node


def test_transform_optimizer_step_no_leading_lines() -> None:
  """Transforms optimizer step no leading lines."""
  ctx: HookContext = MagicMock(spec=HookContext)
  node: cst.Call = cst.Call(func=cst.Name("step"), args=[])
  result: cst.CSTNode | cst.Call = getattr(
    {
      "v": getattr({"v": transform_optimizer_step}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert result is node


def test_strip_zero_grad() -> None:
  """Verifies the behavior of strip zero grad."""
  ctx: HookContext = MagicMock(spec=HookContext)
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("opt"), attr=cst.Name("zero_grad")), args=[])
  new_node: cst.CSTNode | cst.Name | cst.Call = getattr(
    {
      "v": getattr({"v": strip_zero_grad}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(new_node, cst.Call)
  assert getattr(new_node.func, "value") == "None"


def test_optimizer_init_missing_api() -> None:
  """Verify transform_optimizer_init returns unchanged Call node when lookup_api is None."""
  from ml_switcheroo.plugins.optimizer_step import transform_optimizer_init

  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None
  node = cst.ensure_type(cst.parse_expression("opt(lr=0.1)"), cst.Call)
  assert cst.Module(body=[]).code_for_node(transform_optimizer_init(node, ctx)) == "opt(lr=0.1)"
  node_empty = cst.ensure_type(cst.parse_expression("opt()"), cst.Call)
  assert cst.Module(body=[]).code_for_node(transform_optimizer_init(node_empty, ctx)) == "opt()"
