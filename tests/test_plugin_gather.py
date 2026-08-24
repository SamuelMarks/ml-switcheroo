"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.plugins.gather import transform_gather
from ml_switcheroo.core.hooks import HookContext


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_no_api(mock_is_framework):
  """Docstring."""
  node = cst.Call(func=cst.Name("gather"))
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  result = transform_gather(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_method(mock_is_framework):
  """Docstring."""
  mock_is_framework.return_value = False
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("gather")),
    args=[
      cst.Arg(value=cst.Integer("1")),
      cst.Arg(value=cst.Name("idx")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result = transform_gather(node, ctx)
  assert len(result.args) == 3
  assert result.args[0].value.value == "x"
  assert result.args[1].value.value == "idx"
  assert result.args[2].value.value == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_function(mock_is_framework):
  """Docstring."""
  node = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Integer("1")),
      cst.Arg(value=cst.Name("idx")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result = transform_gather(node, ctx)
  assert len(result.args) == 3
  assert result.args[0].value.value == "x"
  assert result.args[1].value.value == "idx"
  assert result.args[2].value.value == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_keywords(mock_is_framework):
  """Docstring."""
  node = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(keyword=cst.Name("index"), value=cst.Name("idx")),
      cst.Arg(keyword=cst.Name("dim"), value=cst.Integer("1")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result = transform_gather(node, ctx)
  assert len(result.args) == 3
  assert result.args[0].value.value == "x"
  assert result.args[1].value.value == "idx"
  assert result.args[2].value.value == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_missing_args(mock_is_framework):
  """Docstring."""
  node = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result = transform_gather(node, ctx)
  assert result is node
