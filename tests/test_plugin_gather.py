"""Docstring."""

from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.gather import transform_gather


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_no_api(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("gather"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  result: cst.CSTNode = transform_gather(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_method(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  mock_is_framework.return_value = False
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("gather")),
    args=[
      cst.Arg(value=cst.Integer("1")),
      cst.Arg(value=cst.Name("idx")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result: cst.CSTNode = transform_gather(node, ctx)
  assert isinstance(result, cst.Call)
  assert len(result.args) == 3
  assert getattr(result.args[0].value, "value", None) == "x"
  assert getattr(result.args[1].value, "value", None) == "idx"
  assert getattr(result.args[2].value, "value", None) == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_function(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Integer("1")),
      cst.Arg(value=cst.Name("idx")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result: cst.CSTNode = transform_gather(node, ctx)
  assert isinstance(result, cst.Call)
  assert len(result.args) == 3
  assert getattr(result.args[0].value, "value", None) == "x"
  assert getattr(result.args[1].value, "value", None) == "idx"
  assert getattr(result.args[2].value, "value", None) == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_keywords(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(keyword=cst.Name("index"), value=cst.Name("idx")),
      cst.Arg(keyword=cst.Name("dim"), value=cst.Integer("1")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result: cst.CSTNode = transform_gather(node, ctx)
  assert isinstance(result, cst.Call)
  assert len(result.args) == 3
  assert getattr(result.args[0].value, "value", None) == "x"
  assert getattr(result.args[1].value, "value", None) == "idx"
  assert getattr(result.args[2].value, "value", None) == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_missing_args(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"

  result: cst.CSTNode = transform_gather(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_module_node(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  mock_is_framework.return_value = True
  node = cst.Call(func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("gather")), args=[])
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"
  transform_gather(node, ctx)


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_other_keyword(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  mock_is_framework.return_value = False
  node = cst.Call(
    func=cst.Name("gather"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Name("idx"), keyword=cst.Name("index")),
      cst.Arg(value=cst.Name("foo"), keyword=cst.Name("other")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"
  res = transform_gather(node, ctx)
  assert isinstance(res, cst.Call)
  assert len(res.args) == 2
