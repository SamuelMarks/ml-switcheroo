"""Test suite for plugin gather."""

from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.gather import transform_gather


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_no_api(mock_is_framework: MagicMock) -> None:
  """Verify `Call` handles transform gather no API as expected.

  Args:
      mock_is_framework: Mock object for is framework.
  """
  node: cst.Call = cst.Call(func=cst.Name("gather"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  result: cst.CSTNode = transform_gather(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_method(mock_is_framework: MagicMock) -> None:
  """Verify `Call` returns `3` for transform gather method.

  Args:
      mock_is_framework: Mock object for is framework.
  """
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
  assert isinstance(result.args[0].value, cst.Name)
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Name)
  assert result.args[1].value.value == "idx"
  assert isinstance(result.args[2].value, cst.Integer)
  assert result.args[2].value.value == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_function(mock_is_framework: MagicMock) -> None:
  """Verify `Call` returns `3` for transform gather function.

  Args:
      mock_is_framework: Mock object for is framework.
  """
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
  assert isinstance(result.args[0].value, cst.Name)
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Name)
  assert result.args[1].value.value == "idx"
  assert isinstance(result.args[2].value, cst.Integer)
  assert result.args[2].value.value == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_keywords(mock_is_framework: MagicMock) -> None:
  """Verify `Call` returns `3` for transform gather keywords.

  Args:
      mock_is_framework: Mock object for is framework.
  """
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
  assert isinstance(result.args[0].value, cst.Name)
  assert result.args[0].value.value == "x"
  assert isinstance(result.args[1].value, cst.Name)
  assert result.args[1].value.value == "idx"
  assert isinstance(result.args[2].value, cst.Integer)
  assert result.args[2].value.value == "1"


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_missing_args(mock_is_framework: MagicMock) -> None:
  """Verify `Call` handles transform gather missing args as expected.

  Args:
      mock_is_framework: Mock object for is framework.
  """
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
  """Verify `Call` handles transform gather module node as expected.

  Args:
      mock_is_framework: Mock object for is framework.
  """
  mock_is_framework.return_value = True
  node = cst.Call(func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("gather")), args=[])
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "take_along_axis"
  _res = transform_gather(node, ctx)
  assert _res is node


@patch("ml_switcheroo.plugins.gather.is_framework_module_node")
def test_transform_gather_other_keyword(mock_is_framework: MagicMock) -> None:
  """Verify `Call` returns `2` for transform gather other keyword.

  Args:
      mock_is_framework: Mock object for is framework.
  """
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
