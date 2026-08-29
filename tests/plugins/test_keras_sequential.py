"""Test suite for the Keras Sequential Extra module."""

from typing import Union
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential


def test_keras_sequential_missing_api() -> None:
  """Verifies the behavior of Keras sequential missing API."""
  node: cst.Call = cst.Call(func=cst.Name("Sequential"), args=[cst.Arg(cst.Name("L"))])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value=None)
  res: Union[cst.CSTNode, cst.Call] = transform_keras_sequential(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Name)
  assert res.func.value.value == "keras"
  assert res.func.attr.value == "Sequential"


def test_keras_sequential_empty_args() -> None:
  """Verifies the behavior of Keras sequential empty arguments."""
  node: cst.Call = cst.Call(func=cst.Name("Sequential"), args=[])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="my.Seq")
  res: Union[cst.CSTNode, cst.Call] = transform_keras_sequential(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Name)
  assert res.func.value.value == "my"
  assert res.func.attr.value == "Seq"
  assert not res.args


def test_keras_sequential_list_args() -> None:
  """Verifies behavior when arguments are already in a list."""
  node: cst.Call = cst.Call(func=cst.Name("Sequential"), args=[cst.Arg(cst.List([cst.Element(cst.Name("L"))]))])
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="my.Seq")
  res: Union[cst.CSTNode, cst.Call] = transform_keras_sequential(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert isinstance(res.func.value, cst.Name)
  assert res.func.value.value == "my"
  assert res.func.attr.value == "Seq"
  assert isinstance(res.args[0].value, cst.List)


def test_keras_sequential_keyword_args() -> None:
  """Verifies the behavior of Keras sequential keyword arguments."""
  node: cst.Call = cst.Call(
    func=cst.Name("Sequential"),
    args=[cst.Arg(value=cst.Name("L1")), cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString("'test'"))],
  )
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="my.Seq")
  res: Union[cst.CSTNode, cst.Call] = transform_keras_sequential(node, ctx)
  assert isinstance(res, cst.Call)
  list_arg: cst.BaseExpression = res.args[0].value
  assert isinstance(list_arg, cst.List)
  assert len(list_arg.elements) == 1
