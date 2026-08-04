"""Test suite for the Keras Sequential Extra module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.keras_sequential import transform_keras_sequential
from ml_switcheroo.core.hooks import HookContext


def test_keras_sequential_missing_api():
  """Verifies the behavior of Keras sequential missing API."""
  node = cst.Call(func=cst.Name("Sequential"), args=[cst.Arg(cst.Name("L"))])
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value=None)
  res = transform_keras_sequential(node, ctx)
  assert res.func.value.value == "keras"
  assert res.func.attr.value == "Sequential"


def test_keras_sequential_empty_args():
  """Verifies the behavior of Keras sequential empty arguments."""
  node = cst.Call(func=cst.Name("Sequential"), args=[])
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="my.Seq")
  res = transform_keras_sequential(node, ctx)
  assert res.func.value.value == "my"
  assert res.func.attr.value == "Seq"
  assert not res.args


def test_keras_sequential_list_args():
  """Verifies behavior when arguments are already in a list."""
  node = cst.Call(func=cst.Name("Sequential"), args=[cst.Arg(cst.List([cst.Element(cst.Name("L"))]))])
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="my.Seq")
  res = transform_keras_sequential(node, ctx)
  assert res.func.value.value == "my"
  assert res.func.attr.value == "Seq"
  assert isinstance(res.args[0].value, cst.List)


def test_keras_sequential_keyword_args():
  """Verifies the behavior of Keras sequential keyword arguments."""
  node = cst.Call(
    func=cst.Name("Sequential"),
    args=[cst.Arg(value=cst.Name("L1")), cst.Arg(keyword=cst.Name("name"), value=cst.SimpleString("'test'"))],
  )
  ctx = HookContext(semantics=MagicMock(), config=MagicMock())
  ctx.lookup_api = MagicMock(return_value="my.Seq")
  res = transform_keras_sequential(node, ctx)
  list_arg = res.args[0].value
  assert isinstance(list_arg, cst.List)
  assert len(list_arg.elements) == 1
