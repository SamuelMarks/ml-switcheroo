"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.inplace_unroll import unroll_inplace_ops, _get_receiver_name, _get_method_name
from ml_switcheroo.core.hooks import HookContext


def test_get_receiver_name():
  """Docstring."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")))
  assert _get_receiver_name(node).value == "x"

  node2 = cst.Call(func=cst.Name("func"))
  assert _get_receiver_name(node2) is None


def test_get_method_name():
  """Docstring."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")))
  assert _get_method_name(node) == "add_"

  node2 = cst.Call(func=cst.Name("func"))
  assert _get_method_name(node2) is None


def test_unroll_inplace_ops_not_method():
  """Docstring."""
  node = cst.Call(func=cst.Name("func"))
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert result is node


def test_unroll_inplace_ops_not_inplace():
  """Docstring."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add")))
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert result is node


def test_unroll_inplace_ops_dunder():
  """Docstring."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__add__")))
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert result is node

  node2 = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  result2 = unroll_inplace_ops(node2, ctx)
  assert result2 is node2


def test_unroll_inplace_ops_infix():
  """Docstring."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")), args=[cst.Arg(value=cst.Name("y"))])
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
  assert result.left.value == "x"
  assert result.right.value == "y"


def test_unroll_inplace_ops_functional_fallback():
  """Docstring."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("relu_")), args=[])
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "relu"
