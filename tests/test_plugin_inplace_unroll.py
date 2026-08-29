"""Docstring."""

from typing import Optional
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.inplace_unroll import _get_method_name, _get_receiver_name, unroll_inplace_ops


def test_get_receiver_name() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")))
  receiver: Optional[cst.BaseExpression] = _get_receiver_name(node)
  assert getattr(receiver, "value", None) == "x"

  node2: cst.Call = cst.Call(func=cst.Name("func"))
  assert _get_receiver_name(node2) is None


def test_get_method_name() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")))
  assert _get_method_name(node) == "add_"

  node2: cst.Call = cst.Call(func=cst.Name("func"))
  assert _get_method_name(node2) is None


def test_unroll_inplace_ops_not_method() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("func"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  result: cst.CSTNode = unroll_inplace_ops(node, ctx)
  assert result is node


def test_unroll_inplace_ops_not_inplace() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add")))
  ctx: MagicMock = MagicMock(spec=HookContext)
  result: cst.CSTNode = unroll_inplace_ops(node, ctx)
  assert result is node


def test_unroll_inplace_ops_dunder() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__add__")))
  ctx: MagicMock = MagicMock(spec=HookContext)
  result: cst.CSTNode = unroll_inplace_ops(node, ctx)
  assert result is node

  node2: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  result2: cst.CSTNode = unroll_inplace_ops(node2, ctx)
  assert result2 is node2


def test_unroll_inplace_ops_infix() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")), args=[cst.Arg(value=cst.Name("y"))]
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  result: cst.CSTNode = unroll_inplace_ops(node, ctx)
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
  assert getattr(result.left, "value", None) == "x"
  assert getattr(result.right, "value", None) == "y"


def test_unroll_inplace_ops_functional_fallback() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("relu_")), args=[])
  ctx: MagicMock = MagicMock(spec=HookContext)
  result: cst.CSTNode = unroll_inplace_ops(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "relu"
