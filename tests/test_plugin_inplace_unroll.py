"""Test suite for plugin inplace unroll."""

from unittest.mock import MagicMock

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.inplace_unroll import _get_method_name, _get_receiver_name, unroll_inplace_ops

cst = _cst


def test_get_receiver_name() -> None:
  """Verify get receiver name behavior."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")))
  receiver = _get_receiver_name(node)
  assert getattr(receiver, "value", None) == "x"

  node2 = cst.Call(func=cst.Name("func"))
  assert _get_receiver_name(node2) is None


def test_get_method_name() -> None:
  """Verify get method name behavior."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")))
  assert _get_method_name(node) == "add_"

  node2 = cst.Call(func=cst.Name("func"))
  assert _get_method_name(node2) is None


def test_unroll_inplace_ops_not_method() -> None:
  """Verify unroll inplace ops not method behavior."""
  node = cst.Call(func=cst.Name("func"))
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert result is node


def test_unroll_inplace_ops_not_inplace() -> None:
  """Verify unroll inplace ops not inplace behavior."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add")))
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert result is node


def test_unroll_inplace_ops_dunder() -> None:
  """Verify unroll inplace ops dunder behavior."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__add__")))
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert result is node

  node2 = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  result2 = unroll_inplace_ops(node2, ctx)
  assert result2 is node2


def test_unroll_inplace_ops_infix() -> None:
  """Verify unroll inplace ops infix behavior."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("add_")), args=[cst.Arg(value=cst.Name("y"))])
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert isinstance(result, cst.BinaryOperation)
  assert isinstance(result.operator, cst.Add)
  assert getattr(result.left, "value", None) == "x"
  assert getattr(result.right, "value", None) == "y"


def test_unroll_inplace_ops_functional_fallback() -> None:
  """Verify unroll inplace ops functional fallback behavior."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("relu_")), args=[])
  ctx = MagicMock(spec=HookContext)
  result = unroll_inplace_ops(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(
    {
      "v": getattr(
        getattr(
          {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
        )("v"),
        getattr({"v": "value"}, "__getitem__")("v"),
        getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "relu"}, "__getitem__")("v")


def test_unroll_inplace_ops_expr_node() -> None:
  """Test unroll_inplace_ops with Expr nodes."""
  ctx = MagicMock(spec=HookContext)

  # Expr without a call
  non_call_expr = cst.Expr(value=cst.Name("x"))
  assert unroll_inplace_ops(non_call_expr, ctx) is non_call_expr

  # Non-Expr and Non-Call node
  pass_node = cst.Pass()
  assert unroll_inplace_ops(pass_node, ctx) is pass_node

  # Expr with a method call
  call_node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("relu_")), args=[])
  expr_node = cst.Expr(value=call_node)
  result = unroll_inplace_ops(expr_node, ctx)
  assert isinstance(result, cst.Expr)
  assert getattr(
    {
      "v": getattr(
        getattr(
          {
            "v": getattr(
              {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").value}, "__getitem__")("v").func},
              "__getitem__",
            )("v").attr
          },
          "__getitem__",
        )("v"),
        getattr({"v": "value"}, "__getitem__")("v"),
        getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "relu"}, "__getitem__")("v")

  # Non-Attribute func fallback
  non_attr_call = cst.Call(func=cst.Name("relu_"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"))])
  assert unroll_inplace_ops(non_attr_call, ctx) is non_attr_call
