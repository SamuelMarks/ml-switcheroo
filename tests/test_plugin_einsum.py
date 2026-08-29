"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.einsum import _create_dotted_name, _is_string, normalize_einsum


def test_is_string() -> None:
  """Docstring."""
  assert _is_string(cst.SimpleString("'a'"))
  assert not _is_string(cst.Name("a"))


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("a.b")
  assert isinstance(node, cst.Attribute)


def test_normalize_einsum_no_args() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("einsum"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result: cst.CSTNode = normalize_einsum(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert len(result.args) == 0


def test_normalize_einsum_already_correct() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Name("einsum"),
    args=[
      cst.Arg(value=cst.SimpleString("'ij,jk->ik'")),
      cst.Arg(value=cst.Name("x")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result: cst.CSTNode = normalize_einsum(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.args[0].value, "value", None) == "'ij,jk->ik'"


def test_normalize_einsum_reorder() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Name("einsum"),
    args=[
      cst.Arg(value=cst.Name("x"), comma=cst.Comma()),
      cst.Arg(value=cst.Name("y"), comma=cst.Comma()),
      cst.Arg(value=cst.SimpleString("'ij,jk->ik'")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result: cst.CSTNode = normalize_einsum(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.args[0].value, "value", None) == "'ij,jk->ik'"
  assert getattr(result.args[1].value, "value", None) == "x"
  assert getattr(result.args[2].value, "value", None) == "y"
  assert result.args[2].comma == cst.MaybeSentinel.DEFAULT


def test_normalize_einsum_no_string() -> None:
  """Docstring."""
  node: cst.Call = cst.Call(
    func=cst.Name("einsum"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Name("y")),
      cst.Arg(value=cst.Name("eq")),
    ],
  )
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result: cst.CSTNode = normalize_einsum(node, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.args[0].value, "value", None) == "x"
