"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.einsum import normalize_einsum, _create_dotted_name, _is_string
from ml_switcheroo.core.hooks import HookContext


def test_is_string():
  """Docstring."""
  assert _is_string(cst.SimpleString("'a'"))
  assert not _is_string(cst.Name("a"))


def test_create_dotted_name():
  """Docstring."""
  node = _create_dotted_name("a.b")
  assert isinstance(node, cst.Attribute)


def test_normalize_einsum_no_args():
  """Docstring."""
  node = cst.Call(func=cst.Name("einsum"))
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result = normalize_einsum(node, ctx)
  assert isinstance(result.func, cst.Attribute)
  assert len(result.args) == 0


def test_normalize_einsum_already_correct():
  """Docstring."""
  node = cst.Call(
    func=cst.Name("einsum"),
    args=[
      cst.Arg(value=cst.SimpleString("'ij,jk->ik'")),
      cst.Arg(value=cst.Name("x")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result = normalize_einsum(node, ctx)
  assert result.args[0].value.value == "'ij,jk->ik'"


def test_normalize_einsum_reorder():
  """Docstring."""
  node = cst.Call(
    func=cst.Name("einsum"),
    args=[
      cst.Arg(value=cst.Name("x"), comma=cst.Comma()),
      cst.Arg(value=cst.Name("y"), comma=cst.Comma()),
      cst.Arg(value=cst.SimpleString("'ij,jk->ik'")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result = normalize_einsum(node, ctx)
  assert result.args[0].value.value == "'ij,jk->ik'"
  assert result.args[1].value.value == "x"
  assert result.args[2].value.value == "y"
  assert result.args[2].comma == cst.MaybeSentinel.DEFAULT


def test_normalize_einsum_no_string():
  """Docstring."""
  node = cst.Call(
    func=cst.Name("einsum"),
    args=[
      cst.Arg(value=cst.Name("x")),
      cst.Arg(value=cst.Name("y")),
      cst.Arg(value=cst.Name("eq")),
    ],
  )
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result = normalize_einsum(node, ctx)
  assert result.args[0].value.value == "x"
