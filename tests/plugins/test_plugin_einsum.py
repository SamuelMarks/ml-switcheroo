"""Test suite for plugin einsum."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.einsum import _create_dotted_name, _is_string, normalize_einsum


def test_is_string() -> None:
  """Verify `_is_string` produces expected results across plugin einsum inputs."""
  assert _is_string(cst.SimpleString("'a'"))
  assert not _is_string(cst.Name("a"))


def test_create_dotted_name() -> None:
  """Verify `_create_dotted_name` produces expected results across plugin einsum inputs."""
  node: cst.BaseExpression = _create_dotted_name("a.b")
  assert isinstance(node, cst.Attribute)
  assert isinstance(node.value, cst.Name)
  assert node.value.value == "a"
  assert node.attr.value == "b"
  assert cst.Module([]).code_for_node(node) == "a.b"


def test_normalize_einsum_no_args() -> None:
  """Verify `Call` returns `0` for normalize einsum no args."""
  node: cst.Call = cst.Call(func=cst.Name("einsum"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.einsum"

  result: cst.CSTNode = normalize_einsum(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert len(result.args) == 0


def test_normalize_einsum_already_correct() -> None:
  """Verify `Call` handles normalize einsum already correct as expected."""
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
  assert cst.ensure_type(result.args[0].value, cst.SimpleString).value == "'ij,jk->ik'"


def test_normalize_einsum_reorder() -> None:
  """Verify `Call` handles normalize einsum reorder as expected."""
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
  assert cst.ensure_type(result.args[0].value, cst.SimpleString).value == "'ij,jk->ik'"
  assert cst.ensure_type(result.args[1].value, cst.Name).value == "x"
  assert cst.ensure_type(result.args[2].value, cst.Name).value == "y"
  assert result.args[2].comma == cst.MaybeSentinel.DEFAULT


def test_normalize_einsum_no_string() -> None:
  """Verify `Call` handles normalize einsum no string as expected."""
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
  assert cst.ensure_type(result.args[0].value, cst.Name).value == "x"
