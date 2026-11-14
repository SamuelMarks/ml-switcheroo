"""Test suite for plugin reshape."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.reshape import _create_dotted_name, transform_view_semantics


def _parse_first_call(code: str) -> cst.Call:
  """Parse a single-statement module and extract its call expression.

  Args:
      code: Python source string containing a call expression statement.

  Returns:
      Extracted ``cst.Call`` node.
  """
  module: cst.Module = cst.parse_module(code)
  stmt: cst.SimpleStatementLine = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr: cst.Expr = cst.ensure_type(stmt.body[0], cst.Expr)
  return cst.ensure_type(expr.value, cst.Call)


def test_create_dotted_name() -> None:
  """Verify `_create_dotted_name` produces expected results across plugin reshape inputs."""
  node: cst.BaseExpression = _create_dotted_name("jax.numpy.reshape")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "reshape"


def test_transform_view_semantics_no_api() -> None:
  """Verify `ensure_type` handles transform view semantics no API as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  call_node: cst.Call = _parse_first_call("x.view(2, 2)")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert transformed is call_node


def test_transform_view_semantics_method() -> None:
  """Verify `ensure_type` returns `2` for transform view semantics method."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  call_node: cst.Call = _parse_first_call("x.view(2, 2)")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert cst.ensure_type(transformed.args[0].value, cst.Name).value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_method_single_int() -> None:
  """Verify `ensure_type` handles transform view semantics method single int as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  call_node: cst.Call = _parse_first_call("x.view(1)")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert cst.ensure_type(transformed.args[0].value, cst.Name).value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_method_tuple_arg() -> None:
  """Verify `ensure_type` handles transform view semantics method tuple arg as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  call_node: cst.Call = _parse_first_call("x.view((2, 2))")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert cst.ensure_type(transformed.args[0].value, cst.Name).value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_no_args() -> None:
  """Verify `ensure_type` handles transform view semantics func no args as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  call_node: cst.Call = _parse_first_call("view()")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert transformed is call_node


def test_transform_view_semantics_func_multiple_args() -> None:
  """Verify `ensure_type` handles transform view semantics func multiple args as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  call_node: cst.Call = _parse_first_call("view(x, 2, 2)")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert cst.ensure_type(transformed.args[0].value, cst.Name).value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_tuple_arg() -> None:
  """Verify `ensure_type` handles transform view semantics func tuple arg as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  call_node: cst.Call = _parse_first_call("view(x, (2, 2))")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert cst.ensure_type(transformed.args[0].value, cst.Name).value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_strict() -> None:
  """Verify `ensure_type` handles transform view semantics strict as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=True)
  ctx.plugin_traits.strict_materialization_method = "block_until_ready"

  call_node: cst.Call = _parse_first_call("x.view(2, 2)")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "block_until_ready"
