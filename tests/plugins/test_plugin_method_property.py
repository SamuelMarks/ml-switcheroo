"""Test suite for plugin method property."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.method_property import transform_method_to_property


def _parse_call(code: str) -> cst.Call:
  """Parse a single-statement call expression into a LibCST Call node.

  Args:
      code: Python call expression source string.

  Returns:
      Parsed `cst.Call` AST node.
  """
  module = cst.parse_module(code)
  stmt = cst.ensure_type(module.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  return cst.ensure_type(expr.value, cst.Call)


def test_not_method_call() -> None:
  """Verify `ensure_type` handles not method call as expected."""
  ctx = MagicMock(spec=HookContext)

  call_node = _parse_call("my_func()")
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_unknown_method() -> None:
  """Verify `ensure_type` handles unknown method as expected."""
  ctx = MagicMock(spec=HookContext)

  call_node = _parse_call("x.unknown_method()")
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_not_tensor() -> None:
  """Verify `ensure_type` handles not tensor as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "List"

  call_node = _parse_call("x.size()")
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_no_target_prop() -> None:
  """Verify `ensure_type` handles no target prop as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = None

  call_node = _parse_call("x.size()")
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_size_no_args() -> None:
  """Verify `ensure_type` handles size no args as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  call_node = _parse_call("x.size()")
  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "shape"


def test_size_with_args() -> None:
  """Verify `ensure_type` handles size with args as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  call_node = _parse_call("x.size(0)")
  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Subscript)
  assert isinstance(transformed.value, cst.Attribute)
  assert transformed.value.attr.value == "shape"


def test_size_with_multiple_args() -> None:
  """Verify `ensure_type` handles size with multiple args as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  call_node = _parse_call("x.size(0, 1)")
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_data_ptr() -> None:
  """Verify `ensure_type` handles data ptr as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "data_ptr"

  call_node = _parse_call("x.data_ptr()")
  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "data_ptr"
