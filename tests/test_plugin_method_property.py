"""Test suite for plugin method property."""

import typing

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.method_property import transform_method_to_property


def test_not_method_call() -> None:
  """Verify not method call behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)

  code: typing.Any = "my_func()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_unknown_method() -> None:
  """Verify unknown method behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)

  code: typing.Any = "x.unknown_method()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_not_tensor() -> None:
  """Verify not tensor behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "List"

  code: typing.Any = "x.size()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_no_target_prop() -> None:
  """Verify no target prop behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = None

  code: typing.Any = "x.size()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_size_no_args() -> None:
  """Verify size no args behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code: typing.Any = "x.size()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "shape"


def test_size_with_args() -> None:
  """Verify size with args behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code: typing.Any = "x.size(0)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Subscript)
  assert isinstance(transformed.value, cst.Attribute)
  assert transformed.value.attr.value == "shape"


def test_size_with_multiple_args() -> None:
  """Verify size with multiple args behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code: typing.Any = "x.size(0, 1)"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_data_ptr() -> None:
  """Verify data ptr behavior."""
  ctx: typing.Any = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "data_ptr"

  code: typing.Any = "x.data_ptr()"
  module: typing.Any = cst.parse_module(code)
  call_node: typing.Any = module.body[0].body[0].value

  transformed: typing.Any = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "data_ptr"
