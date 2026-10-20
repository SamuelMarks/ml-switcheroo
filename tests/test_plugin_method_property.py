"""Test suite for plugin method property."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.method_property import transform_method_to_property


def test_not_method_call() -> None:
  """Verify not method call behavior."""
  ctx = MagicMock(spec=HookContext)

  code = "my_func()"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_unknown_method() -> None:
  """Verify unknown method behavior."""
  ctx = MagicMock(spec=HookContext)

  code = "x.unknown_method()"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_not_tensor() -> None:
  """Verify not tensor behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "List"

  code = "x.size()"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_no_target_prop() -> None:
  """Verify no target prop behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = None

  code = "x.size()"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_size_no_args() -> None:
  """Verify size no args behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code = "x.size()"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "shape"


def test_size_with_args() -> None:
  """Verify size with args behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code = "x.size(0)"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Subscript)
  assert isinstance(transformed.value, cst.Attribute)
  assert transformed.value.attr.value == "shape"


def test_size_with_multiple_args() -> None:
  """Verify size with multiple args behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code = "x.size(0, 1)"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_data_ptr() -> None:
  """Verify data ptr behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "data_ptr"

  code = "x.data_ptr()"
  module = cst.parse_module(code)
  call_node = cst.ensure_type(getattr(getattr(module.body[0], "body")[0], "value"), cst.Call)
  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "data_ptr"
