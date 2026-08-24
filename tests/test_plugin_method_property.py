"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.method_property import transform_method_to_property


def test_not_method_call():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  code = "my_func()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_unknown_method():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  code = "x.unknown_method()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_not_tensor():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "List"

  code = "x.size()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_no_target_prop():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = None

  code = "x.size()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_size_no_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code = "x.size()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "shape"


def test_size_with_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code = "x.size(0)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Subscript)
  assert isinstance(transformed.value, cst.Attribute)
  assert transformed.value.attr.value == "shape"


def test_size_with_multiple_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code = "x.size(0, 1)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_data_ptr():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "data_ptr"

  code = "x.data_ptr()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "data_ptr"
