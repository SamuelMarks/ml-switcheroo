"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.method_property import transform_method_to_property


def test_not_method_call() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "my_func()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_unknown_method() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "x.unknown_method()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_not_tensor() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "List"

  code: str = "x.size()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_no_target_prop() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = None

  code: str = "x.size()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_size_no_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code: str = "x.size()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "shape"


def test_size_with_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code: str = "x.size(0)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Subscript)
  assert isinstance(transformed.value, cst.Attribute)
  assert transformed.value.attr.value == "shape"


def test_size_with_multiple_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "shape"

  code: str = "x.size(0, 1)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert transformed is call_node


def test_data_ptr() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.resolve_type.return_value = "Tensor"
  ctx.lookup_api.return_value = "data_ptr"

  code: str = "x.data_ptr()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_method_to_property(call_node, ctx)
  assert isinstance(transformed, cst.Attribute)
  assert transformed.attr.value == "data_ptr"
