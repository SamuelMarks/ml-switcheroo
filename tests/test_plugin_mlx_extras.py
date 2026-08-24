"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.mlx_extras import _create_dotted_name, transform_compiler, transform_synchronize


def test_create_dotted_name():
  """Docstring."""
  node = _create_dotted_name("mlx.core.compile")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "compile"
  assert node.value.attr.value == "core"
  assert node.value.value.value == "mlx"


def test_transform_compiler_decorator():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mlx.core.compile"

  code = "@torch.compile(fullgraph=True)\ndef foo(): pass"
  module = cst.parse_module(code)
  decorator_node = module.body[0].decorators[0]

  transformed = transform_compiler(decorator_node, ctx)
  assert isinstance(transformed.decorator, cst.Attribute)
  assert transformed.decorator.attr.value == "compile"


def test_transform_compiler_call_with_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mlx.core.compile"

  code = "torch.compile(fn, backend='inductor')"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_compiler(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "compile"
  assert len(transformed.args) == 1
  assert transformed.args[0].value.value == "fn"


def test_transform_compiler_call_without_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  code = "torch.compile()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_compiler(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 0


def test_transform_compiler_not_decorator_or_call():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  code = "torch.compile"
  module = cst.parse_module(code)
  node = module.body[0].body[0].value

  transformed = transform_compiler(node, ctx)
  assert transformed is node


def test_transform_synchronize():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

  code = "torch.cuda.synchronize()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_synchronize(call_node, ctx)

  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Name)
  assert transformed.func.value == "print"
  assert len(transformed.args) == 1
  assert "Global sync requires explicit tensor args" in transformed.args[0].value.value
