"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.mlx_extras import _create_dotted_name, transform_compiler, transform_synchronize


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("mlx.core.compile")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "compile"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "core"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "mlx"


def test_transform_compiler_decorator() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mlx.core.compile"

  code: str = "@torch.compile(fullgraph=True)\ndef foo(): pass"
  module: cst.Module = cst.parse_module(code)
  decorator_node: cst.Decorator = getattr(module.body[0], "decorators")[0]

  transformed_node: cst.CSTNode = transform_compiler(decorator_node, ctx)
  assert isinstance(transformed_node, cst.Decorator)
  assert isinstance(transformed_node.decorator, cst.Attribute)
  assert transformed_node.decorator.attr.value == "compile"


def test_transform_compiler_call_with_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mlx.core.compile"

  code: str = "torch.compile(fn, backend='inductor')"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_compiler(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "compile"
  assert len(transformed.args) == 1
  assert getattr(transformed.args[0].value, "value", None) == "fn"


def test_transform_compiler_call_without_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  code: str = "torch.compile()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_compiler(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 0


def test_transform_compiler_not_decorator_or_call() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  code: str = "torch.compile"
  module: cst.Module = cst.parse_module(code)
  node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_compiler(node, ctx)
  assert transformed is node


def test_transform_synchronize() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  code: str = "torch.cuda.synchronize()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_synchronize(call_node, ctx)

  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Name)
  assert transformed.func.value == "print"
  assert len(transformed.args) == 1
  assert "Global sync requires explicit tensor args" in getattr(transformed.args[0].value, "value", "")
