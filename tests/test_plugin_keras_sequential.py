"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.keras_sequential import _create_dotted_name, transform_keras_sequential


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("keras.models.Sequential")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "Sequential"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "models"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "keras"


def test_transform_keras_sequential_op_api_not_string() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None  # None is not a string

  code: str = "Sequential(Layer1(), Layer2())"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_keras_sequential(call_node, ctx)

  # Should default to "keras.Sequential"
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert isinstance(transformed.func.value, cst.Name)
  assert transformed.func.value.value == "keras"
  assert transformed.func.attr.value == "Sequential"

  # Should pack args into a list
  assert len(transformed.args) == 1
  assert isinstance(transformed.args[0].value, cst.List)
  assert len(transformed.args[0].value.elements) == 2


def test_transform_keras_sequential_empty_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mykeras.Sequential"

  code: str = "Sequential()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_keras_sequential(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 0


def test_transform_keras_sequential_already_list_or_tuple() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mykeras.Sequential"

  code1: str = "Sequential([Layer1(), Layer2()])"
  module1: cst.Module = cst.parse_module(code1)
  call_node1: cst.BaseExpression = module1.body[0].body[0].value

  transformed1: cst.CSTNode = transform_keras_sequential(call_node1, ctx)
  assert isinstance(transformed1, cst.Call)
  assert len(transformed1.args) == 1
  assert isinstance(transformed1.args[0].value, cst.List)

  code2: str = "Sequential((Layer1(), Layer2()))"
  module2: cst.Module = cst.parse_module(code2)
  call_node2: cst.BaseExpression = module2.body[0].body[0].value

  transformed2: cst.CSTNode = transform_keras_sequential(call_node2, ctx)
  assert isinstance(transformed2, cst.Call)
  assert len(transformed2.args) == 1
  assert isinstance(transformed2.args[0].value, cst.Tuple)


def test_transform_keras_sequential_pack_args_ignore_kwargs() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mykeras.Sequential"

  code: str = "Sequential(Layer1(), Layer2(), name='my_model')"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = module.body[0].body[0].value

  transformed: cst.CSTNode = transform_keras_sequential(call_node, ctx)

  # Keyword args should be ignored, positional packed into a list
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 1
  assert isinstance(transformed.args[0].value, cst.List)
  assert len(transformed.args[0].value.elements) == 2
