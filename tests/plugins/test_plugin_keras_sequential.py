"""Test suite for plugin keras sequential."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.keras_sequential import _create_dotted_name, transform_keras_sequential


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


def test_create_dotted_name() -> None:
  """Verify `_create_dotted_name` produces expected results across plugin keras sequential inputs."""
  node = _create_dotted_name("keras.models.Sequential")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "Sequential"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "models"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "keras"


def test_transform_keras_sequential_op_api_not_string() -> None:
  """Verify unmapped Sequential calls abort safely instead of fabricating keras.Sequential."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None
  ctx.semantics = MagicMock()

  call_node = _parse_call("Sequential(Layer1(), Layer2())")
  transformed = transform_keras_sequential(call_node, ctx)
  assert transformed is call_node


def test_transform_keras_sequential_empty_args() -> None:
  """Verify `ensure_type` handles transform keras sequential empty args as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mykeras.Sequential"

  call_node = _parse_call("Sequential()")
  transformed = transform_keras_sequential(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 0


def test_transform_keras_sequential_already_list_or_tuple() -> None:
  """Verify `ensure_type` handles transform keras sequential already list or tuple as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mykeras.Sequential"

  call_node1 = _parse_call("Sequential([Layer1(), Layer2()])")
  transformed1 = transform_keras_sequential(call_node1, ctx)
  assert isinstance(transformed1, cst.Call)
  assert len(transformed1.args) == 1
  assert isinstance(transformed1.args[0].value, cst.List)

  call_node2 = _parse_call("Sequential((Layer1(), Layer2()))")
  transformed2 = transform_keras_sequential(call_node2, ctx)
  assert isinstance(transformed2, cst.Call)
  assert len(transformed2.args) == 1
  assert isinstance(transformed2.args[0].value, cst.Tuple)


def test_transform_keras_sequential_pack_args_ignore_kwargs() -> None:
  """Verify `ensure_type` handles transform keras sequential pack args ignore kwargs as expected."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "mykeras.Sequential"

  call_node = _parse_call("Sequential(Layer1(), Layer2(), name='my_model')")
  transformed = transform_keras_sequential(call_node, ctx)

  # Keyword args should be ignored, positional packed into a list
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 1
  assert isinstance(transformed.args[0].value, cst.List)
  assert len(transformed.args[0].value.elements) == 2
