"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.reshape import transform_view_semantics, _create_dotted_name


def test_create_dotted_name():
  """Docstring."""
  node = _create_dotted_name("jax.numpy.reshape")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "reshape"


def test_transform_view_semantics_no_api():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  code = "x.view(2, 2)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert transformed is call_node


def test_transform_view_semantics_method():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code = "x.view(2, 2)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert transformed.args[0].value.value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_method_single_int():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code = "x.view(1)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert transformed.args[0].value.value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_method_tuple_arg():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code = "x.view((2, 2))"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert transformed.args[0].value.value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_no_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code = "view()"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert transformed is call_node


def test_transform_view_semantics_func_multiple_args():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code = "view(x, 2, 2)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert len(transformed.args) == 2
  assert transformed.args[0].value.value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_tuple_arg():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code = "view(x, (2, 2))"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert len(transformed.args) == 2
  assert transformed.args[0].value.value == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_strict():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=True)
  ctx.plugin_traits.strict_materialization_method = "block_until_ready"

  code = "x.view(2, 2)"
  module = cst.parse_module(code)
  call_node = module.body[0].body[0].value

  transformed = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "block_until_ready"
