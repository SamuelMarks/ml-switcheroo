"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.reshape import _create_dotted_name, transform_view_semantics


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("jax.numpy.reshape")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "reshape"


def test_transform_view_semantics_no_api() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = None

  code: str = "x.view(2, 2)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert transformed is call_node


def test_transform_view_semantics_method() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code: str = "x.view(2, 2)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert getattr(transformed.args[0].value, "value", None) == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_method_single_int() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code: str = "x.view(1)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert getattr(transformed.args[0].value, "value", None) == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_method_tuple_arg() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code: str = "x.view((2, 2))"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert len(transformed.args) == 2
  assert getattr(transformed.args[0].value, "value", None) == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_no_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code: str = "view()"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert transformed is call_node


def test_transform_view_semantics_func_multiple_args() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code: str = "view(x, 2, 2)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert getattr(transformed.args[0].value, "value", None) == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_func_tuple_arg() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=False)

  code: str = "view(x, (2, 2))"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert len(transformed.args) == 2
  assert getattr(transformed.args[0].value, "value", None) == "x"
  assert isinstance(transformed.args[1].value, cst.Tuple)


def test_transform_view_semantics_strict() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.lookup_api.return_value = "jax.numpy.reshape"
  ctx._runtime_config = MagicMock(strict_mode=True)
  ctx.plugin_traits.strict_materialization_method = "block_until_ready"

  code: str = "x.view(2, 2)"
  module: cst.Module = cst.parse_module(code)
  call_node: cst.BaseExpression = getattr(getattr(module.body[0], "body")[0], "value")

  transformed: cst.CSTNode = transform_view_semantics(call_node, ctx)
  assert isinstance(transformed, cst.Call)
  assert isinstance(transformed.func, cst.Attribute)
  assert transformed.func.attr.value == "block_until_ready"
