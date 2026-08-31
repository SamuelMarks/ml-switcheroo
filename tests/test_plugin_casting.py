"""Docstring."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.casting import _create_dotted_name, _supports_numpy_casting, transform_casting


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_supports_numpy_casting() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"

  # no semantics
  ctx.semantics = None
  assert not _supports_numpy_casting(ctx)

  # no config
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = None
  assert not _supports_numpy_casting(ctx)

  # config dict without plugin_traits
  ctx.semantics.get_framework_config.return_value = {}
  assert not _supports_numpy_casting(ctx)

  # plugin_traits dict
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  assert _supports_numpy_casting(ctx)

  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": False}}
  assert not _supports_numpy_casting(ctx)

  # plugin_traits object
  class Traits:
    """Docstring."""

    has_numpy_compatible_arrays: bool = True

  ctx.semantics.get_framework_config.return_value = {"plugin_traits": Traits()}
  assert _supports_numpy_casting(ctx)

  class TraitsFalse:
    """Docstring."""

    has_numpy_compatible_arrays: bool = False

  ctx.semantics.get_framework_config.return_value = {"plugin_traits": TraitsFalse()}
  assert not _supports_numpy_casting(ctx)

  class TraitsMissing:
    """Docstring."""

    pass

  ctx.semantics.get_framework_config.return_value = {"plugin_traits": TraitsMissing()}
  assert not _supports_numpy_casting(ctx)


def test_transform_casting() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}

  # not attribute
  node: cst.Call = cst.Call(func=cst.Name("float"), args=[])
  assert transform_casting(node, ctx) == node

  # attribute but no current_op_id
  node_attr: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("float")), args=[])
  ctx.current_op_id = None
  assert transform_casting(node_attr, ctx) == node_attr

  # current_op_id but no defn
  ctx.current_op_id = "CastFloat"
  ctx.semantics.get_definition_by_id.return_value = None
  assert transform_casting(node_attr, ctx) == node_attr

  # no metadata target_type, inferred
  ctx.semantics.get_definition_by_id.return_value = {"id": "CastFloat"}
  ctx.lookup_api.return_value = "jax.numpy.float32"
  result: cst.CSTNode = transform_casting(node_attr, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(result.func.attr, "value", None) == "astype"
  assert isinstance(result.args[0].value, cst.Attribute)
  assert result.args[0].value.attr.value == "float32"

  # inference map
  ctx.current_op_id = "CastLong"
  ctx.lookup_api.return_value = "jax.numpy.int64"
  result_long: cst.CSTNode = transform_casting(node_attr, ctx)
  assert isinstance(result_long, cst.Call)
  assert isinstance(result_long.args[0].value, cst.Attribute)
  assert result_long.args[0].value.attr.value == "int64"
  ctx.lookup_api.assert_called_with("Int64")

  # inference not found
  ctx.current_op_id = "CastUnknown"
  ctx.lookup_api.return_value = None
  result_unknown: cst.CSTNode = transform_casting(node_attr, ctx)
  assert result_unknown == node_attr

  # inference not Cast
  ctx.current_op_id = "OtherOp"
  result_other: cst.CSTNode = transform_casting(node_attr, ctx)
  assert result_other == node_attr

  # with metadata target_type
  ctx.current_op_id = "CastCustom"
  ctx.semantics.get_definition_by_id.return_value = {"metadata": {"target_type": "Float16"}}
  ctx.lookup_api.return_value = "jax.numpy.float16"
  result_custom: cst.CSTNode = transform_casting(node_attr, ctx)
  assert isinstance(result_custom, cst.Call)
  assert isinstance(result_custom.args[0].value, cst.Attribute)
  assert result_custom.args[0].value.attr.value == "float16"

  # no target api
  ctx.lookup_api.return_value = None
  result_no_api: cst.CSTNode = transform_casting(node_attr, ctx)
  assert result_no_api == node_attr

  # support false
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": False}}
  result_no_support: cst.CSTNode = transform_casting(node_attr, ctx)
  assert result_no_support == node_attr
