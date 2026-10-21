"""Test suite for plugin casting."""

from unittest.mock import MagicMock

import libcst as _cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.casting import _create_dotted_name, _supports_numpy_casting, transform_casting

cst = _cst


def test_create_dotted_name() -> None:
  """Verify create dotted name behavior."""
  node = _create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_supports_numpy_casting() -> None:
  """Verify supports numpy casting behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "jax"

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
    """Test helper class Traits."""

    has_numpy_compatible_arrays = True

  ctx.semantics.get_framework_config.return_value = {"plugin_traits": Traits()}
  assert _supports_numpy_casting(ctx)

  class TraitsFalse:
    """Test helper class TraitsFalse."""

    has_numpy_compatible_arrays = False

  ctx.semantics.get_framework_config.return_value = {"plugin_traits": TraitsFalse()}
  assert not _supports_numpy_casting(ctx)

  class TraitsMissing:
    """Test helper class TraitsMissing."""

    pass

  ctx.semantics.get_framework_config.return_value = {"plugin_traits": TraitsMissing()}
  assert not _supports_numpy_casting(ctx)


def test_transform_casting() -> None:
  """Verify transform casting behavior."""
  ctx = MagicMock(spec=HookContext)
  ctx.target_framework = "jax"
  ctx.semantics = MagicMock()
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}

  # not attribute
  node = cst.Call(func=cst.Name("float"), args=[])
  assert transform_casting(node, ctx) == node

  # attribute but no current_op_id
  node_attr = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("float")), args=[])
  ctx.current_op_id = None
  assert transform_casting(node_attr, ctx) == node_attr

  # current_op_id with semantics=None
  ctx.current_op_id = "CastFloat"
  orig_sem = ctx.semantics
  ctx.semantics = None
  assert transform_casting(node_attr, ctx) == node_attr
  ctx.semantics = orig_sem

  # current_op_id but no defn
  ctx.current_op_id = "CastFloat"
  ctx.semantics.get_definition_by_id.return_value = None
  assert transform_casting(node_attr, ctx) == node_attr

  # no metadata target_type, inferred
  ctx.semantics.get_definition_by_id.return_value = {"id": "CastFloat"}
  ctx.lookup_api.return_value = "jax.numpy.float32"
  result = transform_casting(node_attr, ctx)
  assert isinstance(result, cst.Call)
  assert getattr(result.func, "attr", None) is not None
  assert getattr(
    {
      "v": getattr(
        getattr(
          {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
        )("v"),
        getattr({"v": "value"}, "__getitem__")("v"),
        getattr({"v": None}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr({"v": "astype"}, "__getitem__")("v")
  assert isinstance(result.args[0].value, cst.Attribute)
  assert result.args[0].value.attr.value == "float32"

  # inference map
  ctx.current_op_id = "CastLong"
  ctx.lookup_api.return_value = "jax.numpy.int64"
  result_long = transform_casting(node_attr, ctx)
  assert isinstance(result_long, cst.Call)
  assert isinstance(result_long.args[0].value, cst.Attribute)
  assert result_long.args[0].value.attr.value == "int64"
  ctx.lookup_api.assert_called_with("Int64")

  # inference not found
  ctx.current_op_id = "CastUnknown"
  ctx.lookup_api.return_value = None
  result_unknown = transform_casting(node_attr, ctx)
  assert result_unknown == node_attr

  # inference not Cast
  ctx.current_op_id = "OtherOp"
  result_other = transform_casting(node_attr, ctx)
  assert result_other == node_attr

  # with metadata target_type
  ctx.current_op_id = "CastCustom"
  ctx.semantics.get_definition_by_id.return_value = {"metadata": {"target_type": "Float16"}}
  ctx.lookup_api.return_value = "jax.numpy.float16"
  result_custom = transform_casting(node_attr, ctx)
  assert isinstance(result_custom, cst.Call)
  assert isinstance(result_custom.args[0].value, cst.Attribute)
  assert result_custom.args[0].value.attr.value == "float16"

  # no target api
  ctx.lookup_api.return_value = None
  result_no_api = transform_casting(node_attr, ctx)
  assert result_no_api == node_attr

  # support false
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": False}}
  result_no_support = transform_casting(node_attr, ctx)
  assert result_no_support == node_attr
