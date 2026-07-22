"""Test suite for the Casting Extra module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.casting import transform_casting, _supports_numpy_casting
from ml_switcheroo.core.hooks import HookContext


def test_casting_missing_traits_in_conf():
  """Verifies the behavior of casting missing traits in conf."""
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {"plugin_traits": None}
  assert _supports_numpy_casting(ctx) is False


def test_casting_op_id_not_cast():
  """Verifies the behavior of casting op id not cast."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("float")))
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  ctx.current_op_id = "SomethingElse"
  semantics.get_definition_by_id.return_value = {"metadata": {}}
  result = transform_casting(node, ctx)
  assert result is node
