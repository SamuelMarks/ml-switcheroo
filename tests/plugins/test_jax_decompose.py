import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax


def test_jax_decompose_plugin():
  node = cst.Call(func=cst.Name("UnsupportedOp"), args=[])

  mock_config = RuntimeConfig(target_framework="keras", source_framework="torch")
  mock_semantics = MagicMock()

  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Hardswish"

  result = decompose_via_jax(node, ctx)

  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "hardswish"
  assert isinstance(result.func.value, cst.Attribute)
  assert result.func.value.attr.value == "numpy"


def test_jax_decompose_plugin_no_op_id():
  node = cst.Call(func=cst.Name("UnsupportedOp"), args=[])

  mock_config = RuntimeConfig(target_framework="keras", source_framework="torch")
  mock_semantics = MagicMock()

  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = None

  result = decompose_via_jax(node, ctx)

  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "unknownop"
