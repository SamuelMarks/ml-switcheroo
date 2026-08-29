"""Test suite for the Auto Fsdp Wrapper module."""

import typing
from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OperationDef, OpType
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding


def test_auto_fsdp_wrapper_pytorch() -> None:
  """Verifies the behavior of auto FSDP wrapper pytorch."""
  node = cst.Call(func=cst.Name("Linear"), args=[])
  op_def = OperationDef(
    operation="Linear", description="Linear Layer", op_type=OpType.CLASS, sharding_supported=True, variants={}
  )
  mock_semantics = MagicMock()
  mock_semantics.get_operation.return_value = op_def
  mock_semantics.get_framework_config.return_value = {
    "plugin_traits": {"sharding_wrapper_api": "torch.distributed.fsdp.FSDP"}
  }
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")
  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Linear"
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "FSDP"
  assert isinstance(result.args[0].value, cst.Call)
  assert typing.cast(cst.Name, result.args[0].value.func).value == "Linear"


def test_auto_fsdp_wrapper_jax() -> None:
  """Verifies the behavior of auto FSDP wrapper JAX."""
  node = cst.Call(func=cst.Name("Dense"), args=[])
  op_def = OperationDef(
    operation="Dense", description="Dense Layer", op_type=OpType.CLASS, sharding_supported=True, variants={}
  )
  mock_semantics = MagicMock()
  mock_semantics.get_operation.return_value = op_def
  mock_semantics.get_framework_config.return_value = {
    "plugin_traits": {"sharding_wrapper_api": "jax.experimental.pjit.pjit"}
  }
  mock_config = RuntimeConfig(target_framework="jax", source_framework="torch")
  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Dense"
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "pjit"
  assert typing.cast(cst.Name, typing.cast(cst.Call, result.args[0].value).func).value == "Dense"


def test_auto_fsdp_wrapper_not_supported() -> None:
  """Verifies the behavior of auto FSDP wrapper not supported."""
  node = cst.Call(func=cst.Name("Activation"), args=[])
  op_def = OperationDef(
    operation="Activation", description="Activation Layer", op_type=OpType.CLASS, sharding_supported=False, variants={}
  )
  mock_semantics = MagicMock()
  mock_semantics.get_operation.return_value = op_def
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")
  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Activation"
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert typing.cast(cst.Name, result.func).value == "Activation"


def test_auto_fsdp_wrapper_unknown_api() -> None:
  """Verifies the behavior of auto FSDP wrapper with unknown API."""
  node = cst.Call(func=cst.Name("Linear"), args=[])
  op_def = OperationDef(
    operation="Linear", description="Linear Layer", op_type=OpType.CLASS, sharding_supported=True, variants={}
  )
  mock_semantics = MagicMock()
  mock_semantics.get_operation.return_value = op_def
  mock_semantics.get_framework_config.return_value = {"plugin_traits": {"sharding_wrapper_api": "unknown.api"}}
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")
  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Linear"
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert result is node


def test_auto_fsdp_wrapper_no_op_id() -> None:
  """Verifies the behavior of auto FSDP wrapper no op id."""
  node = cst.Call(func=cst.Name("Unknown"), args=[])
  mock_semantics = MagicMock()
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")
  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = None
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert typing.cast(cst.Name, result.func).value == "Unknown"


# --- Merged from test_auto_fsdp_wrapper_missing_api.py ---


def test_auto_fsdp_wrapper_no_api() -> None:
  """Verifies the behavior of auto FSDP wrapper no API."""
  node = cst.Call(func=cst.Name("Linear"))
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(effective_target="torch"))
  ctx.current_op_id = "Conv2d"
  ctx.semantics.get_operation.return_value = MagicMock(sharding_supported=True)
  ctx.semantics.get_framework_config.return_value = {"plugin_traits": {"sharding_wrapper_api": None}}
  result: typing.Any = wrap_with_sharding(node, ctx)
  assert result is node
