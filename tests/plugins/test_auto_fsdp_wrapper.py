"""Test suite for the Auto Fsdp Wrapper module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.dsl import OperationDef, OpType
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding


def test_auto_fsdp_wrapper_pytorch() -> None:
  """Verify `Call` handles auto FSDP wrapper pytorch as expected."""
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
  result = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "FSDP"
  assert isinstance(result.args[0].value, cst.Call)
  assert cst.ensure_type(result.args[0].value.func, cst.Name).value == "Linear"


def test_auto_fsdp_wrapper_jax() -> None:
  """Verify `Call` handles auto FSDP wrapper JAX as expected."""
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
  result = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "pjit"
  assert cst.ensure_type(cst.ensure_type(result.args[0].value, cst.Call).func, cst.Name).value == "Dense"


def test_auto_fsdp_wrapper_not_supported() -> None:
  """Verify `Call` handles auto FSDP wrapper not supported as expected."""
  node = cst.Call(func=cst.Name("Activation"), args=[])
  op_def = OperationDef(
    operation="Activation", description="Activation Layer", op_type=OpType.CLASS, sharding_supported=False, variants={}
  )
  mock_semantics = MagicMock()
  mock_semantics.get_operation.return_value = op_def
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")
  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Activation"
  result = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert cst.ensure_type(result.func, cst.Name).value == "Activation"


def test_auto_fsdp_wrapper_unknown_api() -> None:
  """Verify `Call` handles auto FSDP wrapper unknown API as expected."""
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
  result = wrap_with_sharding(node, ctx)
  assert result is node


def test_auto_fsdp_wrapper_no_op_id() -> None:
  """Verify `Call` handles auto FSDP wrapper no operation id as expected."""
  node = cst.Call(func=cst.Name("Unknown"), args=[])
  mock_semantics = MagicMock()
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")
  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = None
  result = wrap_with_sharding(node, ctx)
  assert isinstance(result, cst.Call)
  assert cst.ensure_type(result.func, cst.Name).value == "Unknown"


def test_auto_fsdp_wrapper_no_api(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `Call` handles auto FSDP wrapper no API as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  node = cst.Call(func=cst.Name("Linear"))
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="torch"))
  ctx.current_op_id = "Conv2d"
  monkeypatch.setattr(
    semantics.get_operation,
    "return_value",
    MagicMock(sharding_supported=True),
    raising=False,
  )
  monkeypatch.setattr(
    semantics.get_framework_config,
    "return_value",
    {"plugin_traits": {"sharding_wrapper_api": None}},
    raising=False,
  )
  result = wrap_with_sharding(node, ctx)
  assert result is node
