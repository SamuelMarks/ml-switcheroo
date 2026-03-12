import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.dsl import OperationDef, OpType
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.auto_fsdp_wrapper import wrap_with_sharding


def test_auto_fsdp_wrapper_pytorch():
  node = cst.Call(func=cst.Name("Linear"), args=[])

  op_def = OperationDef(
    operation="Linear", description="Linear Layer", op_type=OpType.CLASS, sharding_supported=True, variants={}
  )

  mock_semantics = MagicMock()
  mock_semantics.get_operation.return_value = op_def
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")

  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Linear"

  result = wrap_with_sharding(node, ctx)

  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "FSDP"
  # Verify the first argument is our original node
  assert isinstance(result.args[0].value, cst.Call)
  assert result.args[0].value.func.value == "Linear"


def test_auto_fsdp_wrapper_jax():
  node = cst.Call(func=cst.Name("Dense"), args=[])

  op_def = OperationDef(
    operation="Dense", description="Dense Layer", op_type=OpType.CLASS, sharding_supported=True, variants={}
  )

  mock_semantics = MagicMock()
  mock_semantics.get_operation.return_value = op_def
  mock_config = RuntimeConfig(target_framework="jax", source_framework="torch")

  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Dense"

  result = wrap_with_sharding(node, ctx)

  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "pjit"
  assert result.args[0].value.func.value == "Dense"


def test_auto_fsdp_wrapper_not_supported():
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

  # Unmodified
  assert isinstance(result, cst.Call)
  assert result.func.value == "Activation"


def test_auto_fsdp_wrapper_no_op_id():
  node = cst.Call(func=cst.Name("Unknown"), args=[])

  mock_semantics = MagicMock()
  mock_config = RuntimeConfig(target_framework="torch", source_framework="jax")

  ctx = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = None

  result = wrap_with_sharding(node, ctx)

  # Unmodified
  assert isinstance(result, cst.Call)
  assert result.func.value == "Unknown"
