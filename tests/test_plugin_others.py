import pytest
import libcst as cst
from unittest.mock import MagicMock


def test_plugin_others():
  from ml_switcheroo.plugins.device_allocator import _parse_device_args

  # 56-58, 63-64: kwargs
  node = cst.Call(
    func=cst.Name("dummy"),
    args=[
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("device")),
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("dtype")),
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("other")),
    ],
  )
  _parse_device_args(node)

  # 102-103: return original node
  from ml_switcheroo.plugins.device_allocator import transform_device_allocator

  ctx = MagicMock()
  ctx.semantics.get_operation.return_value = MagicMock(device_allocation_supported=False)
  transform_device_allocator(node, ctx)
  ctx.semantics.get_operation.return_value = MagicMock(device_allocation_supported=True)
  transform_device_allocator(node, ctx)  # target_fw != "jax" etc

  from ml_switcheroo.plugins.device_checks import transform_cuda_check
  from unittest.mock import patch

  # 51-55: Exception in get_device_check_syntax
  with patch("ml_switcheroo.plugins.device_checks.get_adapter") as mock_get:
    mock_adapter = MagicMock()
    mock_adapter.get_device_check_syntax.side_effect = NotImplementedError()
    mock_get.return_value = mock_adapter
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))

    mock_adapter.get_device_check_syntax.side_effect = Exception()
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))

    # 58: empty check_code
    mock_adapter.get_device_check_syntax.side_effect = None
    mock_adapter.get_device_check_syntax.return_value = ""
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))

    # 64-65: invalid CST
    mock_adapter.get_device_check_syntax.return_value = "invalid !@#$ syntax"
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))

  from ml_switcheroo.plugins.shape_packing import transform_shape_packing

  # 68: non-name func
  node = cst.Call(func=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")), args=[])
  ctx.target_fw = "unknown"
  ctx.semantics.get_operation.return_value = MagicMock(requires_shape_packing=True)
  transform_shape_packing(node, ctx)


def get_dummy_ctx(target_fw="torch", current_op_id="dummy", sharding_supported=False):
  ctx = MagicMock()
  ctx.target_fw = target_fw
  ctx.current_op_id = current_op_id
  op_def = MagicMock()
  op_def.sharding_supported = sharding_supported
  ctx.semantics.get_operation.return_value = op_def
  return ctx
