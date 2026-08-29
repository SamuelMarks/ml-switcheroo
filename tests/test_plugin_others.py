"""Test suite for the Plugin Others module."""

from unittest.mock import MagicMock

import libcst as cst


def test_plugin_others() -> None:
  """Verifies the behavior of plugin others."""
  from ml_switcheroo.plugins.device_allocator import _parse_device_args

  node: cst.Call = cst.Call(
    func=cst.Name("dummy"),
    args=[
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("device")),
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("dtype")),
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("other")),
    ],
  )
  _parse_device_args(node)
  from ml_switcheroo.plugins.device_allocator import transform_device_allocator

  ctx: MagicMock = MagicMock()
  ctx.semantics.get_operation.return_value = MagicMock(device_allocation_supported=False)
  transform_device_allocator(node, ctx)
  ctx.semantics.get_operation.return_value = MagicMock(device_allocation_supported=True)
  transform_device_allocator(node, ctx)
  from unittest.mock import patch

  from ml_switcheroo.plugins.device_checks import transform_cuda_check

  with patch("ml_switcheroo.plugins.device_checks.get_adapter") as mock_get:
    mock_adapter: MagicMock = MagicMock()
    mock_adapter.get_device_check_syntax.side_effect = NotImplementedError()
    mock_get.return_value = mock_adapter
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))
    mock_adapter.get_device_check_syntax.side_effect = Exception()
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))
    mock_adapter.get_device_check_syntax.side_effect = None
    mock_adapter.get_device_check_syntax.return_value = ""
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))
    mock_adapter.get_device_check_syntax.return_value = "invalid !@#$ syntax"
    transform_cuda_check(node, get_dummy_ctx(target_fw="jax"))
  from ml_switcheroo.plugins.shape_packing import transform_shape_packing

  node2: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")), args=[])
  ctx.target_fw = "unknown"
  ctx.semantics.get_operation.return_value = MagicMock(requires_shape_packing=True)
  transform_shape_packing(node2, ctx)


def get_dummy_ctx(target_fw: str = "torch", current_op_id: str = "dummy", sharding_supported: bool = False) -> MagicMock:
  """Gets dummy ctx."""
  ctx: MagicMock = MagicMock()
  ctx.target_fw = target_fw
  ctx.current_op_id = current_op_id
  op_def: MagicMock = MagicMock()
  op_def.sharding_supported = sharding_supported
  ctx.semantics.get_operation.return_value = op_def
  return ctx
