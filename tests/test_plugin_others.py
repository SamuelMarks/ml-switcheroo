"""Test suite for the Plugin Others module."""

from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.hooks import HookContext


def test_plugin_others() -> None:
  """Verifies the behavior of plugin others."""
  from ml_switcheroo.plugins.device_allocator import _parse_device_args, transform_device_allocator
  from ml_switcheroo.plugins.device_checks import transform_cuda_check
  from ml_switcheroo.plugins.shape_packing import transform_shape_packing

  node: cst.Call = cst.Call(
    func=cst.Name("dummy"),
    args=[
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("device")),
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("dtype")),
      cst.Arg(value=cst.Name("a"), keyword=cst.Name("other")),
    ],
  )
  dev_type, dev_idx = _parse_device_args(node)
  assert isinstance(dev_type, cst.Name) and dev_type.value == "a"
  assert isinstance(dev_idx, cst.Name) and dev_idx.value == "a"

  ctx = get_dummy_ctx(target_framework="torch")
  res_dev = transform_device_allocator(node, ctx)
  assert cst.Module(body=[]).code_for_node(res_dev) == "torch.device(a, a)"

  with patch("ml_switcheroo.plugins.device_checks.get_adapter") as mock_get:
    mock_adapter: MagicMock = MagicMock()
    mock_adapter.get_device_check_syntax.side_effect = NotImplementedError()
    mock_get.return_value = mock_adapter
    assert transform_cuda_check(node, get_dummy_ctx(target_framework="jax")) is node
    mock_adapter.get_device_check_syntax.side_effect = Exception()
    assert transform_cuda_check(node, get_dummy_ctx(target_framework="jax")) is node
    mock_adapter.get_device_check_syntax.side_effect = None
    mock_adapter.get_device_check_syntax.return_value = ""
    assert transform_cuda_check(node, get_dummy_ctx(target_framework="jax")) is node
    mock_adapter.get_device_check_syntax.return_value = "invalid !@#$ syntax"
    assert transform_cuda_check(node, get_dummy_ctx(target_framework="jax")) is node

  node2: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("a"), attr=cst.Name("b")), args=[])
  ctx_unknown = get_dummy_ctx(target_framework="unknown")
  ctx_unknown.lookup_api.return_value = None
  _res = transform_shape_packing(node2, ctx_unknown)
  assert _res == node2


def get_dummy_ctx(
  target_framework: str = "torch", current_op_id: str = "dummy", sharding_supported: bool = False
) -> MagicMock:
  """Gets dummy ctx.

  Args:
      target_framework: Target framework identifier string.
      current_op_id: The current op id (str) for get dummy ctx.
      sharding_supported: The sharding supported (bool) for get dummy ctx.

  Returns:
      Resulting `MagicMock` from get dummy ctx.
  """
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_framework = target_framework
  ctx.current_op_id = current_op_id
  ctx.semantics = MagicMock()
  ctx.lookup_api.return_value = None
  return ctx
