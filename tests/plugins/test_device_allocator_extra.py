"""Test suite for device allocator extra coverage."""

import libcst as cst
from unittest.mock import patch, MagicMock

from ml_switcheroo.plugins.device_allocator import transform_device_allocator, _parse_device_args
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig


def test_device_allocator_no_adapter():
  """Test device allocator when no adapter is available."""
  node = cst.parse_expression("torch.device('cuda', 0)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  with patch("ml_switcheroo.plugins.device_allocator.get_adapter", return_value=None):
    res = transform_device_allocator(node, ctx)
    assert res is node


def test_parse_device_args_no_args():
  """Test parsing device arguments when no args are provided."""
  node = cst.parse_expression("torch.device()")
  t, idx = _parse_device_args(node)
  assert t is None
  assert idx is None


def test_device_allocator_adapter_exception():
  """Test device allocator adapter exception handling."""
  node = cst.parse_expression("torch.device('cuda')")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  mock_adapter = MagicMock()
  mock_adapter.get_device_syntax.side_effect = Exception("Adapter failure")
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter", return_value=mock_adapter):
    res = transform_device_allocator(node, ctx)
    assert res is node
