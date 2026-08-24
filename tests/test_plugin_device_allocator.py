"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.plugins.device_allocator import transform_device_allocator, _parse_device_args
from ml_switcheroo.core.hooks import HookContext


def test_parse_device_args():
  """Docstring."""
  # torch.device('cuda')
  node = cst.Call(func=cst.Name("device"), args=[cst.Arg(value=cst.SimpleString("'cuda'"))])
  t_node, i_node = _parse_device_args(node)
  assert t_node.value == "'cuda'"
  assert i_node is None

  # torch.device('cuda', 0)
  node2 = cst.Call(
    func=cst.Name("device"),
    args=[
      cst.Arg(value=cst.SimpleString("'cuda'")),
      cst.Arg(value=cst.Integer("0")),
    ],
  )
  t_node2, i_node2 = _parse_device_args(node2)
  assert t_node2.value == "'cuda'"
  assert i_node2.value == "0"

  # torch.device('cuda:0')
  node3 = cst.Call(func=cst.Name("device"), args=[cst.Arg(value=cst.SimpleString("'cuda:0'"))])
  t_node3, i_node3 = _parse_device_args(node3)
  assert t_node3.value == "'cuda'"
  assert i_node3.value == "0"

  # empty
  node4 = cst.Call(func=cst.Name("device"), args=[])
  t_node4, i_node4 = _parse_device_args(node4)
  assert t_node4 is None
  assert i_node4 is None


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_transform_device_allocator(mock_get_adapter):
  """Docstring."""
  node = cst.Call(func=cst.Name("device"), args=[cst.Arg(value=cst.SimpleString("'cuda:0'"))])
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  adapter = MagicMock()
  adapter.get_device_syntax.return_value = "jax.devices('gpu')[0]"
  mock_get_adapter.return_value = adapter

  result = transform_device_allocator(node, ctx)
  assert isinstance(result, cst.Subscript)


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_transform_device_allocator_no_adapter(mock_get_adapter):
  """Docstring."""
  node = cst.Call(func=cst.Name("device"), args=[cst.Arg(value=cst.SimpleString("'cuda:0'"))])
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "unknown"
  mock_get_adapter.return_value = None

  result = transform_device_allocator(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_transform_device_allocator_exception(mock_get_adapter):
  """Docstring."""
  node = cst.Call(func=cst.Name("device"), args=[cst.Arg(value=cst.SimpleString("'cuda:0'"))])
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  adapter = MagicMock()
  adapter.get_device_syntax.side_effect = Exception("Test")
  mock_get_adapter.return_value = adapter

  result = transform_device_allocator(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.device_allocator.get_adapter")
def test_transform_device_allocator_parse_error(mock_get_adapter):
  """Docstring."""
  node = cst.Call(func=cst.Name("device"), args=[cst.Arg(value=cst.SimpleString("'cuda:0'"))])
  ctx = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  adapter = MagicMock()
  adapter.get_device_syntax.return_value = "invalid syntax("
  mock_get_adapter.return_value = adapter

  result = transform_device_allocator(node, ctx)
  assert result is node
