"""Test suite for the Device Allocator Extra module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.device_allocator import transform_device_allocator, _parse_device_args
from ml_switcheroo.core.hooks import HookContext


def test_device_allocator_no_adapter(monkeypatch):
  """Verifies the behavior of device allocator no adapter."""
  monkeypatch.setattr("ml_switcheroo.plugins.device_allocator.get_adapter", lambda fw: None)
  node = cst.Call(func=cst.Name("device"), args=[cst.Arg(value=cst.SimpleString("'cuda'"))])
  ctx = HookContext(semantics=MagicMock(), config=MagicMock(effective_target="unknown_fw"))
  result = transform_device_allocator(node, ctx)
  assert result is node


def test_parse_device_args_no_args():
  """Parses device arguments no arguments."""
  node = cst.Call(func=cst.Name("device"), args=[])
  (dt, di) = _parse_device_args(node)
  assert dt is None
  assert di is None
