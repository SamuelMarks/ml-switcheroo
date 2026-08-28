"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.plugins.device_checks import transform_cuda_check
from ml_switcheroo.core.hooks import HookContext


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_transform_cuda_check(mock_get_adapter: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("is_available"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"

  adapter: MagicMock = MagicMock()
  adapter.get_device_check_syntax.return_value = "len(jax.devices('gpu')) > 0"
  mock_get_adapter.return_value = adapter

  result: cst.CSTNode = transform_cuda_check(node, ctx)
  assert isinstance(result, cst.Comparison)


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_transform_cuda_check_no_adapter(mock_get_adapter: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("is_available"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "unknown"
  mock_get_adapter.return_value = None

  result: cst.CSTNode = transform_cuda_check(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_transform_cuda_check_not_implemented(mock_get_adapter: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("is_available"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  adapter: MagicMock = MagicMock()
  adapter.get_device_check_syntax.side_effect = NotImplementedError
  mock_get_adapter.return_value = adapter

  result: cst.CSTNode = transform_cuda_check(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_transform_cuda_check_exception(mock_get_adapter: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("is_available"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  adapter: MagicMock = MagicMock()
  adapter.get_device_check_syntax.side_effect = Exception("Test")
  mock_get_adapter.return_value = adapter

  result: cst.CSTNode = transform_cuda_check(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_transform_cuda_check_no_syntax(mock_get_adapter: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("is_available"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  adapter: MagicMock = MagicMock()
  adapter.get_device_check_syntax.return_value = ""
  mock_get_adapter.return_value = adapter

  result: cst.CSTNode = transform_cuda_check(node, ctx)
  assert result is node


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_transform_cuda_check_bad_syntax(mock_get_adapter: MagicMock) -> None:
  """Docstring."""
  node: cst.Call = cst.Call(func=cst.Name("is_available"))
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.target_fw = "jax"
  adapter: MagicMock = MagicMock()
  adapter.get_device_check_syntax.return_value = "invalid syntax("
  mock_get_adapter.return_value = adapter

  result: cst.CSTNode = transform_cuda_check(node, ctx)
  assert result is node
