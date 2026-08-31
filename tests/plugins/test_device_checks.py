"""Test suite for the Device Checks module."""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.plugins.device_checks import transform_cuda_check
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Docstring."""
  hooks._HOOKS["cuda_is_available"] = transform_cuda_check
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  op_def: dict[str, typing.Any] = {"variants": {"jax": {"api": "jax.devices", "requires_plugin": "cuda_is_available"}}}
  mgr.get_definition.return_value = ("cuda_is", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: typing.cast(
    typing.Optional[dict[str, typing.Any]], op_def["variants"].get(fw)
  )
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_is_available_transform(rewriter: PivotRewriter) -> None:
  """Checks if is available transform."""
  code: str = "if torch.cuda.is_available(): pass"
  res: str = rewrite_code(rewriter, code)
  assert "len(jax.devices('gpu')) > 0" in res


def test_assignment_transform(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of assignment transform."""
  code: str = "x = torch.cuda.is_available()"
  res: str = rewrite_code(rewriter, code)
  assert "len(jax.devices('gpu')) > 0" in res


def test_ignore_wrong_fw(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore wrong framework."""
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  rewriter.semantics.resolve_variant.side_effect = lambda a, f: None if f == "numpy" else {}
  code: str = "x = torch.cuda.is_available()"
  assert "torch.cuda" in rewrite_code(rewriter, code)


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_device_checks_adapter_error(mock_get_adapter: MagicMock, rewriter: PivotRewriter) -> None:
  """Verifies adapter errors are caught."""

  def raiser(fw: str) -> typing.Any:
    """Docstring."""
    raise Exception("Adapter crashed")

  mock_get_adapter.side_effect = raiser
  code: str = "torch.cuda.is_available()"
  res: str = rewrite_code(rewriter, code)
  assert "torch.cuda.is_available()" in res


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_device_checks_syntax_error(mock_get_adapter: MagicMock, rewriter: PivotRewriter) -> None:
  """Verifies syntax errors are caught."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.return_value = "invalid syntax {{{"
  mock_get_adapter.return_value = mock_adapter
  code: str = "torch.cuda.is_available()"
  res: str = rewrite_code(rewriter, code)
  assert "torch.cuda.is_available()" in res


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_device_checks_not_implemented(mock_get_adapter: MagicMock, rewriter: PivotRewriter) -> None:
  """Verifies NotImplementedError is caught."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.side_effect = NotImplementedError()
  mock_get_adapter.return_value = mock_adapter
  code: str = "torch.cuda.is_available()"
  res: str = rewrite_code(rewriter, code)
  assert "torch.cuda.is_available()" in res


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_device_checks_get_adapter_none(mock_get_adapter: MagicMock, rewriter: PivotRewriter) -> None:
  """Verifies None adapter returns original node."""
  mock_get_adapter.return_value = None
  code: str = "torch.cuda.is_available()"
  res: str = rewrite_code(rewriter, code)
  assert "torch.cuda.is_available()" in res


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_device_checks_empty_code(mock_get_adapter: MagicMock, rewriter: PivotRewriter) -> None:
  """Verifies empty syntax returns original node."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.return_value = ""
  mock_get_adapter.return_value = mock_adapter
  code: str = "torch.cuda.is_available()"
  res: str = rewrite_code(rewriter, code)
  assert "torch.cuda.is_available()" in res


@patch("ml_switcheroo.plugins.device_checks.get_adapter")
def test_device_checks_adapter_exception(mock_get_adapter: MagicMock, rewriter: PivotRewriter) -> None:
  """Verifies Exception is caught."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.side_effect = Exception()
  mock_get_adapter.return_value = mock_adapter
  code: str = "torch.cuda.is_available()"
  res: str = rewrite_code(rewriter, code)
  assert "torch.cuda.is_available()" in res
