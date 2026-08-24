"""Test suite for the Device Checks module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_checks import transform_cuda_check


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["cuda_is_available"] = transform_cuda_check
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  op_def = {"variants": {"jax": {"api": "jax.devices", "requires_plugin": "cuda_is_available"}}}
  mgr.get_definition.return_value = ("cuda_is", op_def)
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_is_available_transform(rewriter):
  """Checks if is available transform."""
  code = "if torch.cuda.is_available(): pass"
  res = rewrite_code(rewriter, code)
  assert "len(jax.devices('gpu')) > 0" in res


def test_assignment_transform(rewriter):
  """Verifies the behavior of assignment transform."""
  code = "x = torch.cuda.is_available()"
  res = rewrite_code(rewriter, code)
  assert "len(jax.devices('gpu')) > 0" in res


def test_ignore_wrong_fw(rewriter):
  """Verifies the behavior of ignore wrong framework."""
  rewriter.context.config.target_framework = "numpy"
  rewriter.context.hook_context.target_fw = "numpy"
  rewriter.semantics.resolve_variant.side_effect = lambda a, f: None if f == "numpy" else {}
  code = "x = torch.cuda.is_available()"
  assert "torch.cuda" in rewrite_code(rewriter, code)


def test_device_checks_adapter_error(rewriter):
  """Verifies adapter errors are caught."""
  import ml_switcheroo.plugins.device_checks as dc

  orig = dc.get_adapter

  def raiser(fw):
    """Docstring."""
    raise Exception("Adapter crashed")

  dc.get_adapter = raiser
  try:
    code = "torch.cuda.is_available()"
    res = rewrite_code(rewriter, code)
    assert "torch.cuda.is_available()" in res
  finally:
    dc.get_adapter = orig


def test_device_checks_syntax_error(rewriter):
  """Verifies syntax errors are caught."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.return_value = "invalid syntax {{{"
  import ml_switcheroo.plugins.device_checks as dc

  orig = dc.get_adapter
  dc.get_adapter = lambda fw: mock_adapter
  try:
    code = "torch.cuda.is_available()"
    res = rewrite_code(rewriter, code)
    assert "torch.cuda.is_available()" in res
  finally:
    dc.get_adapter = orig


def test_device_checks_not_implemented(rewriter):
  """Verifies NotImplementedError is caught."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.side_effect = NotImplementedError()
  import ml_switcheroo.plugins.device_checks as dc

  orig = dc.get_adapter
  dc.get_adapter = lambda fw: mock_adapter
  try:
    code = "torch.cuda.is_available()"
    res = rewrite_code(rewriter, code)
    assert "torch.cuda.is_available()" in res
  finally:
    dc.get_adapter = orig


def test_device_checks_get_adapter_none(rewriter):
  """Verifies None adapter returns original node."""
  import ml_switcheroo.plugins.device_checks as dc

  orig = dc.get_adapter
  dc.get_adapter = lambda fw: None
  try:
    code = "torch.cuda.is_available()"
    res = rewrite_code(rewriter, code)
    assert "torch.cuda.is_available()" in res
  finally:
    dc.get_adapter = orig


def test_device_checks_empty_code(rewriter):
  """Verifies empty syntax returns original node."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.return_value = ""
  import ml_switcheroo.plugins.device_checks as dc

  orig = dc.get_adapter
  dc.get_adapter = lambda fw: mock_adapter
  try:
    code = "torch.cuda.is_available()"
    res = rewrite_code(rewriter, code)
    assert "torch.cuda.is_available()" in res
  finally:
    dc.get_adapter = orig


def test_device_checks_adapter_exception(rewriter):
  """Verifies Exception is caught."""
  mock_adapter = MagicMock()
  mock_adapter.get_device_check_syntax.side_effect = Exception()
  import ml_switcheroo.plugins.device_checks as dc

  orig = dc.get_adapter
  dc.get_adapter = lambda fw: mock_adapter
  try:
    code = "torch.cuda.is_available()"
    res = rewrite_code(rewriter, code)
    assert "torch.cuda.is_available()" in res
  finally:
    dc.get_adapter = orig
