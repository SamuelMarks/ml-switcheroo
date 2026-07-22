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
