"""Test suite for the Clamp module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  clamp_def = {
    "std_args": ["input", "min", "max"],
    "variants": {
      "torch": {"api": "torch.clamp"},
      "jax": {"api": "jax.numpy.clip", "args": {"min": "a_min", "max": "a_max", "input": "a"}},
    },
  }
  mgr.get_definition.side_effect = lambda n: ("Clamp", clamp_def) if "clamp" in n or "clip" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: (
    clamp_def["variants"]["jax"] if aid == "Clamp" and fw == "jax" else None
  )
  mgr.is_verified.return_value = True
  mgr.get_known_apis.return_value = {"Clamp": clamp_def}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_clamp_keyword_rename(rewriter):
  """Verifies the behavior of clamp keyword rename."""
  code = "y = torch.clamp(x, min=0.0, max=1.0)"
  res = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res
  assert "a_min=0.0" in res
  assert "a_max=1.0" in res
  assert " min=" not in res


def test_clip_alias(rewriter):
  """Verifies the behavior of clip alias."""
  code = "y = torch.clip(x, 0, 1)"
  res = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res


def test_method_clamp(rewriter):
  """Verifies the behavior of method clamp."""
  code = "y = x.clamp(min=0)"
  res = rewrite_code(rewriter, code)
  assert "jax.numpy.clip" in res
  assert "a_min=0" in res
