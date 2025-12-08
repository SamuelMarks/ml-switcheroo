"""
Tests for Device Check Shim Plugin.

Verifies that `torch.cuda.is_available()` transforms to `len(jax.devices('gpu')) > 0`.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_checks import transform_cuda_check


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


@pytest.fixture
def rewriter():
  # 1. Register Hook
  hooks._HOOKS["cuda_is_available"] = transform_cuda_check
  hooks._PLUGINS_LOADED = True

  # 2. Mock Semantics
  mgr = MagicMock()

  op_def = {
    "requires_plugin": "cuda_is_available",
    "std_args": [],
    "variants": {
      "torch": {"api": "torch.cuda.is_available"},
      "jax": {"api": "jax.devices", "requires_plugin": "cuda_is_available"},
    },
  }

  def get_def_side_effect(name):
    if name == "torch.cuda.is_available":
      return "cuda_is_available", op_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = {"cuda_is_available": op_def}
  mgr.is_verified.return_value = True

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_is_available_transform(rewriter):
  """
  Scenario: `if torch.cuda.is_available():`
  Expect: `if len(jax.devices('gpu')) > 0:`
  """
  code = "if torch.cuda.is_available():\n    pass"
  result = rewrite_code(rewriter, code)

  assert "len(jax.devices('gpu')) > 0" in result
  assert "torch.cuda.is_available" not in result


def test_assignment_transform(rewriter):
  """
  Scenario: `has_gpu = torch.cuda.is_available()`
  Expect: `has_gpu = len(jax.devices('gpu')) > 0`
  """
  code = "has_gpu = torch.cuda.is_available()"
  result = rewrite_code(rewriter, code)

  assert "len(jax.devices('gpu')) > 0" in result


def test_ignore_wrong_target(rewriter):
  """
  Scenario: Target is not JAX.
  Expect: No change (plugin returns original).
  """
  # Switch target
  rewriter.ctx._runtime_config.target_framework = "numpy"
  rewriter.ctx.target_fw = "numpy"

  code = "check = torch.cuda.is_available()"
  result = rewrite_code(rewriter, code)

  assert "torch.cuda.is_available()" in result.replace(" ", "")  # spaces differ in parser output sometimes
  assert "jax.devices" not in result
