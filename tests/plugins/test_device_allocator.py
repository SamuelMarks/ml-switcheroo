"""
Tests for Device Allocator Plugin.

Verifies that:
1. `torch.device('cuda')` -> `jax.devices('gpu')[0]`.
2. `torch.device('cuda:1')` -> `jax.devices('gpu')[1]`.
3. `torch.device('cpu')` -> `jax.devices('cpu')[0]`.
4. Arguments passed as variables are preserved.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_allocator import transform_device_allocator


# Helper to avoid import errors
def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Register Hook & Prevent automatic loading from disk
  hooks._HOOKS["device_allocator"] = transform_device_allocator
  hooks._PLUGINS_LOADED = True

  # 2. Mock Semantics
  mgr = MagicMock()

  device_def = {
    "requires_plugin": "device_allocator",
    "std_args": ["type"],
    "variants": {
      "torch": {"api": "torch.device"},
      "jax": {"api": "jax.devices", "requires_plugin": "device_allocator"},
    },
  }

  def get_def_side_effect(name):
    if name == "torch.device":
      return "device", device_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = {"device": device_def}
  mgr.is_verified.return_value = True

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_cuda_mapping_default_index(rewriter):
  """
  Scenario: `torch.device('cuda')`.
  Expect: `jax.devices('gpu')[0]`.
  """
  code = "d = torch.device('cuda')"
  result = rewrite_code(rewriter, code)

  assert "jax.devices('gpu')[0]" in result


def test_cuda_mapping_explicit_colon_index(rewriter):
  """
  Scenario: `torch.device('cuda:1')`.
  Expect: `jax.devices('gpu')[1]`.
  """
  code = "d = torch.device('cuda:1')"
  result = rewrite_code(rewriter, code)

  assert "jax.devices('gpu')[1]" in result


def test_cpu_mapping(rewriter):
  """
  Scenario: `torch.device('cpu')`.
  Expect: `jax.devices('cpu')[0]`.
  """
  code = "d = torch.device('cpu')"
  result = rewrite_code(rewriter, code)

  assert "jax.devices('cpu')[0]" in result


def test_variable_passthrough(rewriter):
  """
  Scenario: `torch.device(my_backend)`.
  Expect: `jax.devices(my_backend)[0]`.
  """
  code = "d = torch.device(my_backend)"
  result = rewrite_code(rewriter, code)

  assert "jax.devices(my_backend)[0]" in result


def test_second_arg_index(rewriter):
  """
  Scenario: `torch.device('cuda', 2)`.
  Expect: `jax.devices('gpu')[2]`.
  """
  code = "d = torch.device('cuda', 2)"
  result = rewrite_code(rewriter, code)

  assert "jax.devices('gpu')[2]" in result


def test_mps_mapping(rewriter):
  """
  Scenario: `torch.device('mps')`.
  Expect: `jax.devices('gpu')[0]` (approximate mapping).
  """
  code = "d = torch.device('mps')"
  result = rewrite_code(rewriter, code)

  assert "jax.devices('gpu')[0]" in result


def test_ignore_wrong_fw(rewriter):
  """
  Scenario: Target is 'numpy' (or anything other than jax).
  Expect: Passthrough (no change).
  """
  # Reconfigure context to non-jax
  rewriter.ctx._runtime_config.target_framework = "numpy"
  rewriter.ctx.target_fw = "numpy"

  code = "d = torch.device('cuda')"
  result = rewrite_code(rewriter, code)

  assert "torch.device('cuda')" in result
