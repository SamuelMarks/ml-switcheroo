"""
Tests for Device Allocator Plugin using Real Adapters logic.

Verifies that:
1. `torch.device('cuda')` delegates to JAX Adapter -> `jax.devices('gpu')[0]`.
2. `torch.device('cuda:1')` splits string and delegates.
3. `torch.device('cpu')` works.
4. Variable passing relies on adapter robust string handling.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.device_allocator import transform_device_allocator
from ml_switcheroo.frameworks.jax import JaxAdapter
from ml_switcheroo.frameworks.numpy import NumpyAdapter


# Helper to avoid import errors
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
      "numpy": {"api": "cpu", "requires_plugin": "device_allocator"},
    },
  }

  # Lookup Logic
  mgr.get_definition.side_effect = lambda name: ("device", device_def) if name == "torch.device" else None
  mgr.get_known_apis.return_value = {"device": device_def}
  mgr.is_verified.return_value = True

  # Resolution Logic
  def resolve_variant(aid, fw):
    # Return valid dict if variant exists
    if aid == "device" and fw in device_def["variants"]:
      return device_def["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve_variant

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")

  # 4. Patch get_adapter to return REAL JAX/Numpy adapter logic
  with patch("ml_switcheroo.plugins.device_allocator.get_adapter") as mock_get_adapter:

    def adapter_side_effect(name):
      if name == "jax":
        return JaxAdapter()
      if name == "numpy":
        return NumpyAdapter()
      return None

    mock_get_adapter.side_effect = adapter_side_effect

    yield PivotRewriter(mgr, cfg)


def test_cuda_mapping_default_index(rewriter):
  code = "d = torch.device('cuda')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[0]" in result


def test_cuda_mapping_explicit_colon_index(rewriter):
  code = "d = torch.device('cuda:1')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[1]" in result


def test_cpu_mapping(rewriter):
  code = "d = torch.device('cpu')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('cpu')[0]" in result


def test_variable_passthrough(rewriter):
  code = "d = torch.device(my_backend)"
  result = rewrite_code(rewriter, code)
  assert "jax.devices(my_backend)[0]" in result


def test_second_arg_index(rewriter):
  code = "d = torch.device('cuda', 2)"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[2]" in result


def test_mps_mapping(rewriter):
  code = "d = torch.device('mps')"
  result = rewrite_code(rewriter, code)
  assert "jax.devices('gpu')[0]" in result


def test_ignore_wrong_fw(rewriter):
  # Reconfigure context to generic numpy
  # Note: Must use config setter to update PivotRewriter's property source
  rewriter.ctx._runtime_config.target_framework = "numpy"
  rewriter.ctx.target_fw = "numpy"

  code = "d = torch.device('cuda')"
  result = rewrite_code(rewriter, code)

  # Numpy adapter returns 'cpu' string code.
  assert "'cpu'" in result
