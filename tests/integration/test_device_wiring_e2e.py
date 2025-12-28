"""
Integration Test for Device Allocator Wiring.

Verifies that the "Device" abstract op correctly triggers the
device_allocator plugin and maps platform specific strings.
"""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture(autouse=True)
def ensure_device_plugin():
  """Robustness Fix to ensure plugin is loaded."""
  import ml_switcheroo.core.hooks as hooks
  import ml_switcheroo.plugins.device_allocator

  importlib.reload(ml_switcheroo.plugins.device_allocator)
  hooks._PLUGINS_LOADED = True


@pytest.fixture(scope="module")
def semantics():
  return SemanticsManager()


def test_device_cuda_to_jax(semantics):
  code = "d = torch.device('cuda')"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)

  assert result.success, f"Failed: {result.errors}"
  # Should use JAX specific syntax
  assert "jax.devices('gpu')[0]" in result.code


def test_device_cpu_to_jax(semantics):
  code = "d = torch.device('cpu')"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)

  assert result.success
  # Should use JAX CPU syntax
  assert "jax.devices('cpu')[0]" in result.code
