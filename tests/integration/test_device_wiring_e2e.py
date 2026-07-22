"""Test suite for the Device Wiring E2E module."""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture(autouse=True)
def ensure_device_plugin():
  """Helper to ensure device plugin."""
  import ml_switcheroo.core.hooks as hooks
  import ml_switcheroo.plugins.device_allocator

  importlib.reload(ml_switcheroo.plugins.device_allocator)
  hooks._PLUGINS_LOADED = True


@pytest.fixture(scope="module")
def semantics():
  """Helper to semantics."""
  return SemanticsManager()


def test_device_cuda_to_jax(semantics):
  """Verifies the behavior of device cuda to JAX."""
  code = "d = torch.device('cuda')"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)
  assert result.success, f"Failed: {result.errors}"
  assert "jax.devices('gpu')[0]" in result.code


def test_device_cpu_to_jax(semantics):
  """Verifies the behavior of device cpu to JAX."""
  code = "d = torch.device('cpu')"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)
  assert result.success
  assert "jax.devices('cpu')[0]" in result.code
