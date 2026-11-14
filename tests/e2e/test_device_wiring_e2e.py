"""Test suite for the Device Wiring E2E module."""

import importlib

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture(autouse=True)
def ensure_device_plugin() -> None:
  """Create a configured `None` fixture for device wiring e2e tests."""
  import ml_switcheroo.core.hooks as hooks
  import ml_switcheroo.plugins.device_allocator

  importlib.reload(ml_switcheroo.plugins.device_allocator)
  hooks.set_plugins_loaded(True)


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Create a configured `SemanticsManager` fixture for device wiring e2e tests.

  Returns:
      Configured `semantics` fixture value (SemanticsManager).
  """
  return SemanticsManager()


@pytest.mark.skip(reason="Device definitions removed")
def test_device_cuda_to_jax(semantics: SemanticsManager) -> None:
  """Verify `RuntimeConfig` emits `jax.devices('gpu')[0]` for device cuda to JAX.

  Args:
      semantics: SemanticsManager instance providing operation and trait definitions.
  """
  code: str = "d = torch.device('cuda')"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(code)
  assert result.success, f"Failed: {result.errors}"
  assert "jax.devices('gpu')[0]" in result.code


@pytest.mark.skip(reason="Device definitions removed")
def test_device_cpu_to_jax(semantics: SemanticsManager) -> None:
  """Verify `RuntimeConfig` emits `jax.devices('cpu')[0]` for device cpu to JAX.

  Args:
      semantics: SemanticsManager instance providing operation and trait definitions.
  """
  code: str = "d = torch.device('cpu')"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(code)
  assert result.success
  assert "jax.devices('cpu')[0]" in result.code
