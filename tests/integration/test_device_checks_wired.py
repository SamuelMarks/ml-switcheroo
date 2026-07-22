"""Test suite for the Device Checks Wired module."""

import pytest
import importlib
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
import ml_switcheroo.core.hooks as hooks
import ml_switcheroo.frameworks
import ml_switcheroo.plugins.device_checks


@pytest.fixture
def semantics():
  """Provides a mock semantics for testing."""
  importlib.reload(ml_switcheroo.plugins.device_checks)
  hooks._PLUGINS_LOADED = True
  importlib.reload(ml_switcheroo.frameworks)
  return SemanticsManager()


def test_cuda_check_to_jax(semantics):
  """Verifies the behavior of cuda check to JAX."""
  code = "if torch.cuda.is_available(): run_gpu()"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(code)
  assert result.success, f"Failed: {result.errors}"
  assert "len(jax.devices('gpu')) > 0" in result.code
