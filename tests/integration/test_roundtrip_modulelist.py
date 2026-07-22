"""Test suite for the Roundtrip Modulelist module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader


@pytest.fixture
def semantics_env():
  """Provides a mock semantics environment for testing."""
  mgr = SemanticsManager()
  RegistryLoader(mgr).hydrate()
  return mgr


def test_modulelist_container_flax_to_torch(semantics_env):
  """Verifies the behavior of modulelist container Flax to PyTorch."""
  source = "layers = flax.nnx.List([layer1, layer2])"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "nn.ModuleList" in result.code or "torch.nn.ModuleList" in result.code
  assert "[layer1, layer2]" in result.code


def test_modulelist_container_torch_to_flax(semantics_env):
  """Verifies the behavior of modulelist container PyTorch to Flax."""
  source = "layers = torch.nn.ModuleList([layer1, layer2])"
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "nnx.List" in result.code or "flax.nnx.List" in result.code
  assert "[layer1, layer2]" in result.code


def test_modulelist_missing_support_check(semantics_env):
  """Verifies the behavior of modulelist missing support check."""
  source = "l = torch.nn.ModuleList([])"
  config = RuntimeConfig(source_framework="torch", target_framework="keras", strict_mode=True)
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.has_errors
  assert "Escape Hatches Detected" in str(result.errors)
