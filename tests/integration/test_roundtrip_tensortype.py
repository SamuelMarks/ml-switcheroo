"""Test suite for the Roundtrip Tensortype module."""

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


def test_type_hint_jax_to_torch(semantics_env):
  """Verifies the behavior of type hint JAX to PyTorch."""
  source = "def process(x: jax.Array) -> jax.Array:\n    return x"
  config = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "torch.Tensor" in result.code


def test_type_hint_torch_to_mlx(semantics_env):
  """Verifies the behavior of type hint PyTorch to MLX."""
  source = "def forward(t: torch.Tensor): pass"
  config = RuntimeConfig(source_framework="torch", target_framework="mlx")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "mx.array" in result.code or "mlx.core.array" in result.code
