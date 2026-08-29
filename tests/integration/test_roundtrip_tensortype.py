"""Test suite for the Roundtrip Tensortype module."""

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader


@pytest.fixture
def semantics_env() -> SemanticsManager:
  """Docstring."""
  mgr = SemanticsManager()
  RegistryLoader(mgr).hydrate()
  return mgr


@pytest.mark.skip(reason="Tensor/Array definitions removed")
def test_type_hint_jax_to_torch(semantics_env: SemanticsManager) -> None:
  """Verifies the behavior of type hint JAX to PyTorch."""
  source: str = "def process(x: jax.Array) -> jax.Array:\n    return x"
  config = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result: ConversionResult = engine.run(source)
  assert result.success
  assert "torch.Tensor" in result.code


@pytest.mark.skip(reason="Tensor/Array definitions removed")
def test_type_hint_torch_to_mlx(semantics_env: SemanticsManager) -> None:
  """Verifies the behavior of type hint PyTorch to MLX."""
  source: str = "def forward(t: torch.Tensor): pass"
  config = RuntimeConfig(source_framework="torch", target_framework="mlx")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result: ConversionResult = engine.run(source)
  assert result.success
  assert "mx.array" in result.code or "mlx.core.array" in result.code
