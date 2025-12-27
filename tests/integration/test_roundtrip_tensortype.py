"""
Integration Tests for Tensor Type Hint Rewriting.

Verifies that Python type hints for Tensors are correctly translated
between frameworks during a roundtrip or conversion.

Scenarios:
1.  **JAX -> Torch**: `jax.Array` -> `torch.Tensor`.
2.  **Torch -> MLX**: `torch.Tensor` -> `mlx.core.array`.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader


@pytest.fixture
def semantics_env():
  """
  Creates a SemanticsManager hydrated from the registry.
  Ensures default type mapping definitions are active.
  """
  mgr = SemanticsManager()
  RegistryLoader(mgr).hydrate()
  return mgr


def test_type_hint_jax_to_torch(semantics_env):
  """
  Scenario: Function argument and return type hint using JAX Array.
  Input: `def process(x: jax.Array) -> jax.Array:`
  Expectation: `def process(x: torch.Tensor) -> torch.Tensor:`
  """
  source = "def process(x: jax.Array) -> jax.Array:\n    return x"
  config = RuntimeConfig(source_framework="jax", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "torch.Tensor" in result.code


def test_type_hint_torch_to_mlx(semantics_env):
  """
  Scenario: Function argument hint using Torch Tensor.
  Input: `def forward(t: torch.Tensor):`
  Expectation: `def forward(t: mlx.core.array):` (or `mx.array`).
  """
  source = "def forward(t: torch.Tensor): pass"
  config = RuntimeConfig(source_framework="torch", target_framework="mlx")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  # Update: Relax check for aliased imports due to MLX alias configuration
  assert "mx.array" in result.code or "mlx.core.array" in result.code
