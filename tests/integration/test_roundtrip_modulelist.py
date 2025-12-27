"""
Integration Tests for ModuleList Container Logic.

This module validates the round-trip conversion of container lists for neural layers.
It specifically checks:
1.  **Torch -> Flax**: `torch.nn.ModuleList` -> `flax.nnx.List`
2.  **Flax -> Torch**: `flax.nnx.List` -> `torch.nn.ModuleList`
3.  **Missing Mappings**: Verifies strict mode failure when target (e.g. Keras)
    lacks clear mapping for this container type.
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
  Ensures that default definitions provided by adapters (Torch/Flax) are loaded.
  """
  mgr = SemanticsManager()
  RegistryLoader(mgr).hydrate()
  return mgr


def test_modulelist_container_flax_to_torch(semantics_env):
  """
  Scenario: Converting Flax NNX List container to PyTorch ModuleList.
  Input: `layers = flax.nnx.List([layer1, layer2])`
  Expectation: `layers = torch.nn.ModuleList([layer1, layer2])`
  (or using valid aliases like `nn.ModuleList`).
  """
  source = "layers = flax.nnx.List([layer1, layer2])"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  # Update: check for aliased or full path
  assert "nn.ModuleList" in result.code or "torch.nn.ModuleList" in result.code
  assert "[layer1, layer2]" in result.code


def test_modulelist_container_torch_to_flax(semantics_env):
  """
  Scenario: Converting PyTorch ModuleList to Flax NNX List.
  Input: `layers = torch.nn.ModuleList([layer1, layer2])`
  Expectation: `layers = flax.nnx.List([layer1, layer2])`
  (or using valid aliases).
  """
  source = "layers = torch.nn.ModuleList([layer1, layer2])"
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  # Update: check for aliased or full path
  assert "nnx.List" in result.code or "flax.nnx.List" in result.code
  assert "[layer1, layer2]" in result.code


def test_modulelist_missing_support_check(semantics_env):
  """
  Scenario: Converting ModuleList to a target without mapping (Keras).
  Expectation: Strict mode triggers an error (Escape Hatch detected).
  """
  source = "l = torch.nn.ModuleList([])"
  # Strict mode checks missing mapping (Keras missing ModuleList definition)
  config = RuntimeConfig(source_framework="torch", target_framework="keras", strict_mode=True)
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)

  assert result.has_errors
  # Engine generic error msg
  assert "Escape Hatches Detected" in str(result.errors)
