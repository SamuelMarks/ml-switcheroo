"""
Regression Test for Flax NNX RNG Argument Literal Bug.

Verifies that:
1. `rngs` argument is injected as a variable (`rngs=rngs`), not a string literal (`rngs='rngs'`).
2. This logic relies on `StructuralTraits` injection rather than `inject_args` map.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader

# Source PyTorch Code
SOURCE_TORCH = """ 
import torch.nn as nn

class MyLayer(nn.Module): 
    def __init__(self, in_features, out_features): 
        super().__init__() 
        self.linear = nn.Linear(in_features, out_features) 

    def forward(self, x): 
        return self.linear(x) 
"""


def test_rng_variable_injection():
  """
  Ensures that when converting to Flax NNX, the rngs argument passed to Linear
  uses the variable name `rngs` provided in __init__, not a string literal.
  """
  semantics = SemanticsManager()

  # Force reload from registry to ensure we are testing the actual adapter implementation
  loader = RegistryLoader(semantics)
  loader.hydrate()

  # Note: Removed manual override for semantics._key_origins["Linear"] = "neural"
  # because SemanticsManager now correctly loads NEURAL_OPS as SemanticTier.NEURAL by default.

  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)

  result = engine.run(SOURCE_TORCH)

  assert result.success, f"Conversion failed: {result.errors}"
  code = result.code

  print("\n[Generated Code]:")
  print(code)

  # 1. Verify instantiation
  # Expected: nnx.Linear(..., rngs=rngs)
  assert "rngs=rngs" in code

  # Explicitly check for absence of incorrect string literal
  assert "rngs='rngs'" not in code
  assert 'rngs="rngs"' not in code
