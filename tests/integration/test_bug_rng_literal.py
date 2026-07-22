"""Test suite for the Bug Rng Literal module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader

SOURCE_TORCH = "\nimport torch.nn as nn\n\nclass MyLayer(nn.Module):\n    def __init__(self, in_features, out_features):\n        super().__init__()\n        self.linear = nn.Linear(in_features, out_features)\n\n    def forward(self, x):\n        return self.linear(x)\n"


def test_rng_variable_injection():
  """Verifies the behavior of rng variable injection."""
  semantics = SemanticsManager()
  loader = RegistryLoader(semantics)
  loader.hydrate()
  config = RuntimeConfig(source_framework="torch", target_framework="flax_nnx", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_TORCH)
  assert result.success, f"Conversion failed: {result.errors}"
  code = result.code
  print("\n[Generated Code]:")
  print(code)
  assert "rngs=rngs" in code
  assert "rngs='rngs'" not in code
  assert 'rngs="rngs"' not in code
