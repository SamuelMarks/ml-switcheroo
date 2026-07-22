"""Test suite for the Ex02 Neural Net module."""

import pytest
import textwrap
from ml_switcheroo import RuntimeConfig, ASTEngine, SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier

SOURCE_TORCH = textwrap.dedent(
  "\n    import torch.nn as nn\n\n    class SimplePerceptron(nn.Module):\n        def __init__(self, in_features, out_features):\n            super().__init__()\n            self.layer = nn.Linear(in_features, out_features)\n\n        def forward(self, x):\n            return self.layer(x)\n    "
)


@pytest.fixture(scope="module")
def semantics():
  """Helper to semantics."""
  mgr = SemanticsManager()
  mgr._providers = {}
  mgr._source_registry = {}
  mgr._key_origins["Linear"] = SemanticTier.NEURAL.value
  if not mgr.get_definition_by_id("Linear"):
    mgr.data["Linear"] = {
      "std_args": ["in_features", "out_features"],
      "variants": {
        "torch": {"api": "torch.nn.Linear"},
        "flax_nnx": {"api": "flax.nnx.Linear"},
        "keras": {"api": "keras.layers.Dense", "args": {"out_features": "units"}},
        "mlx": {"api": "mlx.nn.Linear"},
      },
    }
    mgr._reverse_index["torch.nn.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
  mgr._providers["mlx"] = {SemanticTier.NEURAL: {"root": "mlx", "sub": "nn", "alias": "nn"}}
  if "mlx" not in mgr.framework_configs:
    mgr.framework_configs["mlx"] = {"alias": {"module": "mlx.core", "name": "mx"}}
  mgr._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}
  mgr._providers["keras"] = {SemanticTier.NEURAL: {"root": "keras", "sub": None, "alias": None}}
  return mgr


@pytest.mark.parametrize(
  "target_fw, check_strings",
  [
    ("flax_nnx", ["class SimplePerceptron(nnx.Module):", "rngs=rngs", "nnx.Linear"]),
    ("keras", ["class SimplePerceptron(keras.Layer):", "keras.layers.Dense", "def call(self, x):"]),
    ("mlx", ["class SimplePerceptron(nn.Module):", "import mlx.nn as nn", "def __call__(self, x):"]),
  ],
)
def test_torch_to_target_neural(semantics, target_fw, check_strings):
  """Verifies the behavior of PyTorch to target neural."""
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_TORCH)
  assert result.success, f"Conversion Errors: {result.errors}"
  code = result.code
  for s in check_strings:
    assert s in code, f"Missing '{s}' in:\n{code}"
