"""
Integration Tests for Neural Network Conversion (Example 02).

This module validates the end-to-end translation of a Neural Network definition
(stateful class) from PyTorch to various target frameworks.
"""

import ast
import pytest
import textwrap

from ml_switcheroo import RuntimeConfig, ASTEngine, SemanticsManager
from ml_switcheroo.enums import SemanticTier
from tests.utils.ast_utils import cmp_ast

# --- Source Code (PyTorch) ---
SOURCE_TORCH = textwrap.dedent(""" 
    import torch.nn as nn

    class SimplePerceptron(nn.Module): 
        def __init__(self, in_features, out_features): 
            super().__init__() 
            self.layer = nn.Linear(in_features, out_features) 

        def forward(self, x): 
            return self.layer(x) 
    """)


@pytest.fixture(scope="module")
def semantics():
  # Inject the Linear definition manually to ensure test isolation
  # and guarantee RNG injection trigger.
  mgr = SemanticsManager()

  # Feature 22: Import Abstraction Init
  mgr._providers = {}
  mgr._source_registry = {}

  # Ensure Linear is marked as NEURAL to trigger rng injection logic.
  mgr._key_origins["Linear"] = SemanticTier.NEURAL.value

  # Ensure Definition exists if files missing
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
    # Map source paths to abstract ID and definition
    mgr._reverse_index["torch.nn.Linear"] = ("Linear", mgr.data["Linear"])

  # Register Source
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)

  # Register Providers explicitly for ImportResolver
  mgr._providers["mlx"] = {SemanticTier.NEURAL: {"root": "mlx", "sub": "nn", "alias": "nn"}}

  # Register Alias mapping explicitly for the mock semantics if not loaded from adapter
  if "mlx" not in mgr.framework_configs:
    mgr.framework_configs["mlx"] = {"alias": {"module": "mlx.core", "name": "mx"}}

  # Default Provider for Flax
  mgr._providers["flax_nnx"] = {SemanticTier.NEURAL: {"root": "flax", "sub": "nnx", "alias": "nnx"}}

  # Default Provider for Keras
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
  """
  Executes conversion of a Neural Network class from Torch to Target.
  """
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  # Run Transpilation
  result = engine.run(SOURCE_TORCH)

  # Check Validity
  assert result.success, f"Conversion Errors: {result.errors}"
  code = result.code

  for s in check_strings:
    assert s in code, f"Missing '{s}' in:\n{code}"
