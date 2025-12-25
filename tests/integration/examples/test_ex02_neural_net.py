"""
Integration Tests for Neural Network Conversion (Example 02).

This module validates the end-to-end translation of a Neural Network definition
(stateful class) from PyTorch to various target frameworks.

Scope:
1.  **Structure**: Verifies that class inheritance is correctly rewritten
    (e.g., ``nn.Module`` -> ``keras.Layer`` or ``flax.nnx.Module``).
2.  **Layers**: Checks that layer instantiation is mapped correctly
    (e.g., ``nn.Linear`` -> ``keras.layers.Dense``).
3.  **Forward Pass**: Ensures the inference method is renamed correctly
    (e.g., ``forward`` -> ``call`` or ``__call__``).
4.  **State Management**: Validates advanced state handling, such as Flax NNX's
    explicit RNG injection.

Targets Tested:
- **Flax NNX**: Requires ``rngs`` injection and ``__call__``.
- **Keras (v3)**: Requires ``keras.Layer`` base and ``call`` method.
- **MLX**: Requires ``mlx.nn.Module`` base and ``__call__`` method.
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

# --- Expected Outputs (Verified Standards) ---

# 1. Flax NNX
# - Base: flax.nnx.Module
# - Init: Injects 'rngs' because Linear is a stochastic layer (initialization needs RNG).
# - Linear: Requires 'rngs=rngs' pass-through.
# - Inference: '__call__' instead of forward.
# - UPDATED: Reflects actual import fixer output (from flax import nnx)
EXPECTED_FLAX_NNX = textwrap.dedent(""" 
  from flax import nnx
  import flax.nnx as nn

  class SimplePerceptron(nnx.Module): 
      def __init__(self, rngs: nnx.Rngs, in_features, out_features): 
          self.layer = nnx.Linear(in_features, out_features, rngs=rngs) 

      def __call__(self, x): 
          return self.layer(x) 
  """)

# 2. Keras (v3)
EXPECTED_KERAS = textwrap.dedent(""" 
  import keras

  class SimplePerceptron(keras.Layer): 
      def __init__(self, in_features, out_features): 
          super().__init__() 
          self.layer = keras.layers.Dense(in_features, out_features) 

      def call(self, x): 
          return self.layer(x) 
  """)

# 3. Apple MLX
EXPECTED_MLX = textwrap.dedent(""" 
  import mlx.nn as nn

  class SimplePerceptron(nn.Module): 
      def __init__(self, in_features, out_features): 
          super().__init__() 
          self.layer = nn.Linear(in_features, out_features) 

      def __call__(self, x): 
          return self.layer(x) 
  """)


@pytest.fixture(scope="module")
def semantics():
  # Inject the Linear definition manually to ensure test isolation
  # and guarantee RNG injection trigger.
  mgr = SemanticsManager()

  # Ensure Linear is marked as NEURAL to trigger rng injection logic.
  # We FORCE assignment here to robustly handle cases where 'Linear' might have been
  # loaded as 'EXTRAS' by a partial overlay load in the test environment (e.g. Ghost Mode).
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

  # Ensure ReLU definition logic if needed, although Linear is key for state
  if "relu" not in mgr._key_origins and not mgr.get_definition_by_id("relu"):
    mgr.data["relu"] = {"std_args": ["x"], "variants": {}}
    mgr._key_origins["relu"] = SemanticTier.ARRAY_API.value

  return mgr


@pytest.mark.parametrize(
  "target_fw, expected_code",
  [
    ("flax_nnx", EXPECTED_FLAX_NNX),
    ("keras", EXPECTED_KERAS),
    ("mlx", EXPECTED_MLX),
  ],
)
def test_torch_to_target_neural(semantics, target_fw, expected_code):
  """
  Executes conversion of a Neural Network class from Torch to Target.

  Args:
      semantics: The SemanticsManager fixture (loaded with rules).
      target_fw (str): Destination framework key (e.g., 'mlx').
      expected_code (str): The expected Python source code.

  Raises:
      AssertionError: If conversion fails or generated AST mismatch.
  """
  config = RuntimeConfig(source_framework="torch", target_framework=target_fw, strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  # Run Transpilation
  result = engine.run(SOURCE_TORCH)

  # Check Validity
  assert result.success, f"Conversion Errors: {result.errors}"

  # Verify Output (AST Comparison ignores whitespace/comments)
  try:
    generated_ast = ast.parse(result.code)
    expected_ast = ast.parse(expected_code)
    assert cmp_ast(generated_ast, expected_ast)

  except AssertionError:
    # Debug Helper: Print diff if assertion fails
    print(f"\n--- Expected ({target_fw}) ---\n{expected_code}")
    print(f"\n--- Actual ({target_fw}) ---\n{result.code}\n")
    raise
