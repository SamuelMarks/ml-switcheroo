"""
Integration Test for SiLU (Sigmoid Linear Unit) Operation.
Manual fixtures used to ensure robustness against file state.

Verifies:
1.  **Torch Target**: `flax.nnx.silu` -> `torch.nn.functional.silu`.
2.  **NumPy Target**: Macro expansion of SiLU (x * sigmoid(x)) into raw math.
3.  **TensorFlow Target**: `flax.nnx.silu` -> `tensorflow.nn.silu`.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier


@pytest.fixture
def semantics_env():
  """
  Creates a SemanticsManager with explicit SiLU mappings for testing.
  This bypasses external JSON loading to ensure test determinism.
  """
  mgr = SemanticsManager()

  # Manually Inject SiLU Definition
  silu_def = {
    "std_args": [{"name": "x", "type": "Tensor"}],
    "variants": {
      "flax_nnx": {"api": "flax.nnx.silu"},
      "torch": {"api": "torch.nn.functional.silu"},
      # Macro test: explicit math formula
      "numpy": {
        "macro_template": "{x} * (1 / (1 + np.exp(-{x})))",
        "required_imports": ["import numpy as np"],
      },
      "tensorflow": {"api": "tensorflow.nn.silu"},
    },
  }

  # Register in data (Hub)
  mgr.data["SiLU"] = silu_def
  mgr._reverse_index["flax.nnx.silu"] = ("SiLU", silu_def)

  # Ensure frameworks are configured with aliases to allow detection
  # This prevents the 'Macro requires argument x' error by making sure 'nnx' is seen as alias
  if "flax_nnx" not in mgr.framework_configs:
    mgr.framework_configs["flax_nnx"] = {"alias": {"module": "flax.nnx", "name": "nnx"}}

  mgr._key_origins["SiLU"] = SemanticTier.ARRAY_API.value
  return mgr


def test_silu_flax_to_torch(semantics_env):
  """
  Scenario: Flax SiLU -> Torch SiLU.
  Expectation: `torch.nn.functional.silu` (or `F.silu` alias).
  """
  source = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  # Optimized check: Alias F.silu or API
  assert "F.silu(" in result.code or "torch.nn.functional.silu" in result.code


def test_silu_flax_to_numpy_macro(semantics_env):
  """
  Scenario: Flax SiLU -> NumPy (which lacks native SiLU).
  Expectation: Macro expansion into sigmoid logic (`x * (1 / (1 + np.exp(-x)))`).
  """
  source = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="numpy")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "np.exp" in result.code
  assert "1 + np.exp" in result.code


def test_silu_flax_to_tensorflow(semantics_env):
  """
  Scenario: Flax SiLU -> TensorFlow SiLU.
  Expectation: `tensorflow.nn.silu` (or `nn.silu` alias).
  """
  source = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="tensorflow")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  # Aliased tf.nn or full pkg
  assert "nn.silu(x)" in result.code
