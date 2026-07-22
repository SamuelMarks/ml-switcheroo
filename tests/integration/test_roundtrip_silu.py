"""Test suite for the Roundtrip Silu module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier


@pytest.fixture
def semantics_env():
  """Provides a mock semantics environment for testing."""
  mgr = SemanticsManager()
  silu_def = {
    "std_args": [{"name": "x", "type": "Tensor"}],
    "variants": {
      "flax_nnx": {"api": "flax.nnx.silu"},
      "torch": {"api": "torch.nn.functional.silu"},
      "numpy": {"macro_template": "{x} * (1 / (1 + np.exp(-{x})))", "required_imports": ["import numpy as np"]},
      "tensorflow": {"api": "tensorflow.nn.silu"},
    },
  }
  mgr.data["SiLU"] = silu_def
  mgr._reverse_index["flax.nnx.silu"] = ("SiLU", silu_def)
  if "flax_nnx" not in mgr.framework_configs:
    mgr.framework_configs["flax_nnx"] = {"alias": {"module": "flax.nnx", "name": "nnx"}}
  mgr._key_origins["SiLU"] = SemanticTier.ARRAY_API.value
  return mgr


def test_silu_flax_to_torch(semantics_env):
  """Verifies the behavior of silu Flax to PyTorch."""
  source = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="torch")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "F.silu(" in result.code or "torch.nn.functional.silu" in result.code


def test_silu_flax_to_numpy_macro(semantics_env):
  """Verifies the behavior of silu Flax to NumPy macro."""
  source = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="numpy")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "np.exp" in result.code
  assert "1 + np.exp" in result.code


def test_silu_flax_to_tensorflow(semantics_env):
  """Verifies the behavior of silu Flax to TensorFlow."""
  source = "y = flax.nnx.silu(x)"
  config = RuntimeConfig(source_framework="flax_nnx", target_framework="tensorflow")
  engine = ASTEngine(semantics=semantics_env, config=config)
  result = engine.run(source)
  assert result.success
  assert "nn.silu(x)" in result.code
