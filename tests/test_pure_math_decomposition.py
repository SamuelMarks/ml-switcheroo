"""Tests for pure math framework decomposition handling."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


def test_pure_math_rejection_numpy():
  """Test rejection of NN nodes to numpy in strict mode."""
  code = "import torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config = RuntimeConfig(source_framework="torch", target_framework="numpy", strict_mode=True)
  semantics = SemanticsManager()
  engine = ASTEngine(semantics, config)
  result = engine.run(code)

  assert "Cannot map neural network abstraction" in result.code
  assert "numpy" in result.code
  assert "Use a framework like Flax or Keras" in result.code


def test_pure_math_rejection_jax():
  """Test rejection of NN nodes to jax in strict mode."""
  code = "import torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  semantics = SemanticsManager()
  engine = ASTEngine(semantics, config)
  result = engine.run(code)

  assert "Cannot map neural network abstraction" in result.code
  assert "jax" in result.code
  assert "Use a framework like Flax or Keras" in result.code
