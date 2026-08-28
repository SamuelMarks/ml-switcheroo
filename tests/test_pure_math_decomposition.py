"""Tests for pure math framework decomposition handling."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from typing import Any


def test_pure_math_rejection_numpy() -> None:
  """Test rejection of NN nodes to numpy in strict mode."""
  code: str = "import torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="numpy", strict_mode=True)
  semantics: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(semantics, config)
  result: Any = engine.run(code)

  assert "No mapping available for" in result.code
  assert "numpy" in result.code


def test_pure_math_rejection_jax() -> None:
  """Test rejection of NN nodes to jax in strict mode."""
  code: str = "import torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
  semantics: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(semantics, config)
  result: Any = engine.run(code)

  assert "Cannot map neural network abstraction" in result.code
  assert "jax" in result.code
