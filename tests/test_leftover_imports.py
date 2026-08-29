"""Tests for leftover import removal after transformation."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager


def test_leftover_import_tensorflow() -> None:
  """Docstring."""
  code: str = "import torch\nimport torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="tensorflow")
  semantics: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(semantics, config)
  result: ConversionResult = engine.run(code)

  assert "import torch.nn as nn" not in result.code
  assert "import torch" not in result.code
  assert "import tensorflow as tf" in result.code


def test_leftover_import_keras() -> None:
  """Docstring."""
  code: str = "import torch\nimport torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="keras")
  semantics: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(semantics, config)
  result: ConversionResult = engine.run(code)

  assert "import torch.nn as nn" not in result.code
  assert "import torch" not in result.code
  assert "import keras" in result.code
