"""Tests for PaxML layer mappings."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.conversion_result import ConversionResult


def test_conv2d_paxml_positional() -> None:
  """Test positional mapping for Conv2d to PaxML."""
  code: str = "import torch.nn as nn\nself.conv = nn.Conv2d(1, 32, 3)"
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="paxml")
  semantics: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(semantics, config)
  result: ConversionResult = engine.run(code)

  assert "pl.Conv2D(32, 3)" in result.code


def test_linear_paxml_positional() -> None:
  """Test positional mapping for Linear to PaxML."""
  code: str = "import torch.nn as nn\nself.fc = nn.Linear(32 * 26 * 26, 10)"
  config: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="paxml")
  semantics: SemanticsManager = SemanticsManager()
  engine: ASTEngine = ASTEngine(semantics, config)
  result: ConversionResult = engine.run(code)

  assert "pl.Linear(32 * 26 * 26, 10)" in result.code
