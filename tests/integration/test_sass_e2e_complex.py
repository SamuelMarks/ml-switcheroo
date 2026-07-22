"""Test suite for the Sass E2E Complex module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

CONVNET_SOURCE = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # k=3 implies kernel size 3 for macro\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


@pytest.fixture
def sass_engine() -> ASTEngine:
  """Provides a mock SASS engine for testing."""
  semantics = SemanticsManager()
  config = RuntimeConfig(source_framework="torch", target_framework="sass", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


def test_convnet_macro_expansion(sass_engine: ASTEngine) -> None:
  """Verifies the behavior of convnet macro expansion."""
  result = sass_engine.run(CONVNET_SOURCE)
  assert result.success, f"Compilation failed with errors: {result.errors}"
  code = result.code
  assert "BEGIN Conv2d (conv)" in code
  assert "L_KY_conv:" in code
  assert "L_KX_conv:" in code
  assert "BRA L_KX_conv" in code
  assert "IMAD" in code
  assert "LDG.E.F32" in code
  assert "FFMA" in code
  assert "Unmapped Op" in code or "//" in code
  assert "BEGIN Linear (fc)" in code
  assert "L_GEMM_fc:" in code
  assert code.count("LDG.E.F32") >= 4
  assert "IADD3" in code
  assert "R0" in code
  assert "RZ" in code
  assert "PT" in code
  assert "Return:" in code
