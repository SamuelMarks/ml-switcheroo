"""Test suite for the Html Roundtrip module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader

SOURCE_CODE = "\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        x = F.relu(x)\n        return self.fc(x)\n"
HTML_INPUT = '\n<h3>Model: RestoredNet</h3>\n<div class="sw-grid">\n  <!-- Attribute: self.conv = Conv2d(...) -->\n  <div class="box r">\n     <span class="header-txt">conv: Conv2d</span>\n     <code>i=1, o=32, k=3</code>\n  </div>\n\n  <!-- Operation: Call self.conv(x) -->\n  <div class="box b">\n     <span class="header-txt">Call (conv)</span>\n     <code>args: x</code>\n  </div>\n\n  <!-- Data Flow: Output of conv (Green Box) -->\n  <!-- This should be IGNORED by the parser -->\n  <div class="box g">\n     <span class="header-txt">out_conv</span>\n     <code>[_]</code>\n  </div>\n\n  <!-- Functional Operation -->\n  <div class="box b">\n     <span class="header-txt">Flatten</span>\n     <code>start_dim=1</code>\n  </div>\n</div>\n'


@pytest.fixture
def semantics() -> SemanticsManager:
  """Provides a mock semantics for testing."""
  mgr = SemanticsManager()
  RegistryLoader(mgr).hydrate()
  return mgr


def test_torch_to_html_generation(semantics):
  """Verifies the behavior of PyTorch to HTML generation."""
  config = RuntimeConfig(source_framework="torch", target_framework="html", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_CODE)
  assert result.success, f"To-HTML failed: {result.errors}"
  html = result.code
  assert '<div class="sw-grid">' in html
  assert "Model: Net</h3>" in html
  assert "marker-end" in html
  assert "<style>" in html
  assert "z-index" in html
  assert 'class="box r"' in html
  assert "Conv2d" in html
  assert "32" in html
  assert 'class="box b"' in html
  assert "Call (conv)" in html
  assert 'class="sw-arrow"' in html
  assert "Flatten" in html
  blocks = html.split('<div class="box ')
  flatten_block = next((b for b in blocks if "Flatten" in b), None)
  assert flatten_block is not None
  assert flatten_block.startswith('b"')
  assert "1" in flatten_block
  assert 'class="box g"' in html


def test_html_to_python_parsing(semantics):
  """Verifies the behavior of HTML to python parsing."""
  config = RuntimeConfig(source_framework="html", target_framework="torch", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(HTML_INPUT)
  assert result.success, f"To-Python failed: {result.errors}"
  py_code = result.code
  assert "class RestoredNet(" in py_code
  assert "nn.Module" in py_code or "torch.nn.Module" in py_code
  assert "self.conv =" in py_code
  assert "Conv2d" in py_code or "conv2d" in py_code
  assert "self.conv(x)" in py_code
  assert "flatten" in py_code.lower()
  assert "out_conv(" not in py_code
  assert "dsl.out_conv" not in py_code
  assert "return" in py_code
