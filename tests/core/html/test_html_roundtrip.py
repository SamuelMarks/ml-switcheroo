"""
Integration Tests for HTML DSL Roundtrip.

Verifies:
1. Python -> HTML generation (Red/Blue Boxes, SVG Arrows, Functional Ops classification).
2. HTML -> Python parsing (Reconstruction of Class structure and Logic, ignoring Data/Green boxes).
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader
import ml_switcheroo.frameworks.html_dsl  # Ensure adapter registration works

# Sample Python source (PyTorch style)
# Updated to include functional calls (flatten, relu) to test classification logic
SOURCE_CODE = """ 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv = nn.Conv2d(1, 32, 3) 
        self.fc = nn.Linear(32, 10) 
    
    def forward(self, x): 
        x = self.conv(x) 
        x = torch.flatten(x, 1) 
        x = F.relu(x) 
        return self.fc(x) 
"""

# Sample HTML Input logic
# Update: 'in' keyword replaced by 'i' to ensure valid python generation by Parser
# Includes a Green box to verify it is ignored by the parser
HTML_INPUT = """ 
<h3>Model: RestoredNet</h3>
<div class="sw-grid">
  <!-- Attribute: self.conv = Conv2d(...) -->
  <div class="box r">
     <span class="header-txt">conv: Conv2d</span>
     <code>i=1, o=32, k=3</code>
  </div>
  
  <!-- Operation: Call self.conv(x) -->
  <div class="box b">
     <span class="header-txt">Call (conv)</span>
     <code>args: x</code>
  </div>

  <!-- Data Flow: Output of conv (Green Box) -->
  <!-- This should be IGNORED by the parser -->
  <div class="box g">
     <span class="header-txt">out_conv</span>
     <code>[_]</code>
  </div>
  
  <!-- Functional Operation -->
  <div class="box b">
     <span class="header-txt">Flatten</span>
     <code>start_dim=1</code>
  </div>
</div>
"""


@pytest.fixture
def semantics() -> SemanticsManager:
  """
  Returns a hydrated SemanticsManager.
  Ideally loads definitions from the HTML adapter to ensure mappings match.
  """
  mgr = SemanticsManager()
  RegistryLoader(mgr).hydrate()
  return mgr


def test_torch_to_html_generation(semantics):
  """
  Scenario: Convert Python class to HTML Grid.
  Expect: Valid HTML structure with styled boxes and connections.
  Verifies that functional ops like 'flatten' are correctly placed in Blue boxes,
  not Red attribute boxes.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="html", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result = engine.run(SOURCE_CODE)

  assert result.success, f"To-HTML failed: {result.errors}"
  html = result.code

  # 1. Check Document Structure
  assert '<div class="sw-grid">' in html
  assert "Model: Net</h3>" in html
  assert "marker-end" in html  # SVG logic
  assert "<style>" in html  # CSS Injection
  assert "z-index" in html  # Stacking context fix

  # 2. Check Red Box (Conv)
  assert 'class="box r"' in html
  assert "Conv2d" in html
  assert "32" in html

  # 3. Check Blue Box for Stateful Call
  assert 'class="box b"' in html
  assert "Call (conv)" in html

  # 4. Check that SVG arrows have correct classes
  assert 'class="sw-arrow"' in html

  # 5. Verify Functional Op Logic (Flatten placement)
  # Must be 'box b'
  assert "Flatten" in html
  blocks = html.split('<div class="box ')
  flatten_block = next((b for b in blocks if "Flatten" in b), None)
  assert flatten_block is not None
  assert flatten_block.startswith('b"')
  assert "1" in flatten_block

  # 6. Check Green Data Box validation
  assert 'class="box g"' in html


def test_html_to_python_parsing(semantics):
  """
  Scenario: Convert HTML Grid back to Python.
  Expect: Valid Python class with `html_dsl` alias (rewriter will pivot to target if configured).
  Verifies that Green 'Data' boxes are ignored and do not produce function calls.
  """
  # Source is 'html', Target is 'torch'
  config = RuntimeConfig(source_framework="html", target_framework="torch", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)

  # We pass the raw HTML snippet
  result = engine.run(HTML_INPUT)

  assert result.success, f"To-Python failed: {result.errors}"
  py_code = result.code

  # 1. Check Structural Traits
  assert "class RestoredNet(" in py_code
  # ImportFixer + Rewriter should map html_dsl.Module -> torch.nn.Module
  assert "nn.Module" in py_code or "torch.nn.Module" in py_code

  # 2. Check Init Logic
  assert "self.conv =" in py_code
  assert "Conv2d" in py_code

  # 3. Check Forward Logic
  # Parser synthesized `conv_out = self.conv(x)`
  assert "self.conv(x)" in py_code

  # 4. Check for Functional Op parsing (Flatten)
  # Logic: `op_N = dsl.Flatten(start_dim=1)` -> `torch.flatten(...)`
  assert "flatten" in py_code.lower()

  # 5. Verify Green Box Exclusion
  assert "out_conv(" not in py_code
  assert "dsl.out_conv" not in py_code

  # 6. Return statement
  assert "return" in py_code
