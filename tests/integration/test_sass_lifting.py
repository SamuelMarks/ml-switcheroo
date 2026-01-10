"""
Integration Tests for SASS Decompilation (Lifting).
"""

import ast
import pytest
from textwrap import dedent

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks import register_framework

# --- Source Snippet (SASS) ---
SASS_SOURCE = dedent(""" 
    // Input x -> R0
    // BEGIN Conv2d (conv) 
    MOV R1, RZ; 
    MOV R2, RZ; 
L_KY_conv: 
    IADD3 R3, R3, 1, RZ; 
    // END Conv2d (conv) 
    // Unmapped Op: torch.flatten (func_flatten) 
    // BEGIN Linear (fc) 
    MOV R7, RZ; 
L_GEMM_fc: 
    FFMA R7, R9, R10, R7; 
    // END Linear (fc) 
    // Return: R7
""")


@pytest.fixture
def lifting_engine() -> ASTEngine:
  # Ensure registration
  register_framework("sass")(SassAdapter)
  register_framework("torch")(TorchAdapter)

  semantics = SemanticsManager()
  config = RuntimeConfig(source_framework="sass", target_framework="torch", strict_mode=False)

  return ASTEngine(semantics=semantics, config=config)


def test_sass_lifting_e2e(lifting_engine: ASTEngine) -> None:
  """
  Verifies that the SASS snippet is successfully decompiled into a PyTorch class.
  """
  # 1. Run Conversion
  result = lifting_engine.run(SASS_SOURCE)

  assert result.success, f"Decompilation failed: {result.errors}"
  code = result.code

  # 2. Structural Assertions
  # Class Definition
  assert "class DecompiledModel(nn.Module):" in code

  # Init Layer Definitions
  assert "self.conv = nn.Conv2d(" in code
  assert "self.fc = nn.Linear(" in code

  # Forward Pass Logic
  assert "def forward(self, x):" in code
  assert "x = self.conv(x)" in code
  assert "x = torch.flatten(x, 1)" in code
  assert "x = self.fc(x)" in code
  assert "return x" in code

  # 3. Syntax Verification
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_sass_lifting_no_structural_markers(lifting_engine: ASTEngine) -> None:
  """
  Verifies fallback if SASS contains no high-level markers.
  Should produce low-level python calls (asm.FADD) inside a generic class wrapper.
  """
  raw_sass = "FADD R0, R1, R2;"
  result = lifting_engine.run(raw_sass)

  assert result.success
  code = result.code

  # Check that basic class structure remains (PythonBackend default)
  assert "class DecompiledModel" in code

  # Check for low-level call
  # Assuming 'asm' alias is used for SASS opcodes
  assert "asm.FADD" in code
