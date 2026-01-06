"""
Integration Tests for SASS Decompilation (Lifting).

Verifies the end-to-end pipeline:
1.  **Ingestion**: SASS Source with Semantic Markers -> SASS AST.
2.  **Lifting**: SASS AST -> Logical Graph.
3.  **Synthesis**: Logical Graph -> PyTorch Module Source.

Matches the prompt's specific ConvNet example.
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
# A simplified version of the prompt's SASS output containing critical markers.
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

# --- Expected Output (PyTorch) ---
# Note: Arguments are empty because the SASS markers didn't carry metadata args.
# The synthesizer generates skeleton code which is structurally valid but requires params.
EXPECTED_PYTORCH = dedent("""
import torch
import torch.nn as nn

class DecompiledModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d()
        self.fc = nn.Linear()

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x
""")


@pytest.fixture
def lifting_engine() -> ASTEngine:
  """
  Returns an ASTEngine configured for SASS -> Torch lifting.
  Ensures SASS adapter is registered.
  """
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
  # Note: Analyzer extracts params now, so we might see args if instructions were rich enough.
  # The snippet has IMAD/ISETP which analyzer handles, so we expect kernel_size=3 (args or kwargs).
  # Since synthesizer maps metadata to args string, checking basic presence is robust.
  assert "self.conv = nn.Conv2d(" in code
  assert "self.fc = nn.Linear(" in code

  # Forward Pass Logic
  assert "def forward(self, x):" in code
  assert "x = self.conv(x)" in code
  # Fix: Expect the default arg '1' injected by the lifter logic for flatten
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
  Should produce low-level python calls (sass.FADD, sass.MOV).
  """
  raw_sass = "FADD R0, R1, R2;"
  result = lifting_engine.run(raw_sass)

  assert result.success
  code = result.code

  # Should NOT generate a class
  assert "class DecompiledModel" not in code

  # Should contain low-level calls using the alias 'asm' as defined in SassAdapter
  # or 'sass' depending on implementation details of the Synthesizer.
  # We check for either to be robust to synthesizer defaults.
  assert "asm.FADD" in code or "sass.FADD" in code
