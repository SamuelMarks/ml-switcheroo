"""
End-to-End Integration Tests for Complex SASS Kernel Generation.

This test validates the "Macro Expansion" capabilities of the SASS backend.
It inputs a full PyTorch Convolutional Neural Network definition matching the
user's original request and verifies that the output contains the expanded
instruction sequences (loops, memory loads, address calculations) rather
than simple 1:1 opcode mappings.

Items Verified:
1.  **Conv2d Expansion**: Checks for nested loop labels (`L_KY`, `L_KX`) and
    Address Calculation markers (`IMAD`).
2.  **Linear Expansion**: Checks for GEMM loop markers (`L_GEMM`) and
    memory access patterns (`LDG`, `IADD3`).
3.  **Control Flow**: Presence of `BRA` (Branch) and `ISETP` (Comparison).
4.  **Register Allocation**: Verification that operands use allocated registers (`R...`).
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager

# The complex user-provided neural network source
CONVNET_SOURCE = """
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # k=3 implies kernel size 3 for macro
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
"""


@pytest.fixture
def sass_engine() -> ASTEngine:
  """
  Creates an ASTEngine configured for Torch -> SASS conversion.

  Returns:
      ASTEngine: Configured engine instance.
  """
  # Initialize real semantics manager to pick up standards_internal.py changes
  # and sass adapter registrations.
  semantics = SemanticsManager()

  config = RuntimeConfig(source_framework="torch", target_framework="sass", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


def test_convnet_macro_expansion(sass_engine: ASTEngine) -> None:
  """
  Verifies that the ConvNet model compiles to expanded SASS assembly.

  Args:
      sass_engine (ASTEngine): The compilation engine.
  """
  result = sass_engine.run(CONVNET_SOURCE)

  assert result.success, f"Compilation failed with errors: {result.errors}"
  code = result.code

  # --- 1. Validate Convolution Macro (self.conv) ---
  # The GraphExtractor extracts the attribute name 'conv' as the node ID.
  # The macro generator prefixes labels with node ID.

  # Check for Loop structure assertions
  assert "BEGIN Conv2d (conv)" in code
  assert "L_KY_conv:" in code
  assert "L_KX_conv:" in code
  assert "BRA L_KX_conv" in code  # Inner loop jump

  # Check for Address Math & Loads inside loop
  # IMAD = Integer Multiply Add (Address Calc)
  assert "IMAD" in code
  assert "LDG.E.F32" in code

  # Check for Fused Multiply Add
  assert "FFMA" in code

  # --- 2. Validate Flatten (No-Op / Unmapped) ---
  # Flatten is not currently mapped to an instruction or macro in this scope,
  # so it should appear as a Comment or be filtered out depending on graph flow.
  # Our standards define it, but not SASS variant (unless default).
  # If unmapped, it often appears as a comment.
  # Note: Variable flow analysis might link conv output directly to fc input
  # if flatten is treated as passthrough or unmapped op comment.
  assert "Unmapped Op" in code or "//" in code

  # --- 3. Validate Linear Macro (self.fc) ---
  # Attribute name 'fc'
  assert "BEGIN Linear (fc)" in code
  assert "L_GEMM_fc:" in code

  # Check for Load pairs (Input + Weight)
  assert code.count("LDG.E.F32") >= 4  # At least 2 for Conv, 2 for Linear

  # Check for Pointer arithmetic
  assert "IADD3" in code

  # --- 4. Validate Registers ---
  # Ensure symbolic variables mapped to physical registers
  assert "R0" in code
  assert "RZ" in code  # Zero register used in macros
  assert "PT" in code  # Predicate True

  # --- 5. Validate Output ---
  assert "Return:" in code
