"""
Integration Test for RDNA Round-Trip (PyTorch -> RDNA -> PyTorch).

This test verifies the complete "Shift-Left" compiler/decompiler pipeline:
1.  **Compilation**: Converts high-level PyTorch code into low-level AMD RDNA assembly.
    It verifies that architectural logic (Loops, Memory Loads) is expanded via macros.
2.  **Reconstruction**: Reads the generated Assembly back into the Engine.
    It uses semantic markers to "Lift" the assembly logic back into a high-level
    Python `nn.Module` definition.

This guarantees that the Transpiler is reversible even across abstraction layers.
"""

import pytest
import textwrap
import ast
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.enums import SemanticTier

# --- 1. Source Input (ConvNet) ---
CONVNET_SOURCE = textwrap.dedent("""
    import torch
    import torch.nn as nn

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 32, 3)
            self.fc = nn.Linear(32 * 26 * 26, 10)

        def forward(self, x):
            x = self.conv(x)
            x = torch.flatten(x, 1)
            return self.fc(x)
    """).strip()


@pytest.fixture
def semantics_mgr():
  """
  Sets up a Semantic knowledge base supporting both Torch and RDNA definitions.
  We mock the definitions usually found in JSON files to ensure test stability.
  """
  mgr = SemanticsManager()

  # 1. Define Conv2d
  mgr.data["Conv2d"] = {"std_args": ["in", "out", "k"], "variants": {"torch": {"api": "torch.nn.Conv2d"}}}
  mgr._reverse_index["torch.nn.Conv2d"] = ("Conv2d", mgr.data["Conv2d"])
  mgr._key_origins["Conv2d"] = SemanticTier.NEURAL.value

  # 2. Define Linear
  mgr.data["Linear"] = {"std_args": ["in", "out"], "variants": {"torch": {"api": "torch.nn.Linear"}}}
  mgr._reverse_index["torch.nn.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._key_origins["Linear"] = SemanticTier.NEURAL.value

  # 3. Define Flatten
  # Note: RDNA doesn't have an instruction, but GraphExtractor extracts it as 'torch.flatten'
  mgr.data["Flatten"] = {"std_args": ["start", "end"], "variants": {"torch": {"api": "torch.flatten"}}}
  mgr._reverse_index["torch.flatten"] = ("Flatten", mgr.data["Flatten"])

  # Register safety lookups
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
  mgr.framework_configs["rdna"] = {}

  return mgr


def test_rdna_roundtrip_logic(semantics_mgr):
  """
  Executes the full Round-Trip Pipeline.
  """
  print("\n--- [Phase 1] Compilation (Torch -> RDNA) ---")

  # 1. Setup Compiler
  config_compile = RuntimeConfig(source_framework="torch", target_framework="rdna", strict_mode=False)
  engine_compile = ASTEngine(semantics=semantics_mgr, config=config_compile)

  # 2. Execute Compilation
  res_compile = engine_compile.run(CONVNET_SOURCE)
  assert res_compile.success, f"Compilation failed: {res_compile.errors}"
  rdna_code = res_compile.code

  print(rdna_code)

  # 2a. Verify Compilation Artifacts
  # Header
  assert "; RDNA Code Generation Initialized" in rdna_code
  # Loop Labels
  assert "L_KY_conv:" in rdna_code
  # Instructions
  assert "v_fmac_f32" in rdna_code
  # Semantic Markers
  assert "; BEGIN Conv2d (conv)" in rdna_code

  print("\n--- [Phase 2] Decompilation (RDNA -> Torch) ---")

  # 3. Setup Decompiler
  config_decompile = RuntimeConfig(source_framework="rdna", target_framework="torch", strict_mode=False)
  engine_decompile = ASTEngine(semantics=semantics_mgr, config=config_decompile)

  # 4. Execute Decompilation
  res_decompile = engine_decompile.run(rdna_code)
  assert res_decompile.success, f"Decompilation failed: {res_decompile.errors}"

  reconstructed_code = res_decompile.code
  print(reconstructed_code)

  # 5. Verify Structure
  # Class Name: The lifter currently defaults to "DecompiledNet"
  assert "class DecompiledNet(nn.Module):" in reconstructed_code

  # Init Method
  assert "def __init__(self):" in reconstructed_code
  assert "super().__init__()" in reconstructed_code

  # Layer reconstruction (based on markers)
  # Metadata recovery via Analyzer should have found kernel size
  assert "self.conv = nn.Conv2d(" in reconstructed_code
  assert "self.fc = nn.Linear(" in reconstructed_code

  # Forward Method
  assert "def forward(self, x):" in reconstructed_code

  # Logic flow
  assert "x = self.conv(x)" in reconstructed_code
  assert "x = self.fc(x)" in reconstructed_code
  # Flatten might be explicit or implied depending on unmapped handling in lifter
  # Our test case expects it to be preserved as 'torch.flatten' (unmapped default)
  # and injected into forward.
  # GraphSynthesizer outputs 'flatten(x' if no module alias, or 'torch.flatten(x'.
  # Because lifter uses kind="torch.flatten", synthesizer likely outputs "torch.flatten(x, ...)"
  assert "flatten(x" in reconstructed_code

  # 6. Syntax Check
  try:
    ast.parse(reconstructed_code)
  except SyntaxError as e:
    pytest.fail(f"Reconstructed code is invalid Python: {e}")
