"""Test suite for the Integration Rdna Roundtrip module."""

import pytest
import textwrap
import ast
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier

CONVNET_SOURCE = textwrap.dedent(
  "\n    import torch\n    import torch.nn as nn\n\n    class ConvNet(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.conv = nn.Conv2d(1, 32, 3)\n            self.fc = nn.Linear(32 * 26 * 26, 10)\n\n        def forward(self, x):\n            x = self.conv(x)\n            x = torch.flatten(x, 1)\n            return self.fc(x)\n    "
).strip()


@pytest.fixture
def semantics_mgr():
  """Provides a mock semantics mgr for testing."""
  mgr = SemanticsManager()
  mgr.data["Conv2d"] = {"std_args": ["in", "out", "k"], "variants": {"torch": {"api": "torch.nn.Conv2d"}}}
  mgr._reverse_index["torch.nn.Conv2d"] = ("Conv2d", mgr.data["Conv2d"])
  mgr._key_origins["Conv2d"] = SemanticTier.NEURAL.value
  mgr.data["Linear"] = {"std_args": ["in", "out"], "variants": {"torch": {"api": "torch.nn.Linear"}}}
  mgr._reverse_index["torch.nn.Linear"] = ("Linear", mgr.data["Linear"])
  mgr._key_origins["Linear"] = SemanticTier.NEURAL.value
  mgr.data["Flatten"] = {"std_args": ["start", "end"], "variants": {"torch": {"api": "torch.flatten"}}}
  mgr._reverse_index["torch.flatten"] = ("Flatten", mgr.data["Flatten"])
  mgr._source_registry["torch.nn"] = ("torch", SemanticTier.NEURAL)
  mgr.framework_configs["rdna"] = {}
  return mgr


def test_rdna_roundtrip_logic(semantics_mgr):
  """Verifies the behavior of RDNA roundtrip logic."""
  print("\n--- [Phase 1] Compilation (Torch -> RDNA) ---")
  config_compile = RuntimeConfig(source_framework="torch", target_framework="rdna", strict_mode=False)
  engine_compile = ASTEngine(semantics=semantics_mgr, config=config_compile)
  res_compile = engine_compile.run(CONVNET_SOURCE)
  assert res_compile.success, f"Compilation failed: {res_compile.errors}"
  rdna_code = res_compile.code
  print(rdna_code)
  assert "; RDNA Code Generation Initialized" in rdna_code
  assert "L_KY_conv:" in rdna_code
  assert "v_fmac_f32" in rdna_code
  assert "; BEGIN Conv2d (conv)" in rdna_code
  print("\n--- [Phase 2] Decompilation (RDNA -> Torch) ---")
  config_decompile = RuntimeConfig(source_framework="rdna", target_framework="torch", strict_mode=False)
  engine_decompile = ASTEngine(semantics=semantics_mgr, config=config_decompile)
  res_decompile = engine_decompile.run(rdna_code)
  assert res_decompile.success, f"Decompilation failed: {res_decompile.errors}"
  reconstructed_code = res_decompile.code
  print(reconstructed_code)
  assert "class DecompiledNet(nn.Module):" in reconstructed_code
  assert "def __init__(self):" in reconstructed_code
  assert "super().__init__()" in reconstructed_code
  assert "self.conv = nn.Conv2d(" in reconstructed_code
  assert "self.fc = nn.Linear(" in reconstructed_code
  assert "def forward(self, x):" in reconstructed_code
  assert "x = self.conv(x)" in reconstructed_code
  assert "x = self.fc(x)" in reconstructed_code
  assert "flatten(x" in reconstructed_code
  try:
    ast.parse(reconstructed_code)
  except SyntaxError as e:
    pytest.fail(f"Reconstructed code is invalid Python: {e}")
