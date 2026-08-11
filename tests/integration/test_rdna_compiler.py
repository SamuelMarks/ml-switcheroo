"""Test suite for the Rdna Compiler module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier

CONVNET_SOURCE = "\nimport torch\nimport torch.nn as nn\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, 3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = torch.flatten(x, 1)\n        return self.fc(x)\n"


@pytest.fixture
def compiler_semantics():
  """Provides a mock compiler semantics for testing."""
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


def test_rdna_compiler_pipeline(compiler_semantics):
  """Verifies the behavior of RDNA compiler pipeline."""
  config = RuntimeConfig(source_framework="torch", target_framework="rdna", strict_mode=False)
  engine = ASTEngine(semantics=compiler_semantics, config=config)
  result = engine.run(CONVNET_SOURCE)
  assert result.success, f"Compilation failed: {result.errors}"
  code = result.code
  print(code)
  assert "; RDNA Code Generation Initialized (Arch: gfx1030)" in code
  assert "; BEGIN Conv2d (conv)" in code
  assert "L_KY_conv:" in code
  assert "L_KX_conv:" in code
  assert "v_fmac_f32" in code
  assert "global_load_dword" in code
  assert "s_waitcnt" in code
  assert "; END Conv2d (conv)" in code
  assert "; BEGIN Flatten (func_flatten)" in code
  assert "; END Flatten (func_flatten)" in code
  assert "; BEGIN Linear (fc)" in code
  assert "L_GEMM_fc:" in code
  assert "v0" in code
  assert "s0" in code
