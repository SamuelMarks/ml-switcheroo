"""Test suite for the Latex Roundtrip module."""

import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader

SOURCE_TORCH: str = "\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass ConvNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv = nn.Conv2d(1, 32, kernel_size=3)\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        x = self.conv(x)\n        x = F.relu(x)\n        x = self.fc(x)\n        return x\n"


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Helper to semantics."""
  mgr = SemanticsManager()
  RegistryLoader(mgr).hydrate()
  return mgr


def test_torch_to_latex_generation(semantics: SemanticsManager) -> None:
  """Verifies the behavior of PyTorch to LaTeX generation."""
  config = RuntimeConfig(source_framework="torch", target_framework="latex_dsl", strict_mode=False)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(SOURCE_TORCH)
  assert result.success, f"To-LaTeX Conversion Failed: {result.errors}"
  latex_code: str = result.code
  print(latex_code)
  assert "\\documentclass[tikz" in latex_code
  assert "\\begin{DefModel}{ConvNet}" in latex_code
  assert "\\Attribute{conv}{Conv2d}" in latex_code
  assert "kernel_size=3" in latex_code or "arg_2=3" in latex_code
  assert "\\Attribute{fc}{Linear}" in latex_code
  assert "\\Input{input}{[_]}" in latex_code
  assert "\\StateOp{op_conv}{conv}{input}{[_]}" in latex_code
  assert "\\Op{op_func_relu}{Relu}{op_conv, x}{[_]}" in latex_code
  assert "\\StateOp{op_fc}{fc}{op_func_relu}{[_]}" in latex_code
  assert "\\Return{op_fc}" in latex_code
  assert "% [Requirement] midl.sty" in latex_code
  assert "% Ensure 'midl.sty' is in your LaTeX path." in latex_code


def test_latex_to_flax_generation(semantics: SemanticsManager) -> None:
  """Verifies the behavior of LaTeX to Flax generation."""
  latex_source: str = "\n\\documentclass[tikz]{standalone}\n\\begin{DefModel}{ConvNet}\n    \\Attribute{conv}{Conv2d}{in=1, out=32, k=3}\n    \\Attribute{fc}{Linear}{in=21632, out=10}\n    \\Input{x}{[_]}\n\n    \\StateOp{op_conv}{conv}{x}{[_]}\n    \\Op{op_act}{ReLU}{op_conv}{[_]}\n    \\StateOp{op_fc}{fc}{op_act}{[_]}\n    \\Return{op_fc}\n\\end{DefModel}\n"
  config = RuntimeConfig(source_framework="latex_dsl", target_framework="flax_nnx", strict_mode=True)
  engine = ASTEngine(semantics=semantics, config=config)
  result: ConversionResult = engine.run(latex_source)
  assert result.success, f"To-Flax Conversion Failed: {result.errors}"
  flax_code: str = result.code
  assert "class ConvNet(nnx.Module):" in flax_code
  assert "def __init__(self, rngs: nnx.Rngs):" in flax_code
  assert "nnx.Conv(" in flax_code
  assert "rngs=rngs" in flax_code


def test_latex_roundtrip_complex_args(semantics: SemanticsManager) -> None:
  """Verifies the behavior of LaTeX roundtrip complex arguments."""
  source_code: str = "\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass ComplexNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # Complex expression in arguments\n        self.fc = nn.Linear(32 * 26 * 26, 10)\n\n    def forward(self, x):\n        return self.fc(x)\n"
  config_t2l = RuntimeConfig(source_framework="torch", target_framework="latex_dsl", strict_mode=False)
  engine_t2l = ASTEngine(semantics=semantics, config=config_t2l)
  res_latex: ConversionResult = engine_t2l.run(source_code)
  assert res_latex.success, f"Torch->Latex Failed: {res_latex.errors}"
  latex: str = res_latex.code
  assert "32 * 26 * 26" in latex
  config_l2t = RuntimeConfig(source_framework="latex_dsl", target_framework="torch", strict_mode=False)
  engine_l2t = ASTEngine(semantics=semantics, config=config_l2t)
  res_torch: ConversionResult = engine_l2t.run(latex)
  assert res_torch.success, f"Latex->Torch Failed: {res_torch.errors}"
  torch_code: str = res_torch.code
  clean_code: str = torch_code.replace(" ", "")
  assert "nn.Linear(32*26*26,10)" in clean_code
