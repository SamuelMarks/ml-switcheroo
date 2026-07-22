"""Test suite for the Sass Lifting module."""

import ast
import pytest
from textwrap import dedent
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.frameworks.sass import SassAdapter
from ml_switcheroo.frameworks.torch import TorchAdapter
from ml_switcheroo.frameworks import register_framework

SASS_SOURCE = dedent(
  "\n    // Input x -> R0\n    // BEGIN Conv2d (conv)\n    MOV R1, RZ;\n    MOV R2, RZ;\nL_KY_conv:\n    IADD3 R3, R3, 1, RZ;\n    // END Conv2d (conv)\n    // Unmapped Op: torch.flatten (func_flatten)\n    // BEGIN Linear (fc)\n    MOV R7, RZ;\nL_GEMM_fc:\n    FFMA R7, R9, R10, R7;\n    // END Linear (fc)\n    // Return: R7\n"
)


@pytest.fixture
def lifting_engine() -> ASTEngine:
  """Provides a mock lifting engine for testing."""
  register_framework("sass")(SassAdapter)
  register_framework("torch")(TorchAdapter)
  semantics = SemanticsManager()
  config = RuntimeConfig(source_framework="sass", target_framework="torch", strict_mode=False)
  return ASTEngine(semantics=semantics, config=config)


def test_sass_lifting_e2e(lifting_engine: ASTEngine) -> None:
  """Verifies the behavior of SASS lifting end-to-end."""
  result = lifting_engine.run(SASS_SOURCE)
  assert result.success, f"Decompilation failed: {result.errors}"
  code = result.code
  assert "class DecompiledModel(nn.Module):" in code
  assert "self.conv = nn.Conv2d(" in code
  assert "self.fc = nn.Linear(" in code
  assert "def forward(self, x):" in code
  assert "x = self.conv(x)" in code
  assert "x = torch.flatten(x, 1)" in code
  assert "x = self.fc(x)" in code
  assert "return x" in code
  try:
    ast.parse(code)
  except SyntaxError as e:
    pytest.fail(f"Generated Invalid Python:\n{e}\n\nCode:\n{code}")


def test_sass_lifting_no_structural_markers(lifting_engine: ASTEngine) -> None:
  """Verifies the behavior of SASS lifting no structural markers."""
  raw_sass = "FADD R0, R1, R2;"
  result = lifting_engine.run(raw_sass)
  assert result.success
  code = result.code
  assert "class DecompiledModel" in code
  assert "asm.FADD" in code
