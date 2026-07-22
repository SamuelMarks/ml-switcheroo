"""Test suite for the Latex Source To Torch module."""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader
from ml_switcheroo.core.escape_hatch import EscapeHatch

LATEX_SOURCE_CONVNET = "\n\\documentclass[tikz]{standalone}\n\\begin{DefModel}{ConvNet}\n    % arg_0=in_channels, arg_1=out_channels, arg_2=kernel_size\n    \\Attribute{conv}{Conv2d}{arg_0=1, arg_1=32, arg_2=3}\n\n    % arg_0=in_features, arg_1=out_features\n    \\Attribute{fc}{Linear}{arg_0=128, arg_1=10}\n\n    % Input tensor\n    \\Input{x}{[B, 1, 28, 28]}\n\n    % Forward Pass\n    \\StateOp{h1}{conv}{x}{[_]}\n    \\Op{h2}{relu}{h1}{[_]}\n\n    % Flatten: arg_0=start_dim\n    \\Op{flat}{Flatten}{h2, arg_0=1}{[_]}\n\n    \\StateOp{out}{fc}{flat}{[_]}\n    \\Return{out}\n\\end{DefModel}\n"


@pytest.fixture
def hydrated_semantics():
  """Provides a mock hydrated semantics for testing."""
  mgr = SemanticsManager()
  loader = RegistryLoader(mgr)
  loader.hydrate()
  return mgr


def test_latex_to_torch_architecture_conversion(hydrated_semantics):
  """Verifies the behavior of LaTeX to PyTorch architecture conversion."""
  config = RuntimeConfig(source_framework="latex_dsl", target_framework="torch", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)
  result = engine.run(LATEX_SOURCE_CONVNET)
  if not result.success:
    pytest.fail(f"Transpilation failed. Errors: {result.errors}")
  code = result.code
  print(f"\n[Generated Output]\n{code}")
  assert "import midl" not in code
  assert "import torch" in code
  assert "class ConvNet(" in code
  assert "Module):" in code
  assert "super().__init__()" in code
  assert "Conv2d" in code or "conv2d" in code
  clean_conv = code.replace("in_channels=", "").replace("out_channels=", "").replace("kernel_size=", "")
  assert "(1, 32, 3" in clean_conv
  assert "self.conv =" in code
  assert "Linear" in code
  assert "(128, 10)" in code.replace("in_features=", "").replace("out_features=", "")
  assert "def forward(self, x):" in code
  assert "h1 = self.conv(x)" in code
  assert "h2 = " in code
  assert "relu(h1)" in code
  assert "flat =" in code
  assert "Flatten" in code
  assert "1)" in code
  assert "out = self.fc(flat)" in code
  assert "return out" in code


def test_missing_mapping_fails_strict_mode(hydrated_semantics):
  """Verifies the behavior of missing mapping fails strict mode."""
  bad_source = "\n    \\documentclass{standalone}\n    \\begin{DefModel}{BadNet}\n        \\Attribute{x}{UnknownLayer}{arg_0=1}\n    \\end{DefModel}\n    "
  config = RuntimeConfig(source_framework="latex_dsl", target_framework="torch", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)
  result = engine.run(bad_source)
  assert EscapeHatch.START_MARKER in result.code
  assert "midl.UnknownLayer" in result.code


def test_argument_value_mapping(hydrated_semantics):
  """Verifies the behavior of argument value mapping."""
  source = "\n    \\documentclass{standalone}\n    \\begin{DefModel}{KeywordNet}\n        % Using direct key-value pairs supported by parser\n        \\Attribute{drop}{Dropout}{p=0.5}\n        \\Input{x}{_}\n        \\StateOp{y}{drop}{x}{_}\n        \\Return{y}\n    \\end{DefModel}\n    "
  config = RuntimeConfig(source_framework="latex_dsl", target_framework="torch", strict_mode=False)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)
  result = engine.run(source)
  assert result.success
  assert "dropout" in result.code.lower()
  assert "p=0.5" in result.code
