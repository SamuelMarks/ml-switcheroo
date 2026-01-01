"""
Integration Tests for LaTeX DSL (MIDL) to PyTorch Conversion.

This module verifies the "Reverse Pivot" logic:
Input:  Declarative LaTeX describing a neural network.
Output: Valid PyTorch `nn.Module` code using `torch.nn.Conv2d`, etc.

Key Verification Points:
1.  **Parsing**: The LaTeX Parser correctly synthesizes a Python AST using the virtual `midl` namespace.
2.  **Semantic Mapping**: The `LatexDSLAdapter` definitions correctly map positional LaTeX arguments
    (e.g., `arg_0`) to semantic names (e.g., `in_channels`).
3.  **Rewriting**: The AST Engine transforms `midl.Conv2d` into `torch.nn.Conv2d`.
4.  **Integration**: Arguments, imports, and class structure are generated correctly.
"""

import pytest
import textwrap

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.registry_loader import RegistryLoader
from ml_switcheroo.core.escape_hatch import EscapeHatch

# Ensure the adapter is registered in the global scope
import ml_switcheroo.frameworks.latex_dsl

# --- Test Constants ---

# A complete ConvNet definition in MIDL LaTeX
# Uses positional argument mapping (arg_0, arg_1) defined in frameworks/latex_dsl.py
LATEX_SOURCE_CONVNET = r"""
\documentclass[tikz]{standalone}
\begin{DefModel}{ConvNet}
    % arg_0=in_channels, arg_1=out_channels, arg_2=kernel_size
    \Attribute{conv}{Conv2d}{arg_0=1, arg_1=32, arg_2=3}
    
    % arg_0=in_features, arg_1=out_features
    \Attribute{fc}{Linear}{arg_0=128, arg_1=10}
    
    % Input tensor
    \Input{x}{[B, 1, 28, 28]}

    % Forward Pass
    \StateOp{h1}{conv}{x}{[_]}
    \Op{h2}{relu}{h1}{[_]}
    
    % Flatten: arg_0=start_dim
    \Op{flat}{Flatten}{h2, arg_0=1}{[_]}
    
    \StateOp{out}{fc}{flat}{[_]}
    \Return{out}
\end{DefModel}
"""

# --- Fixtures ---


@pytest.fixture
def hydrated_semantics():
  """
  Returns a SemanticsManager formatted with all registered adapters.
  This ensures that the mappings defined in `LatexDSLAdapter.definitions` are loaded.
  """
  mgr = SemanticsManager()
  loader = RegistryLoader(mgr)
  loader.hydrate()
  return mgr


# --- Tests ---


def test_latex_to_torch_architecture_conversion(hydrated_semantics):
  """
  Verifies that a MIDL string converts to valid PyTorch code.
  """
  # 1. Configure Engine (Source: latex_dsl -> Target: torch)
  config = RuntimeConfig(source_framework="latex_dsl", target_framework="torch", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)

  # 2. Run Transpilation
  result = engine.run(LATEX_SOURCE_CONVNET)

  # 3. Validation
  if not result.success:
    pytest.fail(f"Transpilation failed. Errors: {result.errors}")

  code = result.code

  # Debug Output
  print(f"\n[Generated Output]\n{code}")

  # A. Check Imports
  # The 'midl' mock import injected by the parser should be stripped by ImportFixer
  assert "import midl" not in code
  # Torch headers should be injected
  assert "import torch" in code

  # B. Check Class Definition
  # midl.Module -> torch.nn.Module (or aliased variants)
  # Note: ImportFixer and Traits logic might result in 'nn.Module' or 'torch.nn.Module'
  # depending on specific aliasing resolution in the environment.
  # We accept valid PyTorch definitions.
  assert "class ConvNet(" in code
  assert "Module):" in code

  # Check Init
  assert "super().__init__()" in code

  # C. Check Conv2d Mapping (Argument Resolution)
  # The crucial check: did arg_0=1 become in_channels=1 (or positional 1)?
  assert "Conv2d" in code
  # Search for (1, 32, 3) pattern regardless of keyword/positional
  # This covers both explicit keywords (in_channels=1) and normalized positional args (1, 32, 3)
  assert "(1, 32, 3)" in code.replace("in_channels=", "").replace("out_channels=", "").replace("kernel_size=", "")

  # Ensure it was assigned to 'self.conv' as defined in \Attribute{conv}
  assert "self.conv =" in code

  # D. Check Linear Mapping
  assert "Linear" in code
  assert "(128, 10)" in code.replace("in_features=", "").replace("out_features=", "")

  # E. Check Forward Pass Structure
  assert "def forward(self, x):" in code

  # StateOp h1
  assert "h1 = self.conv(x)" in code

  # functional relu
  # The parser emits midl.ReLU, which maps to torch.functional.relu or F.relu
  # Accepting both forms
  assert "h2 = " in code
  assert "relu(h1)" in code

  # Flatten (Functional Op with args)
  # \Op{flat}{Flatten}{h2, arg_0=1} -> torch.flatten(h2, start_dim=1)
  # We check that arg_0 was consumed/mapped or passed through if matching default
  assert "flat =" in code
  assert "flatten(h2," in code
  assert "1)" in code

  # Final Output
  assert "out = self.fc(flat)" in code
  assert "return out" in code


def test_missing_mapping_fails_strict_mode(hydrated_semantics):
  """
  Verifies that if LaTeX uses an unknown operation (e.g. \Attribute{x}{UnknownOp}),
  strict mode catches it.
  """
  bad_source = r"""
    \documentclass{standalone}
    \begin{DefModel}{BadNet}
        \Attribute{x}{UnknownLayer}{arg_0=1}
    \end{DefModel}
    """
  config = RuntimeConfig(source_framework="latex_dsl", target_framework="torch", strict_mode=True)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)

  result = engine.run(bad_source)

  # Should produce code but wrap it in EscapeHatch because 'midl.UnknownLayer' is not in logic
  assert EscapeHatch.START_MARKER in result.code
  assert "midl.UnknownLayer" in result.code


def test_argument_value_mapping(hydrated_semantics):
  """
  Verify that string values in LaTeX config are preserved.
  """
  source = r"""
    \documentclass{standalone}
    \begin{DefModel}{KeywordNet}
        % Using direct key-value pairs supported by parser
        \Attribute{drop}{Dropout}{p=0.5}
        \Input{x}{_}
        \StateOp{y}{drop}{x}{_}
        \Return{y}
    \end{DefModel}
    """

  config = RuntimeConfig(source_framework="latex_dsl", target_framework="torch", strict_mode=False)
  engine = ASTEngine(semantics=hydrated_semantics, config=config)

  result = engine.run(source)
  assert result.success

  # torch.nn.Dropout(p=0.5)
  # Allow for aliasing (nn.Dropout)
  assert "Dropout" in result.code
  # 0.5 can be float or string depending on parser safety, rewriter handles both
  assert "p=0.5" in result.code
