"""Test suite for static analysis limits and impasse guards in deep learning transpilation.

Verifies detection and handling of paradigms that exceed pure compile-time static AST/CST
analysis as defined in STATIC_ANALYSIS_LIMITS.md:
1. Data-dependent shapes (boolean masking)
2. Value-dependent dynamic control flow (tensor branch conditions)
3. Unbounded data-dependent loops (while loops over tensor state)
4. Imperative state mutations and side effects (PurityScanner & EscapeHatch)
"""

import libcst as cst

from ml_switcheroo.analysis.purity import PurityScanner
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch


def test_purity_scanner_mutation_detection() -> None:
  """Verifies that PurityScanner detects in-place container mutations."""
  tree = cst.parse_module("items.append(tensor)\n")
  scanner = PurityScanner()
  modified = tree.visit(scanner)
  assert EscapeHatch.START_MARKER in modified.code
  assert "append" in modified.code


def test_purity_scanner_io_detection() -> None:
  """Verifies that PurityScanner flags arbitrary I/O side effects."""
  tree = cst.parse_module("print(x)\n")
  scanner = PurityScanner()
  modified = tree.visit(scanner)
  assert EscapeHatch.START_MARKER in modified.code
  assert "print" in modified.code


def test_dynamic_shape_masking_transpilation_safety() -> None:
  """Verifies that data-dependent boolean masking is processed with preservation or warning."""
  code = """
import torch
import torch.nn as nn

class MaskingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        out = self.fc(x)
        mask = out > 0
        filtered = out[mask]
        return filtered
"""
  engine = ASTEngine(source="torch", target="jax", config=RuntimeConfig(strict_mode=False))
  res: ConversionResult = engine.run(code)
  assert res.code is not None
  assert len(res.code.strip()) > 0
  assert "MaskingNet" in res.code


def test_value_dependent_control_flow_handling() -> None:
  """Verifies that value-dependent tensor branching preserves source AST structure."""
  code = """
import torch
import torch.nn as nn

class DynamicRoutingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(10, 5)
        self.e2 = nn.Linear(10, 5)

    def forward(self, x):
        if x.mean() > 0.5:
            return self.e1(x)
        else:
            return self.e2(x)
"""
  engine = ASTEngine(source="torch", target="jax", config=RuntimeConfig(strict_mode=False))
  res: ConversionResult = engine.run(code)
  assert res.code is not None
  assert "if " in res.code
  assert "else:" in res.code


def test_unbounded_while_loop_handling() -> None:
  """Verifies that data-dependent while loops retain loop structure without corruption."""
  code = """
import torch
import torch.nn as nn

class WhileLoopNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        state = x
        while state.sum() < 10.0:
            state = self.fc(state)
        return state
"""
  engine = ASTEngine(source="torch", target="mlx", config=RuntimeConfig(strict_mode=False))
  res: ConversionResult = engine.run(code)
  assert res.code is not None
  assert "while " in res.code
