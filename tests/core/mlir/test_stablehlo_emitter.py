"""
Tests for the StableHLO Emitter.

Verifies:
1.  Structure: Python Classes map to `module`, Functions to `func.func`.
2.  Semantics: Known math ops (torch.abs) map to `stablehlo.abs`.
3.  Type conversion: Python hints to MLIR types.
4.  Fallback: Unknown ops remain `sw.op`.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock semantics manager to provide stablehlo mappings."""

  def __init__(self):
    # Skip super init
    self.data = {}
    self._reverse_index = {}

    # Define Abs
    abs_def = {
      "variants": {
        "torch": {"api": "torch.abs"},
        "stablehlo": {"api": "stablehlo.abs"},
      }
    }
    self._inject("Abs", "torch.abs", abs_def)

    # Define Add
    add_def = {
      "variants": {
        "torch": {"api": "torch.add"},
        "stablehlo": {"api": "stablehlo.add"},
      }
    }
    self._inject("Add", "torch.add", add_def)

  def _inject(self, name, api, defn):
    self._reverse_index[api] = (name, defn)

  def get_definition(self, name):
    return self._reverse_index.get(name)


def emit_code(code: str) -> str:
  """Helper to parse and emit."""
  tree = cst.parse_module(code.strip())
  semantics = MockSemantics()
  emitter = StableHloEmitter(semantics)
  mlir_node = emitter.convert(tree)
  return mlir_node.to_text()


def test_module_structure():
  """Verify class -> module translation."""
  code = """
class MyNet:
    pass
"""
  mlir = emit_code(code)
  assert 'module {sym_name = "MyNet"}' in mlir


def test_func_structure_and_types():
  """Verify function signature translation and type mapping."""
  code = """
def forward(x: Tensor, i: int) -> float:
    return x
"""
  mlir = emit_code(code)
  assert "func.func" in mlir
  assert 'sym_name = "forward"' in mlir
  # Check args
  # %x_0: tensor<*xf32>, %i_1: i32
  assert "tensor<*xf32>" in mlir
  assert "i32" in mlir
  # Check return type
  assert ") -> f32" in mlir or ": f32" in mlir  # Depending on node format


def test_stablehlo_op_resolution():
  """
  Scenario: torch.abs(x) -> stablehlo.abs(%x)
  """
  code = "y = torch.abs(x)"
  mlir = emit_code(code)

  # Should NOT contain sw.op
  assert "sw.op" not in mlir
  # Should contain stablehlo.abs
  assert "stablehlo.abs" in mlir
  # Should have result type
  assert ": tensor<*xf32>" in mlir


def test_unknown_op_fallback():
  """
  Scenario: torch.unknown(x) (No semantic mapping) -> sw.op
  """
  code = "y = torch.unknown(x)"
  mlir = emit_code(code)

  assert "stablehlo" not in mlir
  # Fallback to base emitter behavior
  assert "sw.op" in mlir
  assert 'type = "torch.unknown"' in mlir


def test_return_statement():
  """Verify func.return usage."""
  code = "return x"
  mlir = emit_code(code)
  assert "func.return" in mlir


def test_expression_chaining():
  """Verify nested expression resolution."""
  code = "y = torch.add(torch.abs(x), x)"
  mlir = emit_code(code)

  assert "stablehlo.abs" in mlir
  assert "stablehlo.add" in mlir
  # Verify SSA nesting structure roughly by checking intermediate assignment
  # The base emitter creates SSA for nested calls
  assert mlir.count("=") >= 2  # One for abs, one for add
