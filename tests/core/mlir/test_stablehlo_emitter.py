"""Test suite for the Stablehlo Emitter module."""

import libcst as cst
from ml_switcheroo.core.mlir.stablehlo_emitter import StableHloEmitter
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  """Mock Semantics class for testing purposes."""

  def __init__(self):
    """Initializes the MockSemantics instance."""
    self.data = {}
    self._reverse_index = {}
    abs_def = {"variants": {"torch": {"api": "torch.abs"}, "stablehlo": {"api": "stablehlo.abs"}}}
    self._inject("Abs", "torch.abs", abs_def)
    add_def = {"variants": {"torch": {"api": "torch.add"}, "stablehlo": {"api": "stablehlo.add"}}}
    self._inject("Add", "torch.add", add_def)

  def _inject(self, name, api, defn):
    """Mock implementation of  inject."""
    self._reverse_index[api] = (name, defn)

  def get_definition(self, name):
    """Mock implementation of get definition."""
    return self._reverse_index.get(name)


def emit_code(code: str) -> str:
  """Emits code."""
  tree = cst.parse_module(code.strip())
  semantics = MockSemantics()
  emitter = StableHloEmitter(semantics)
  mlir_node = emitter.convert(tree)
  return mlir_node.to_text()


def test_module_structure():
  """Verifies the behavior of module structure."""
  code = "\nclass MyNet:\n    pass\n"
  mlir = emit_code(code)
  assert 'module {sym_name = "MyNet"}' in mlir


def test_func_structure_and_types():
  """Verifies the behavior of function structure and types."""
  code = "\ndef forward(x: Tensor, i: int) -> float:\n    return x\n"
  mlir = emit_code(code)
  assert "func.func" in mlir
  assert 'sym_name = "forward"' in mlir
  assert "tensor<*xf32>" in mlir
  assert "i32" in mlir
  assert ") -> f32" in mlir or ": f32" in mlir


def test_stablehlo_op_resolution():
  """Verifies the behavior of StableHLO op resolution."""
  code = "y = torch.abs(x)"
  mlir = emit_code(code)
  assert "sw.op" not in mlir
  assert "stablehlo.abs" in mlir
  assert ": tensor<*xf32>" in mlir


def test_unknown_op_fallback():
  """Verifies the behavior of unknown op fallback."""
  code = "y = torch.unknown(x)"
  mlir = emit_code(code)
  assert "stablehlo" not in mlir
  assert "sw.op" in mlir
  assert 'type = "torch.unknown"' in mlir


def test_return_statement():
  """Verifies the behavior of return statement."""
  code = "return x"
  mlir = emit_code(code)
  assert "func.return" in mlir


def test_expression_chaining():
  """Verifies the behavior of expression chaining."""
  code = "y = torch.add(torch.abs(x), x)"
  mlir = emit_code(code)
  assert "stablehlo.abs" in mlir
  assert "stablehlo.add" in mlir
  assert mlir.count("=") >= 2
