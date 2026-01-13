"""
Tests for Extra Backends (MLIR, StableHLO).

Verifies that graph-based compilation backends produce valid structural text
even if simplified compared to CST rewrite.
"""

from ml_switcheroo.compiler.backends.extras import MlirBackend, StableHloBackend
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode


def test_mlir_backend_compile_structure():
  """Verify MLIR backend produces module/func structure."""
  g = LogicalGraph()
  g.nodes = [
    LogicalNode(id="in1", kind="Input", metadata={"value": "100"}),
    LogicalNode(id="op1", kind="MyOp", metadata={"attr": "val"}),
    LogicalNode(id="out", kind="Output"),
  ]

  backend = MlirBackend()
  code = backend.compile(g)

  assert "module {" in code
  assert "func.func @main" in code
  assert '%in1 = "sw.constant"() {value = 100}' in code
  assert '%op1 = "sw.op"()' in code
  assert 'type = "MyOp"' in code
  assert 'attr = "val"' in code
  assert '"sw.return"' in code


def test_stablehlo_backend_compile_structure():
  """Verify StableHLO backend produces module/func structure."""
  g = LogicalGraph()
  g.nodes = [LogicalNode(id="in1", kind="Input"), LogicalNode(id="add", kind="Add"), LogicalNode(id="out", kind="Output")]

  backend = StableHloBackend()
  code = backend.compile(g)

  assert "module {" in code
  assert "stablehlo.constant" in code
  assert "stablehlo.custom_call @add" in code
  assert "return" in code
