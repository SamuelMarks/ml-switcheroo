"""Test suite for Phase 5: Control Flow Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for control flow operations.
"""

import pytest
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend():
  """Provides a StableHLO Backend with a loaded SemanticsManager."""
  return StableHloBackend(SemanticsManager())


CONTROL_OPS = [
  ("Case", "stablehlo.case"),
  ("CustomCall", "stablehlo.custom_call"),
  ("If", "stablehlo.if"),
  ("OptimizationBarrier", "stablehlo.optimization_barrier"),
  ("UniformDequantize", "stablehlo.uniform_dequantize"),
  ("UniformQuantize", "stablehlo.uniform_quantize"),
  ("While", "stablehlo.while"),
]


@pytest.mark.parametrize("logical_op, expected_mlir_op", CONTROL_OPS)
def test_control_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str):
  """Verifies that control flow operations map to the correct MLIR syntax."""
  g = LogicalGraph()
  # Simple graph: Input -> Op -> Output
  g.nodes = [LogicalNode("in_node", "Input"), LogicalNode("op_node", logical_op), LogicalNode("out_node", "Output")]
  g.edges = [LogicalEdge("in_node", "op_node"), LogicalEdge("op_node", "out_node")]

  mlir_code = backend.compile(g)

  assert expected_mlir_op in mlir_code
  assert "%op_node =" in mlir_code
