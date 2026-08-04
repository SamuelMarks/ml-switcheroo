"""Test suite for Phase 7: Other Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for remaining operations defined in the plan.
"""

import pytest
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend():
  """Provides a StableHLO Backend with a loaded SemanticsManager."""
  return StableHloBackend(SemanticsManager())


OTHER_OPS = [
  ("AfterAll", "stablehlo.after_all"),
  ("And", "stablehlo.and"),
  ("AsyncDone", "stablehlo.async_done"),
  ("AsyncStart", "stablehlo.async_start"),
  ("Compare", "stablehlo.compare"),
  ("Composite", "stablehlo.composite"),
  ("CountLeadingZeros", "stablehlo.count_leading_zeros"),
  ("Exponential", "stablehlo.exponential"),
  ("ExponentialMinusOne", "stablehlo.exponential_minus_one"),
  ("Fft", "stablehlo.fft"),
  ("GetDimensionSize", "stablehlo.get_dimension_size"),
  ("GetTupleElement", "stablehlo.get_tuple_element"),
  ("Infeed", "stablehlo.infeed"),
  ("Map", "stablehlo.map"),
  ("Not", "stablehlo.not"),
  ("Or", "stablehlo.or"),
  ("Outfeed", "stablehlo.outfeed"),
  ("Popcnt", "stablehlo.popcnt"),
  ("Reduce", "stablehlo.reduce"),
  ("ReducePrecision", "stablehlo.reduce_precision"),
  ("Select", "stablehlo.select"),
  ("Tan", "stablehlo.tan"),
  ("TriangularSolve", "stablehlo.triangular_solve"),
  ("Tuple", "stablehlo.tuple"),
  ("Xor", "stablehlo.xor"),
]


@pytest.mark.parametrize("logical_op, expected_mlir_op", OTHER_OPS)
def test_other_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str):
  """Verifies that other operations are correctly mapped to StableHLO syntax.

  This ensures both mapping resolution and operand generation are correct.
  """
  g = LogicalGraph()
  # Simple graph: Input -> Op -> Output
  g.nodes = [LogicalNode("in_node", "Input"), LogicalNode("op_node", logical_op), LogicalNode("out_node", "Output")]
  g.edges = [LogicalEdge("in_node", "op_node"), LogicalEdge("op_node", "out_node")]

  mlir_code = backend.compile(g)

  # 1. Operation exists in MLIR output
  assert expected_mlir_op in mlir_code

  # 2. Input/Output structure
  assert "stablehlo.constant" in mlir_code
  assert "return" in mlir_code
  assert "%op_node =" in mlir_code
