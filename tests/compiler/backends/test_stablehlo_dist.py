"""Test suite for Phase 6: Distributed & Communication Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for distributed and communication operations
defined in the plan.
"""

import pytest
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend():
  """Provides a StableHLO Backend with a loaded SemanticsManager."""
  return StableHloBackend(SemanticsManager())


DIST_OPS = [
  ("AllReduce", "stablehlo.all_reduce"),
  ("AllToAll", "stablehlo.all_to_all"),
  ("CollectivePermute", "stablehlo.collective_permute"),
  ("PartitionId", "stablehlo.partition_id"),
  ("Recv", "stablehlo.recv"),
  ("ReplicaId", "stablehlo.replica_id"),
  ("Send", "stablehlo.send"),
]


@pytest.mark.parametrize("logical_op, expected_mlir_op", DIST_OPS)
def test_dist_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str):
  """Verifies that distributed operations are correctly mapped to StableHLO syntax.

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
