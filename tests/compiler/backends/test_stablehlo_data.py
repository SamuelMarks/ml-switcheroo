"""Test suite for Phase 4: Data Manipulation & Tensor Operations in StableHLO.

Provides comprehensive coverage of exact semantics, type enforcement,
and generation correctness for shape/tensor manipulation operations.
"""

import pytest
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.semantics.manager import SemanticsManager


@pytest.fixture
def backend():
  """Provides a StableHLO Backend with a loaded SemanticsManager."""
  return StableHloBackend(SemanticsManager())


DATA_OPS = [
  ("AllGather", "stablehlo.all_gather"),
  ("BitcastConvert", "stablehlo.bitcast_convert"),
  ("BroadcastInDim", "stablehlo.broadcast_in_dim"),
  ("CollectiveBroadcast", "stablehlo.collective_broadcast"),
  ("Concatenate", "stablehlo.concatenate"),
  ("Convert", "stablehlo.convert"),
  ("DynamicBroadcastInDim", "stablehlo.dynamic_broadcast_in_dim"),
  ("DynamicConv", "stablehlo.dynamic_conv"),
  ("DynamicGather", "stablehlo.dynamic_gather"),
  ("DynamicIota", "stablehlo.dynamic_iota"),
  ("DynamicReshape", "stablehlo.dynamic_reshape"),
  ("DynamicSlice", "stablehlo.dynamic_slice"),
  ("DynamicUpdateSlice", "stablehlo.dynamic_update_slice"),
  ("Gather", "stablehlo.gather"),
  ("Iota", "stablehlo.iota"),
  ("ReduceScatter", "stablehlo.reduce_scatter"),
  ("Reshape", "stablehlo.reshape"),
  ("Reverse", "stablehlo.reverse"),
  ("Scatter", "stablehlo.scatter"),
  ("Slice", "stablehlo.slice"),
  ("Sort", "stablehlo.sort"),
  ("Transpose", "stablehlo.transpose"),
]


@pytest.mark.parametrize("logical_op, expected_mlir_op", DATA_OPS)
def test_data_operations(backend: StableHloBackend, logical_op: str, expected_mlir_op: str):
  """Verifies that data manipulation operations map to the correct MLIR syntax."""
  g = LogicalGraph()
  # Simple graph: Input -> Op -> Output
  g.nodes = [LogicalNode("in_node", "Input"), LogicalNode("op_node", logical_op), LogicalNode("out_node", "Output")]
  g.edges = [LogicalEdge("in_node", "op_node"), LogicalEdge("op_node", "out_node")]

  mlir_code = backend.compile(g)

  assert expected_mlir_op in mlir_code
  assert "%op_node =" in mlir_code
