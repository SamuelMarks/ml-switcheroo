"""Test suite for CF, Comm, RNG, and Misc operations in the StableHLO backend."""

import pytest
from typing import List, Tuple, Dict, Any, Optional
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge

OPS: List[Tuple[str, str]] = [
  # CF
  ("Case", "stablehlo.case"),
  ("If", "stablehlo.if"),
  ("Map", "stablehlo.map"),
  ("While", "stablehlo.while"),
  # Comm
  ("AfterAll", "stablehlo.after_all"),
  ("AllGather", "stablehlo.all_gather"),
  ("AllReduce", "stablehlo.all_reduce"),
  ("AllToAll", "stablehlo.all_to_all"),
  ("AsyncDone", "stablehlo.async_done"),
  ("AsyncStart", "stablehlo.async_start"),
  ("CollectiveBroadcast", "stablehlo.collective_broadcast"),
  ("CollectivePermute", "stablehlo.collective_permute"),
  ("Infeed", "stablehlo.infeed"),
  ("Outfeed", "stablehlo.outfeed"),
  ("PartitionId", "stablehlo.partition_id"),
  ("Recv", "stablehlo.recv"),
  ("Reduce", "stablehlo.reduce"),
  ("ReducePrecision", "stablehlo.reduce_precision"),
  ("ReduceScatter", "stablehlo.reduce_scatter"),
  ("ReduceWindow", "stablehlo.reduce_window"),
  ("ReplicaId", "stablehlo.replica_id"),
  ("Send", "stablehlo.send"),
  # RNG
  ("Rng", "stablehlo.rng"),
  ("RngBitGenerator", "stablehlo.rng_bit_generator"),
  # Misc
  ("Clamp", "stablehlo.clamp"),
  ("Compare", "stablehlo.compare"),
  ("Composite", "stablehlo.composite"),
  ("Constant", "stablehlo.constant"),
  ("CustomCall", "stablehlo.custom_call"),
  ("IsFinite", "stablehlo.is_finite"),
  ("OptimizationBarrier", "stablehlo.optimization_barrier"),
  ("Select", "stablehlo.select"),
  ("SelectAndScatter", "stablehlo.select_and_scatter"),
  ("UniformDequantize", "stablehlo.uniform_dequantize"),
  ("UniformQuantize", "stablehlo.uniform_quantize"),
]


class RestSemanticsMock:
  """Mock semantics."""

  def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Get mock definition for kind.

    Args:
        kind (str): Kind string.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    for abstract, stablehlo_api in OPS:
      if kind == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize("abstract_name, expected_api", OPS)
def test_rest_operations_backend(abstract_name: str, expected_api: str) -> None:
  """Test generating StableHLO via backend.

  Args:
      abstract_name (str): Abstract name.
      expected_api (str): Expected API string.
  """
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="op1", kind=abstract_name),
    LogicalNode(id="out1", kind="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="in1", target="op1"),
    LogicalEdge(source="op1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=RestSemanticsMock())  # type: ignore
  code: str = backend.compile(graph)
  assert expected_api in code
