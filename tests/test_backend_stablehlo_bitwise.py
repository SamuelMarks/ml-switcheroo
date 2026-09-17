"""Test suite for Bitwise & Logic operations in the StableHLO backend."""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode

BITWISE_OPS: List[Tuple[str, str]] = [
  ("And", "stablehlo.and"),
  ("CountLeadingZeros", "stablehlo.count_leading_zeros"),
  ("Not", "stablehlo.not"),
  ("Or", "stablehlo.or"),
  ("Popcnt", "stablehlo.popcnt"),
  ("ShiftLeft", "stablehlo.shift_left"),
  ("ShiftRightArithmetic", "stablehlo.shift_right_arithmetic"),
  ("ShiftRightLogical", "stablehlo.shift_right_logical"),
  ("Xor", "stablehlo.xor"),
]


class BitwiseSemanticsMock:
  """Mock semantics manager for bitwise ops."""

  def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Mock lookup.

    Args:
        kind (str): Kind string.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    for abstract, stablehlo_api in BITWISE_OPS:
      if kind == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize("abstract_name, expected_api", BITWISE_OPS)
def test_bitwise_operations_backend(abstract_name: str, expected_api: str) -> None:
  """Test generating StableHLO for bitwise operations via backend.

  Args:
      abstract_name: The Abstract operation name.
      expected_api: The expected stablehlo string.
  """
  nodes: List[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="op1", op_type=abstract_name),
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="in1", target="op1"),
    LogicalEdge(source="op1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=BitwiseSemanticsMock())  # type: ignore
  code: str = backend.compile(graph)

  assert expected_api in code
