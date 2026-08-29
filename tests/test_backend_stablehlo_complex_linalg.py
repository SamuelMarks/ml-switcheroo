"""Test suite for Complex & Linalg operations in the StableHLO backend."""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode

COMPLEX_LINALG_OPS: List[Tuple[str, str]] = [
  ("Complex", "stablehlo.complex"),
  ("Imag", "stablehlo.imag"),
  ("Real", "stablehlo.real"),
  ("Cholesky", "stablehlo.cholesky"),
  ("DotGeneral", "stablehlo.dot_general"),
  ("TriangularSolve", "stablehlo.triangular_solve"),
  ("Fft", "stablehlo.fft"),
]


class ComplexLinalgSemanticsMock:
  """Mock semantics."""

  def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Get mock definition for kind.

    Args:
        kind (str): Kind string.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition tuple.
    """
    for abstract, stablehlo_api in COMPLEX_LINALG_OPS:
      if kind == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize("abstract_name, expected_api", COMPLEX_LINALG_OPS)
def test_complex_linalg_operations_backend(abstract_name: str, expected_api: str) -> None:
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
  backend: StableHloBackend = StableHloBackend(semantics=ComplexLinalgSemanticsMock())  # type: ignore
  code: str = backend.compile(graph)
  assert expected_api in code
