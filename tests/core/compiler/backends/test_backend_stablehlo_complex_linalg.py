"""Test suite for Complex & Linalg operations in the StableHLO backend."""

import pytest
from pydantic import JsonValue

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.semantics.manager import SemanticsManager

COMPLEX_LINALG_OPS: list[tuple[str, str]] = [
  ("Complex", "stablehlo.complex"),
  ("Imag", "stablehlo.imag"),
  ("Real", "stablehlo.real"),
  ("Cholesky", "stablehlo.cholesky"),
  ("DotGeneral", "stablehlo.dot_general"),
  ("TriangularSolve", "stablehlo.triangular_solve"),
  ("Fft", "stablehlo.fft"),
]


class ComplexLinalgSemanticsMock(SemanticsManager):
  """Mock semantics."""

  def __init__(self) -> None:
    """Initialize empty semantics manager without disk hydration."""
    super().__init__(load_defaults=False)

  def get_definition(self, api_name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Get mock definition for kind.

    Args:
        api_name: Kind string.

    Returns:
        Definition tuple if matched, otherwise None.
    """
    for abstract, stablehlo_api in COMPLEX_LINALG_OPS:
      if api_name == abstract:
        return abstract, {"variants": {"stablehlo": {"api": stablehlo_api}}}
    return None


@pytest.mark.parametrize(("abstract_name", "expected_api"), COMPLEX_LINALG_OPS)
def test_complex_linalg_operations_backend(abstract_name: str, expected_api: str) -> None:
  """Test generating StableHLO via backend.

  Args:
      abstract_name (str): Abstract name.
      expected_api (str): Expected API string.
  """
  nodes: list[LogicalNode] = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="op1", op_type=abstract_name),
    LogicalNode(id="out1", op_type="Output"),
  ]
  edges: list[LogicalEdge] = [
    LogicalEdge(source="in1", target="op1"),
    LogicalEdge(source="op1", target="out1"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  backend: StableHloBackend = StableHloBackend(semantics=ComplexLinalgSemanticsMock())
  code: str = backend.compile(graph)
  assert expected_api in code
