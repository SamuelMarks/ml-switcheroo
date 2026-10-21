"""Test suite for Complex & Linalg operations in the StableHLO backend."""

from pydantic import JsonValue

import pytest

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode

COMPLEX_LINALG_OPS: list[tuple[str, str]] = [
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

  def get_definition(self, kind: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Get mock definition for kind.

    Args:
        kind (str): Kind string.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: Definition tuple.
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
  backend: StableHloBackend = getattr(
    {
      "v": getattr({"v": StableHloBackend}, "__getitem__")("v")(
        semantics=getattr({"v": getattr({"v": ComplexLinalgSemanticsMock}, "__getitem__")("v")()}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code: str = backend.compile(graph)
  assert expected_api in code
