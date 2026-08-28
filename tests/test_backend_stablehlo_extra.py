"""Extra tests for stablehlo backend."""

from typing import Dict, Any, Tuple, List
from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


class DummyStrSemantics:
  """Dummy semantics."""

  def get_definition(self, kind: str) -> Tuple[str, Dict[str, Any]]:
    """Get mock definition.

    Args:
        kind (str): Kind string.

    Returns:
        Tuple[str, Dict[str, Any]]: Definition tuple.
    """
    return kind, {"variants": {"stablehlo": {"api": "stablehlo.dummy"}}}


def test_stablehlo_backend_str_attribute() -> None:
  """Test that stablehlo backend handles string attributes."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="op1", kind="Dummy", metadata={"str_attr": "hello_world", "quoted_attr": '"quoted"'})
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  backend: StableHloBackend = StableHloBackend(semantics=DummyStrSemantics())  # type: ignore
  code: str = backend.compile(graph)
  assert 'str_attr = "hello_world"' in code
  assert 'quoted_attr = "quoted"' in code
