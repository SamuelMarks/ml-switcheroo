"""Test suite for the StableHLO backend."""

from ml_switcheroo.core.compiler.backends.stablehlo import StableHloBackend
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


class DummySemantics:
  """Test element."""

  def get_definition(self, kind):
    """Test element."""
    if kind == "Add":
      return None, {"variants": {"stablehlo": {"api": "stablehlo.add"}}}
    return None


def test_stablehlo_backend_init():
  """Test element."""
  backend = StableHloBackend(semantics="dummy")
  assert backend.semantics == "dummy"


def test_stablehlo_backend_compile():
  """Test element."""
  nodes = [
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="in2", kind="Input"),
    LogicalNode(id="add1", kind="Add"),
    LogicalNode(id="mul1", kind="Mul"),  # fallback to custom_call
    LogicalNode(id="out1", kind="Output"),
  ]
  edges = [
    LogicalEdge(source="in1", target="add1"),
    LogicalEdge(source="in2", target="add1"),
    LogicalEdge(source="add1", target="mul1"),
    LogicalEdge(source="in2", target="mul1"),
    LogicalEdge(source="mul1", target="out1"),
  ]
  graph = LogicalGraph(nodes=nodes, edges=edges)
  backend = StableHloBackend(semantics=DummySemantics())
  code = backend.compile(graph)

  assert "Graph -> StableHLO compilation output" in code
  assert "stablehlo.constant" in code
  assert "stablehlo.add" in code
  assert "stablehlo.custom_call" in code
  assert '"@mul"' in code
  assert "return" in code


def test_stablehlo_backend_compile_no_semantics():
  """Test element."""
  nodes = [
    LogicalNode(id="add1", kind="Add"),
  ]
  edges = []
  graph = LogicalGraph(nodes=nodes, edges=edges)
  backend = StableHloBackend(semantics=None)
  code = backend.compile(graph)

  assert "stablehlo.custom_call" in code
  assert '"@add"' in code
