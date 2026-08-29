"""Test suite for the Visual Backends module."""

from ml_switcheroo.core.compiler.backends.visual_backends import LatexBackend, TikzBackend
from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode


def create_sample_graph() -> LogicalGraph:
  """Creates sample graph."""
  graph = LogicalGraph("TestGraph")
  graph.nodes = [
    LogicalNode("in", "Input", {"shape": "[10]"}),
    LogicalNode("l1", "Linear", {"features": "20", "bias": "True"}),
    LogicalNode("func_relu", "func_relu", {"arg1": "1.0"}),
    LogicalNode("out", "Output", {}),
  ]
  graph.edges = [LogicalEdge("in", "l1"), LogicalEdge("l1", "func_relu"), LogicalEdge("func_relu", "out")]
  return graph


def create_disconnected_graph() -> LogicalGraph:
  """Creates disconnected graph."""
  graph = LogicalGraph("DisGraph")
  graph.nodes = [LogicalNode("in", "Input", {}), LogicalNode("out", "Output", {})]
  graph.edges = []
  return graph


def test_tikz_backend_empty() -> None:
  """Verifies the behavior of TikZ backend empty."""
  backend = TikzBackend()
  graph = LogicalGraph("Empty")
  res: str = backend.compile(graph)
  assert "begin{tikzpicture}" in res


def test_tikz_backend_sample() -> None:
  """Verifies the behavior of TikZ backend sample."""
  backend = TikzBackend()
  graph: LogicalGraph = create_sample_graph()
  res: str = backend.compile(graph)
  assert "begin{tikzpicture}" in res
  assert "Linear" in res


def test_tikz_backend_disconnected() -> None:
  """Verifies the behavior of TikZ backend disconnected."""
  backend = TikzBackend()
  graph: LogicalGraph = create_disconnected_graph()
  res: str = backend.compile(graph)
  assert "in" in res


def test_latex_backend_empty() -> None:
  """Verifies the behavior of LaTeX backend empty."""
  backend = LatexBackend()
  graph = LogicalGraph()
  res: str = backend.compile(graph)
  assert "documentclass" in res
  assert "Model" in res


def test_latex_backend_sample() -> None:
  """Verifies the behavior of LaTeX backend sample."""
  backend = LatexBackend()
  graph: LogicalGraph = create_sample_graph()
  res: str = backend.compile(graph)
  assert "documentclass" in res
  assert "TestGraph" in res
  assert "Linear" in res
  assert "Relu" in res


def test_latex_backend_custom() -> None:
  """Verifies the behavior of LaTeX backend custom."""
  backend = LatexBackend()
  graph = LogicalGraph("Custom")
  graph.nodes = [LogicalNode("in", "Input", {}), LogicalNode("output", "Output", {})]
  graph.edges = [LogicalEdge("in", "some.op.Missing"), LogicalEdge("some.op.Missing", "output")]
  res: str = backend.compile(graph)
  assert "Missing" in res


def test_latex_backend_no_out_edges() -> None:
  """Verifies the behavior of LaTeX backend no output edges."""
  backend = LatexBackend()
  graph = LogicalGraph("Custom")
  graph.nodes = [LogicalNode("in", "Input", {}), LogicalNode("some_mod.foo", "some_mod.foo", {})]
  graph.edges = [LogicalEdge("in", "some_mod.foo")]
  res: str = backend.compile(graph)
  assert "ReturnNode" not in res or "last_step" in res
